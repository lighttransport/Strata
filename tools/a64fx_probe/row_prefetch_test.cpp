#include "strata/cpu_runner/row_prefetch.hpp"
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
using strata::cpu_runner::RowPrefetch;
static uint32_t bits(size_t row, size_t word, int generation) {
    constexpr uint32_t edge[] = {0,0x80000000u,0x7fc12345u,0x7fa12345u,1,0x7f800000u};
    return word < 6 ? edge[(word + size_t(generation)) % 6] : uint32_t(row * 977 + word * 131 + size_t(generation) * 65537);
}
int main() {
    int bad = 0;
    for (int workers : {1, 3, 8, 16}) {
        for (int generation = 0; generation < 6; ++generation) {
            constexpr size_t n = 513, width = 39;
            constexpr uint32_t guard = 0xdeadbeefu;
            std::vector<std::array<uint32_t,width>> data(n);
            std::array<std::atomic<int>,n> visits{};
            std::atomic<int> setup{0};
            for (auto &row : data) row.fill(guard);
            // First-use order differs from the immutable sorted row lookup.
            std::vector<size_t> order(n);
            for (size_t j=0;j<n;++j) order[j]=(j*17)%n;
            RowPrefetch p(n,workers,[&](size_t job) {
                size_t row=order[job], len=row%37;
                ++visits[row];
                if (job%13==0) std::this_thread::yield();
                for(size_t w=0;w<len;++w) data[row][w+1]=bits(row,w,generation);
                return true;
            },[&](int){++setup;return true;});
            // Multiple consumers may acquire and use the same row concurrently.
            std::atomic<int> concurrent_bad{0};
            auto consume=[&] {
                for(size_t job=0;job<n;++job) {
                    if(!p.wait(job)){++concurrent_bad;continue;}
                    const size_t row=order[job],len=row%37;
                    for(size_t w=0;w<len;++w) if(data[row][w+1]!=bits(row,w,generation))++concurrent_bad;
                    for(size_t w=len+1;w<width;++w) if(data[row][w]!=guard)++concurrent_bad;
                    if(data[row][0]!=guard)++concurrent_bad;
                }
            };
            std::thread other(consume);consume();other.join();
            bad+=!p.finish()||p.completed()!=n||setup!=workers||concurrent_bad!=0;
            for(auto &v:visits) bad+=v!=1;
            bad+=p.wait(n);
        }
        RowPrefetch empty(0,workers,[](size_t){return false;},[](int){return true;});
        bad+=!empty.finish()||empty.wait(0);
    }
    for(int mode:{0,1,2}) {
        RowPrefetch p(20,1,[&](size_t i){if(i==3){if(mode==1)throw std::runtime_error("read");return false;}return true;},
            [&](int){if(mode==2)throw std::runtime_error("setup");return true;});
        bad+=p.wait(19)||p.finish();
    }
    // Destruction drains readers even if a consumer never waited.
    std::atomic<int> drained{0};
    {RowPrefetch p(100,3,[&](size_t){++drained;return true;},[](int){return true;});}
    bad+=drained!=100;
    bool rejected=false;
    try {RowPrefetch p(1,0,[](size_t){return true;},[](int){return true;});}catch(const std::invalid_argument&){rejected=true;}
    bad+=!rejected;
    std::printf("ROW_PREFETCH_TEST fails=%d\n",bad);
    return bad!=0;
}
