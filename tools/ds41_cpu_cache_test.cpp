// CPU-only: leased discontiguous expert bytes must preserve the mapped lane's arithmetic.
#include "strata/ds41/expert_lane.hpp"
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    constexpr int H=64, F=64, E=4;
    constexpr size_t matrix=H*F;
    std::array<std::vector<float>,3> mapped;
    std::array<std::array<std::vector<float>,3>,E> copied;
    std::array<std::array<const uint8_t*,3>,E> spans{};
    for (int p=0;p<3;++p) {
        mapped[p].resize(E*matrix);
        for (size_t i=0;i<mapped[p].size();++i) mapped[p][i]=float(int((i*37+p*11)%127)-63)/2048.f;
        for (int e=0;e<E;++e) {
            copied[e][p].assign(mapped[p].begin()+e*matrix,mapped[p].begin()+(e+1)*matrix);
            spans[e][p]=reinterpret_cast<const uint8_t*>(copied[e][p].data());
        }
    }
    strata::ds41::LaneLayer plain;
    plain.f.gu_type=plain.f.d_type=plain.f.gu_act=plain.f.d_act=0;
    plain.f.n_embd=H; plain.f.n_ff=F;
    plain.f.gu_row=H*sizeof(float); plain.f.d_row=F*sizeof(float);
    plain.f.act_bytes=H*sizeof(float); plain.f.h_bytes=F*sizeof(float);
    plain.gate=reinterpret_cast<const uint8_t*>(mapped[0].data());
    plain.up=reinterpret_cast<const uint8_t*>(mapped[1].data());
    plain.down=reinterpret_cast<const uint8_t*>(mapped[2].data());
    auto cached=plain;
    cached.expert_spans=spans.data(); cached.expert_span_count=E;
    // Poison the fallback bases: a cached job must read only its leased spans.
    cached.gate=cached.up=cached.down=nullptr;
    for (int nt : {1,2,8}) for (int threads : {1,3}) for (int top : {3,6}) {
        std::vector<float> x(nt*H), weights(nt*top), expected(nt*H), actual(nt*H);
        std::vector<int32_t> ids(nt*top);
        for (size_t i=0;i<x.size();++i) x[i]=float(int(i%13)-6)/16.f;
        for (size_t i=0;i<ids.size();++i) { ids[i]=i%7==0 ? -1 : int(i%E); weights[i]=float(i%5+1)/13.f; }
        strata::ds41::ExpertLane lane(H,F,threads,false);
        lane.run(plain,x.data(),nt,ids.data(),weights.data(),top,expected.data());
        lane.run(cached,x.data(),nt,ids.data(),weights.data(),top,actual.data());
        assert(!std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)));
        // Compute ready routes first, missing routes later, with a second
        // external source standing in for the canonical GPU cache. Merge once
        // in the original rank order, including repeats and negative routes.
        std::vector<int32_t> hot_ids(ids.size(),-1), cold_ids=ids, external_ids(ids.size(),-1);
        std::vector<int32_t> hot_entries, external_entries;
        for (size_t i=0;i<ids.size();++i) if (ids[i]>=0) {
            if (i%3==0) { hot_ids[i]=ids[i]; hot_entries.push_back(int32_t(i)); cold_ids[i]=-1; }
            else if (i%3==1) { external_ids[i]=ids[i]; external_entries.push_back(int32_t(i)); cold_ids[i]=-1; }
        }
        std::vector<float> hot(hot_entries.size()*H), external(external_entries.size()*H);
        for (int repeat=0;repeat<100;++repeat) {
            if (!hot.empty()) lane.run_raw(cached,x.data(),nt,hot_ids.data(),top,hot.data());
            if (!external.empty()) lane.run_raw(cached,x.data(),nt,external_ids.data(),top,external.data());
            lane.begin(cached,x.data(),nt,cold_ids.data(),weights.data(),top,actual.data());
            lane.finish_with_external2(ids.data(),hot_entries.data(),hot.data(),int(hot_entries.size()),
                external_entries.data(),external.data(),int(external_entries.size()),top);
            assert(!std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)));
        }
        lane.begin(cached,x.data(),nt,ids.data(),weights.data(),top,actual.data());
        lane.wait_pending(); // lease lifetime barrier must not collect or destroy outputs
        lane.finish();
        assert(!std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)));
        lane.begin(cached,x.data(),nt,ids.data(),weights.data(),top,actual.data());
        lane.finish();
        lane.wait_pending(); // already-finished jobs do not run or merge again
        assert(!std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)));
    }
    strata::ds41::ExpertLane lane(H,F,1,false);
    std::vector<float> x(H), y(H), weight{1.f};
    int32_t id=E;
    bool bounds=false;
    try { lane.run(cached,x.data(),1,&id,weight.data(),1,y.data()); }
    catch(const std::out_of_range&) { bounds=true; }
    assert(bounds);
    id=2; spans[2][1]=nullptr;
    bool missing=false;
    try { lane.run(cached,x.data(),1,&id,weight.data(),1,y.data()); }
    catch(const std::invalid_argument&) { missing=true; }
    assert(missing);
    std::cout << "DS41 cached spans: byte parity, hot/cold/external merge, repeats, sync/async leases and bounds passed\n";
}
