#include "tools/glm_fp4_service_lane.hpp"
#include <cstdio>
int main() {
    int checks=0;
    ServiceLane a,b;
    for(int i=0;i<1000;++i) {
        auto x=a.submit([i]{return std::vector<float>{float(i),float(i+1)};});
        auto y=b.submit([i]{return std::vector<float>{float(-i)};});
        auto xx=x.get(),yy=y.get();
        if(xx!=std::vector<float>{float(i),float(i+1)}||yy!=std::vector<float>{float(-i)})return 1;
        checks+=2;
    }
    auto fail=a.submit([]()->std::vector<float>{throw std::runtime_error("expected");});
    try {fail.get();return 2;}catch(const std::runtime_error&) {++checks;}
    if(a.submit([]{return std::vector<float>{42};}).get()[0]!=42)return 3;
    std::future<std::vector<float>> drained;
    {ServiceLane transient;drained=transient.submit([]{return std::vector<float>{17};});}
    if(drained.get()[0]!=17)return 4;
    std::promise<void> started,release;
    auto gate=release.get_future();
    auto held=a.submit([&]{started.set_value();gate.wait();return std::vector<float>{1};});
    started.get_future().wait();
    auto queued=a.submit([]{return std::vector<float>{2};});
    bool rejected=false;
    try {a.submit([]{return std::vector<float>{3};});}catch(const std::logic_error&) {rejected=true;}
    release.set_value();
    if(!rejected||held.get()[0]!=1||queued.get()[0]!=2)return 5;
    std::printf("PASS %d result/exception checks; post-exception reuse and destruction drain and full-queue rejection passed\n",checks);
}
