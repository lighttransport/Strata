#include "strata/spec/layer_pipeline.hpp"
#include <array>
#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

// A recurrent head and a tail with a four-token open pool and an 8192-entry
// ring. Rolled-back slots deliberately retain stale values, as GPU memory does.
struct Backend {
    struct State { int position; uint64_t recurrent; std::array<uint64_t,4> open; };
    struct Hop { uint64_t generation=0, value=0; bool live=false; };
    State head{}, saved{}, tail{};
    std::array<Hop,2> hops;
    std::array<uint64_t,8192> ring{};
    bool miss;
    int draft_calls=0, head_calls=0, fail_at=-1, window=0, policy=0;
    int head_routes=0,saved_routes=0,tail_routes=0;
    explicit Backend(int pos,bool misses,int w=0):miss(misses),window(w) {head.position=tail.position=pos;}
    static void step(State &s,int token) {
        s.recurrent=s.recurrent*6364136223846793005ULL+unsigned(token)+s.position;
        s.open[s.position%4]=s.recurrent;
        ++s.position;
    }
    static int next(int token) {return (token*13+7)%113;}
    std::future<void> start_head(int token,int slot,uint64_t gen) {
        if(hops[slot].live)throw std::logic_error("overwrite live slot");
        hops[slot].live=true;hops[slot].generation=gen;
        return std::async(std::launch::async,[=,this]{
            ++head_calls;
            ++head_routes;step(head,token+policy*131);ring[(head.position-1)%8192]=head.recurrent;
            hops[slot].value=head.recurrent;
            if(head_calls==fail_at)throw std::runtime_error("forced head failure");
        });
    }
    void save_head(){saved=head;saved_routes=head_routes;}
    void restore_head(){head=saved;head_routes=saved_routes;}
    bool window_due(uint64_t consumed)const{return window && consumed%(unsigned)window==0;}
    void rebalance(){
        if(head_routes!=tail_routes||head.position!=tail.position||head.recurrent!=tail.recurrent)
            throw std::logic_error("policy updated outside confirmed frontier");
        head_routes=tail_routes=0;++policy;
    }
    void release(int slot,uint64_t gen){
        if(!hops[slot].live||hops[slot].generation!=gen)throw std::logic_error("stale hop");
        hops[slot].live=false;
    }
    int draft(int token){++draft_calls;return next(token)+(miss&&draft_calls%2?1:0);}
    std::future<void> start_draft(int token,int &out){return std::async(std::launch::async,[&,token]{out=draft(token);});}
    int confirm(int token,int slot,uint64_t gen){
        if(!hops[slot].live||hops[slot].generation!=gen)throw std::logic_error("unpublished hop");
        ++tail_routes;step(tail,token+policy*131);
        if(tail.recurrent!=hops[slot].value)throw std::logic_error("unconfirmed input reached tail");
        release(slot,gen);return next(token);
    }
};
struct PrefillBackend {
    struct Hop{uint64_t generation=0,value=0;bool live=false;};
    std::vector<std::array<Hop,2>> hops;
    std::vector<size_t> positions;
    std::vector<uint64_t> output;
    std::atomic<int> active{0};
    bool fail=false;
    explicit PrefillBackend(size_t stages,size_t panels):hops(stages-1),positions(stages),output(panels){}
    static uint64_t expected(size_t stages,size_t panel){return (stages*(stages+1)/2)*100+stages*panel;}
    std::future<void> start(size_t stage,size_t panel){
        const uint64_t gen=panel+1;const size_t slot=gen&1;
        uint64_t input=0;
        if(stage){auto &h=hops[stage-1][slot];if(!h.live||h.generation!=gen)throw std::logic_error("stale prefill source");input=h.value;}
        if(stage+1<positions.size()){auto &h=hops[stage][slot];if(h.live)throw std::logic_error("prefill slot overwritten");h.live=true;h.generation=gen;}
        return std::async(std::launch::async,[=,this]{
            ++active;
            if(positions[stage]++!=panel){--active;throw std::logic_error("prefill reordered a layer");}
            uint64_t result=input+(stage+1)*100+panel;
            if(stage+1<positions.size())hops[stage][slot].value=result;else output[panel]=result;
            --active;
            if(fail&&stage==1&&panel==1)throw std::runtime_error("forced prefill failure");
        });
    }
    void complete(size_t stage,size_t panel){if(stage)hops[stage-1][(panel+1)&1].live=false;}
};
int main(){
    try {
        for(int pos:{0,2,3,4,254,255,256,8190,8191,8192})for(bool misses:{false,true})for(int count:{1,2,7,17})for(int window:{0,2,7}) {
            Backend b(pos,misses,window);int token=11, emitted=0;
            auto stats=strata::spec::run_layer_pipeline(b,token,[&](int actual){
                token=Backend::next(token);
                if(actual!=token)throw std::runtime_error("wrong output");
                return ++emitted==count;
            });
            if(b.head.position!=b.tail.position||b.head.recurrent!=b.tail.recurrent||b.head.open!=b.tail.open)
                throw std::runtime_error("frontiers differ");
            if(b.head_routes!=b.tail_routes || (window && b.policy!=(count-1)/window))throw std::runtime_error("uncommitted route statistics");
            if(stats.proposed!=(unsigned)count||(!misses&&stats.accepted!=stats.proposed))throw std::runtime_error("wrong accounting");
            for(const auto &h:b.hops)if(h.live)throw std::runtime_error("leaked hop");
        }
        for(int pos:{3,255,8191}) {
            Backend b(pos,true);int emitted=0;
            try {strata::spec::run_layer_pipeline(b,11,[&](int){if(++emitted==4)throw std::runtime_error("cancelled");return false;});}
            catch(const std::runtime_error &e){if(std::string(e.what())!="cancelled")throw;}
            if(b.head.position!=b.tail.position||b.head.recurrent!=b.tail.recurrent||b.head.open!=b.tail.open)throw std::runtime_error("cancel rollback differs");
        }
        for(size_t stages:{2,3,4,16})for(size_t panels:{1,2,3,17}) {
            PrefillBackend b(stages,panels);
            strata::spec::run_layer_prefill(b,panels,stages);
            for(size_t p=0;p<panels;++p)if(b.output[p]!=PrefillBackend::expected(stages,p))throw std::runtime_error("prefill result differs");
            for(const auto &pair:b.hops)for(const auto &h:pair)if(h.live)throw std::runtime_error("prefill slot leaked");
        }
        PrefillBackend failed(4,8);failed.fail=true;
        try{strata::spec::run_layer_prefill(failed,8,4);return 4;}
        catch(const std::runtime_error &e){if(std::string(e.what())!="forced prefill failure")throw;}
        if(failed.active.load())throw std::runtime_error("prefill failure left an active job");
        Backend b(8191,false);b.fail_at=2;
        try {strata::spec::run_layer_pipeline(b,11,[](int){return false;});return 2;}
        catch(const std::runtime_error &e){if(std::string(e.what())!="forced head failure")throw;}
        if(b.head.position!=b.saved.position||b.head.recurrent!=b.saved.recurrent)return 3;
        std::cout<<"layer pipeline: 240 decode traces, 16 prefill traces, cancellation and failures passed\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
