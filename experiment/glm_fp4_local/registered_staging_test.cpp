// CPU-only fault injection: no transport, model, or GPU required.
#include "tools/glm_fp4_registered_staging.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
struct ucomm { int id; };
struct ucomm_mr { unsigned char* bytes; size_t size; };
static int calls=0, fail_at=0, live=0, released=0;
extern "C" ucomm_status ucomm_mr_reg(ucomm_t*, void* bytes, size_t size, ucomm_mr_t** out) {
    if (++calls==fail_at) return UCOMM_ERR_NOMEM;
    *out=new ucomm_mr{static_cast<unsigned char*>(bytes),size}; ++live;
    return UCOMM_OK;
}
extern "C" void ucomm_mr_dereg(ucomm_mr_t* mr) {
    // ASan detects freeing backing storage before deregistration.
    std::memset(mr->bytes,0xa5,mr->size);
    delete mr; --live; ++released;
}
extern "C" const char* ucomm_strerror(ucomm_status) { return "injected registration failure"; }
template<class E,class F> void rejects(F f) {
    bool caught=false; try { f(); } catch(const E&) { caught=true; } assert(caught);
}
int main() {
    using strata::experiment::RegisteredStaging;
    ucomm a{1},b{2};
    {
        RegisteredStaging buffers;
        rejects<std::invalid_argument>([&]{buffers.add(nullptr,32);});
        rejects<std::invalid_argument>([&]{buffers.add(&a,0);});
        assert(calls==0 && buffers.find(&a)==nullptr);
        for(int failure: {1,2}) {
            calls=0;fail_at=failure;
            rejects<std::runtime_error>([&]{buffers.add(&a,32);});
            assert(live==0 && buffers.size()==0 && buffers.find(&a)==nullptr);
        }
        fail_at=0;
        buffers.add(&a,32);buffers.add(&b,64);
        assert(live==4 && buffers.size()==2);
        auto* first=buffers.find(&a);
        assert(first && first->context()==&a && first->tx(32)!=first->rx(32));
        std::memset(first->tx(32),7,32);std::memset(first->rx(32),9,32);
        assert(first->tx(0)[31]==7 && first->rx(0)[31]==9);
        rejects<std::length_error>([&]{first->tx(33);});
        rejects<std::length_error>([&]{first->rx(33);});
        rejects<std::invalid_argument>([&]{buffers.add(&a,128);});
        assert(live==4 && buffers.find(&a)==first);
        buffers.clear();buffers.clear();assert(live==0 && buffers.size()==0);
        buffers.add(&b,16);assert(live==2);
    }
    assert(live==0 && released==7);
    std::puts("PASS registered staging bounds, isolation, fault cleanup, and backing lifetime");
}
