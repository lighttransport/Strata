#include "strata/spec/bounded_accept.hpp"
#include <array>
#include <stdexcept>
#include <limits>
int main() {
    using namespace strata::spec;
    auto require=[](bool v){if(!v)throw std::runtime_error("bounded accept contract failed");};
    std::array<float,4> logits{3,2.8f,2.8f,1};
    std::array<int,1> stop{3};
    BoundedAcceptPolicy p;
    require(bounded_accept(logits,0,0,p,0,stop).accepted);
    require(!bounded_accept(logits,1,0,p,0,stop).accepted);
    p.margin=.3f;p.top_k=2;
    require(bounded_accept(logits,1,0,p,0,stop).soft);
    require(!bounded_accept(logits,2,0,p,0,stop).accepted); // lower ID wins tied rank
    require(!bounded_accept(logits,1,0,p,1,stop).accepted); // per-window cap
    require(!bounded_accept(logits,1,3,p,0,stop).accepted);
    require(!bounded_accept(logits,-1,0,p,0,stop).accepted);
    logits[2]=std::numeric_limits<float>::quiet_NaN();
    require(!bounded_accept(logits,1,0,p,0,stop).accepted);
    std::array<int,5> history{1,2,3,1,2};
    require(repeats_ngram(history,3,3));
    require(!repeats_ngram(history,4,3));
    require(!repeats_ngram(history,3,6));
}
