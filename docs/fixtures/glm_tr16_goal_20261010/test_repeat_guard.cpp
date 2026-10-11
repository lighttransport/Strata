#include "strata/spec/bounded_accept.hpp"
#include <cassert>
#include <vector>
using strata::spec::repeats_ngram;
int main() {
    assert(!repeats_ngram({}, 1, 4));
    const std::vector<int> h{1,2,3,4,9,1,2,3};
    assert(repeats_ngram(h, 4, 4));
    assert(!repeats_ngram(h, 5, 4));
    assert(!repeats_ngram(h, 4, 0));
    assert(!repeats_ngram(h, 4, 9));
    assert(repeats_ngram(std::vector<int>{1,2,1}, 2, 2));
    assert(repeats_ngram(std::vector<int>{1,1,1,1}, 1, 4));
    assert(!repeats_ngram(std::vector<int>{1,1,1}, 1, 4));
    assert(repeats_ngram(std::vector<int>{1,2,3,4,5,6,7,8,1,2,3,4,5,6,7}, 8, 8));
    // Differential oracle enumerates every candidate n-gram in emitted history.
    for (unsigned bits=0; bits<1024; ++bits) {
        std::vector<int> history;
        for (int i=0; i<10; ++i) history.push_back((bits>>i)&1);
        for (int n=2; n<=8; ++n) for (int token=0; token<3; ++token) {
            std::vector<int> candidate(history.end()-(n-1), history.end());
            candidate.push_back(token);
            bool expected=false;
            for (int i=0; i+n<=int(history.size()); ++i) {
                bool same=true;
                for (int j=0; j<n; ++j) same &= history[i+j]==candidate[j];
                expected |= same;
            }
            assert(repeats_ngram(history,token,n)==expected);
        }
    }
}
