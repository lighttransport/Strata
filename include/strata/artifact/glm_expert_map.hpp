#pragma once
// FP8 source expert IDs for a pruned GLM model. An explicitly supplied map must never fall back to identity.
#include "strata/host/glm_expert_pack.h"
#include <algorithm>
#include <cstdint>
#include <istream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::artifact {
struct GlmExpertMap {
    std::map<int, std::vector<int>> layers;
    std::string plan_text;
};

inline GlmExpertMap read_glm_expert_map(std::istream& in, uint64_t experts, int first, int last) {
    if (!experts || experts > 288 || first < STRATA_EP_FIRST_LAYER || last > STRATA_EP_LAST_LAYER || first > last)
        throw std::runtime_error("invalid expert map geometry");
    GlmExpertMap result;
    bool header = false;
    for (std::string line; std::getline(in, line);) {
        std::istringstream ls(line);
        ls >> std::ws;
        if (ls.eof() || ls.peek() == '#') continue;
        if (ls.peek() == 'e') {
            std::string word;
            uint64_t source = 0, target = 0;
            if (!(ls >> word >> source >> target) || word != "experts" || source != 288 || target != experts ||
                header || !result.layers.empty())
                throw std::runtime_error("expert map header must be: experts 288 " + std::to_string(experts));
            ls >> std::ws;
            if (!ls.eof()) throw std::runtime_error("unexpected text in expert map header");
            header = true;
            continue;
        }
        int layer = 0;
        // Pruning plans may also contain layer 45 (MTP); the packer only requires its selected main layers.
        if (!(ls >> layer) || layer < STRATA_EP_FIRST_LAYER || layer > STRATA_EP_LAST_LAYER + 1)
            throw std::runtime_error("invalid expert map layer");
        std::vector<int> ids;
        for (;;) {
            ls >> std::ws;
            if (ls.eof()) break;
            int expert;
            if (!(ls >> expert)) throw std::runtime_error("invalid expert map index");
            ids.push_back(expert);
            if (ids.size() > experts) throw std::runtime_error("too many expert map indices");
        }
        auto sorted = ids;
        std::sort(sorted.begin(), sorted.end());
        if (ids.size() != experts || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() ||
            sorted.front() < 0 || sorted.back() >= 288)
            throw std::runtime_error("expert map layer " + std::to_string(layer) + ": need " +
                                     std::to_string(experts) + " distinct indices below 288");
        if (!result.layers.emplace(layer, std::move(ids)).second)
            throw std::runtime_error("duplicate expert map layer " + std::to_string(layer));
        // Preserve the old resume-plan identity for valid maps: only mapping rows were hashed.
        result.plan_text += line + "\n";
    }
    if (in.bad()) throw std::runtime_error("cannot read expert map");
    for (int layer = first; layer <= last; ++layer)
        if (!result.layers.count(layer)) throw std::runtime_error("expert map lacks layer " + std::to_string(layer));
    return result;
}
}  // namespace strata::artifact
