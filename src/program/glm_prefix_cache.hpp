// src/program/glm_prefix_cache.hpp - which saved prefix a GLM serve request resumes from (serve --prefix-cache).
//
// One sequence slot holds `held`: the token ids its MLA rows were written from (prompt plus generated tokens, in
// order).  Entries are checkpoints taken at positions inside `held` (the end of a prompt, a pinned prefix).  A new
// prompt resumes from the latest entry at or below its longest common prefix with `held`, and below its own length
// (at least one prompt token is read, for the first logits).  Restoring position r makes every entry above r stale:
// the rows above r are written again from the new prompt.  Entries are evicted least recently used; pinned entries
// (a shared system prompt) only when every entry is pinned.  Store is the checkpoint storage; it is reused in place.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::glm {

template <class Store>
struct PrefixCache {
    struct Entry {
        int pos = 0;
        bool pinned = false, live = false;
        uint64_t used = 0;
        Store store{};
    };
    std::vector<int> held;
    std::vector<Entry> entries;
    uint64_t clock = 0;

    explicit PrefixCache(size_t capacity = 0) : entries(capacity) {}

    static size_t common_prefix(const std::vector<int> &a, const std::vector<int> &b) {
        const size_t n = std::min(a.size(), b.size());
        size_t i = 0;
        while (i < n && a[i] == b[i]) ++i;
        return i;
    }
    // The entry to resume `prompt` from, or nullptr (start from zero).
    Entry *choose(const std::vector<int> &prompt) {
        if (prompt.empty()) return nullptr;
        const size_t limit = std::min(common_prefix(held, prompt), prompt.size() - 1);
        Entry *best = nullptr;
        for (auto &e : entries)
            if (e.live && e.pos > 0 && size_t(e.pos) <= limit && (!best || e.pos > best->pos)) best = &e;
        if (best) best->used = ++clock;
        return best;
    }
    // After restoring position r: rows above r are rewritten, so are their checkpoints and `held`.
    void truncate(int r) {
        for (auto &e : entries)
            if (e.live && e.pos > r) e.live = false;
        if (held.size() > size_t(r)) held.resize(size_t(r));
    }
    void clear() {
        for (auto &e : entries) e.live = false;
        held.clear();
    }
    // The entry to save position `pos` into: the one already at pos, else a free one, else the least recently used
    // (an unpinned one first).  nullptr when the cache has no entries.
    Entry *slot_for(int pos, bool pinned) {
        if (entries.empty()) return nullptr;
        Entry *pick = nullptr;
        for (auto &e : entries)
            if (e.live && e.pos == pos) pick = &e;
        if (!pick)
            for (auto &e : entries)
                if (!e.live) { pick = &e; break; }
        if (!pick) {
            for (auto &e : entries)
                if (!e.pinned && (!pick || e.used < pick->used)) pick = &e;
            if (!pick)
                for (auto &e : entries)
                    if (!pick || e.used < pick->used) pick = &e;
        }
        pick->pos = pos;
        pick->pinned = pinned;
        pick->live = false;          // live once the caller has saved into it (commit)
        pick->used = ++clock;
        return pick;
    }
    void commit(Entry *e) { if (e) e->live = true; }
};

}  // namespace strata::glm
