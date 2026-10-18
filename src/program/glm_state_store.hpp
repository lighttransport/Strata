// src/program/glm_state_store.hpp - GLM serve state that outlives its slot: a conversation's decoder state up to a
// position (GlmStateImage), kept in host RAM and on disk by the token ids it was computed from (GlmStateStore).
//
// An image holds what a resume needs at position P: the fixed per-sequence buffers (KDA recurrent and convolution
// states, the MLA index rings, the draft block's), the MLA latent rows [0, P) and pooled index rows [0, P / pool) of
// every MLA layer (the draft layer's too), the last target hidden row (MTP continues from it) and the route counts.
// Images are taken at a sequence's current position, or - for a shared prefix inside a prompt - assembled from the
// fixed buffers taken at P during the prefill and the append-only rows copied after it (P a multiple of the index
// pool size, so the index rings hold nothing).
//
// Store keys: (identity, hash of the token ids [0, P), P).  A different model or engine setting is a different
// identity: never a hit.  RAM tier: LRU under a byte cap.  Disk tier: one file per image in a directory, LRU by
// access under a byte cap, written by a background thread (a request never waits for a disk write); a file that does
// not verify (header, sizes, payload hash) is a miss and is removed.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::glm {

struct GlmStateImage {
    int position = 0, mtp_position = 0;      // mtp_position == position: the draft block is synced to it
    std::vector<int> tokens;                 // the ids [0, position) the state was computed from
    std::vector<uint8_t> fixed;              // the fixed buffers, concatenated in the decoder's order
    std::vector<std::vector<uint8_t>> rows;  // per MLA layer (main layers, then the draft): latent rows, pooled rows
    std::vector<float> last_hidden;          // target hidden row position - 1 (empty without MTP capture)
    std::vector<uint64_t> routes;            // route counts, flattened (may be empty)
    std::vector<uint64_t> prefill_routes;    // the prefill's route counts, flattened: they decide expert placement
    size_t bytes() const {
        size_t n = fixed.size() + last_hidden.size() * 4 + (routes.size() + prefill_routes.size()) * 8 + tokens.size() * 4;
        for (const auto &r : rows) n += r.size();
        return n;
    }
    bool draft_synced() const { return mtp_position == position; }
};

// The byte form of an image (the disk payload): little-endian, length-prefixed sections.
std::vector<uint8_t> serialize_image(const GlmStateImage &image);
bool deserialize_image(const uint8_t *data, size_t size, GlmStateImage &image);

class GlmStateStore {
public:
    struct Options {
        size_t ram_bytes = 0;            // 0: no RAM tier
        std::string disk_dir;            // empty: no disk tier
        uint64_t disk_bytes = 0;
        uint64_t identity_model = 0, identity_config = 0;
    };
    explicit GlmStateStore(Options options);
    ~GlmStateStore();
    GlmStateStore(const GlmStateStore &) = delete;
    GlmStateStore &operator=(const GlmStateStore &) = delete;

    static uint64_t prefix_hash(const int *tokens, size_t n);
    // The longest stored prefix of `prompt` among `candidates` (positions, any order) and at most prompt.size() - 1:
    // the image, or nullptr.  `tier` says where it came from ("ram" or "disk").
    std::shared_ptr<const GlmStateImage> find(const std::vector<int> &prompt, std::vector<int> candidates,
                                              std::string *tier = nullptr);
    // Keep an image (RAM tier now, disk tier in the background).  Images whose prefix is already stored are skipped.
    void put(std::shared_ptr<const GlmStateImage> image);
    bool contains(const std::vector<int> &tokens, int position);
    void flush();                        // wait until queued disk writes are done (tests, shutdown)
    struct Stats { uint64_t ram_bytes = 0, disk_bytes = 0, ram_entries = 0, disk_entries = 0,
                   ram_hits = 0, disk_hits = 0, misses = 0, written = 0, write_errors = 0; };
    Stats stats();
    // The file the disk tier keeps an image in (exposed for tests).
    std::string file_for(uint64_t hash, int position) const;
    // Explicit session files (SAVE / RESTORE): one image, the same container, any path.
    static bool write_file(const std::string &path, const GlmStateImage &image, uint64_t model, uint64_t config,
                           std::string &error);
    static bool read_file(const std::string &path, uint64_t model, uint64_t config, GlmStateImage &image,
                          std::string &error, uint64_t max_bytes = uint64_t(64) << 30);

private:
    struct Key {
        uint64_t hash; int position;
        bool operator<(const Key &o) const { return hash != o.hash ? hash < o.hash : position < o.position; }
    };
    std::shared_ptr<const GlmStateImage> load_disk(const Key &key);
    void evict_ram();
    void evict_disk();
    void writer();

    Options options_;
    std::mutex mutex_;
    std::list<Key> ram_lru_;                                        // front: most recent
    std::map<Key, std::pair<std::shared_ptr<const GlmStateImage>, std::list<Key>::iterator>> ram_;
    std::map<Key, std::pair<uint64_t, uint64_t>> disk_;            // key -> (file bytes, last use tick)
    uint64_t tick_ = 0, ram_used_ = 0, disk_used_ = 0;
    Stats stats_;
    std::deque<std::shared_ptr<const GlmStateImage>> queue_;
    std::condition_variable queue_cv_, idle_cv_;
    bool stop_ = false, writing_ = false;
    std::thread thread_;
};

}  // namespace strata::glm
