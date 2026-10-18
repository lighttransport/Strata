// src/program/glm_state_store.cpp - see glm_state_store.hpp.
#include "glm_state_store.hpp"

#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::glm {
namespace {

constexpr char kMagic[8] = {'G', 'L', 'M', 'S', 'T', 'A', 'T', '1'};
constexpr uint32_t kVersion = 2;
// header: magic, version, header bytes, model fp, config fp, position, payload bytes, payload hash, header hash
struct Header {
    char magic[8];
    uint32_t version, header_bytes;
    uint64_t model, config;
    int64_t position;
    uint64_t payload_bytes, payload_hash, header_hash;
};
static_assert(sizeof(Header) == 64, "GLM state header layout");
constexpr uint64_t kSeed = 0x474c4d5354415445ull;   // "GLMSTATE"

template <class T> void put(std::vector<uint8_t> &out, const T &v) {
    const auto *p = reinterpret_cast<const uint8_t *>(&v);
    out.insert(out.end(), p, p + sizeof v);
}
void put_bytes(std::vector<uint8_t> &out, const void *data, uint64_t n) {
    put(out, n);
    const auto *p = static_cast<const uint8_t *>(data);
    out.insert(out.end(), p, p + n);
}
struct Reader {
    const uint8_t *p; size_t left;
    template <class T> bool get(T &v) {
        if (left < sizeof v) return false;
        std::memcpy(&v, p, sizeof v); p += sizeof v; left -= sizeof v; return true;
    }
    bool bytes(std::vector<uint8_t> &v) {
        uint64_t n;
        if (!get(n) || n > left) return false;
        v.assign(p, p + n); p += n; left -= n; return true;
    }
    template <class T> bool array(std::vector<T> &v) {
        uint64_t n;
        if (!get(n) || n % sizeof(T) || n > left) return false;
        v.resize(n / sizeof(T));
        std::memcpy(v.data(), p, n); p += n; left -= n; return true;
    }
};

bool write_all(int fd, const void *data, size_t n) {
    const auto *p = static_cast<const char *>(data);
    while (n) {
        const ssize_t w = ::write(fd, p, n);
        if (w < 0) { if (errno == EINTR) continue; return false; }
        p += w; n -= size_t(w);
    }
    return true;
}
bool read_all(int fd, void *data, size_t n) {
    auto *p = static_cast<char *>(data);
    while (n) {
        const ssize_t r = ::read(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        p += r; n -= size_t(r);
    }
    return true;
}
std::string hex(uint64_t v) {
    char s[17];
    std::snprintf(s, sizeof s, "%016llx", (unsigned long long)v);
    return s;
}

}  // namespace

std::vector<uint8_t> serialize_image(const GlmStateImage &image) {
    std::vector<uint8_t> out;
    out.reserve(image.bytes() + 256);
    put(out, int64_t(image.position));
    put(out, int64_t(image.mtp_position));
    put_bytes(out, image.tokens.data(), image.tokens.size() * sizeof(int));
    put_bytes(out, image.fixed.data(), image.fixed.size());
    put(out, uint64_t(image.rows.size()));
    for (const auto &r : image.rows) put_bytes(out, r.data(), r.size());
    put_bytes(out, image.last_hidden.data(), image.last_hidden.size() * 4);
    put_bytes(out, image.routes.data(), image.routes.size() * 8);
    put_bytes(out, image.prefill_routes.data(), image.prefill_routes.size() * 8);
    return out;
}

bool deserialize_image(const uint8_t *data, size_t size, GlmStateImage &image) {
    Reader r{data, size};
    int64_t position, mtp;
    uint64_t layers;
    if (!r.get(position) || !r.get(mtp) || position < 0 || mtp < 0 || mtp > position) return false;
    image.position = int(position); image.mtp_position = int(mtp);
    if (!r.array(image.tokens) || image.tokens.size() != size_t(position)) return false;
    if (!r.bytes(image.fixed) || !r.get(layers) || layers > 4096) return false;
    image.rows.resize(layers);
    for (auto &row : image.rows)
        if (!r.bytes(row)) return false;
    if (!r.array(image.last_hidden) || !r.array(image.routes) || !r.array(image.prefill_routes)) return false;
    return r.left == 0;
}

bool GlmStateStore::write_file(const std::string &path, const GlmStateImage &image, uint64_t model, uint64_t config,
                               std::string &error) {
    const auto payload = serialize_image(image);
    Header h{};
    std::memcpy(h.magic, kMagic, 8);
    h.version = kVersion; h.header_bytes = sizeof h; h.model = model; h.config = config;
    h.position = image.position; h.payload_bytes = payload.size();
    h.payload_hash = core::session_hash64(payload.data(), payload.size(), kSeed);
    h.header_hash = core::session_hash64(&h, offsetof(Header, header_hash), kSeed);
    const std::filesystem::path target(path);
    const auto temp = (target.parent_path() / ("." + target.filename().string() + ".tmp")).string();
    const int fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) { error = "cannot create " + temp + ": " + std::strerror(errno); return false; }
    const bool ok = write_all(fd, &h, sizeof h) && write_all(fd, payload.data(), payload.size()) && ::fsync(fd) == 0;
    const int saved = errno;
    ::close(fd);
    if (!ok) { ::unlink(temp.c_str()); error = std::string("write failed: ") + std::strerror(saved); return false; }
    if (::rename(temp.c_str(), path.c_str()) != 0) {
        error = std::string("rename failed: ") + std::strerror(errno); ::unlink(temp.c_str()); return false;
    }
    const auto dir = target.parent_path().empty() ? std::string(".") : target.parent_path().string();
    if (const int d = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC); d >= 0) { ::fsync(d); ::close(d); }
    return true;
}

bool GlmStateStore::read_file(const std::string &path, uint64_t model, uint64_t config, GlmStateImage &image,
                              std::string &error, uint64_t max_bytes) {
    struct stat st{};
    if (::lstat(path.c_str(), &st) != 0) { error = "no such file"; return false; }
    if (!S_ISREG(st.st_mode) || st.st_nlink != 1) { error = "not a regular file with one link"; return false; }
    if (uint64_t(st.st_size) < sizeof(Header) || uint64_t(st.st_size) > max_bytes) { error = "bad size"; return false; }
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) { error = std::string("cannot open: ") + std::strerror(errno); return false; }
    struct Close { int fd; ~Close() { ::close(fd); } } close_fd{fd};
    Header h{};
    if (!read_all(fd, &h, sizeof h)) { error = "short header"; return false; }
    if (std::memcmp(h.magic, kMagic, 8) || h.version != kVersion || h.header_bytes != sizeof h ||
        h.header_hash != core::session_hash64(&h, offsetof(Header, header_hash), kSeed)) { error = "not a GLM state file"; return false; }
    if (h.model != model || h.config != config) { error = "saved with another model or engine settings"; return false; }
    if (h.payload_bytes != uint64_t(st.st_size) - sizeof h) { error = "payload size mismatch"; return false; }
    std::vector<uint8_t> payload(h.payload_bytes);
    if (!read_all(fd, payload.data(), payload.size())) { error = "short payload"; return false; }
    if (core::session_hash64(payload.data(), payload.size(), kSeed) != h.payload_hash) { error = "payload hash mismatch"; return false; }
    if (!deserialize_image(payload.data(), payload.size(), image) || image.position != h.position) { error = "bad payload"; return false; }
    return true;
}

GlmStateStore::GlmStateStore(Options options) : options_(std::move(options)) {
    if (!options_.disk_dir.empty()) {
        std::filesystem::create_directories(options_.disk_dir);
        // index the files of this identity that are already there (a restart keeps its cache)
        const std::string prefix = hex(options_.identity_model) + "-" + hex(options_.identity_config) + "-";
        for (const auto &entry : std::filesystem::directory_iterator(options_.disk_dir)) {
            const auto name = entry.path().filename().string();
            if (!entry.is_regular_file() || !name.starts_with(prefix) || !name.ends_with(".glmstate")) continue;
            uint64_t hash = 0; long position = 0;
            if (std::sscanf(name.c_str() + prefix.size(), "%16llx-%ld.glmstate", (unsigned long long *)&hash, &position) != 2) continue;
            const uint64_t bytes = entry.file_size();
            disk_[{hash, int(position)}] = {bytes, ++tick_};
            disk_used_ += bytes;
        }
        stats_.disk_entries = disk_.size();
        thread_ = std::thread([this] { writer(); });
    }
}

GlmStateStore::~GlmStateStore() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    queue_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

uint64_t GlmStateStore::prefix_hash(const int *tokens, size_t n) {
    return core::session_hash64(tokens, n * sizeof(int), kSeed ^ n);
}

std::string GlmStateStore::file_for(uint64_t hash, int position) const {
    return (std::filesystem::path(options_.disk_dir) /
            (hex(options_.identity_model) + "-" + hex(options_.identity_config) + "-" + hex(hash) + "-" +
             std::to_string(position) + ".glmstate")).string();
}

bool GlmStateStore::contains(const std::vector<int> &tokens, int position) {
    if (position <= 0 || size_t(position) > tokens.size()) return false;
    const Key key{prefix_hash(tokens.data(), size_t(position)), position};
    std::lock_guard lock(mutex_);
    return ram_.count(key) || disk_.count(key);
}

std::shared_ptr<const GlmStateImage> GlmStateStore::find(const std::vector<int> &prompt, std::vector<int> candidates,
                                                         std::string *tier) {
    std::sort(candidates.begin(), candidates.end(), std::greater<int>());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    for (int position : candidates) {
        if (position <= 0 || size_t(position) >= prompt.size()) continue;
        const Key key{prefix_hash(prompt.data(), size_t(position)), position};
        std::shared_ptr<const GlmStateImage> image;
        bool on_disk = false;
        {
            std::lock_guard lock(mutex_);
            if (auto it = ram_.find(key); it != ram_.end()) {
                ram_lru_.splice(ram_lru_.begin(), ram_lru_, it->second.second);
                image = it->second.first;
            } else if (auto d = disk_.find(key); d != disk_.end()) {
                d->second.second = ++tick_;
                on_disk = true;
            }
        }
        if (!image && on_disk) image = load_disk(key);
        // a hash collision is not a hit: the ids must match
        if (image && image->position == position &&
            std::equal(image->tokens.begin(), image->tokens.end(), prompt.begin())) {
            std::lock_guard lock(mutex_);
            ++(on_disk ? stats_.disk_hits : stats_.ram_hits);
            if (tier) *tier = on_disk ? "disk" : "ram";
            return image;
        }
    }
    std::lock_guard lock(mutex_);
    ++stats_.misses;
    return nullptr;
}

std::shared_ptr<const GlmStateImage> GlmStateStore::load_disk(const Key &key) {
    auto image = std::make_shared<GlmStateImage>();
    std::string error;
    const auto path = file_for(key.hash, key.position);
    if (!read_file(path, options_.identity_model, options_.identity_config, *image, error)) {
        std::lock_guard lock(mutex_);
        if (auto d = disk_.find(key); d != disk_.end()) { disk_used_ -= d->second.first; disk_.erase(d); }
        ::unlink(path.c_str());           // a file that does not verify is dropped
        stats_.disk_entries = disk_.size();
        return nullptr;
    }
    // promote to the RAM tier: the next turn of this conversation finds it there
    std::shared_ptr<const GlmStateImage> shared = image;
    if (options_.ram_bytes) {
        std::lock_guard lock(mutex_);
        if (!ram_.count(key) && shared->bytes() <= options_.ram_bytes) {
            ram_lru_.push_front(key);
            ram_[key] = {shared, ram_lru_.begin()};
            ram_used_ += shared->bytes();
            evict_ram();
        }
    }
    return shared;
}

void GlmStateStore::put(std::shared_ptr<const GlmStateImage> image) {
    if (!image || image->position <= 0 || image->tokens.size() != size_t(image->position)) return;
    const Key key{prefix_hash(image->tokens.data(), image->tokens.size()), image->position};
    std::lock_guard lock(mutex_);
    if (options_.ram_bytes && image->bytes() <= options_.ram_bytes) {
        if (auto it = ram_.find(key); it != ram_.end()) {
            ram_lru_.splice(ram_lru_.begin(), ram_lru_, it->second.second);
        } else {
            ram_lru_.push_front(key);
            ram_[key] = {image, ram_lru_.begin()};
            ram_used_ += image->bytes();
            evict_ram();
        }
    }
    if (!options_.disk_dir.empty() && !disk_.count(key)) {
        // one queued write per key; the queue holds at most a few images (each is large)
        const bool queued = std::any_of(queue_.begin(), queue_.end(), [&](const auto &q) {
            return q->position == image->position && prefix_hash(q->tokens.data(), q->tokens.size()) == key.hash;
        });
        if (!queued && queue_.size() < 4) { queue_.push_back(image); queue_cv_.notify_one(); }
    }
    stats_.ram_entries = ram_.size();
}

void GlmStateStore::evict_ram() {   // with mutex_ held
    while (ram_used_ > options_.ram_bytes && !ram_lru_.empty()) {
        const Key key = ram_lru_.back();
        ram_lru_.pop_back();
        auto it = ram_.find(key);
        ram_used_ -= it->second.first->bytes();
        ram_.erase(it);
    }
    stats_.ram_entries = ram_.size();
}

void GlmStateStore::evict_disk() {  // with mutex_ held
    while (disk_used_ > options_.disk_bytes && !disk_.empty()) {
        auto victim = std::min_element(disk_.begin(), disk_.end(),
                                       [](const auto &a, const auto &b) { return a.second.second < b.second.second; });
        ::unlink(file_for(victim->first.hash, victim->first.position).c_str());
        disk_used_ -= victim->second.first;
        disk_.erase(victim);
    }
    stats_.disk_entries = disk_.size();
}

void GlmStateStore::writer() {
    for (;;) {
        std::shared_ptr<const GlmStateImage> image;
        {
            std::unique_lock lock(mutex_);
            queue_cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;          // stop_ with nothing left
            image = queue_.front();
            queue_.pop_front();
            writing_ = true;
        }
        const Key key{prefix_hash(image->tokens.data(), image->tokens.size()), image->position};
        std::string error;
        const auto path = file_for(key.hash, key.position);
        const bool ok = write_file(path, *image, options_.identity_model, options_.identity_config, error);
        std::lock_guard lock(mutex_);
        writing_ = false;
        if (ok) {
            const uint64_t bytes = std::filesystem::file_size(path);
            disk_[key] = {bytes, ++tick_};
            disk_used_ += bytes;
            ++stats_.written;
            evict_disk();
        } else {
            ++stats_.write_errors;
        }
        idle_cv_.notify_all();
    }
}

void GlmStateStore::flush() {
    std::unique_lock lock(mutex_);
    idle_cv_.wait(lock, [&] { return queue_.empty() && !writing_; });
}

GlmStateStore::Stats GlmStateStore::stats() {
    std::lock_guard lock(mutex_);
    Stats s = stats_;
    s.ram_bytes = ram_used_; s.disk_bytes = disk_used_;
    s.ram_entries = ram_.size(); s.disk_entries = disk_.size();
    return s;
}

}  // namespace strata::glm
