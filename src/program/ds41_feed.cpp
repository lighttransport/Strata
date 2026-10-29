// strata-ds41-feed: shards a DeepSeek-V4.1 model into another node's memory over ucomm. The node that has the files serves
// the byte ranges a decoder needs; the receiving node writes them at their offsets into sparse files of the same sizes
// (on a tmpfs, so the pages stay in RAM and a decoder can map the files as the model). Everything the decoder reads from
// disk except the Engram tables and the experts the other node owns is sent: every file's header and tensor table, all
// dense tensors, and the hottest --keep-mib of routed experts by the route profile (the order the decoder's
// STRATA_DS41_REMOTE_OWN uses with STRATA_DS41_REMOTE_OWN_SKIP_MIB, which makes the other node own the rest).
//   serve (node with the model):  strata-ds41-feed serve --root HOST:PORT --model FIRST_SHARD.gguf --profile ROUTES.i32 --keep-mib N
//                                                       [--top-k 6] [--extra FILE]... [--backend=auto|ib|udp]
//   recv  (node that decodes):    strata-ds41-feed recv --listen HOST:PORT --dir DIR [--backend=auto|ib|udp]
#include "strata/core/model.hpp"
#include "ucomm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace {
constexpr uint32_t kTagManifest = 40, kTagData = 41;
constexpr size_t kChunk = size_t(8) << 20;
struct DataHeader {
    uint32_t file;
    uint32_t len;
    uint64_t offset;
};
void ok(ucomm_status s, const char* what) {
    if (s != UCOMM_OK) throw std::runtime_error(std::string(what) + ": " + ucomm_strerror(s));
}
ucomm_t* connect(int rank, const std::string& addr, const std::string& backend) {
    ucomm_config cfg;
    ucomm_config_default(&cfg);
    cfg.rank = rank;
    cfg.world = 2;
    cfg.root_addr = addr.c_str();
    cfg.timeout_ms = 0;
    cfg.backend = backend == "ib" ? UCOMM_BACKEND_IB : backend == "udp" ? UCOMM_BACKEND_UDP : UCOMM_BACKEND_AUTO;
    ucomm_t* c = nullptr;
    ok(ucomm_init(&cfg, &c), "ucomm_init");
    return c;
}
bool engram_tensor(const std::string& name) { return name.find("engram_embd") != std::string::npos || name == "per_layer_token_embd.weight"; }
bool expert_tensor(const std::string& name) {
    return name.find("ffn_gate_exps.weight") != std::string::npos || name.find("ffn_up_exps.weight") != std::string::npos ||
           name.find("ffn_down_exps.weight") != std::string::npos;
}

struct Range {
    uint32_t file;
    uint64_t offset, len;
};

int serve(int argc, char** argv) {
    std::string root, model, profile, backend = "auto";
    double keep_mib = 0;
    int top_k = 6;
    std::vector<std::string> extra;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--keep-mib" && i + 1 < argc) keep_mib = std::stod(argv[++i]);
        else if (a == "--top-k" && i + 1 < argc) top_k = std::stoi(argv[++i]);
        else if (a == "--extra" && i + 1 < argc) extra.push_back(argv[++i]);
        else if (a.starts_with("--backend=")) backend = a.substr(10);
        else throw std::runtime_error("serve: unknown argument " + a);
    }
    if (root.empty() || model.empty() || profile.empty()) throw std::runtime_error("serve needs --root, --model and --profile");
    strata::core::ModelArtifact art{std::filesystem::path(model)};
    // files in the artifact's order, plus the extras (whole files)
    std::vector<std::string> paths;
    std::map<std::string, uint32_t> index;
    std::map<std::string, const strata::GgufFile*> gguf;
    for (const auto& [name, t] : art.tensors()) {
        const std::string path = t.file->path();
        if (!index.count(path)) { index[path] = uint32_t(paths.size()); paths.push_back(path); gguf[path] = t.file; }
    }
    // shards without tensors (the first of a split model holds only metadata) are sent whole
    std::vector<std::string> bare;
    {
        const std::string name = std::filesystem::path(model).filename().string();
        const size_t at = name.rfind("-00001-of-");
        if (at != std::string::npos) {
            const int count = std::atoi(name.c_str() + at + 10);
            for (int i = 1; i <= count; ++i) {
                char suffix[40];
                std::snprintf(suffix, sizeof suffix, "-%05d-of-%05d.gguf", i, count);
                const std::string path = (std::filesystem::path(model).parent_path() / (name.substr(0, at) + suffix)).string();
                if (!index.count(path)) { index[path] = uint32_t(paths.size()); paths.push_back(path); bare.push_back(path); }
            }
        } else if (!index.count(model)) { index[model] = uint32_t(paths.size()); paths.push_back(model); bare.push_back(model); }
    }
    // the layers' expert tensors and their per-expert byte counts
    int layers = 0;
    std::vector<size_t> expert_bytes;
    int experts = 0;
    for (;; ++layers) {
        const std::string p = "blk." + std::to_string(layers) + ".ffn_";
        auto g = art.tensors().find(p + "gate_exps.weight"), u = art.tensors().find(p + "up_exps.weight"), d = art.tensors().find(p + "down_exps.weight");
        if (g == art.tensors().end() || u == art.tensors().end() || d == art.tensors().end()) break;
        const int e = int(g->second.tensor->shape.back());
        if (experts && e != experts) throw std::runtime_error("layers disagree on the expert count");
        experts = e;
        expert_bytes.push_back((g->second.bytes + u->second.bytes + d->second.bytes) / size_t(e));
    }
    if (!layers || !experts) throw std::runtime_error("the model has no routed-expert layers");
    // the decoder's ownership order (ds41_decode.cpp connect_own): hits per byte of the profile, hottest first, stable
    std::vector<double> count(size_t(layers) * experts, 0);
    {
        std::ifstream g(profile, std::ios::binary);
        if (!g) throw std::runtime_error("cannot read the route profile");
        std::vector<int32_t> v(size_t(layers) * top_k);
        while (g.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4)))
            for (int l = 0; l < layers; ++l)
                for (int k = 0; k < top_k; ++k) {
                    const int e = v[size_t(l) * top_k + k];
                    if (e >= 0 && e < experts) count[size_t(l) * experts + e] += 1;
                }
    }
    std::vector<std::tuple<double, int, int>> order;
    for (int l = 0; l < layers; ++l)
        for (int e = 0; e < experts; ++e) order.emplace_back(count[size_t(l) * experts + e] / double(expert_bytes[size_t(l)]), l, e);
    std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    std::vector<char> keep(size_t(layers) * experts, 0);
    double kept = 0;
    uint64_t kept_hash = 14695981039346656037ull;
    size_t kept_n = 0;
    for (const auto& [score, l, e] : order) {
        if (kept + double(expert_bytes[size_t(l)]) > keep_mib * double(1 << 20)) break;
        kept += double(expert_bytes[size_t(l)]);
        keep[size_t(l) * experts + e] = 1;
        kept_hash = (kept_hash ^ uint64_t(l * 65536 + e)) * 1099511628211ull;
        ++kept_n;
        (void)score;
    }
    std::vector<Range> ranges;
    double dense = 0, routed = 0;
    for (uint32_t f = 0; f < paths.size(); ++f)
        if (gguf.count(paths[f])) ranges.push_back({f, 0, gguf[paths[f]]->data_start()});
    for (const auto& path : bare) ranges.push_back({index[path], 0, std::filesystem::file_size(path)});
    for (const auto& [name, t] : art.tensors()) {
        const uint32_t f = index[t.file->path()];
        const uint64_t base = t.file->data_start() + t.tensor->offset;
        if (engram_tensor(name)) continue;
        if (!expert_tensor(name)) { ranges.push_back({f, base, t.bytes}); dense += double(t.bytes); continue; }
        int l = 0;
        std::sscanf(name.c_str(), "blk.%d.", &l);
        if (l >= layers) { ranges.push_back({f, base, t.bytes}); dense += double(t.bytes); continue; }   // layers past the main stack (draft): whole
        const size_t stride = t.bytes / size_t(experts);
        for (int e = 0; e < experts; ++e)
            if (keep[size_t(l) * experts + e]) { ranges.push_back({f, base + uint64_t(e) * stride, stride}); routed += double(stride); }
    }
    for (const auto& path : extra) {
        index[path] = uint32_t(paths.size());
        paths.push_back(path);
        ranges.push_back({index[path], 0, std::filesystem::file_size(path)});
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) { return std::tie(a.file, a.offset) < std::tie(b.file, b.offset); });
    std::fprintf(stderr, "DS41_FEED layers=%d experts=%d keep %zu experts (%.2f GiB, hash %016llx), dense %.2f GiB, %zu ranges\n", layers, experts, kept_n,
                 routed / double(1ull << 30), (unsigned long long)kept_hash, dense / double(1ull << 30), ranges.size());
    ucomm_t* c = connect(1, root, backend);
    std::fprintf(stderr, "DS41_FEED connected backend=%s\n", ucomm_backend_name(c));
    std::string manifest;
    {
        uint32_t nfiles = uint32_t(paths.size());
        manifest.append(reinterpret_cast<const char*>(&nfiles), 4);
        for (const auto& path : paths) {
            const std::string base = std::filesystem::path(path).filename().string();
            const uint64_t size = std::filesystem::file_size(path);
            const uint32_t nlen = uint32_t(base.size());
            manifest.append(reinterpret_cast<const char*>(&nlen), 4);
            manifest.append(base);
            manifest.append(reinterpret_cast<const char*>(&size), 8);
        }
    }
    ok(ucomm_send(c, 0, kTagManifest, manifest.data(), manifest.size()), "manifest");
    std::vector<int> fds;
    for (const auto& path : paths) {
        fds.push_back(::open(path.c_str(), O_RDONLY));
        if (fds.back() < 0) throw std::runtime_error("cannot open " + path);
    }
    std::vector<uint8_t> staging[2] = {std::vector<uint8_t>(sizeof(DataHeader) + kChunk), std::vector<uint8_t>(sizeof(DataHeader) + kChunk)};
    ucomm_req_t* inflight[2] = {nullptr, nullptr};
    ucomm_mr_t* registered[2] = {nullptr, nullptr};
    for (int k = 0; k < 2; ++k)
        if (ucomm_mr_reg(c, staging[k].data(), staging[k].size(), &registered[k]) != UCOMM_OK) registered[k] = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    double sent = 0;
    int which = 0;
    for (const auto& r : ranges) {
        for (uint64_t at = 0; at < r.len; at += kChunk) {
            const uint32_t len = uint32_t(std::min<uint64_t>(kChunk, r.len - at));
            if (inflight[which]) ok(ucomm_wait(inflight[which], nullptr), "data");
            inflight[which] = nullptr;
            auto& buf = staging[which];
            const DataHeader h{r.file, len, r.offset + at};
            std::memcpy(buf.data(), &h, sizeof h);
            for (size_t got = 0; got < len;) {
                const ssize_t k = pread(fds[r.file], buf.data() + sizeof h + got, len - got, off_t(h.offset + got));
                if (k <= 0) throw std::runtime_error("read failed");
                got += size_t(k);
            }
            ok(ucomm_isend(c, 0, kTagData, buf.data(), sizeof h + len, &inflight[which]), "data");
            which ^= 1;
            sent += double(len);
        }
    }
    for (auto* q : inflight) if (q) ok(ucomm_wait(q, nullptr), "data");
    DataHeader end{0xFFFFFFFFu, 0, 0};
    ok(ucomm_send(c, 0, kTagData, &end, sizeof end), "done");
    std::fprintf(stderr, "DS41_FEED sent %.2f GiB in %.1f s\n", sent / double(1ull << 30), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    for (int fd : fds) ::close(fd);
    return 0;
}

int recv(int argc, char** argv) {
    std::string listen_addr, dir, backend = "auto";
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--listen" && i + 1 < argc) listen_addr = argv[++i];
        else if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a.starts_with("--backend=")) backend = a.substr(10);
        else throw std::runtime_error("recv: unknown argument " + a);
    }
    if (listen_addr.empty() || dir.empty()) throw std::runtime_error("recv needs --listen and --dir");
    std::filesystem::create_directories(dir);
    ucomm_t* c = connect(0, listen_addr, backend);
    std::fprintf(stderr, "DS41_FEED connected backend=%s\n", ucomm_backend_name(c));
    std::vector<uint8_t> manifest(1 << 20);
    size_t n = 0;
    ok(ucomm_recv(c, 1, kTagManifest, manifest.data(), manifest.size(), &n), "manifest");
    size_t at = 0;
    auto take = [&](void* dst, size_t bytes) {
        if (at + bytes > n) throw std::runtime_error("short manifest");
        std::memcpy(dst, manifest.data() + at, bytes);
        at += bytes;
    };
    uint32_t nfiles = 0;
    take(&nfiles, 4);
    std::vector<int> fds;
    for (uint32_t f = 0; f < nfiles; ++f) {
        uint32_t nlen = 0;
        take(&nlen, 4);
        std::string name(nlen, '\0');
        take(name.data(), nlen);
        uint64_t size = 0;
        take(&size, 8);
        if (name.find('/') != std::string::npos || name.empty() || name[0] == '.') throw std::runtime_error("bad file name in the manifest");
        const std::string path = dir + "/" + name;
        const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0 || ftruncate(fd, off_t(size)) != 0) throw std::runtime_error("cannot create " + path);
        fds.push_back(fd);
    }
    std::vector<uint8_t> buf(sizeof(DataHeader) + kChunk);
    ucomm_mr_t* region = nullptr;
    if (ucomm_mr_reg(c, buf.data(), buf.size(), &region) != UCOMM_OK) region = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    double got = 0;
    for (;;) {
        size_t m = 0;
        // the data and the end marker use different tags: poll the data tag first, then check for the end after each message
        ok(ucomm_recv(c, 1, kTagData, buf.data(), buf.size(), &m), "data");
        DataHeader h;
        std::memcpy(&h, buf.data(), sizeof h);
        if (h.file == 0xFFFFFFFFu) break;
        if (h.file >= fds.size() || m != sizeof h + h.len) throw std::runtime_error("bad data message");
        for (size_t done = 0; done < h.len;) {
            const ssize_t k = pwrite(fds[h.file], buf.data() + sizeof h + done, h.len - done, off_t(h.offset + done));
            if (k <= 0) throw std::runtime_error("write failed");
            done += size_t(k);
        }
        got += double(h.len);
    }
    std::fprintf(stderr, "DS41_FEED received %.2f GiB in %.1f s\n", got / double(1ull << 30), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    for (int fd : fds) ::close(fd);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "serve") return serve(argc, argv);
        if (argc >= 2 && std::string(argv[1]) == "recv") return recv(argc, argv);
        std::fprintf(stderr, "usage: %s serve|recv ...\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "DS41_FEED error: %s\n", e.what());
        return 1;
    }
}
