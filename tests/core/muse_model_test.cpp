// tests/core/muse_model_test.cpp - the Muse Spark model checker (strata/artifact/muse_model.hpp) on a synthetic
// two-layer `muse-spark` GGUF written here (tiny dimensions, the release's formats), no model and no GPU:
//
//   1. a complete file: the keys read, every tensor has its shape and format, and the footprint adds up;
//   2. a missing tensor, a projection in a format the engine does not read for it, another architecture: refused,
//      each naming what is wrong;
//   3. the two-node row split for the UD-IQ1_S release's numbers: whole blocks, the largest share that fits node B.
//
// STRATA_MUSE_HEADERS=DIR (optional): DIR holds the release's shard headers (s1.gguf h2.gguf ... h5.gguf, see
// tools/muse_gguf_census.py); they must check clean and add up to the 175,626,404,864 bytes the census lists.
#include "strata/artifact/muse_model.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-70s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

struct Kv {
    Kv(std::string k, int t) : key(std::move(k)), type(t) {}
    std::string key;
    int type;                 // 4 u32, 6 f32, 7 bool, 8 string, 9 array (of bools: 7, of strings: 8)
    uint32_t u = 0;
    float f = 0;
    std::string s;
    int elem = 0;
    std::vector<uint8_t> bools;
    std::vector<std::string> strings;
};
struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type;
};

uint64_t bytes_of(const Tensor& t) {
    uint64_t n = 1;
    for (auto d : t.shape) n *= d;
    switch (t.type) {
        case 0: return n * 4;
        case 8: return n / 32 * 34;
        case 12: return n / 256 * 144;
        case 14: return n / 256 * 210;
        case 19: return n / 256 * 50;
        default: return 0;
    }
}

std::filesystem::path write(const std::vector<Kv>& kvs, const std::vector<Tensor>& ts, const char* file) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);
    put<uint32_t>(b, 3);
    put<uint64_t>(b, ts.size());
    put<uint64_t>(b, kvs.size());
    for (const Kv& kv : kvs) {
        put_str(b, kv.key);
        put<uint32_t>(b, (uint32_t) kv.type);
        switch (kv.type) {
            case 4: put<uint32_t>(b, kv.u); break;
            case 6: put<float>(b, kv.f); break;
            case 7: put<uint8_t>(b, (uint8_t) kv.u); break;
            case 8: put_str(b, kv.s); break;
            case 9:
                put<uint32_t>(b, (uint32_t) kv.elem);
                if (kv.elem == 7) {
                    put<uint64_t>(b, kv.bools.size());
                    for (uint8_t v : kv.bools) put<uint8_t>(b, v);
                } else {
                    put<uint64_t>(b, kv.strings.size());
                    for (const auto& s : kv.strings) put_str(b, s);
                }
                break;
        }
    }
    uint64_t off = 0;
    for (const Tensor& t : ts) {
        put_str(b, t.name);
        put<uint32_t>(b, (uint32_t) t.shape.size());
        for (auto d : t.shape) put<uint64_t>(b, d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, off);
        off += (bytes_of(t) + 31) / 32 * 32;
    }
    b.resize((b.size() + 31) / 32 * 32 + off, 0);
    const auto path = std::filesystem::temp_directory_path() / file;
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize) b.size());
    return path;
}

// hidden 256, latent 256, 4 experts of 256 (top-2), 4/2 heads of 64, window 16, 2 layers (SWA, full), vocab 8
constexpr uint64_t H = 256, L = 256, FF = 256, E = 4, QH = 4, KVH = 2, HD = 64, V = 8;

std::vector<Kv> keys(const std::string& arch = STRATA_MUSE_ARCH) {
    auto u = [](const std::string& k, uint32_t v) { Kv kv{k, 4}; kv.u = v; return kv; };
    auto f = [](const std::string& k, float v) { Kv kv{k, 6}; kv.f = v; return kv; };
    const std::string p = "muse-spark.";
    std::vector<Kv> kvs;
    Kv a{"general.architecture", 8};
    a.s = arch;
    kvs.push_back(a);
    for (auto kv : {u(p + "block_count", 2), u(p + "context_length", 1 << 20), u(p + "embedding_length", H),
                    u(p + "feed_forward_length", 16384), u(p + "attention.head_count", QH),
                    u(p + "attention.head_count_kv", KVH), u(p + "attention.key_length", HD),
                    u(p + "attention.value_length", HD), u(p + "attention.sliding_window", 16), u(p + "expert_count", E),
                    u(p + "expert_used_count", 2), u(p + "expert_feed_forward_length", FF), u(p + "moe_latent_size", L),
                    u(p + "expert_gating_func", 2), f(p + "rope.freq_base", 5e5f), f(p + "rope.freq_base_swa", 5e5f),
                    f(p + "attention.layer_norm_rms_epsilon", 1e-5f), f(p + "post_norm_epsilon", 1e-8f),
                    f(p + "logit_scale", 0.17677669f), f(p + "final_logit_softcapping", 20.f),
                    f(p + "expert_weights_scale", 1.f)})
        kvs.push_back(kv);
    Kv norm{p + "expert_weights_norm", 7};
    norm.u = 1;
    kvs.push_back(norm);
    Kv pat{p + "attention.sliding_window_pattern", 9};
    pat.elem = 7;
    pat.bools = {1, 0};
    kvs.push_back(pat);
    Kv tok{"tokenizer.ggml.tokens", 9};
    tok.elem = 8;
    for (uint64_t i = 0; i < V; ++i) tok.strings.push_back("t" + std::to_string(i));
    kvs.push_back(tok);
    return kvs;
}

std::vector<Tensor> tensors() {
    std::vector<Tensor> ts = {{"token_embd.weight", {H, V}, 12}, {"output.weight", {H, V}, 14}, {"output_norm.weight", {H}, 0}};
    const strata_muse_dims d{(int64_t) H, (int64_t) L, (int64_t) FF, (int64_t) E, (int64_t) QH, (int64_t) KVH, (int64_t) HD};
    for (int l = 0; l < 2; ++l)
        for (int k = 0; k < STRATA_MUSE_TENSOR_COUNT; ++k) {
            const auto kind = (strata_muse_tensor) k;
            char name[STRATA_NAME_MAX];
            strata_muse_tensor_name(name, sizeof name, l, kind);
            int64_t s[3];
            const int nd = strata_muse_shape(&d, kind, s);
            uint32_t type = 0;
            if (strata_muse_is_expert(kind)) type = 19;
            else if (kind == STRATA_MUSE_ATTN_Q || kind == STRATA_MUSE_ATTN_OUTPUT) type = 14;
            else if (nd == 2 && kind != STRATA_MUSE_FFN_GATE_INP) type = 8;
            ts.push_back({name, std::vector<uint64_t>(s, s + nd), type});
        }
    return ts;
}
}  // namespace

int main() {
    using namespace strata::muse;
    {
        const auto ts = tensors();
        const auto path = write(keys(), ts, "strata_muse_model_test.gguf");
        uint64_t want_experts = 0, want_total = 0;
        for (const auto& t : ts) {
            want_total += bytes_of(t);
            if (t.name.find("_exps") != std::string::npos) want_experts += bytes_of(t);
        }
        {
            const strata::GgufModel m(std::vector<std::string>{path.string()});
            MuseConfig c;
            const std::string err = read_config(m.meta(), c);
            check(err.empty(), "a complete muse-spark file reads: " + (err.empty() ? std::string("") : err));
            check(c.layers == 2 && c.latent == (int64_t) L && c.top_k == 2 && c.vocab == (int64_t) V && c.full_layers() == 1 &&
                      c.swa[0] == 1 && c.swa[1] == 0 && c.expert_weights_norm, "the geometry and the SWA pattern");
            Footprint fp;
            const auto problems = check_tensors(m, c, fp);
            check(problems.empty(), "every tensor in shape and format" + (problems.empty() ? std::string("") : ": " + problems[0]));
            check(fp.experts == want_experts && fp.total() == want_total, "the footprint adds up");
            check(fp.expert_bytes == 3 * L * FF / 256 * 50, "one expert's bytes (IQ1_S gate + up + down)");
        }
        std::filesystem::remove(path);
    }
    {   // a missing tensor and a projection in IQ1_S
        auto ts = tensors();
        ts.erase(std::remove_if(ts.begin(), ts.end(), [](const Tensor& t) { return t.name == "blk.1.attn_gate.weight"; }), ts.end());
        for (auto& t : ts) if (t.name == "blk.0.attn_q.weight") t.type = 19;
        const auto path = write(keys(), ts, "strata_muse_model_test_bad.gguf");
        {
            const strata::GgufModel m(std::vector<std::string>{path.string()});
            MuseConfig c;
            read_config(m.meta(), c);
            Footprint fp;
            const auto problems = check_tensors(m, c, fp);
            bool missing = false, format = false;
            for (const auto& p : problems) {
                missing |= p == "missing blk.1.attn_gate.weight";
                format |= p.find("blk.0.attn_q.weight is IQ1_S") == 0;
            }
            check(missing, "a missing tensor is named");
            check(format, "an IQ1_S attention projection is refused");
        }
        std::filesystem::remove(path);
    }
    {   // another architecture
        const auto path = write(keys("muse-glimmer"), tensors(), "strata_muse_model_test_arch.gguf");
        {
            const strata::GgufModel m(std::vector<std::string>{path.string()});
            MuseConfig c;
            check(read_config(m.meta(), c).find("not muse-spark") != std::string::npos, "another architecture is refused");
        }
        std::filesystem::remove(path);
    }
    {   // the release's row split: 156.03 GB of experts, 56 GiB on node B
        MuseConfig c;
        c.expert_ff = 4096;
        Footprint fp;
        fp.experts = 156028108800ull;
        const RowSplit r = plan_row_split(c, fp, 56ull << 30);
        check(r.first_b == 2560 && r.share_b == 0.375, "56 GiB on node B: rows [2560, 4096) of every expert (6 of 16 blocks)");
        check(r.bytes_b <= (56ull << 30) && r.bytes_a + r.bytes_b == fp.experts, "node B's share fits and both add up");
    }
    if (const char* dir = std::getenv("STRATA_MUSE_HEADERS")) {
        std::vector<std::string> shards;
        for (const char* f : {"s1.gguf", "h2.gguf", "h3.gguf", "h4.gguf", "h5.gguf"}) shards.push_back(std::string(dir) + "/" + f);
        const strata::GgufModel m(shards);
        MuseConfig c;
        const std::string err = read_config(m.meta(), c);
        Footprint fp;
        const auto problems = err.empty() ? check_tensors(m, c, fp) : std::vector<std::string>{err};
        check(problems.empty(), "the UD-IQ1_S release's headers check clean" + (problems.empty() ? std::string("") : ": " + problems[0]));
        check(fp.total() == 175626404864ull && c.layers == 62 && c.full_layers() == 16, "and add up to 175,626,404,864 bytes");
    }
    std::printf("muse_model_test: %d failures\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
