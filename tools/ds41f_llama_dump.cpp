// tools/ds41f_llama_dump.cpp - run a DeepSeek-V4.1-Flash GGUF (llama.cpp PR #30817 schema, arch deepseek41) through
// llama.cpp on the CPU and dump its named intermediate tensors, the reference for Strata's DS4.1F decode path.
// Built against that llama.cpp checkout, not against Strata:
//
//     g++ -O2 -std=c++17 tools/ds41f_llama_dump.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
//         -L$LLAMA/build-cpu/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build-cpu/bin -o ds41f_llama_dump
//     ds41f_llama_dump MODEL-00001-of-0000N.gguf OUTDIR "prompt text" [--serial] [--names a,b,...]
//
// Default: one batch of the whole prompt, logits at every position, tensors in OUTDIR/<name>.bin.  --serial decodes
// one token per llama_decode (the decode path: window ring, compressor state, top-k reuse) and writes step k's
// tensors to OUTDIR/s<k>/<name>.bin.  OUTDIR/tokens.txt holds the token ids.  A .bin file is int32 type (0 f32,
// 26 i32), int64 ne[4], then the elements in ne order.  The KV cache stays F32 so the dump has no f16 rounding.
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::string g_dir;
std::set<std::string> g_names = {"hc_init", "engram_out", "hc_attn_pre", "attn_norm", "qr_norm", "q", "kv",
                                 "comp_latent", "idx_top_k", "attn_derope", "attn_out", "hc_attn_post",
                                 "hc_ffn_pre", "ffn_norm", "ffn_moe_logits", "ffn_moe_topk", "ffn_moe_weights_norm",
                                 "ffn_moe_weights_scaled", "ffn_moe_out", "ffn_shexp", "l_out", "hc_out",
                                 "result_norm", "result_output"};

bool wanted(const char* name) {
    std::string n(name);
    const size_t dash = n.rfind('-');
    const std::string base = dash == std::string::npos ? n : n.substr(0, dash);
    return g_names.count(base) || g_names.count(n);
}

bool on_eval(struct ggml_tensor* t, bool ask, void*) {
    if (ask) return wanted(t->name);
    if (!wanted(t->name) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32)) return true;
    if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
        std::fprintf(stderr, "skip %s: not in host memory\n", t->name);
        return true;
    }
    std::ofstream f(g_dir + "/" + t->name + ".bin", std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot write %s/%s.bin\n", g_dir.c_str(), t->name); std::exit(1); }
    const int32_t type = (int32_t) t->type;
    f.write((const char*) &type, 4);
    for (int i = 0; i < 4; ++i) f.write((const char*) &t->ne[i], 8);
    const char* base = (const char*) t->data;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0)
                    f.write(base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], 4);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: ds41f_llama_dump MODEL.gguf OUTDIR \"prompt\" [--serial] [--names a,b,...]\n");
        return 2;
    }
    const std::string out = argv[2];
    bool serial = false;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--serial")) serial = true;
        else if (!std::strcmp(argv[i], "--names") && i + 1 < argc) {
            g_names.clear();
            std::stringstream ss(argv[++i]);
            for (std::string n; std::getline(ss, n, ',');) g_names.insert(n);
        } else {
            std::fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    std::filesystem::create_directories(out);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(4096);
    const int n = llama_tokenize(vocab, argv[3], (int32_t) std::strlen(argv[3]), toks.data(), (int32_t) toks.size(), true, true);
    if (n <= 0) return 1;
    toks.resize((size_t) n);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = serial ? 1 : 4096;
    cp.n_threads = cp.n_threads_batch = 16;
    cp.cb_eval = on_eval;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    const int steps = serial ? n : 1;
    for (int s = 0; s < steps; ++s) {
        g_dir = serial ? out + "/s" + std::to_string(s) : out;
        std::filesystem::create_directories(g_dir);
        const int first = serial ? s : 0, count = serial ? 1 : n;
        llama_batch b = llama_batch_init(count, 0, 1);
        for (int i = 0; i < count; ++i) {
            b.token[i] = toks[(size_t) (first + i)];
            b.pos[i] = first + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        b.n_tokens = count;
        if (llama_decode(ctx, b) != 0) return 1;
        llama_batch_free(b);
    }
    std::ofstream tf(out + "/tokens.txt");
    if (!tf) { std::fprintf(stderr, "cannot write %s/tokens.txt\n", out.c_str()); return 1; }
    for (llama_token t : toks) tf << t << "\n";
    std::printf("%d tokens dumped to %s%s\n", n, out.c_str(), serial ? " (serial)" : "");
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
