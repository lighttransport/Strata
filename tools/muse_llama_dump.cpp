// tools/muse_llama_dump.cpp - run a Muse Spark GGUF (typically a few-layer slice, tools/muse_layer_slice.py) through
// llama.cpp on the CPU and dump the block's named tensors for tools/muse_reference_check.cpp.  Built against a
// llama.cpp checkout that has src/models/muse-spark.cpp, not against Strata:
//
//     g++ -O2 -std=c++17 tools/muse_llama_dump.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
//         -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o muse_llama_dump
//     muse_llama_dump MODEL.gguf OUTDIR "prompt text"
//
// One batch of the whole prompt with logits at every position.  OUTDIR/tokens.txt holds the token ids; every dumped
// tensor is OUTDIR/<name>.bin: int32 type (0 f32, 26 i32), int64 ne[4], then the elements in ne order.
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {
std::string g_out;
const std::set<std::string> kPrefixes = {"embd_norm", "attn_post_norm", "ffn_inp", "ffn_moe_topk", "ffn_moe_weights_norm",
                                         "ffn_moe_out", "ffn_post_norm", "l_out", "result_norm", "result_output"};

bool wanted(const char* name) {
    std::string n(name);
    const size_t dash = n.rfind('-');
    const std::string base = dash == std::string::npos ? n : n.substr(0, dash);
    return kPrefixes.count(base) || kPrefixes.count(n);
}

bool on_eval(struct ggml_tensor* t, bool ask, void*) {
    if (ask) return wanted(t->name);
    if (!wanted(t->name) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32)) return true;
    if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
        std::fprintf(stderr, "skip %s: not in host memory\n", t->name);
        return true;
    }
    std::ofstream f(g_out + "/" + t->name + ".bin", std::ios::binary);
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
        std::fprintf(stderr, "usage: muse_llama_dump MODEL.gguf OUTDIR \"prompt\"\n");
        return 2;
    }
    g_out = argv[2];
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(512);
    const int n = llama_tokenize(vocab, argv[3], (int32_t) std::strlen(argv[3]), toks.data(), (int32_t) toks.size(), true, true);
    if (n <= 0) return 1;
    toks.resize((size_t) n);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 16;
    cp.cb_eval = on_eval;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[(size_t) i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) return 1;
    std::ofstream tf(g_out + "/tokens.txt");
    for (llama_token t : toks) tf << t << "\n";
    std::printf("%d tokens dumped to %s\n", n, g_out.c_str());
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
