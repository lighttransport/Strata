/* tests/host/muse_names_test.c - Muse Spark's block names, shapes, formats and expert row split (muse_names.h), on
   synthetic data: the names and shapes are the ones the UD-IQ1_S release's headers list (tools/muse_gguf_census.py,
   tools/sim/data/census-muse-spark-ud-iq1s.json). */
#include "strata/host/muse_names.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, what)                                              \
    do {                                                               \
        int ok_ = (cond);                                              \
        printf("  %-64s %s\n", what, ok_ ? "ok" : "FAIL");             \
        if (!ok_) ++failures;                                          \
    } while (0)

int main(void) {
    const strata_muse_dims d = {8192, 4096, 4096, 256, 128, 16, 64};
    char name[128];
    int64_t s[3];

    CHECK(strata_muse_tensor_name(name, sizeof name, 61, STRATA_MUSE_FFN_DOWN_EXPS) > 0 &&
              strcmp(name, "blk.61.ffn_down_exps.weight") == 0, "blk.61.ffn_down_exps.weight");
    CHECK(strata_muse_tensor_name(name, sizeof name, 0, STRATA_MUSE_EXP_PROBS_B) > 0 &&
              strcmp(name, "blk.0.exp_probs_b.bias") == 0, "the router bias is a .bias");
    CHECK(strata_muse_tensor_name(name, 8, 0, STRATA_MUSE_ATTN_Q) == -1, "a name that does not fit is refused");
    CHECK(strata_muse_suffix(STRATA_MUSE_TENSOR_COUNT) == NULL, "out-of-range kind has no suffix");

    /* every kind has a distinct suffix */
    int distinct = 1;
    for (int i = 0; i < STRATA_MUSE_TENSOR_COUNT; ++i)
        for (int j = i + 1; j < STRATA_MUSE_TENSOR_COUNT; ++j)
            distinct &= strcmp(strata_muse_suffix((strata_muse_tensor) i), strata_muse_suffix((strata_muse_tensor) j)) != 0;
    CHECK(distinct, "23 distinct suffixes");

    /* shapes as the release lists them */
    CHECK(strata_muse_shape(&d, STRATA_MUSE_FFN_GATE_EXPS, s) == 3 && s[0] == 4096 && s[1] == 4096 && s[2] == 256,
          "ffn_gate_exps [4096, 4096, 256]");
    CHECK(strata_muse_shape(&d, STRATA_MUSE_ATTN_K, s) == 2 && s[0] == 8192 && s[1] == 1024, "attn_k [8192, 1024]");
    CHECK(strata_muse_shape(&d, STRATA_MUSE_ATTN_GATE, s) == 2 && s[0] == 8192 && s[1] == 8192, "attn_gate [8192, 8192]");
    CHECK(strata_muse_shape(&d, STRATA_MUSE_FFN_LATENT_DOWN, s) == 2 && s[0] == 8192 && s[1] == 4096,
          "ffn_latent_down [8192, 4096]");
    CHECK(strata_muse_shape(&d, STRATA_MUSE_FFN_LATENT_UP, s) == 2 && s[0] == 4096 && s[1] == 8192,
          "ffn_latent_up [4096, 8192]");
    CHECK(strata_muse_shape(&d, STRATA_MUSE_FFN_GATE_INP, s) == 2 && s[0] == 8192 && s[1] == 256,
          "ffn_gate_inp [8192, 256] (router on the full stream)");
    CHECK(strata_muse_shape(&d, STRATA_MUSE_ATTN_Q_NORM, s) == 1 && s[0] == 64, "attn_q_norm [64]");
    CHECK(strata_muse_shape(&d, STRATA_MUSE_FFN_POST_EXP_NORM, s) == 1 && s[0] == 4096, "ffn_post_exp_norm [4096]");
    CHECK(strata_muse_shape(&d, STRATA_MUSE_TENSOR_COUNT, s) == 0, "out-of-range kind has no shape");

    /* formats: UD-IQ1_S's experts IQ1_S, attention Q6_K/Q8_0, the rest F32 */
    CHECK(strata_muse_type_ok(STRATA_MUSE_FFN_UP_EXPS, 19), "IQ1_S experts");
    CHECK(strata_muse_type_ok(STRATA_MUSE_ATTN_Q, 14) && strata_muse_type_ok(STRATA_MUSE_ATTN_GATE, 8), "Q6_K q, Q8_0 gate");
    CHECK(!strata_muse_type_ok(STRATA_MUSE_ATTN_Q, 19), "IQ1_S attention is not read");
    CHECK(strata_muse_type_ok(STRATA_MUSE_FFN_GATE_INP, 0) && !strata_muse_type_ok(STRATA_MUSE_FFN_GATE_INP, 8),
          "the router is F32 only");
    CHECK(strata_muse_is_expert(STRATA_MUSE_FFN_DOWN_EXPS) && !strata_muse_is_expert(STRATA_MUSE_FFN_LATENT_UP),
          "expert kinds");

    /* RoPE on the sliding-window layers (pattern True), NoPE on 1, 5, ..., 61 */
    uint8_t pattern[62];
    for (int l = 0; l < 62; ++l) pattern[l] = (l % 4) != 1;
    CHECK(strata_muse_layer_rope(pattern, 62, 0) == 1 && strata_muse_layer_rope(pattern, 62, 1) == 0 &&
              strata_muse_layer_rope(pattern, 62, 61) == 0 && strata_muse_layer_rope(pattern, 62, 62) == -1,
          "RoPE on SWA layers, NoPE on the full ones");

    /* expert row split: whole 256-value blocks, nearest to the share */
    CHECK(strata_muse_row_split(4096, 256, 0.0) == 4096, "share 0: the primary keeps every row");
    CHECK(strata_muse_row_split(4096, 256, 1.0) == 0, "share 1: the worker holds every row");
    CHECK(strata_muse_row_split(4096, 256, 0.385) == 2560, "share 0.385 -> 6 of 16 blocks on the worker");
    CHECK(strata_muse_row_split(4096, 256, 0.35) % 256 == 0, "the cut is block-aligned");
    CHECK(strata_muse_row_split(4096, 256, 1.5) == -1 && strata_muse_row_split(4000, 256, 0.3) == -1,
          "bad share or partial blocks refused");

    printf("muse_names_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
