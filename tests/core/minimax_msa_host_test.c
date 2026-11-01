#include "strata/host/minimax_msa.h"

#include <math.h>
#include <stdio.h>

static int failures;

static void check(int ok, const char *name) {
    printf("  %-72s %s\n", name, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

int main(void) {
    // Two indexer heads query eight keys in four blocks of two. The second query
    // is causal at position five, so block three is still invisible.
    const float q[] = {1, 0, 0, 1, 1, 0, 0, 1};
    const float k[] = {1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0, 6, 7, 0, 0, 8};
    const int32_t qpos[] = {3, 5};
    const int32_t kpos[] = {0, 1, 2, 3, 4, 5, 6, 7};
    float scores[2 * 2 * 6];
    for (int i = 0; i < 24; ++i) scores[i] = 123.0f;
    check(strata_minimax_msa_block_scores_f32(q, k, qpos, kpos, 2, 2, 2, 8, 2, 4, scores, 6) ==
              STRATA_MINIMAX_MSA_OK,
          "block score computation accepts valid grouped queries");
    check(scores[0] == 1.0f && scores[1] == 3.0f && isinf(scores[2]) && scores[2] < 0 &&
              isinf(scores[3]) && scores[3] < 0 && scores[4] == 123.0f,
          "per-head block max uses causal visibility and preserves row padding");
    check(scores[6] == 2.0f && scores[7] == 4.0f && isinf(scores[8]) && scores[8] < 0,
          "index heads score independently");
    check(scores[12] == 1.0f && scores[13] == 3.0f && scores[14] == 5.0f &&
              isinf(scores[15]) && scores[15] < 0,
          "later query sees keys through its own position only");

    const float rank_scores[] = {
        10, 9, 8, 7, -INFINITY, -INFINITY,
        5, 5, 4, 1, -INFINITY, -INFINITY,
    };
    const int32_t one_pos[] = {5};
    int32_t ids[2 * 3];
    check(strata_minimax_msa_select_blocks(rank_scores, one_pos, 1, 2, 4, 2, 3, 1, 6, ids, 3) ==
              STRATA_MINIMAX_MSA_OK,
          "top block selection accepts masked entries and local-block forcing");
    check(ids[0] == 0 && ids[1] == 1 && ids[2] == 2,
          "forced current block is retained and score ties prefer the lower ID");
    check(ids[3] == 0 && ids[4] == 1 && ids[5] == 2,
          "selection is independent per index head and returned in canonical order");

    const float aq[] = {1, 0, 0, 1};
    const float ak[] = {1, 0, 0, 1, 100, 100, 100, 100};
    const float av[] = {1, 0, 0, 2, 50, 50, 50, 50};
    const int32_t apos[] = {1};
    const int32_t akeypos[] = {0, 1, 2, 3};
    const int32_t aselected[] = {0, 1};
    float attn[4] = {0};
    check(strata_minimax_msa_attention_f32(aq, ak, av, apos, akeypos, aselected,
              1, 2, 1, 2, 4, 2, 2, 2, 2, attn) == STRATA_MINIMAX_MSA_OK,
          "sparse grouped attention accepts selected causal blocks");
    const float p = 1.0f / (1.0f + expf(-1.0f / sqrtf(2.0f)));
    check(fabsf(attn[0] - p) < 1e-6f && fabsf(attn[1] - 2.0f * (1.0f - p)) < 1e-6f &&
              fabsf(attn[2] - (1.0f - p)) < 1e-6f && fabsf(attn[3] - 2.0f * p) < 1e-6f,
          "grouped attention maps heads to KV correctly and masks future values");

    float bad_q[] = {NAN, 0, 0, 1, 1, 0, 0, 1};
    check(strata_minimax_msa_block_scores_f32(bad_q, k, qpos, kpos, 2, 2, 2, 8, 2, 4, scores, 6) ==
              STRATA_MINIMAX_MSA_NONFINITE_VALUE,
          "nonfinite indexer values are rejected");
    check(strata_minimax_msa_select_blocks(rank_scores, one_pos, 1, 2, 4, 2, 3, 4, 6, ids, 3) ==
              STRATA_MINIMAX_MSA_INVALID_ARGUMENT,
          "impossible forced-local width is rejected");
    if (failures) return 1;
    puts("MiniMax MSA host reference passed");
    return 0;
}
