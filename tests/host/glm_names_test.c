// tests/host/glm_names_test.c - the naming module (no model, no GPU).
#include "strata/host/glm_names.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c)                                                   \
    do {                                                           \
        if (!(c)) {                                                \
            (void)printf("FAIL line %d: %s\n", __LINE__, #c);      \
            ++fails;                                               \
        }                                                          \
    } while (0)

static int same(const char *got, const char *want) { return got != NULL && strcmp(got, want) == 0; }

int main(void) {
    char buf[STRATA_NAME_MAX];
    int layer = -1;
    const char *suffix = NULL;

    /* The spellings every caller used to write by hand. */
    CHECK(strata_blk_name(buf, sizeof buf, 44, "ffn_gate_exps.weight") == (int)strlen(buf));
    CHECK(same(buf, "blk.44.ffn_gate_exps.weight"));
    CHECK(strata_blk_name(buf, sizeof buf, 3, "attn_qkv.weight") == (int)strlen(buf));
    CHECK(same(buf, "blk.3.attn_qkv.weight"));
    CHECK(strata_blk_name(buf, sizeof buf, 47, "ffn_down_exps.weight") == (int)strlen(buf));
    CHECK(same(buf, "blk.47.ffn_down_exps.weight"));
    CHECK(strata_blk_name(buf, sizeof buf, 0, "ffn_gate_exps.weight") == 26);
    CHECK(same(buf, "blk.0.ffn_gate_exps.weight"));
    CHECK(strata_blk_prefix(buf, sizeof buf, 7) == 6);
    CHECK(same(buf, "blk.7."));
    CHECK(strata_blk_prefix(buf, sizeof buf, 0) == 6);
    CHECK(same(buf, "blk.0."));

    /* Truncation is an error, never a short name: a short name resolves to nothing, or to something else. */
    CHECK(strata_blk_name(buf, 8, 44, "ffn_gate_exps.weight") == -1);
    CHECK(strata_blk_name(buf, 0, 1, "x") == -1);
    CHECK(strata_blk_name(NULL, sizeof buf, 1, "x") == -1);
    CHECK(strata_blk_name(buf, sizeof buf, -1, "x") == -1);

    /* Parsing: 1 owns the name, 0 is "not a block name", -1 is "malformed block name". */
    CHECK(strata_parse_blk("blk.12.attn_q.weight", &layer, &suffix) == 1);
    CHECK(layer == 12);
    CHECK(same(suffix, "attn_q.weight"));
    CHECK(strata_parse_blk("blk.0.", &layer, &suffix) == -1); /* names no tensor */
    CHECK(strata_parse_blk("blk.12", &layer, &suffix) == -1);
    CHECK(strata_parse_blk("blk.x.y", &layer, &suffix) == -1);
    CHECK(strata_parse_blk("blk.99999999.y", &layer, &suffix) == -1); /* a layer that cannot be right */
    CHECK(strata_parse_blk("blk.-1.y", &layer, &suffix) == -1);
    CHECK(strata_parse_blk("tokenizer.ggml.tokens", &layer, &suffix) == 0);
    CHECK(strata_parse_blk("blkx.1.y", &layer, &suffix) == 0);
    CHECK(strata_parse_blk(NULL, &layer, &suffix) == -1);

    /* Round trip: what the scan builds, the parser reads back the same way. */
    CHECK(strata_blk_name(buf, sizeof buf, 44, "ffn_down_exps.weight") == (int)strlen(buf));
    CHECK(strata_parse_blk(buf, &layer, &suffix) == 1);
    CHECK(layer == 44);
    CHECK(same(suffix, "ffn_down_exps.weight"));

    if (fails != 0) return 1;
    (void)printf("names OK %zu\n", strlen(buf));
    return 0;
}
