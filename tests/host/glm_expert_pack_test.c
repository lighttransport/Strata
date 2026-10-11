// tests/host/glm_expert_pack_test.c - the assembled pack's rules (no model, no GPU).
//
// The interesting cases are the ones where the two programs used to disagree: a projection the overlay
// converted but not for the part it is allowed to be, a gate/up pair where only one came back original, and
// a recorded count that no longer matches the file.
#include "strata/host/glm_expert_pack.h"

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

/* A synthetic artifact: every projection is either converted-and-hashed or left original, decided by layer
   and part so the tests can poke one cell at a time. */
typedef struct {
    int hashed[STRATA_EP_PROJECTIONS];
    int type[STRATA_EP_PROJECTIONS];
    int missing; /* index of the projection to hide */
} artifact;

static int lookup(void *context, int layer, const char *part, strata_ep_projection *out) {
    artifact *a = (artifact *)context;
    int part_index = 0;
    size_t at;
    if (a == NULL) return -1; /* no artifact: nothing to answer about */
    if (strcmp(part, "gate") == 0) part_index = STRATA_EP_PART_GATE;
    else if (strcmp(part, "up") == 0) part_index = STRATA_EP_PART_UP;
    else if (strcmp(part, "down") == 0) part_index = STRATA_EP_PART_DOWN;
    at = (size_t)(layer - STRATA_EP_FIRST_LAYER) * STRATA_EP_PART_COUNT + (size_t)part_index;
    if (at >= STRATA_EP_PROJECTIONS || (int)at == a->missing) return -1;
    out->type = a->type[at];
    out->hashed = a->hashed[at];
    out->hash_type = STRATA_EP_META_U64;
    return 0;
}

static strata_ep_key key(int present, int type, uint64_t u) {
    strata_ep_key k;
    k.present = present;
    k.type = type;
    k.u = u;
    return k;
}

static void fill(artifact *a, int hashed, int converted_type, int original_type) {
    size_t i;
    for (i = 0; i < STRATA_EP_PROJECTIONS; ++i) {
        int part = (int)(i % STRATA_EP_PART_COUNT);
        a->hashed[i] = hashed;
        a->type[i] = hashed ? (part == STRATA_EP_PART_DOWN ? converted_type : STRATA_EP_TYPE_Q2_K) : original_type;
    }
    a->missing = -1;
}

int main(void) {
    artifact a;
    strata_ep_projection list[STRATA_EP_PROJECTIONS];
    size_t count = 0;
    uint32_t retained = 0, declared = 0;
    char message[256];

    /* The scan walks the layers and parts, names every projection, and reports how many there are. */
    fill(&a, 1, STRATA_EP_TYPE_Q3_K, STRATA_EP_TYPE_IQ2_XS);
    CHECK(strata_ep_scan(&a, lookup, list, STRATA_EP_PROJECTIONS, &count, message, sizeof message) ==
          STRATA_EP_OK);
    CHECK(count == STRATA_EP_PROJECTIONS);
    CHECK(strcmp(list[0].name, "blk.3.ffn_gate_exps.weight") == 0);
    CHECK(strcmp(list[1].name, "blk.3.ffn_up_exps.weight") == 0);
    CHECK(strcmp(list[2].name, "blk.3.ffn_down_exps.weight") == 0);
    CHECK(strcmp(list[STRATA_EP_PROJECTIONS - 1].name, "blk.44.ffn_down_exps.weight") == 0);
    CHECK(list[0].part == STRATA_EP_PART_GATE);
    CHECK(list[1].part == STRATA_EP_PART_UP);
    CHECK(list[2].part == STRATA_EP_PART_DOWN);
    CHECK(strata_ep_scan(&a, lookup, list, 3, &count, message, sizeof message) == STRATA_EP_ERR_ARGS);

    /* A projection the artifact does not have is named in the message: it is the file's problem. */
    a.missing = 5;
    CHECK(strata_ep_scan(&a, lookup, list, STRATA_EP_PROJECTIONS, &count, message, sizeof message) ==
          STRATA_EP_ERR_MISSING);
    CHECK(strstr(message, "blk.4.ffn_down_exps.weight") != NULL);
    a.missing = -1;

    /* Fully converted: nothing retained, so the recorded count must be 0. */
    CHECK(strata_ep_count_retained(list, count, &retained, message, sizeof message) == STRATA_EP_OK);
    CHECK(retained == 0);
    CHECK(strata_ep_check_projections(list, count, 0, &retained, message, sizeof message) == STRATA_EP_OK);

    /* The overlapping layer 11/12/44 case of UD-Q2_K_XL: one whole layer left original. */
    fill(&a, 0, STRATA_EP_TYPE_Q3_K, STRATA_EP_TYPE_Q4_K);
    CHECK(strata_ep_scan(&a, lookup, list, STRATA_EP_PROJECTIONS, &count, message, sizeof message) ==
          STRATA_EP_OK);
    CHECK(strata_ep_count_retained(list, count, &retained, message, sizeof message) == STRATA_EP_OK);
    CHECK(retained == STRATA_EP_PROJECTIONS);
    CHECK(strata_ep_check_projections(list, count, (uint32_t)STRATA_EP_PROJECTIONS, &retained, message,
                                      sizeof message) == STRATA_EP_OK);

    /* One layer's gate/up hashed and down original: allowed (down may stay original on its own). */
    fill(&a, 1, STRATA_EP_TYPE_Q3_K, STRATA_EP_TYPE_IQ3_XXS);
    CHECK(strata_ep_scan(&a, lookup, list, STRATA_EP_PROJECTIONS, &count, message, sizeof message) ==
          STRATA_EP_OK);
    {
        size_t at = 2u * STRATA_EP_PART_COUNT + STRATA_EP_PART_DOWN; /* layer 5, down */
        list[at].hashed = 0;
        list[at].type = STRATA_EP_TYPE_IQ3_XXS;
    }
    CHECK(strata_ep_count_retained(list, count, &retained, message, sizeof message) == STRATA_EP_OK);
    CHECK(retained == 1);

    /* Gate/up must be paired: they share one NativeFmt, so one of them coming back original is refused. */
    list[2u * STRATA_EP_PART_COUNT + STRATA_EP_PART_DOWN].hashed = 1;
    list[2u * STRATA_EP_PART_COUNT + STRATA_EP_PART_DOWN].type = STRATA_EP_TYPE_Q3_K;
    list[2u * STRATA_EP_PART_COUNT + STRATA_EP_PART_UP].hashed = 0;
    list[2u * STRATA_EP_PART_COUNT + STRATA_EP_PART_UP].type = STRATA_EP_TYPE_IQ2_XS;
    CHECK(strata_ep_check_projections(list, count, 0, &retained, message, sizeof message) ==
          STRATA_EP_ERR_PAIR);
    CHECK(strstr(message, "paired") != NULL);

    /* A checksum only certifies bytes the overlay wrote: hashed gate/up must be Q2_K, hashed down Q3_K or
       Q2_K.  A retained projection must be a source format the loader accepts. */
    list[2u * STRATA_EP_PART_COUNT + STRATA_EP_PART_UP].hashed = 1;
    list[2u * STRATA_EP_PART_COUNT + STRATA_EP_PART_UP].type = STRATA_EP_TYPE_Q4_K;
    CHECK(strata_ep_check_projections(list, count, 0, &retained, message, sizeof message) ==
          STRATA_EP_ERR_INCOMPLETE);
    list[2u * STRATA_EP_PART_COUNT + STRATA_EP_PART_UP].type = STRATA_EP_TYPE_Q3_K; /* gate/up may not be Q3_K */
    CHECK(strata_ep_check_projections(list, count, 0, &retained, message, sizeof message) ==
          STRATA_EP_ERR_INCOMPLETE);
    list[2u * STRATA_EP_PART_COUNT + STRATA_EP_PART_UP].type = STRATA_EP_TYPE_Q2_K;
    CHECK(strata_ep_count_retained(list, count, &retained, message, sizeof message) == STRATA_EP_OK);
    CHECK(retained == 0);
    fill(&a, 0, STRATA_EP_TYPE_Q3_K, STRATA_EP_TYPE_Q2_K); /* Q2_K is a target, not a retained source */
    CHECK(strata_ep_scan(&a, lookup, list, STRATA_EP_PROJECTIONS, &count, message, sizeof message) ==
          STRATA_EP_OK);
    CHECK(strata_ep_count_retained(list, count, &retained, message, sizeof message) == STRATA_EP_ERR_INCOMPLETE);

    /* A recorded count that the file does not support, and the key that is not a U32. */
    fill(&a, 1, STRATA_EP_TYPE_Q3_K, STRATA_EP_TYPE_IQ2_XS);
    CHECK(strata_ep_scan(&a, lookup, list, STRATA_EP_PROJECTIONS, &count, message, sizeof message) ==
          STRATA_EP_OK);
    CHECK(strata_ep_check_projections(list, count, 3, &retained, message, sizeof message) ==
          STRATA_EP_ERR_COUNT);
    CHECK(strstr(message, "count mismatch") != NULL);
    CHECK(strata_ep_check_retained_key(key(1, STRATA_EP_META_U64, 3), &declared, message, sizeof message) ==
          STRATA_EP_ERR_RETAINED);
    CHECK(strata_ep_check_retained_key(key(1, STRATA_EP_META_U32, 3), &declared, message, sizeof message) ==
          STRATA_EP_OK);
    CHECK(declared == 3);
    CHECK(strata_ep_check_retained_key(key(0, 0, 0), &declared, message, sizeof message) == STRATA_EP_OK);
    CHECK(declared == 0); /* an assembly from before the key existed: nothing retained */

    /* The provenance keys: assembled BOOL 1, version U32 1, source U64 whatever the fingerprint is. */
    CHECK(strata_ep_check_provenance(key(1, STRATA_EP_META_BOOL, 1), key(1, STRATA_EP_META_U32, 1),
                                     key(1, STRATA_EP_META_U64, 0xdeadbeefull), message, sizeof message) ==
          STRATA_EP_OK);
    CHECK(strata_ep_check_provenance(key(1, STRATA_EP_META_BOOL, 0), key(1, STRATA_EP_META_U32, 1),
                                     key(1, STRATA_EP_META_U64, 1), message, sizeof message) ==
          STRATA_EP_ERR_PROVENANCE);
    CHECK(strata_ep_check_provenance(key(1, STRATA_EP_META_U32, 1), key(1, STRATA_EP_META_U32, 1),
                                     key(1, STRATA_EP_META_U64, 1), message, sizeof message) ==
          STRATA_EP_ERR_PROVENANCE);
    CHECK(strata_ep_check_provenance(key(0, STRATA_EP_META_BOOL, 1), key(1, STRATA_EP_META_U32, 1),
                                     key(1, STRATA_EP_META_U64, 1), message, sizeof message) ==
          STRATA_EP_ERR_PROVENANCE);
    CHECK(strata_ep_check_provenance(key(1, STRATA_EP_META_BOOL, 1), key(0, STRATA_EP_META_U32, 1),
                                     key(1, STRATA_EP_META_U64, 1), message, sizeof message) ==
          STRATA_EP_ERR_PROVENANCE);
    CHECK(strata_ep_check_provenance(key(1, STRATA_EP_META_BOOL, 1), key(1, STRATA_EP_META_U32, 1),
                                     key(0, STRATA_EP_META_U64, 1), message, sizeof message) ==
          STRATA_EP_ERR_PROVENANCE);
    CHECK(strata_ep_check_provenance(key(1, STRATA_EP_META_BOOL, 1), key(1, STRATA_EP_META_U32, 2),
                                     key(1, STRATA_EP_META_U64, 1), message, sizeof message) ==
          STRATA_EP_ERR_PROVENANCE);

    /* No argument is trusted to be non-null.  An absent artifact answers "no such projection", like a file
       that does not have the tensor. */
    CHECK(strata_ep_scan(NULL, lookup, list, STRATA_EP_PROJECTIONS, &count, message, sizeof message) ==
          STRATA_EP_ERR_MISSING);
    CHECK(strata_ep_scan(&a, NULL, list, STRATA_EP_PROJECTIONS, &count, message, sizeof message) ==
          STRATA_EP_ERR_ARGS);
    CHECK(strata_ep_scan(&a, lookup, NULL, 0, &count, message, sizeof message) == STRATA_EP_ERR_ARGS);
    CHECK(strata_ep_scan(&a, lookup, list, STRATA_EP_PROJECTIONS, NULL, message, sizeof message) ==
          STRATA_EP_ERR_ARGS);
    CHECK(strata_ep_count_retained(list, count, NULL, message, sizeof message) == STRATA_EP_ERR_ARGS);

    if (fails != 0) return 1;
    (void)printf("expert_pack OK %lld projections\n", (long long)STRATA_EP_PROJECTIONS);
    return 0;
}
