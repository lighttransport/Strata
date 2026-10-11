// src/host/glm_expert_pack.c - the assembled expert pack's rules, in one place.  See the header for why.
#include "strata/host/glm_expert_pack.h"

#include <stdio.h>

/* Layer order, then part order.  Both sides of the round trip iterate in this exact order, so the array one
   builds is the array the other checks. */
static const char *const kParts[STRATA_EP_PART_COUNT] = {"gate", "up", "down"};

static int fail(char *err, size_t cap, int code, const char *message) {
    if (err != NULL && cap != 0) (void)snprintf(err, cap, "%s", message);
    return code;
}

static int fail_named(char *err, size_t cap, int code, const char *what, const char *name) {
    if (err != NULL && cap != 0) (void)snprintf(err, cap, "%s %s", what, name);
    return code;
}

static int accepted_retained_type(int type) {
    return type == STRATA_EP_TYPE_IQ2_XS || type == STRATA_EP_TYPE_IQ3_XXS || type == STRATA_EP_TYPE_IQ4_XS ||
           type == STRATA_EP_TYPE_Q4_K || type == STRATA_EP_TYPE_Q5_K || type == STRATA_EP_TYPE_Q6_K;
}

int strata_ep_scan(void *context, strata_ep_lookup lookup, strata_ep_projection *out, size_t cap, size_t *count,
                   char *err, size_t err_cap) {
    size_t at = 0;
    int layer, part_index;
    int status = STRATA_EP_OK;
    if (count == NULL || lookup == NULL || (out == NULL && cap != 0))
        return fail(err, err_cap, STRATA_EP_ERR_ARGS, "invalid routed-projection scan");
    for (layer = STRATA_EP_FIRST_LAYER; layer <= STRATA_EP_LAST_LAYER; ++layer) {
        for (part_index = 0; part_index < STRATA_EP_PART_COUNT; ++part_index) {
            char suffix[STRATA_NAME_MAX];
            strata_ep_projection entry;
            int written;
            if (at >= cap) {
                status = fail(err, err_cap, STRATA_EP_ERR_ARGS, "too many routed projections");
                break;
            }
            written = snprintf(suffix, sizeof suffix, "ffn_%s_exps.weight", kParts[part_index]);
            if (written < 0 || (size_t)written >= sizeof suffix) {
                status = fail(err, err_cap, STRATA_EP_ERR_ARGS, "routed projection name does not fit");
                break;
            }
            if (strata_blk_name(entry.name, sizeof entry.name, layer, suffix) < 0) {
                status = fail(err, err_cap, STRATA_EP_ERR_ARGS, "routed projection name does not fit");
                break;
            }
            /* The lookup answers the artifact's part of the question: the tensor type and whether the pack
               recorded a checksum.  It leaves entry.name and entry.part alone. */
            entry.part = part_index;
            if (lookup(context, layer, kParts[part_index], &entry) != 0) {
                status = fail_named(err, err_cap, STRATA_EP_ERR_MISSING, "missing routed projection", entry.name);
                break;
            }
            out[at++] = entry;
        }
        if (status != STRATA_EP_OK) break;
    }
    if (status != STRATA_EP_OK) return status;
    *count = at;
    return STRATA_EP_OK;
}

int strata_ep_count_retained(const strata_ep_projection *projections, size_t count, uint32_t *retained,
                             char *err, size_t err_cap) {
    size_t layer;
    if (projections == NULL || (count != 0 && retained == NULL))
        return fail(err, err_cap, STRATA_EP_ERR_ARGS, "invalid routed-projection list");
    if (retained != NULL) *retained = 0;
    for (layer = 0; layer < count; ++layer) {
        const strata_ep_projection *p = &projections[layer];
        if (p->name[0] == '\0') return fail(err, err_cap, STRATA_EP_ERR_ARGS, "unnamed routed projection");
        if (p->hashed) {
            /* A checksum only certifies a payload the overlay wrote itself: Q2_K, or Q3_K on the down
               projections that are allowed it.  Anything else after a checksum is a pack that recorded a
               number for bytes it did not convert. */
            if (p->hash_type != STRATA_EP_META_U64 ||
                (p->type != STRATA_EP_TYPE_Q2_K && !(p->part == STRATA_EP_PART_DOWN && p->type == STRATA_EP_TYPE_Q3_K)))
                return fail(err, err_cap, STRATA_EP_ERR_INCOMPLETE, "incomplete assembled Q23 experts");
            continue;
        }
        if (!accepted_retained_type(p->type)) return fail(err, err_cap, STRATA_EP_ERR_INCOMPLETE,
                                                          "incomplete assembled Q23 experts");
    }
    /* Gate/up share one NativeFmt, so they stay paired: either the overlay converted both, or neither.  Parts
       sit in gate,up,down order, hence stride 3. */
    for (layer = 0; layer + 1 < count; layer += STRATA_EP_PART_COUNT) {
        if (projections[layer].hashed != projections[layer + 1].hashed)
            return fail(err, err_cap, STRATA_EP_ERR_PAIR, "assembled gate/up must be paired");
    }
    for (layer = 0; layer < count; ++layer) {
        if (!projections[layer].hashed && retained != NULL) ++*retained;
    }
    return STRATA_EP_OK;
}

int strata_ep_check_projections(const strata_ep_projection *projections, size_t count, uint32_t declared,
                                uint32_t *retained, char *err, size_t err_cap) {
    uint32_t counted = 0;
    int status = strata_ep_count_retained(projections, count, &counted, err, err_cap);
    if (status != STRATA_EP_OK) return status;
    if (counted != declared) return fail(err, err_cap, STRATA_EP_ERR_COUNT,
                                        "assembled retained-projection count mismatch");
    if (retained != NULL) *retained = counted;
    return STRATA_EP_OK;
}

int strata_ep_check_provenance(strata_ep_key assembled, strata_ep_key version, strata_ep_key source, char *err,
                               size_t err_cap) {
    if (!assembled.present || assembled.type != STRATA_EP_META_BOOL || assembled.u != 1) goto bad;
    if (!version.present || version.type != STRATA_EP_META_U32 || version.u != 1) goto bad;
    if (!source.present || source.type != STRATA_EP_META_U64) goto bad;
    return STRATA_EP_OK;
bad:
    return fail(err, err_cap, STRATA_EP_ERR_PROVENANCE, "invalid assembled expert provenance");
}

int strata_ep_check_retained_key(strata_ep_key retained, uint32_t *declared, char *err, size_t err_cap) {
    if (declared != NULL) *declared = 0;
    if (!retained.present) return STRATA_EP_OK; /* older assemblies predate the key: nothing retained */
    if (retained.type != STRATA_EP_META_U32 || retained.u > 0xffffffffu)
        return fail(err, err_cap, STRATA_EP_ERR_RETAINED, "invalid assembled retained-projection count");
    if (declared != NULL) *declared = (uint32_t)retained.u;
    return STRATA_EP_OK;
}
