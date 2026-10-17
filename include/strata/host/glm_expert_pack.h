// include/strata/host/glm_expert_pack.h - the assembled expert pack's rules, in one place.
//
// An "assembled" model is one GGUF: the original file's other tensors plus the expert overlay's payload,
// with `strata.expert_pack.*` keys recording what was done.  Two programs write and read that agreement:
// `tools/glm_assemble.cpp` counts the routed projections the overlay left in their original format and
// records the number, and `ModelArtifact` (include/strata/core/model.hpp) re-derives that count when the file
// is opened and refuses a mismatch.
//
// They used to spell the rules twice.  The assembler's loop was `for(int l=3;l<=44;++l) for(part :
// {"gate","up","down"})`, the loader's was `for (int l=3;l<=44;++l)` with an inner range-for over the same
// three strings, the accepted type sets were two separate literals, and neither could reference the other.
// That is the shape this project's own comments keep naming as the expensive failure: two places computing
// one thing, free to disagree.
//
// So the rules live here, in C11, and both sides iterate through `strata_ep_scan` - which also names each
// projection through `glm_names` rather than through ad-hoc `+` concatenation.  Portable subset, same
// reason as glm_names.h.
#ifndef STRATA_HOST_GLM_EXPERT_PACK_H_
#define STRATA_HOST_GLM_EXPERT_PACK_H_

#include <stddef.h>
#include <stdint.h>

#include "strata/host/glm_names.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The routed projections live on layers 3..44 (0..2 are dense, 45+ carry no routed experts). */
#define STRATA_EP_FIRST_LAYER 3
#define STRATA_EP_LAST_LAYER 44

/* Gate/up become Q2_K and down becomes Q3_K (or Q2_K in the UD-Q2_K_XL pack), so the rules above differ by
   part and cannot be written without naming it. */
#define STRATA_EP_PART_COUNT 3
#define STRATA_EP_PART_GATE 0
#define STRATA_EP_PART_UP 1
#define STRATA_EP_PART_DOWN 2

/* One entry per routed projection, in layer-then-gate/up/down order. */
#define STRATA_EP_PROJECTIONS ((size_t)(STRATA_EP_LAST_LAYER - STRATA_EP_FIRST_LAYER + 1) * STRATA_EP_PART_COUNT)

/* GGUF metadata types (`enum class MetaType` in gguf_reader.hpp).  Different numbering from the GGML tensor
   types below - 10 is U64 here and Q2_K there, which is exactly why both sets are named here rather than
   written as literals at their use site. */
#define STRATA_EP_META_U32 4
#define STRATA_EP_META_BOOL 7
#define STRATA_EP_META_U64 10

/* GGML tensor types.  The overlay converts gate/up to Q2_K and down to Q3_K (or Q2_K, the UD-Q2_K_XL pack);
   a projection the overlay left ALONE is still one of the source formats it accepts. */
#define STRATA_EP_TYPE_Q2_K 10
#define STRATA_EP_TYPE_Q3_K 11
#define STRATA_EP_TYPE_Q4_K 12
#define STRATA_EP_TYPE_Q5_K 13
#define STRATA_EP_TYPE_Q6_K 14
#define STRATA_EP_TYPE_IQ2_XXS 16
#define STRATA_EP_TYPE_IQ2_XS 17
#define STRATA_EP_TYPE_IQ3_XXS 18
#define STRATA_EP_TYPE_IQ3_S 21
#define STRATA_EP_TYPE_IQ2_S 22
#define STRATA_EP_TYPE_IQ4_XS 23

/* A GGUF metadata value this module is asked about.  `type` is the MetaType the file claims, ignored when
   `present` is 0 - a key that is absent is not an error by itself, except where the pack requires it. */
typedef struct {
    int present;
    int type;
    uint64_t u;
} strata_ep_key;

/* One routed projection in engine terms. */
typedef struct {
    char name[STRATA_NAME_MAX]; /* "blk.<l>.ffn_gate_exps.weight" */
    int type;                   /* the artifact's tensor type */
    int part;                   /* STRATA_EP_PART_GATE / _UP / _DOWN */
    int hashed;                 /* the pack recorded a payload checksum for this projection */
    int hash_type;              /* that checksum key's metadata type, meaningful only when hashed */
    int from_fp8;               /* this projection's pack has a BOOL true strata.expert_pack.fp8 key */
} strata_ep_projection;

/* Bytes per 256 weights for a supported FP8-derived expert target, or 0 for an unsupported type. */
uint32_t strata_ep_fp8_block_bytes(int type);

/* How a caller resolves a projection.  The artifact is C++ on the loader side and raw bytes on any other, so
   the container is the caller's: return 0 and fill `out`'s type/hashed/hash_type (the scan has already set
   `out`'s name and part), or -1 if the projection is not in the artifact. */
typedef int (*strata_ep_lookup)(void *context, int layer, const char *part, strata_ep_projection *out);

enum {
    STRATA_EP_OK = 0,
    STRATA_EP_ERR_ARGS,        /* null pointer, table too small, count above capacity */
    STRATA_EP_ERR_MISSING,     /* the artifact has no such routed projection */
    STRATA_EP_ERR_PROVENANCE,  /* assembled / version / source are missing, or not the type they must be */
    STRATA_EP_ERR_RETAINED,    /* the retained-count key is present but not a U32 */
    STRATA_EP_ERR_INCOMPLETE,  /* a projection is neither a hashed Q2/Q3 target nor an accepted source type */
    STRATA_EP_ERR_PAIR,        /* gate and up do not both stay original (they share a NativeFmt) */
    STRATA_EP_ERR_COUNT        /* the counted retained projections do not equal the recorded key */
};

/* Fills `out` (cap entries) with every routed projection, naming each one, and reports how many there were.
   The only place the layer range, the part order and the `ffn_<part>_exps.weight` spelling are written; the
   callback itself is the caller's container. */
int strata_ep_scan(void *context, strata_ep_lookup lookup, strata_ep_projection *out, size_t cap, size_t *count,
                   char *err, size_t err_cap);

/* The count of projections the overlay left in their original format. */
int strata_ep_count_retained(const strata_ep_projection *projections, size_t count, uint32_t *retained,
                             char *err, size_t err_cap);

/* The provenance keys: assembled must be BOOL 1, version U32 1, source U64 (any value). */
int strata_ep_check_provenance(strata_ep_key assembled, strata_ep_key version, strata_ep_key source, char *err,
                               size_t err_cap);

/* The recorded retained count, when the assembly has the key at all: a U32, and 0 when it is absent (older
   assemblies predate the key). */
int strata_ep_check_retained_key(strata_ep_key retained, uint32_t *declared, char *err, size_t err_cap);

/* The whole agreement: every projection is either a hashed target or an accepted retained source, gate and up
   are paired, and the counted retained number equals the recorded one.  A caller that only wants the count
   (the assembler, before it writes the key) calls strata_ep_count_retained directly. */
int strata_ep_check_projections(const strata_ep_projection *projections, size_t count, uint32_t declared,
                                uint32_t *retained, char *err, size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif // STRATA_HOST_GLM_EXPERT_PACK_H_
