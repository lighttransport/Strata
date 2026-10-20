// include/strata/host/ds41f_names.h - DeepSeek V4.1 Flash tensor names, spelled once (the glm_names.h pattern).
//
// The release checkpoint (safetensors) names tensors `layers.<L>.ffn.experts.<E>.w{1,2,3}.{weight,scale}`,
// `layers.<L>.engram.embed.{weight,scale}`, `layers.<L>.attn.*`, `embed.weight`, `head.weight`. The Strata sidecar
// pack and the trunk GGUF use GGUF-style names `blk.<L>.ffn_{gate,up,down}_exps.weight` for the routed experts
// (w1 = gate, w3 = up, w2 = down, as in the reference forward) and the safetensors names verbatim for everything
// else. Both spellings live here so the pack tool, its verifier and a later loader cannot drift apart.
#ifndef STRATA_HOST_DS41F_NAMES_H_
#define STRATA_HOST_DS41F_NAMES_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STRATA_DS41F_NAME_MAX 128
#define STRATA_DS41F_LAYERS 40
#define STRATA_DS41F_EXPERTS 384
#define STRATA_DS41F_HIDDEN 5120
#define STRATA_DS41F_EXPERT_FF 2304
#define STRATA_DS41F_ENGRAM_LAYER_COUNT 2

/* Projection index: 0 = gate (w1), 1 = up (w3), 2 = down (w2). */
enum strata_ds41f_proj { STRATA_DS41F_GATE = 0, STRATA_DS41F_UP = 1, STRATA_DS41F_DOWN = 2 };

/* `layers.<L>.ffn.experts.<E>.w<N>.weight` or `.scale` (part = "weight" | "scale"). Returns the length or -1. */
int strata_ds41f_expert_source_name(char *out, size_t cap, int layer, int expert, enum strata_ds41f_proj proj, const char *part);

/* `blk.<L>.ffn_<gate|up|down>_exps.weight`. Returns the length or -1. */
int strata_ds41f_expert_pack_name(char *out, size_t cap, int layer, enum strata_ds41f_proj proj);

/* `blk.<L>.ffn_exps_keep` : the I32 list of kept source expert ids of a REAP-pruned layer. */
int strata_ds41f_keep_name(char *out, size_t cap, int layer);

/* Splits `blk.<L>.ffn_<gate|up|down>_exps.weight`: 1 and sets *layer / *proj on a match, 0 if not an expert pack
   name, -1 if malformed. */
int strata_ds41f_parse_pack_name(const char *name, int *layer, enum strata_ds41f_proj *proj);

/* The safetensors source checkpoint's w<N> for a projection: gate -> 1, up -> 3, down -> 2. */
int strata_ds41f_source_w(enum strata_ds41f_proj proj);

/* Engram layers (1 and 14) and whether a safetensors name is an Engram table (`layers.<L>.engram.embed.*`). */
int strata_ds41f_engram_layer(int index);
int strata_ds41f_is_engram_table(const char *name);

/* Whether a safetensors name is a routed-expert tensor (`layers.<L>.ffn.experts.<E>.*`). */
int strata_ds41f_is_routed_expert(const char *name);

#ifdef __cplusplus
}
#endif

#endif // STRATA_HOST_DS41F_NAMES_H_
