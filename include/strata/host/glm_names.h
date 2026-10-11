// include/strata/host/glm_names.h - the tensor names, spelled once.
//
// Every host-side caller that reaches for a GLM tensor builds `blk.<layer>.<suffix>` out of pieces:
// `"blk." + std::to_string(l) + "."`, then a suffix appended per call site.  Outside the decoder that is
// 15 places, and each one allocates a `std::string` to name a tensor that is then looked up by comparing
// strings again - on a path that runs once per LOAD, which is exactly why it has never cost anything and
// exactly why it has drifted: three of those spellings are the pack metadata prefix
// (`strata.expert_pack.hash.<tensor>`), and a typo there fails as a missing checksum rather than as a
// name.
//
// This is C11 so it is callable from the decoder's host code and from any future C caller without a
// shim, and so the spelling is checked in one unit test rather than in 15 readers' heads.  The portable
// subset is deliberate: no `_Generic`, no VLAs, no `<threads.h>`, nothing MSVC lacks - this file builds
// on Windows next to the C++ engine rather than behind a #ifdef.
#ifndef STRATA_HOST_GLM_NAMES_H_
#define STRATA_HOST_GLM_NAMES_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Longest name this module will build.  `blk.44.ffn_gate_exps.weight` is 26 bytes; 128 leaves room for the
   longer per-layer suffixes (`blk.7.ssm_conv1d_q.weight`, `blk.47.attn_q_a.weight`) and for a caller to
   suffix something else on top.  A name that does not fit returns -1 rather than being truncated, because
   a truncated tensor name is a name that resolves to nothing, or worse, to something else. */
#define STRATA_NAME_MAX 128

/* `blk.<layer>.<suffix>` -> out.  Returns the length written, or -1 if it does not fit in cap
   (suffix == NULL writes the prefix alone, as strata_blk_prefix does). */
int strata_blk_name(char *out, size_t cap, int layer, const char *suffix);

/* `blk.<layer>.` -> out.  The trailing dot is part of the name: every caller suffixes onto it. */
int strata_blk_prefix(char *out, size_t cap, int layer);

/* Splits `blk.<layer>.<suffix>` back apart.  Returns 1 and sets *layer / *suffix (suffix points into
   `name`, not into a copy), 0 if the name is not a `blk.` name at all, and -1 if it starts the right way
   but the layer is not a number or the suffix is empty.  The distinction matters: 0 means "not mine, look
   elsewhere", -1 means "this is malformed and should not be silently skipped". */
int strata_parse_blk(const char *name, int *layer, const char **suffix);

#ifdef __cplusplus
}
#endif

#endif // STRATA_HOST_GLM_NAMES_H_
