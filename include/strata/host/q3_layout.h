#ifndef STRATA_HOST_Q3_LAYOUT_H
#define STRATA_HOST_Q3_LAYOUT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define STRATA_Q3K_GGML_BLOCK_BYTES 110u
#define STRATA_Q3K_EXPANDED_BLOCK_BYTES 114u
/* Experimental CPU-only Q3_K scale layout, per 256 weights:
 * original: 96 code/high-mask bytes, 12 packed scale bytes, 2 FP16 scale bytes;
 * expanded: same 96 bytes, 16 signed two's-complement scale bytes (-32..31),
 * same 2 opaque FP16 bytes. Existing GGML/GPU kernels cannot consume it.
 * Extents must describe the same whole-block count. Buffers must be disjoint;
 * in-place/partial overlap, null nonempty buffers, address/size overflow and
 * invalid extents fail before writes. Empty input/output accepts null buffers.
 * Compact additionally rejects invalid signed scale bytes before any writes.
 * Return 0 on success, -1 on invalid input. No allocations or float rounding. */
int strata_q3k_expand_scales(const uint8_t *input, size_t input_bytes,
                           uint8_t *output, size_t output_bytes);
int strata_q3k_compact_scales(const uint8_t *input, size_t input_bytes,
                            uint8_t *output, size_t output_bytes);
/* Complete Q2_K gate/up + Q3_K down blob geometry. Dimensions must be
 * positive multiples of 256. Checked arithmetic; failure leaves layout intact. */
typedef struct strata_q2q3_layout {
    size_t gu_row, d_row, up_off, down_off, bytes;
} strata_q2q3_layout;
int strata_q2q3_scale_layout(size_t embedding, size_t hidden, int expanded,
                            strata_q2q3_layout *layout);
/* Copy gate/up bytes and expand only down scales. Exact complete-blob extents,
 * disjoint nonnull buffers required. Any invalid argument fails before writes. */
int strata_q2q3_expand_down(size_t embedding, size_t hidden,
                           const uint8_t *input, size_t input_bytes,
                           uint8_t *output, size_t output_bytes);
#ifdef __cplusplus
}
#endif
#endif
