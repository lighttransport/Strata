#ifndef STRATA_HOST_GLM_FP4_H
#define STRATA_HOST_GLM_FP4_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* csf_32x2_v1 NVFP4 scales. Components are little-endian byte streams.
 * Output is logical row-major E4M3 bytes; no rounding or requantization.
 * Returns zero on success. Malformed input is rejected before output writes. */
int strata_glm_csf32_decode(size_t rows, size_t columns,
    const uint8_t *bases, size_t bases_bytes, const uint8_t *codes, size_t codes_bytes,
    const uint8_t *pages, size_t pages_bytes, const uint8_t *offsets, size_t offsets_bytes,
    const uint8_t *exceptions, size_t exceptions_bytes, uint8_t *out, size_t out_bytes);
float strata_glm_e2m1(uint8_t code);
float strata_glm_e4m3(uint8_t code);
float strata_glm_ue8m0(uint8_t code);
/* CPU-only, size-preserving K32 layout. Each 16-byte output block stores
 * values 0..15 in low nibbles and 16..31 in high nibbles. inverse=1 restores
 * adjacent nibbles. bytes must be a multiple of 16. Exact in-place conversion
 * is supported; partial overlap is rejected before writes. Returns 0 on success. */
int strata_glm_fp4_repack_k32(const uint8_t *input, uint8_t *output,
    size_t bytes, int inverse);
/* Experimental CPU-only NVFP4 layout conversion. Input/output contain complete
 * 36-byte K64 blocks (four scale bytes, 32 code bytes), excluding global tails.
 * Each K32 pair changes between GGML's two split-8 groups and low-16/high-16
 * nibbles. This permutation is its own inverse. Scale bytes are opaque and
 * unchanged. Existing GGML/GPU kernels cannot consume the converted layout.
 * Exact in-place conversion is supported; partial overlap, null nonempty
 * buffers, address overflow and non-multiple-of-36 lengths fail before writes.
 * Zero bytes accepts null buffers. Returns 0 on success, -1 on invalid input. */
int strata_glm_nvfp4_repack_k32(const uint8_t *input, uint8_t *output, size_t bytes);
/* Exact local adjacent-nibble codes + row-major E4M3 scales to GGML NVFP4:
 * 64 values per block, four scale bytes then 32 code bytes. Columns must be
 * divisible by 64; scales must be finite nonnegative codes 0..126. Global
 * scale remains separate. Buffer extents must be exact; output must not
 * overlap either input. All validation precedes output writes. */
int strata_glm_fp4_pack_nvfp4(size_t rows, size_t columns,
    const uint8_t *codes, size_t codes_bytes, const uint8_t *scales, size_t scales_bytes,
    uint8_t *output, size_t output_bytes);
/* Lossless ModelOpt weight codecs shared by the native loader and uploaders (origin/glm53f). */
float strata_fp4_e2m1(uint8_t code);
float strata_fp4_e4m3(uint8_t code);
float strata_fp4_e8m0(uint8_t code);

/* CSF32 stores 32-row x 2-column blocks, in 128-row x 4-column tiles.
 * Every 32 tiles start a new exception page. Offsets are page-relative;
 * page_bases are BYTE offsets into the U16 exception stream. Output is logical row-major
 * E4M3 bytes, without rounding or conversion to another weight format.
 * All lengths are elements, rather than bytes. Returns zero on success. */
int strata_fp4_csf32_decode(size_t rows, size_t columns,
    const uint8_t *bases, size_t bases_count,
    const uint8_t *codes, size_t codes_count,
    const uint16_t *offsets, size_t offsets_count,
    const uint32_t *page_bases, size_t pages_count,
    const uint16_t *exceptions, size_t exceptions_count,
    uint8_t *output, size_t output_count);

/* Check structure and exception bounds without expanding the scale plane.
 * This validates the same metadata as decode; scale values are checked by
 * the uploader after applying the exceptions. */
int strata_fp4_csf32_validate(size_t rows, size_t columns,
    const uint8_t *bases, size_t bases_count,
    const uint8_t *codes, size_t codes_count,
    const uint16_t *offsets, size_t offsets_count,
    const uint32_t *page_bases, size_t pages_count,
    const uint16_t *exceptions, size_t exceptions_count);
#ifdef __cplusplus
}
#endif
#endif
