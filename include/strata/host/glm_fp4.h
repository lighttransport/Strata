/* Lossless ModelOpt weight codecs shared by the native loader and uploaders. */
#ifndef STRATA_HOST_GLM_FP4_H
#define STRATA_HOST_GLM_FP4_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

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
