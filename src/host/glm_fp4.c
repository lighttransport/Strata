#include "strata/host/glm_fp4.h"
#include <math.h>
#include <stdint.h>

float strata_fp4_e2m1(uint8_t c) {
    static const float values[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return c & 8 ? -values[c & 7] : values[c & 7];
}
float strata_fp4_e4m3(uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    float v = e == 15 && m == 7 ? NAN : e ? ldexpf(1.f + m * .125f, e - 7)
                                                        : ldexpf(m * .125f, -6);
    return c & 128 ? -v : v;
}
float strata_fp4_e8m0(uint8_t c) { return c == 255 ? NAN : ldexpf(1.f, (int)c - 127); }

static int csf32(size_t rows, size_t columns,
    const uint8_t *bases, size_t bases_count, const uint8_t *codes, size_t codes_count,
    const uint16_t *offsets, size_t offsets_count, const uint32_t *page_bases, size_t pages_count,
    const uint16_t *exceptions, size_t exceptions_count, uint8_t *output, size_t output_count) {
    if (!rows || rows % 32 || !columns || rows > SIZE_MAX / columns ||
        !bases || !codes || !offsets || !page_bases ||
        (exceptions_count && !exceptions)) return -1;
    const size_t rb = rows / 32, cb = columns / 2 + columns % 2;
    if (rb > SIZE_MAX / cb) return -1;
    const size_t tiles = ((rb + 3) / 4) * ((cb + 1) / 2);
    const size_t pages = tiles / 32 + (tiles % 32 != 0), n = rows * columns;
    if (bases_count != rows || codes_count != n / 2 + n % 2 ||
        (output && output_count != n) || pages_count != pages ||
        rb * cb > SIZE_MAX - pages || offsets_count != rb * cb + pages) return -1;
    size_t field = 0, block = 0, tile = 0, consumed = 0;
    for (size_t r0 = 0; r0 < rb; r0 += 4) {
        for (size_t c0 = 0; c0 < cb; c0 += 2, ++tile) {
            const size_t page = tile / 32;
            if (page_bases[page] % sizeof(uint16_t)) return -2;
            for (size_t c = c0; c < cb && c < c0 + 2; ++c) {
                const size_t width = columns - c * 2 < 2 ? 1 : 2;
                for (size_t r = r0; r < rb && r < r0 + 4; ++r, ++block) {
                    if(output) for (size_t rr = 0; rr < 32; ++rr) {
                        for (size_t cc = 0; cc < width; ++cc, ++field) {
                            const unsigned nibble = (codes[field / 2] >> ((field % 2) * 4)) & 15;
                            output[(r * 32 + rr) * columns + c * 2 + cc] = (uint8_t)(bases[r * 32 + rr] + nibble);
                        }
                    }
                    if(!output)field+=32*width;
                    const size_t index = block + page;
                    if (offsets[index + 1] < offsets[index]) return -2;
                    const size_t first = (size_t)page_bases[page] / sizeof(uint16_t) + offsets[index];
                    const size_t last = (size_t)page_bases[page] / sizeof(uint16_t) + offsets[index + 1];
                    if (last > exceptions_count || first != consumed) return -2;
                    uint64_t slots = 0;
                    for (size_t i = first; i < last; ++i) {
                        const unsigned slot = exceptions[i] & 255;
                        if (slot >= 64 || slot % 2 >= width || (slots & (UINT64_C(1) << slot))) return -3;
                        slots |= UINT64_C(1) << slot;
                        if(output)output[(r * 32 + slot / 2) * columns + c * 2 + slot % 2] = (uint8_t)(exceptions[i] >> 8);
                    }
                    consumed = last;
                }
            }
        }
    }
    return field == n && consumed == exceptions_count ? 0 : -2;
}

int strata_fp4_csf32_decode(size_t rows, size_t columns,
    const uint8_t *bases, size_t bases_count, const uint8_t *codes, size_t codes_count,
    const uint16_t *offsets, size_t offsets_count, const uint32_t *page_bases, size_t pages_count,
    const uint16_t *exceptions, size_t exceptions_count, uint8_t *output, size_t output_count) {
    if(!output)return -1;
    return csf32(rows,columns,bases,bases_count,codes,codes_count,offsets,offsets_count,
                 page_bases,pages_count,exceptions,exceptions_count,output,output_count);
}
int strata_fp4_csf32_validate(size_t rows, size_t columns,
    const uint8_t *bases, size_t bases_count, const uint8_t *codes, size_t codes_count,
    const uint16_t *offsets, size_t offsets_count, const uint32_t *page_bases, size_t pages_count,
    const uint16_t *exceptions, size_t exceptions_count) {
    return csf32(rows,columns,bases,bases_count,codes,codes_count,offsets,offsets_count,
                 page_bases,pages_count,exceptions,exceptions_count,NULL,0);
}
