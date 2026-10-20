#include "strata/host/glm_fp4.h"
#include <math.h>
#include <stdint.h>
/* Storage contract: flashinfer's csf_32x2_v1, 32-row/two-column blocks,
 * column-pair-major within 128-row/four-column tiles; 32 tiles per page. */
static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | (uint16_t)p[1] << 8); }
static uint32_t le32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24; }
static size_t lesser(size_t a,size_t b) { return a<b?a:b; }
int strata_glm_csf32_decode(size_t rows,size_t cols,
    const uint8_t *bases,size_t nb,const uint8_t *codes,size_t nc,
    const uint8_t *pages,size_t np,const uint8_t *offsets,size_t no,
    const uint8_t *exceptions,size_t ne,uint8_t *out,size_t ny) {
    size_t groups,blocks,page_count,pass;
    if(!rows || rows%32 || !cols || rows>UINT32_MAX-127 || cols>UINT32_MAX-3 || rows>SIZE_MAX/cols) return 1;
    if(rows*cols>UINT32_MAX) return 1;
    groups=((rows+127)/128)*((cols+3)/4); blocks=(rows/32)*((cols+1)/2); page_count=(groups+31)/32;
    if(nb!=rows || nc!=rows*cols/2 || np!=page_count*4 || no!=(blocks+page_count)*2 || ne%2 || ny!=rows*cols) return 2;
    if(!bases||!codes||!pages||!offsets||(!exceptions&&ne)||!out) return 3;
    for(size_t r=0;r<rows;++r) if(bases[r]>240) return 4;
    for(pass=0;pass<2;++pass) {
        size_t group=0,block=0,code=0,record=0,page=0,page_record=0;
        for(size_t tr=0;tr<rows;tr+=128) for(size_t tc=0;tc<cols;tc+=4,++group) {
            if(group%32==0) {
                page=group/32;page_record=record;
                if(le32(pages+page*4)!=record*2 || le16(offsets+(block+page)*2)!=0) return 5;
            }
            for(size_t pair=0;pair<lesser(4,cols-tc);pair+=2) {
                size_t width=lesser(2,cols-tc-pair);
                for(size_t rr=0;rr<lesser(128,rows-tr);rr+=32,++block) {
                    size_t first=le16(offsets+(block+page)*2),last=le16(offsets+(block+page+1)*2);
                    if(first!=record-page_record || last<first || last-first>32*width || last>ne/2-page_record) return 6;
                    int previous=-1;
                    for(size_t e=first;e<last;++e) {
                        uint16_t entry=le16(exceptions+2*(page_record+e));unsigned pos=entry&255;
                        if(pos>=64 || pos%2>=width || (int)pos<=previous) return 7;
                        previous=(int)pos;
                    }
                    if(pass) {
                        if(width==2) {
                            /* The common 32x2 block stores one byte per row.
                             * Avoid variable-width indexing and the inner loop;
                             * retain the odd-column tail and validation pass. */
                            const uint8_t *b=bases+tr+rr,*q=codes+code;
                            uint8_t *dst=out+(tr+rr)*cols+tc+pair;
                            for(size_t r=0;r<32;++r,dst+=cols) {
                                unsigned v=q[r],base=b[r];
                                dst[0]=(uint8_t)(base+(v&15));
                                dst[1]=(uint8_t)(base+(v>>4));
                            }
                        } else {
                            for(size_t r=0;r<32;++r) {
                                unsigned delta=(codes[code+r/2]>>((r%2)*4))&15;
                                out[(tr+rr+r)*cols+tc+pair]=(uint8_t)(bases[tr+rr+r]+delta);
                            }
                        }
                        for(size_t e=first;e<last;++e) {
                            uint16_t entry=le16(exceptions+2*(page_record+e));unsigned pos=entry&255;
                            out[(tr+rr+pos/2)*cols+tc+pair+pos%2]=(uint8_t)(entry>>8);
                        }
                    }
                    code+=32*width/2;record=page_record+last;
                }
            }
        }
        if(record!=ne/2 || code!=nc || block!=blocks) return 8;
    }
    return 0;
}
float strata_glm_e2m1(uint8_t code) {
    static const float values[8]={0,.5f,1,1.5f,2,3,4,6};
    float x=values[code&7];return code&8?-x:x;
}
float strata_glm_e4m3(uint8_t code) {
    unsigned mag=code&127,exp=mag>>3,mant=mag&7;
    float x=mag==127?NAN:exp?ldexpf(1.0f+mant*.125f,(int)exp-7):mant/512.0f;
    return code&128?-x:x;
}
float strata_glm_ue8m0(uint8_t code) { return code==255?NAN:ldexpf(1.0f,(int)code-127); }
int strata_glm_fp4_repack_k32(const uint8_t *input,uint8_t *output,size_t bytes,int inverse) {
    if(bytes%16 || (inverse!=0 && inverse!=1))return -1;
    if(!bytes)return 0;
    if(!input || !output)return -1;
    const uintptr_t a=(uintptr_t)input,b=(uintptr_t)output;
    if(a!=b && (a<b?b-a:a-b)<bytes)return -1;
    for(size_t offset=0;offset<bytes;offset+=16) {
        uint8_t block[16];
        const uint8_t *src=input+offset;
        if(!inverse) {
            for(unsigned j=0;j<8;++j) {
                block[2*j]=(uint8_t)((src[j]&15)|((src[j+8]&15)<<4));
                block[2*j+1]=(uint8_t)((src[j]>>4)|(src[j+8]&240));
            }
        } else {
            for(unsigned j=0;j<8;++j) {
                block[j]=(uint8_t)((src[2*j]&15)|((src[2*j+1]&15)<<4));
                block[j+8]=(uint8_t)((src[2*j]>>4)|(src[2*j+1]&240));
            }
        }
        for(unsigned j=0;j<16;++j)output[offset+j]=block[j];
    }
    return 0;
}

static int fp4_ranges_overlap(const void *a,size_t na,const void *b,size_t nb) {
    const uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return x<=y ? y-x<na : x-y<nb;
}
int strata_glm_nvfp4_repack_k32(const uint8_t *input,uint8_t *output,size_t bytes) {
    if(bytes%36)return -1;
    if(!bytes)return 0;
    if(!input || !output)return -1;
    const uintptr_t a=(uintptr_t)input,b=(uintptr_t)output;
    if(bytes>UINTPTR_MAX-a || bytes>UINTPTR_MAX-b)return -1;
    if(a!=b && fp4_ranges_overlap(input,bytes,output,bytes))return -1;
    for(size_t offset=0;offset<bytes;offset+=36) {
        uint8_t block[36];
        const uint8_t *src=input+offset;
        for(unsigned j=0;j<4;++j)block[j]=src[j];
        for(unsigned pair=0;pair<2;++pair) {
            const uint8_t *q=src+4+pair*16;
            uint8_t *dst=block+4+pair*16;
            for(unsigned j=0;j<8;++j) {
                dst[j]=(uint8_t)((q[j]&15)|((q[j+8]&15)<<4));
                dst[j+8]=(uint8_t)((q[j]>>4)|(q[j+8]&240));
            }
        }
        for(unsigned j=0;j<36;++j)output[offset+j]=block[j];
    }
    return 0;
}

int strata_glm_fp4_pack_nvfp4(size_t rows,size_t cols,
    const uint8_t *codes,size_t nc,const uint8_t *scales,size_t ns,uint8_t *out,size_t no) {
    if(!rows || !cols || cols%64 || rows>SIZE_MAX/cols) return 1;
    const size_t values=rows*cols,blocks=values/64;
    if(nc!=values/2 || ns!=values/16 || no!=blocks*36) return 2;
    if(!codes || !scales || !out) return 3;
    if(fp4_ranges_overlap(codes,nc,out,no) || fp4_ranges_overlap(scales,ns,out,no)) return 4;
    for(size_t i=0;i<ns;++i) if(scales[i]>126) return 5;
    for(size_t b=0;b<blocks;++b) {
        uint8_t *dst=out+b*36;
        for(size_t s=0;s<4;++s) {
            dst[s]=scales[b*4+s];
            const uint8_t *src=codes+b*32+s*8;
            for(unsigned j=0;j<8;++j) {
                const unsigned shift=(j%2)*4;
                dst[4+s*8+j]=(uint8_t)(((src[j/2]>>shift)&15)|(((src[4+j/2]>>shift)&15)<<4));
            }
        }
    }
    return 0;
}

/* origin/glm53f: ModelOpt codecs for the native FP4 loader */
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