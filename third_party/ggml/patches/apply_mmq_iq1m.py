#!/usr/bin/env python3
"""Add an IQ1_M tile loader to llama.cpp's CUDA MMQ (the ggml tree Strata builds against; pinned in VERSION.txt).

Idempotent: a tree that already names GGML_TYPE_IQ1_M in mmq.cuh is left alone.  CMake runs it at configure time
(STRATA_GGML_PATCH, default ON) on the fetched tree or on STRATA_GGML_DIR.

IQ1_M (1.75 bpw) has no MMQ kernel upstream.  Its weights are d * (2 ls + 1) * (g + delta) with g in {-1, +1} from
the IQ1_S grid, ls a 3-bit scale per 16 weights, delta = +-1/8 per 8 weights and d one fp16 per 256: that is exactly a
Q6_K block with int8 values 8 g + sign(delta) in {-9, -7, 7, 9}, per-16 scales 2 ls + 1 and block scale d / 8 (exact).
So the loader fills Q6_K's tile (x_qs, x_sc, x_df) and the Q6_K dot products (dp4a and mma) run unchanged; the
per-architecture configs take Q6_K's entries.

usage: apply_mmq_iq1m.py <llama.cpp or ggml root>   (the directory holding ggml/src/ggml-cuda, or src/ggml-cuda)
"""
import pathlib
import re
import sys

LOADER = r'''
// ---------------------------------------------------------------------------------------------
// IQ1_M on Q6_K's tile (Strata, third_party/ggml/patches/apply_mmq_iq1m.py): int8 8*g + sign(delta) in {-9, -7, 7, 9},
// the 3-bit scales per 16 weights as 2*ls + 1, the block's fp16 d as d/8 - the same values as the dequantizer's.

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq1_m(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    int   * x_sc = (int   *) (x_df + MMQ_TILE_NE_K/QI6_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    // 32 groups of 8 weights per block: one grid lookup each (qs byte + 3 bits of qh), the delta sign in qh's 4th bit
    constexpr int threads_per_row = QK_K / 8;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq1_m * bxi = (const block_iq1_m *) x + kbx0 + i*stride;

        const int qh   = (bxi->qh[kqsx/2] >> (4*(kqsx%2))) & 0x0F;
        const int grid = iq1s_grid_gpu[bxi->qs[kqsx] | ((qh & 0x07) << 8)];   // 8 nibbles, g + 1 in {0, 2}
        const int sub  = (qh & 0x08) ? 0x01010101 : 0xFFFFFFFF;               // -sign(delta) per byte

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int v0 = __vsubss4(__vsubss4(grid0 << 3, 0x08080808), sub);    // 8 g + sign(delta)
        const int v1 = __vsubss4(__vsubss4(grid1 << 3, 0x08080808), sub);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + 2*kqsx + 0] = v0;
        x_qs[i*sram_stride + 2*kqsx + 1] = v1;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 2*kqsx + 0] = v0;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 2*kqsx + 1] = v1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq1_m * bxi = (const block_iq1_m *) x + kbx0 + i*stride;
        const uint16_t * sc = (const uint16_t *) bxi->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d8 = __half2float(scale.f16) * 0.125f;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride]                     = d8;
#else
        x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K] = d8;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    // the 16 scales of a block (2 ls + 1 per 16 weights), 4 per int as Q6_K keeps them
    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq1_m * bxi = (const block_iq1_m *) x + kbx0 + i*stride;
        const uint16_t * sc = (const uint16_t *) bxi->scales;
        const int m = threadIdx.x % (MMQ_TILE_NE_K/8);   // 16-weight groups 4m..4m+3 = 32-weight sub-blocks 2m, 2m+1
        int packed = 0;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const int ib = 2*m + h/2;                                        // the 32-weight sub-block
            const int ls = 2*((sc[ib/2] >> (6*(ib%2) + 3*(h%2))) & 0x07) + 1;
            packed |= ls << (8*h);
        }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_sc[i*sram_stride + m] = packed;
#else
        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + m] = packed;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}
'''

INSTANCE = '''// IQ1_M on Q6_K's tile: third_party/ggml/patches/apply_mmq_iq1m.py (Strata).

#include "../mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_IQ1_M);
'''


def replace_once(text, old, new, what):
    if text.count(old) != 1:
        raise SystemExit(f"apply_mmq_iq1m: expected one '{what}' anchor, found {text.count(old)}")
    return text.replace(old, new)


def main(root):
    root = pathlib.Path(root)
    cuda = root / "ggml" / "src" / "ggml-cuda"
    if not cuda.is_dir():
        cuda = root / "src" / "ggml-cuda"
    if not cuda.is_dir():
        raise SystemExit(f"apply_mmq_iq1m: no ggml-cuda under {root}")
    done = []
    # mmq.cuh: the type switches (IQ1_M takes Q6_K's answers) and the instance declaration
    mmq = cuda / "mmq.cuh"
    text = mmq.read_text()
    if "GGML_TYPE_IQ1_M" not in text:
        text = replace_once(text, "        case GGML_TYPE_Q6_K:\n        case GGML_TYPE_IQ2_XXS:\n",
                            "        case GGML_TYPE_Q6_K:\n        case GGML_TYPE_IQ1_M:\n        case GGML_TYPE_IQ2_XXS:\n", "ds layout")
        text = replace_once(text, "        case GGML_TYPE_Q6_K:    return MMQ_DP4A_TXS_Q6_K;\n",
                            "        case GGML_TYPE_Q6_K:    return MMQ_DP4A_TXS_Q6_K;\n        case GGML_TYPE_IQ1_M:   return MMQ_DP4A_TXS_Q6_K;\n",
                            "dp4a tile sizes")
        dp4a = ("            case GGML_TYPE_Q6_K:\n"
                "                return ggml_cuda_mmq_util_funcs(\n"
                "                    VDR_Q6_K_Q8_1_MMQ,\n"
                "                    ggml_cuda_mmq_load_tiles_q6_K<type, J, fallback>,\n"
                "                    ggml_cuda_mmq_vec_dot_q6_K_q8_1_dp4a<type, J, fallback>,\n"
                "                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);\n")
        text = replace_once(text, dp4a, dp4a + dp4a.replace("GGML_TYPE_Q6_K", "GGML_TYPE_IQ1_M").replace("load_tiles_q6_K", "load_tiles_iq1_m"),
                            "dp4a util funcs")
        mma = ("        case GGML_TYPE_Q6_K:\n"
               "            return ggml_cuda_mmq_util_funcs(\n"
               "                -1,\n"
               "                ggml_cuda_mmq_load_tiles_q6_K<type, J, fallback>,\n"
               "                ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma<type, J, fallback>,\n"
               "                ggml_cuda_mmq_write_back_mma<type, J, fallback>);\n")
        text = replace_once(text, mma, mma + mma.replace("GGML_TYPE_Q6_K", "GGML_TYPE_IQ1_M").replace("load_tiles_q6_K", "load_tiles_iq1_m"),
                            "mma util funcs")
        text = replace_once(text, "extern DECL_MMQ_CASE(GGML_TYPE_IQ1_S);\n",
                            "extern DECL_MMQ_CASE(GGML_TYPE_IQ1_S);\nextern DECL_MMQ_CASE(GGML_TYPE_IQ1_M);\n", "instance declaration")
        mmq.write_text(text)
        done.append(mmq.name)

    # the loader, after IQ1_S's
    tiles = cuda / "mmq-load-tiles.cuh"
    t = tiles.read_text()
    if "load_tiles_iq1_m(" not in t:
        anchor = "template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xxs("
        t = replace_once(t, anchor, LOADER.lstrip("\n") + "\n" + anchor, "iq2_xxs loader")
        tiles.write_text(t)
        done.append(tiles.name)

    # every architecture's config: Q6_K's entries, for IQ1_M (a config without Q6_K, Blackwell's NVFP4 one, has none)
    for cfg in sorted(cuda.glob("mmq-config-*.cuh")):
        c = cfg.read_text()
        if "GGML_TYPE_IQ1_M" in c or "CASE(GGML_TYPE_Q6_K," not in c:
            continue
        out, q6 = [], []
        for line in c.split("\n"):
            out.append(line)
            if "CASE(GGML_TYPE_Q6_K," in line:
                q6.append(line.replace("GGML_TYPE_Q6_K,", "GGML_TYPE_IQ1_M,"))
            elif q6:
                out[-1:-1] = q6
                q6 = []
        cfg.write_text("\n".join(out))
        done.append(cfg.name)

    # the dispatch of ggml's own mul_mat_q (not what Strata calls, kept consistent)
    mmq_cu = cuda / "mmq.cu"
    if mmq_cu.exists() and "GGML_TYPE_IQ1_M" not in mmq_cu.read_text():
        c = mmq_cu.read_text()
        old = "        case GGML_TYPE_IQ1_S:\n            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);\n            break;\n"
        if c.count(old) == 1:
            mmq_cu.write_text(c.replace(old, old + old.replace("IQ1_S", "IQ1_M")))
            done.append(mmq_cu.name)

    inst = cuda / "template-instances" / "mmq-instance-iq1_m.cu"
    if not inst.exists():
        inst.write_text(INSTANCE)
        done.append(inst.name)
    if not done:
        print(f"apply_mmq_iq1m: {cuda} already patched")
        return 0
    print(f"apply_mmq_iq1m: patched {cuda}: " + " ".join(done))
    return 0



if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
