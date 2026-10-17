# llvq Tetra for GLM-5.3-Flash experts: C++/CUDA/HIP encoder and decoder (2026-10-16)

llvq (github.com/pjmalandrino/llvq, Rust, MIT OR Apache-2.0) quantizes 24 weights to one 48-bit word: 47 bits of
Leech-lattice point (the "Tetra" map) and one gain bit, 2.06 bits per weight with a 16-column tail. The one-layer
study ([GLM_QUANT_LEECH_GSQ_STUDY.md](GLM_QUANT_LEECH_GSQ_STUDY.md)) used the Rust crates. This is a native port
pinned to llvq ddecad31, so the full model can be encoded on TR16 without PyTorch or Rust.

## What is ported

| Part | File | Reference |
|---|---|---|
| Golay code, trellis, rank table, word fields, decode, point -> word | `include/strata/llvq/tetra.hpp` | `llvq-core/src/golay.rs`, `llvq-search/src/tetra/{mod,trellis,rank,word}.rs` |
| Production encoder (CPU, f64) | `include/strata/llvq/tetra_encoder.hpp` | `llvq-search/src/tetra/encoder.rs` |
| Encoder core for host and device, f64 or f32 | `include/strata/llvq/tetra_encode_core.hpp`, `tetra_view.hpp` | the same |
| GPU encoder, one warp per block (CUDA and HIP) | `src/kernels/cuda/llvq_tetra.cu` | the same |
| SplitMix64, Rotation, row scale, gain centroids, gain choice, reconstruct | `include/strata/llvq/quantize.hpp` | `llvq-quant/src/{quantizer,rotation}.rs` |
| Decode: GEMV and FP16 dequant (CPU AVX2, CUDA, HIP), activation rotation | `include/strata/llvq/tetra_decode.hpp`, `src/kernels/cuda/llvq_tetra.cu` | `llvq-cuda/kernels/llvq_tetra48.cuh` (layout) |
| Full-model tool | `tools/glm_llvq_encode.cpp` (`strata-glm-llvq-encode`) | |
| Rust reference dumps | `tools/quant_study/llvq_layer/src/dump.rs` (`llvq_dump`) | |

The tables are rebuilt at construction and checked against llvq's constants: the trio partition, the linear columns of
the trellis map, the class and mixed bounds, the N0 split. `Tetra::fingerprint()` equals llvq-artifact's published
`tetra_fingerprint()`, `0x9c3060086c7a13e6`.

## Bit-exactness

The encoder keeps llvq's f64 arithmetic in the same operation order. It is compiled without floating-point contraction:
`-ffp-contract=off` on the host and on HIP, `-fmad=false` on CUDA, set per target and per source file in
`CMakeLists.txt`. Two changes speed it up without changing any result:

- The row index is a direct base-5 table instead of a binary search.
- The GPU warp reduces its joins by (cost, path index), so the first minimum still wins.

min() over non-NaN values does not depend on order, so the bound pass can be split across lanes. Resolves are
independent of each other.

| Check | Blocks | Result |
|---|---|---|
| `llvq_dump` Gaussian blocks scaled 10^-3..10^3, with zero, 1e-300, 1e250 and sparse blocks | 200,000 | C++ CPU, CUDA and HIP: every word and every t bit-identical |
| The same with two thirds rounded to a coarse grid (exact distance ties) | 100,000 | identical |
| Whole matrix pipeline (rotation, centroids, row scales, words with gain) on a real FP8 expert, layer 20: gate 2048x4096 and down 4096x2048 | 2 x 348,160 | the C++ CPU, CUDA and HIP tool outputs equal `llvq_dump matrix` byte for byte |
| ctest `llvq_tetra_test`, `llvq_tetra_gpu_test`: golden hash of 3,000 `llvq_dump` blocks regenerated in C++ | 3,000 | pass, no data files needed |

The f32 GPU mode uses the same search in single precision. It is format-compatible but not bit-exact. On the tie-heavy
set it picks a different word for 9.9% of blocks (2.1% on the Gaussian set). No differing word lowers t, the encoder's
objective <x, y>/|y|, by more than 1e-6 relative. The mean loss is 5e-10 and the worst 3e-7.

## Throughput

TR16: Threadripper 1950X (Zen1, 16 cores), RX 9070 XT, RTX 5060 Ti. B550: Ryzen 9 3950X (Zen2). Blocks of 24
weights per second, f64 unless noted.

| Encoder | Blocks/s |
|---|---|
| Rust llvq, 16 threads (study) | ~25,000 |
| C++ CPU, TR16, 16 threads / 32 threads | 32,350 / 44,700 |
| C++ CPU, B550, 16 / 32 threads | 28,300 / 47,700 |
| RX 9070 XT, f64, real layer 10 (16 units of 16.7 M blocks each) | 153,800 sustained |
| RX 9070 XT, f32 | 290,000-315,000 |
| RTX 5060 Ti, f64 / f32 (shared with a GPU process at 100% load: lower bounds) | 26,000 / 68,000 |

The GPU encoder runs one 32-lane warp per block. In each of the four passes (two scales x two parities), the lanes
build the coordinate table, the 512 base entries, the 4,096-path joins and the bound pass. The warm-join resolves run
one lane per (pattern, shrink layer). The bound-pass resolves are compacted into a list and run one resolve per lane.
Profiling the earlier serial warm phase found it took 32% of the cycles; making it cooperative took the RX 9070 XT
from 112,000 to 150,000 blocks/s. A cooperative bound pass was slower (145,000) and was not kept. Four warps per CTA
fit gfx1201's 64 KiB LDS; CUDA uses three (48 KiB static shared memory).

The RTX 5060 Ti has 1/64-rate FP64, the RX 9070 XT 1/32, so f64 is the limit on the NVIDIA card.

## Full model

42 routed layers (3-44) x 288 experts x 3 matrices = 12.63 G blocks; under 12 h needs 292,000 blocks/s. With the
measured rates on TR16:

| Setup | Blocks/s | Hours |
|---|---|---|
| RX 9070 XT f64 + CPU f64 (20-26 threads beside it: 31,000) | ~185,000 | ~19 |
| + RTX 5060 Ti f64 (shared GPU, 26,000) | ~211,000 | ~17 |
| + RTX 5060 Ti f32 (shared GPU, 68,000) | ~253,000 | ~14 |

All-f64 on TR16 is short of the target. The f32 fallback on the 5060 Ti reaches it only with that GPU to itself; its
unshared f32 rate was not measured. The run scripts are in the output's parent directory
(`models/glm53f/llvq-{run,stop,progress}.sh`):

- The 9070 XT and the CPU run in f64. An NVIDIA GPU, if present, runs the f32 fallback.
- The workers share the queue through claim files.
- Each unit records its precision (`.json` sidecar and header).
- When all units exist, `--redo-f32` re-encodes the f32 units in f64.

The tool is resumable:

- A unit file is renamed into place only when complete.
- A claim holds the pid and the boot id, so claims from a dead process or an earlier boot are taken over.
- SIGTERM (`llvq-stop.sh`) abandons the unit in progress at the next matrix boundary and removes its claim.

### Matrix pipeline

Per matrix (`strata-glm-llvq-encode`):

1. Dequantize FP8 E4M3 with its 128x128 `weight_scale_inv` to f32 (the reference dequantizer's rounding), then f64.
2. Rotate rows with llvq's `Rotation(cols, seed)`: Gaussian signs, then a scaled Walsh-Hadamard transform. GLM's
   widths 4096 and 2048 are powers of two.
3. Fit two gain centroids by Lloyd, 40 iterations.
4. Compute row scales.
5. Encode every full block (Tetra word) and pick its gain bit (nearest centroid, first wins). Zero blocks become word 0.
6. Store the 16- or 8-column tail as BF16.

The Hessian is the identity. llvq's GPTQ loop then propagates nothing, so each matrix is one batch for the encoder.
The study's best setting, a diagonal Hessian with rotation (8.0% against 8.5%), becomes a dense Hessian after
rotation. Its error feedback costs rows x cols^2 per matrix, about 1.5 PFLOP for the 36,288 expert matrices, which is
not feasible in 12 h.

Quality: unit `L10.e000` decodes (`--check`) to a relative error of 0.2915 against the rotated FP8 weights for all 48
matrices, normalized MSE 0.085. That matches the study's 8.5% for this setting.

### File format

One unit file `Lxx.eyyy.llvq` holds 16 experts of one layer, written to `.tmp` and renamed when complete. Layout:

- Header: `LLVQTET1`, then u32 version 1, layer, first expert, expert count, Lloyd iterations, precision (0 f64, 1 f32),
  then the u64 rotation seed.
- Matrix records, gate, up, down per expert:
  - u32 rows, u32 cols
  - f64 centroids[2]
  - f32 row scales[rows]
  - words: 6 bytes little endian, row stride padded to 8 bytes (llvq's tetra48 layout)
  - BF16 tail [rows][cols % 24]

The `.json` sidecar has the block count, an FNV-1a 64 checksum, device, precision, seed and time.

```sh
strata-glm-llvq-encode --model FP8_DIR --out OUT [--device cpu|gpu] [--precision f64|f32] [--threads N] [--layers A-B]
strata-glm-llvq-encode --model FP8_DIR --check OUT/L10.e000.llvq     # decode and compare against FP8
strata-glm-llvq-encode --matrix ROWS COLS in.f32 out.bin            # one matrix, llvq_dump's matrix format
llvq_dump matrix ROWS COLS in.f32 SEED ITERS out.bin                 # the Rust reference (tools/quant_study/llvq_layer)
```

## Decode

W' = W Q^T is stored, so y = W x = W' (Q x). `llvq_tetra_prepare` rotates each token once (signs, shared-memory
WHT) and permutes each block into trio order, which makes every word a contiguous 24-wide dot. Words decode with no
gathers: three table lookups for the pattern bytes, three row-table lookups, and per value a 16-entry
(rank, pattern bit) table per parity.

| Kernel, matrix | 1 token | 8 tokens |
|---|---|---|
| CPU AVX2, one Zen1 thread | 14.0 ns/word, 0.43 GB/s of words | 25.7 ns/word |
| CPU AVX2, one Zen2 thread | 10.5 ns/word, 0.58 GB/s | 17.2 ns/word |
| RX 9070 XT, 2048x4096 (2 MB of words) | 19.7 us, 107 GB/s | 51.7 us |
| RX 9070 XT, 16384x4096 | 117 us, 144 GB/s | 383 us |

Errors against a double reference of the dequantized rows: CPU 3.7e-7, GPU 1.2e-7. The FP16 dequant has a
maximum relative error of 4.6e-4, from FP16 itself. The CPU kernel takes the activation's pattern bits as +8 on the
rank nibble, one `pshufb` per section into the parity's 16-value table, and a 1/sqrt(16m) table by shell; it avoids
`pdep`, which is microcoded on Zen1/Zen2. The GPU GEMV keeps the 16 KB row table in global memory for one token and in
shared memory for more. A byte-permute decode with no table loads was tried and was not faster. The GEMV is not wired
into the engine.

## Reproduce

```sh
cmake --build build-glm-release --target llvq_tetra_test llvq_tetra_gpu_test llvq_tetra_decode_test strata-glm-llvq-encode
ctest --test-dir build-glm-release -R llvq        # the same in the HIP build
cd tools/quant_study/llvq_layer && cargo +1.93.0 build --release   # llvq_dump, Rust reference vectors
llvq_dump 200000 0x0f1b20260904 ref.bin && build-glm-release/llvq_tetra_test ref.bin 16
```
