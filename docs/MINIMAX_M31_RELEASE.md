# MiniMax M3.1 Flash with strata-m3: release notes (November 2026)

What this branch adds to the MiniMax M3.1 Flash engine (`strata-m3`), with the measured speed and quality of each part. Every
number below was measured on the machine named here. The full records, including what was tried and did not help, are in
[MINIMAX_M31_SIM.md](MINIMAX_M31_SIM.md).

**Machine.** One allocation of a V100 server: 2x Xeon Gold 6240, 181 GB RAM, one or two Tesla V100 PCIe 32 GB (pinned host to device
12.3 GB/s). **Model.** Unsloth UD-Q2_K_XL. The 124.7 GiB expert pack is pinned in RAM, and the DSpark draft head is Q8_0. Unless a row
says otherwise: one GPU, FP16 K/V (`--kv-f16`), the draft head loaded (`--dspark FILE`, which turns on speculation with 6 drafts).

## The default path is unchanged

Every new feature is opt-in. With no new flags, the four reference prompts (exact routing, 200 tokens) give the same tokens as before
the branch: dijkstra 1ee67a8d, lru_cache 38fb7620, knapsack cd04cbf5, eval_expr e30269a7 (md5 of the whole `M3_TOKENS` line). This was
checked after each kernel change, and again on 2026-11-04 after the C++ benchmark. Decode in that last check was 27.9 / 26.5 / 24.9 / 23.4 tok/s.

## Decode

| configuration | decode tok/s | 46 C++ tasks passed |
|---|---:|---:|
| exact routing (`--speed exact`, the default) | 27.0 / 25.8 (two prompts) | 41 |
| `--speed fast` (top-3 experts) | 32.9 / 33.3 | 43 (30.3 tok/s on the tasks) |
| `--speed turbo` (top-2) | 39.2 (tasks) | 37 |
| `--speed max` (top-2, experts that miss VRAM dropped) | 56.7 / 55.5 | 36-39 |
| two GPUs, `STRATA_M3_GPU1=1`, exact routing | 38.8 / 38.7 | 41 (37.6 tok/s on the tasks) |
| two GPUs, `--speed fast` | 46.7 / 46.8 | not run |
| two GPUs, `--speed max` | 62.7 / 59.3 | 38 (62.7 tok/s on the tasks) |
| batched decode, `--seqs 8`, `--speed max` settings, different prompts | 81.2 aggregate | 39 (batches of 4, 72.7 tok/s aggregate) |

The C++ benchmark (`tools/minimax_cpp_eval.py`) has 46 single-function tasks with unit tests. The answers are greedy with thinking off,
and a task passes if the first code block compiles with g++ and every assert holds. It is one sample per setting, so a difference of
one or two tasks is within noise. Spot check of the final build (2026-11-04), exact routing: 41 of 46 at 25.2 tok/s mean, the same
pass count as the earlier exact runs (41 of 46 at 24.1-25.5 tok/s).

With a second GPU, that GPU holds only expert slots (1800 more). On a check run, all 20,487 jobs it ran matched GPU 0 bit for bit
(`STRATA_M3_GPU1_CHECK=1`). Exact-routing tokens on two GPUs equal one GPU's when the expert placement is equal. In normal use they
differ only through CPU/GPU placement.

## Prefill

| prompt | before | now |
|---|---:|---:|
| 16K tokens, one chunk | 537 tok/s | 634-640 tok/s |
| 8K tokens, one chunk | 497 tok/s | 541-547 tok/s |

Settings: `--pf-gemm --pf-prefetch --kv-f16`, draft head on, `STRATA_M3_PF_INJECT=1 STRATA_MSA_TC2=1` (both opt-in). The block-score
skip of invisible key blocks is exact and always on. Quality of `STRATA_MSA_TC2`, teacher-forced over 64 steps against the default
q8_1 path:
- short prompt: KL 5.4e-4, top-1 64/64
- 7K prompt: KL 1.37e-3 and 1.67e-3 in two runs (without the flag: 1.36e-3 and 1.16e-3), top-1 47/47

The 7K prompt sits at the 1.6e-3 gate line. The 600 tok/s target is met at 16K but not at 8K.

## Long context

New opt-in switches, which all need `--kv-f16`:
- `--kv-fp8`: E4M3 K/V with a scale per row.
- `--kv-host`: K/V in pinned host memory. With `--kv-fp8` it adds a VRAM block cache sized by `STRATA_M3_KV_BCACHE_GIB`.
- `--idx-fp8`: FP8 indexer keys.

Needle-in-a-haystack, with the answer at the given depth:

| prompt | K/V | prefill | decode | answer |
|---|---|---:|---:|---|
| 32K | FP16, host | 425 tok/s | 2.8 tok/s | correct |
| 128K | FP16, host | 398 tok/s | 2.75 tok/s | correct |
| 255K, depths 25 % and 75 % | FP16, host | 329 tok/s | 2.65 tok/s | correct |
| 32K | FP8, VRAM | 537 tok/s | 12.4 tok/s | correct, same tokens as FP16 |
| 128K | FP8 K/V and FP8 index keys, VRAM | 434 tok/s | **12.8 tok/s** | correct |
| 255K | FP8, host + 2 GiB block cache | 350 tok/s | 8.5 tok/s | correct, same tokens as FP16 |
| 512K | FP8, host + block cache | 251 tok/s (34 min) | 6.0 tok/s | correct |

Teacher-forced quality on the 7K prompt against the default path:

| K/V | KL | top-1 |
|---|---:|---:|
| FP16 in host memory | 1.29e-3 | 47/47 |
| FP8 | 1.59e-3 | 47/47 |
| FP8 with FP8 index keys | 1.06e-3 | 47/47 |

The 128K decode rate (FP8 in VRAM) went from 10.3 to 12.8 tok/s through five kernel changes:
- block scores for all index heads from one key read
- a wider FP8 expand step that decodes through a table
- a table decode for the FP8 index keys
- a block selection without block-wide barriers
- block scores summed with a transposed butterfly, every query of a window from one key read

All of them give bitwise-identical tokens. 12.8 tok/s is 86 % of the same engine's short-context speed (14.8 tok/s) and 72-81 % of an
FP16 engine sized for 16K context (15.9-17.7 tok/s in different sessions). 1M context does not fit next to the pinned pack on a 181 GB
node and was not tested.

## Tools

- `tools/minimax_cpp_eval.py` (`selftest`, `prepare`, `score`): the C++ benchmark.
- `tools/sim/minimax_step.py`, `tools/minimax_m31_fit.py`: the step-time model fitted on the measured path, with batched-decode and
  prefill projections. Projections are labeled as such.
- `minimax_msa_parity` (selection checked against the host reference up to 8193 blocks), `tc_mmvq_test`, `dense_mmvq_bench`.

## Not tested, and known limits

- Only on V100 (CUDA, sm_70). The HIP build compiles every changed file but was not run on an AMD GPU.
- No benchmark other than the 46 C++ tasks: no other languages, no sampling with temperature, one sample per setting.
- `--kv-host` without the FP8 block cache decodes at about 2.7 tok/s at every length, because the reads are latency-bound PCIe reads.
- Batched decode (`--seqs`) is not combined with the long-context switches, and has not been run on two GPUs.
- The 8K prefill target (600 tok/s) is not met.
