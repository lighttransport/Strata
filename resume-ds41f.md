# Resume: DeepSeek-V4.1-Flash on 2x V100

## State as of 2026-10-26

Branch `glm53f`, recent commits (not pushed): ba2ec970 / 7c9443a5 / e49e31cd (docs 8j), 69ea6521 (S27 expert kernels,
STRATA_EXPERT_V3), b925c45d (pack split tool), 76f10a05 / 625649f0 (GSQ docs, KL tool), 550274aa (GSQ-RCO loading).

Best so far (docs/DS41F_PLAN.md 8j), Engram RAM 56 GiB (node-safe; peak RSS ~155-163 GiB):
- IQ2_XXSE3 + S27 (`STRATA_EXPERT_V3=1 STRATA_DS41_DECODE_GROUPED=1 STRATA_DS41_GROWN_GROUPED=1`), GROW, DSpark:
  wiki600 32-33, code600 36.0, code4k 33.2 tok/s (KL 0.314 vs Q8_0).
- IQ2_XXSE4 baseline at the same budget: code600 24.2, wiki600 26.0, code4k 24.7 (KL 0.290).
- GSQ-RCO IQ2_XXS: KL 0.379, slower; not pursued.

Memory rule: the decoder holds ~104 GiB + Engram RAM. Engram 76 GiB (~180 GiB) took nodes down (Slurm NODE_FAIL);
use <= 56-60 GiB. Run scripts with the memory guard (`runm` in `~/work/Strata-ds41/ds41-s76.sh` / `s78.sh`: stops
the decoder by PID when MemAvailable < 12 GiB; its sampler variable must not be `p`).

Cluster: `fugaku1:~/work/Strata-ds41` (build-v100), data in `ds41data/`; sessions via
`slurm-gpu/ds41/gpu_session.py send/status/capture`, keeper `slurm-gpu/ds41/keep2.sh <hours>`; gpu1 queue can be
full (6 nodes). Split packs (backbone + shared Engram files) in `~/models/ds41f-split/` (tool
`tools/ds41f_split_engram.py`; `header` / `rewrite` convert already downloaded old shards).

## Remaining items

1. Long-context levers never measured on the current code (scripts staged on the cluster: `ds41-s61.sh` 106K part,
   `ds41-s62.sh`): REFILL + GROWN_GROUPED at 106K / 128 tokens, batched rebalance swaps, PEER_SLOTS 2 vs 8 at
   PROMPT_BATCH 4096, 256K needle at batch 4096 with peak RSS. Re-run at Engram RAM <= 56 GiB, with S27 and on E3.
2. Prefill sweep (Round 1 part 6, never completed): 4K / 8K prompts, PROMPT_BATCH 2048 / 4096 / 8192, PEER_SLOTS 2 vs
   8, INDEX_GEMM_MIN=64 (`ds41data/t8192.ids` is created by `ds41-s70.sh`).
3. S27 acceptance drift on code (4.27 -> 4.00 tokens per step on code600, deterministic): test more code prompts
   (tokenize `ds41data/code_src.txt` / `code_src4k.txt` first - the decoder reads token ids), and whether making
   S27's sums match MMQ's rounding (or the drafter's features from the same path) recovers it.
4. Draft width on E3 + S27 (DSPARK_N 4 looked neutral-to-better on E4: 28.4 vs 26.7, one run).
5. CPU lane (26-29 ms per pass) is now the limit with S27: fewer lane experts (decode rebalance at short context),
   or an IQ2_XXS / Q2_K lane kernel check.
6. Engram ranked rows from a profile closer to the workload (currently wiki-103 + llama.cpp code).
7. Simulator: records for E3 / E43 / S27 at Engram 56 GiB and the memory budget (`tools/sim/ds41f.py` V100 model
   assumes Engram 76); test counts in `tools/test_sim_ds41f.py`.
8. Saved logits `ds41data/ref/{gsq,e4}_kl_wiki2.f32` (17 GB each) can be deleted.

## TR16 (RX 9070 XT 16 GB + 128 GB RAM), state as of 2026-10-27

Commit 371687ae (docs/DS41F_PLAN.md 8k). Model: Unsloth UD-IQ1_S, `/mnt/nvme02/models/ds41f-oct/iq1s/`; build
`build-hip-rocm101-tr16` (`LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib`). Run scripts and logs:
`/home/syoyo/.claude/jobs/c6566765/tmp/tr16/` (run5-run21.sh; `routes_ref1008.i32` is the tier profile, `cmp_f32` the
numpy-free KL tool, `kl_off.f32` an old exact reference - regenerate it with the current flags before new KL runs).

Best so far (code600):
- Exact decode: ~13-14 tok/s (estimated: no defer, all lossless levers on; not measured as one run).
  Lossless levers: `STRATA_DS41_DECODE_GROUPED=1 STRATA_DS41_GPU_RESERVE_MIB=1536 STRATA_DS41_EXPERT_PROFILE=<routes>
  STRATA_DS41_FUSE=1 STRATA_DS41_SHARE_Q8=1 STRATA_IQ1S_FAST=1 STRATA_Q6_V2=1 STRATA_DS41_GPU_ROUTE=1` (GROW off),
  plus `numactl --interleave=all`, `--threads=15`.
- With `STRATA_DS41_DEFER_RANK=3` (KL 0.078): 15.9-16.0 tok/s.
- `DEFER_ALL=1 CPU_TOPK=4`: 18.7 tok/s (lossy, KL ~0.2).
- Prefill with `STRATA_DS41_PIN_GB=100 STRATA_DS41_STREAM_MIN=256 STRATA_DS41_PREFILL_HYBRID=0 STRATA_DS41_KV=f16
  STRATA_DS41_INDEX_GEMM_MIN=64`: 600 tokens 52-62, 4K 249, 8K ~330 tok/s warm.
- 8K prefill, 2026-10-27 (docs/DS41F_PLAN.md 8k): 329 tok/s warm (24.9 s) with `PREFILL_HYBRID` unset (gated copies
  idle the copy engine), `STAGE_SLOTS=8 ENGRAM_THREADS=256`, against 308 before. The pass is expert-DMA bound: 116.1 GB
  per 4096-token pass at 12.5 GB/s (the 1950X root port is PCIe 3.0 x16, 8 GT/s), 9.3 s of copy stream per pass; the
  expert MMQ (2.4 s) hides under it. `PREFILL_TIMING=1` prints the split; `EXPERT_DEBUG=1|2` times the DMA / the
  products alone.
- VRAM safety (2026-10-27): the decoder leaves `STRATA_DS41_DESKTOP_MIB` (default 2048) MiB free - every allocation
  refuses below it, the tier/grow/slot budgets are clamped to it, and the stream slots halve to 32 MiB if needed (a
  run that grew past it took the GNOME session down once). With the reserve: 8K 306, 16K 307 tok/s (8 x 128 MiB ring).
- Scratch tiles (opt-in, byte-identical off): `ENGRAM_TILE=256 MIX_TILE=1024 SHEXP_TILE=1024 OUTA_TILE=1024
  SCORE_MIB=16` cut a 8192-token pass's buffers ~1.6 GB.
- PREFILL_HALVES (2026-10-27): trunk per 4096-tile, FFN once per layer, one expert stream per prompt, per-layer row
  truncation. **8K 20.1 s / 408 tok/s, 16K 41.5 s / 394** (two passes: 306/307), KL 0.0008 vs the two-pass
  path. The truncation needed the indexer selection indexed by the pass's absolute row (`sel_base_`; it is written at
  one layer and read by later ones, so a per-layer kept range read the wrong rows: KL 1.06) plus a 1024-row boundary
  slack (0.03 -> 0.0007). The pass is no longer DMA bound: 9.5 s of DMA in 20.6 s, compute = attention 6.2 + output
  1.5 + FFN 11.0. 500 needs ~1.4x faster attention/expert kernels. Splitting hot experts into their own MMQ products
  cost 2.5 s and was reverted.
  Flags: `STRATA_DS41_PREFILL_HALVES=1 STRATA_DS41_PREFILL_TILE=4096 PROMPT_BATCH=8192 ENGRAM_TILE=256
  SHEXP_TILE=2048 OUTA_TILE=4096 SCORE_MIB=16` (OUTA_TILE below the tile picks a split-K GEMM and costs KL 0.2).

Safety: run every decoder in its own `systemd-run --user --scope` (systemd-oomd killed this session's whole tmux scope
once: RAM_COPY, 86.6 GB anonymous). Never RAM_COPY on TR16. `STRATA_DS41_PREFILL_HYBRID>=1` froze the machine (a
reboot); do not run it until the cause is found. The machine is a shared desktop: noise about +-1 tok/s, check `top`
first.

### TR16 remaining items

1. Measure the exact (no defer) decode with every lossless lever on, 128 tokens x2, code600 and t600; record it in 8k.
2. The CPU lane in decode runs ~12 GB/s against 20-26 GB/s in the lane bench (`STRATA_DS41_LANE_BENCH=map`). Find
   the gap: per-job phase times on the deferred lane (gate/up, quant, down), the main thread / HIP runtime threads on
   the lane cores' SMT siblings, the futex wake per job. The lane is the decode limit (~50 ms a token).
3. Lossy budget for 25-30 tok/s: KL of the combinations (DEFER_ALL + CPU_TOPK=4/5, GPU_SUBST=0.8 with rank 4-5)
   against a fresh exact reference; pick the best tok/s within an agreed KL (ask the user for the gate).
4. Remaining GPU bubbles: the "experts" phase still ~10 ms a token with GPU_ROUTE (the host's route sync and lane
   setup). Next: let the host enqueue layer l+1's attention before it waits for layer l's ids when every CPU route is
   deferred (DEFER_ALL), and a deferral depth of 2 layers.
5. Q6_V2 as the default on HIP after a per-shape check (q_b / kv / out_b win; out_a whole-matrix slower in the bench,
   per-group 1024 rows in decode); dense Q6_K -> Q4_K requant as a lossy pack variant (-2 GB VRAM, -35 % dense bytes).
6. PREFILL_HYBRID>=1 freeze: reproduce on a 2-layer-equivalent small prompt with a watchdog (or under a remote
   session with a hardware reset path), suspects: the lane's begin() during a streamed pass with 100 GB of pinned
   page cache under concurrent DMA.
7. Prefill is at 408 tok/s at 8K and 394 at 16K (PREFILL_HALVES, exact). The FP16 KV is now read directly by the
   attention kernel (comp_half), no per-launch float gather. 500 needs the attention kernel ported to gfx12 WMMA (the
   WMMA attention is CUDA-only) and ~1.4x on the expert MMQ. The port is designed in docs/DS41F_PLAN.md 8k from
   ~/work/gemm/pixal3d/rdna4 (gfx12 f16 WMMA builtin, FA2 at 37 TF/s of a 190 peak, LDS XOR swizzle, k must stream in
   64-dim chunks at d512, sync WMMA caps ~30-40 TF/s): expected ~3x on the attention (5.6 -> ~2 s), i.e. 460-480 at 8K.
   Historical note: 400 at 8K was the ceiling of this
   configuration and 500 needs ~1.4x faster kernels, in order: (a) `ds41_attention_batched` (6.2 s per 8K pass,
   ~10 TFLOPS) and the IQ1_S expert MMQ (11.0 s, ~10 TOPS); (b) a smaller per-pass DMA (the transfers are 9.5 s of
   20.6 s, so overlapping them fully is worth at most ~1 s now); (c) the CPU lane for a share of the experts (still
   blocked on the PREFILL_HYBRID freeze), (d) a finer row truncation (the 1024-row slack costs ~1 s a pass).
   Pin the experts at load (first streamed prompt pays ~5 s), and a GPU_BENCH run of MMQ vs dequant-F16 + hipBLASLt
   for IQ1_S at M 16-256 on gfx1201.
8. Upstream Strata v0.2.6 `strata-dsv41` on TR16 for reference: 4.5-5.6 tok/s on the GSQ 2.00 set
   (`/mnt/nvme01/models/ds41f-gsq/IQ2_XXS/maya-ds4`, worktree `/mnt/nvme02/work/Strata-upstream`).
9. TR16 records in `tools/sim/ds41f.py` (lane 12 GB/s in decode, GPU ~25 ms, prefill stream ~8.7 GB/s) and its tests.

V100 note: the research-levers run (s80, 2026-10-27) measured DSPARK_SCHED code600 35.7 -> 39.9 tok/s (wiki
neutral), deferral KL off 0.121 / R5 0.130 / R4 0.124 / R3 0.132, LANE_NUMA neutral, DECODE_STREAM slower, F16 GEMM
ahead of MMQ at M >= 128 (gate/up) and M >= 32 (down) - not yet in docs/DS41F_PLAN.md.

## Resuming prompt

> Resume the DS4.1-Flash V100 work from resume-ds41f.md. Start the keeper (`slurm-gpu/ds41/keep2.sh 4`). Use
> IQ2_XXSE3 + S27 at Engram RAM 56 GiB with the memory guard. Run item 1 and item 2, report decode / prefill tok/s
> and peak RSS, and update docs/DS41F_PLAN.md. Commit only when I ask; do not push.

TR16:

> Resume the DS4.1-Flash TR16 work from resume-ds41f.md (TR16 section). Run every decoder in its own
> `systemd-run --user --scope`, never RAM_COPY or PREFILL_HYBRID>=1. Do items 1 and 2, report decode tok/s and the
> lane's GB/s, and update docs/DS41F_PLAN.md 8k. Commit only when I ask; do not push.
