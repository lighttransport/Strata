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

## Resuming prompt

> Resume the DS4.1-Flash V100 work from resume-ds41f.md. Start the keeper (`slurm-gpu/ds41/keep2.sh 4`). Use
> IQ2_XXSE3 + S27 at Engram RAM 56 GiB with the memory guard. Run item 1 and item 2, report decode / prefill tok/s
> and peak RSS, and update docs/DS41F_PLAN.md. Commit only when I ask; do not push.
