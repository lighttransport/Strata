# B550 decode levers, round 2 (2026-10-09/10)

Machine: Ryzen 9 3950X, 62 GB DDR4 **at 2133 MT/s** (JEDEC; the DIMMs are rated 2666, XMP off), RX 9070 XT 16 GB,
ROCm 7.14. Model: GLM-5.3-Flash REAP-50 Q23 (`/mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf`).
Engine env: the step-1 BASE15 config (`../step1/step1-b15360-01-base15.config.json`) with `--prefill-experts=mmq`,
`--gpu-budget-mib=15360`, `--decode-cache-mib=12288`, 12 threads. Speed = 128 greedy tokens after a 2048-token
prefill (`STEP_TRACE` per-token split). Decode with mmq prefill is deterministic per prompt.

Starting point: route affinity 0.08, 17.2 tok/s. Memory bandwidth is the wall: `tools/membw` reads 30 GB/s, the CPU
expert phase moves 0.85 GB/token. GPU streaming of cold experts over PCIe (zero-copy or staged) does not help: CPU
reads and DMA together top out at 31-37 GB/s.

## Opt-in kernel and transfer switches, aff 0.08

| Switch | tok/s | per token (ms): CPU / GPU wait / head+tail |
|---|---:|---|
| none | 17.2 | 32.5 / 19.4 / 5.5 |
| `STRATA_GLM_HC_FUSED=1` | 18.1 | 31.0 / 18.0 / 5.4 |
| + `STRATA_GLM_TIER_DIRECT=1` | 19.0 | 30.4 / 17.8 / 4.1 |
| + `STRATA_GLM_MLA_KERNELS=1 STRATA_GLM_HEADS_BLOCKS=1` | 19.5 | 30.8 / 16.3 / 3.8 |
| + `STRATA_GLM_FUSE_ROUTE=1` | 19.5 | within noise (+0.05-0.1) |

These kernel switches preserve the equations but some change FP32 summation order. They are included in the
B550 optimization commit; the earlier local commit identifier is not present in this checkout.

Teacher-forced KL (width 1, 4 x 512 tokens, reference = the same binary at affinity 0): all four switches 0.0247
vs 0.0190 for the fused-only build; the per-head GEMV alone is 0.0180 (fp-order noise ~0.001), the MLA kernels
reorder the attention sums (+0.005 KL, perplexity unchanged, kernels match a double-precision reference).

## Lossy levers

KL is against affinity 0 (teacher-forced, width 1). HumanEval = pass@1 on the first 90 tasks (chat, greedy,
1024-token cap, isolated tests; `tools/humaneval_b550.py` = `tools/glm_q2_humaneval.py` plus the `<|user|>` stop
token 154827, which this GGUF's eos metadata lacks). "csv" = the 1024-token `tools/glm_q2_coding_bench.py` csv
fixture (one long generation; prime and json_escape pass everywhere).

| Config | tok/s | CPU / wait ms | KL | top-1 | HumanEval | csv |
|---|---:|---|---:|---:|---:|---|
| aff 0.08 (reference) | 19.5 | 30.8 / 16.3 | 0.154 | 0.862 | 81/90 | pass |
| aff 0.10 | 19.1 | 31.9 / 16.2 | 0.175 | 0.864 | | |
| aff 0.12 | 21.8 | 25.1 / 16.7 | 0.196 | 0.856 | | pass |
| profile 0.06/0.10/0.14 (layers 3-16 / 17-23 / 24-46) | 20.5 | 28.2 / 16.6 | 0.158 | 0.864 | | |
| profile 0.08/0.14/0.18 ("mix2") | 23.3 | 21.5 / 17.4 | 0.187 | 0.847 | 83/90 | pass |
| profile 0.06/0.12/0.20 | 22.8 | 22.4 / 17.5 | 0.178 | 0.861 | | |
| profile 0.08/0.16/0.22 | 22.2 | 24.1 / 16.9 | 0.200 | 0.847 | | |
| aff 0.08 + defer 1 | 23.4 | 31.2 / 7.4 | 0.221 | 0.845 | 28/30 | pass |
| aff 0.08 + defer 2 | 23.2 | 32.7 / 6.2 | 0.273 | 0.816 | | |
| aff 0.08 + defer 3 | 24.2 | 30.8 / 6.4 | 0.318 | 0.805 | | |
| aff 0.08 + full deferral | 21.1 | 31.3 / 10.2 | 0.343 | 0.799 | | |
| aff 0.12 + defer 1 | 25.9 | 25.7 / 8.9 | 0.247 | 0.832 | 83/90 | compile error (`&& ...` placeholder) |
| aff 0.12 + defer 1, cap 0.12 | 23.0 | 24.4 / 14.9 | 0.193 | 0.845 | 29/30 | fail |
| aff 0.12 + defer 1, cap 0.16 | 24.6 | 24.3 / 12.4 | 0.197 | 0.845 | | |
| aff 0.12 + defer 1, cap 0.20 | 24.6 | 26.1 / 10.4 | 0.196 | 0.844 | 81/90 | compile error |
| aff 0.12 + defer 2 | 27.0 | 24.6 / 8.4 | 0.315 | 0.813 | | no code in 1024 tokens |
| aff 0.12 + defer 3 | 28.4 | 21.4 / 9.7 | 0.345 | 0.796 | | |
| mix2 + defer 1 | 26.7 | 23.4 / 10.0 | 0.237 | 0.838 | 82/90 | degenerate (`"` loop) |
| mix2 + defer 1, cap 0.16 | 25.0 | 22.6 / 13.4 | 0.194 | 0.841 | 83/90 | fail |
| mix2 + defer 1, layers >= 17 | 24.6 | 23.4 / 13.1 | 0.203 | 0.852 | 28/30 | fail |
| mix2 + defer 1, layers >= 24 | 23.9 | 23.3 / 14.6 | 0.186 | 0.853 | | |
| mix2 + defer 2 | 28.2 | 21.2 / 10.3 | 0.286 | 0.813 | 54/60 | |

Switches: `STRATA_GLM_ROUTE_AFFINITY_PROFILE=<json>` (`{"layers": {"3": 0.08, ...}}`),
`STRATA_GLM_DEFER_EXPERTS=1 STRATA_GLM_DEFER_ROUTES=n` (partial deferral: the n lowest-weight cold routes of a
layer are computed one mixer late, via the group-B mailbox slots), `STRATA_GLM_DEFER_MAX_SHARE=x` (defer only
routes below x of the layer's routed weight), `STRATA_GLM_DEFER_MIN_LAYER=l`. Check mode
(`STRATA_GLM_DEFER_EXPERTS=2`) reproduces the undeferred KL (0.1541 vs 0.1538); `DEFER_ROUTES=8` reproduces full
deferral bit for bit (KL 0.343360504056 both).

Readings:
- Layer-profiled affinity (early layers low, late layers high; the aff-0.16 scan gave KL 0.129 / 0.122 / 0.084 /
  0.043 / 0.041 / 0.030 for layer groups 3-9 / 10-16 / 17-23 / 24-30 / 31-37 / 38-46) beats flat affinity:
  mix2 gives aff-0.12 hit rates (80 %) at lower KL.
- Deferral and heavy affinity conflict: the affinity bonus steers the router to resident experts, so a cold route
  that is still selected carries a high weight, and deferring it costs KL. Capping the deferred route's weight
  share at 0.16-0.20 removes the KL cost (0.19-0.20 = the affinity setting's own KL) and keeps about two thirds
  of the speed gain.
- HumanEval (90 tasks, short answers) shows no loss for any setting, deferral included. The one long csv
  generation fails under every deferral setting and passes without; one prompt, deterministic, so it is a weak
  signal, but it is the only long-form evidence so far.
- Confirmation (two alternating rounds, 2026-10-10): aff 0.08 19.48/19.51, mix2 23.29/23.30, mix2 + defer 1 cap 0.16
  25.01/25.00, mix2 + defer 1 26.78/26.79, mix2 + defer 2 28.20/28.20 tok/s.
- Long C++ review samples (1280 tokens, `tools/glm_generate.py`, `b550:~/work/kexp/sample-*.txt`): baseline,
  mix2 + defer 1 and mix2 + defer 2 all read coherently and all three, baseline included, fall into the same
  "let me recall UINT64_MAX" loop, so that failure mode is the model's, not the levers'.
- Dead ends: MTP (CPU bytes grow with verify width), tier policy knobs, CPU thread count (10-16), MLA graph
  capture, cold-route pruning (`STRATA_GLM_ROUTE_COLD_MIN_SHARE`, KL 0.20+ before it saves time; patch kept at
  `b550:~/work/kexp/cold-prune.patch`), GPU PCIe streaming of cold experts (patch at
  `b550:~/work/kexp/with-cold-stream.patch`).

## Projection for DDR4-2666 (CPU phase x 0.8, nothing else changes)

mix2 ~26 tok/s, mix2 + defer 1 cap 0.16 ~28, mix2 + defer 1 ~31, mix2 + defer 2 ~32. To be measured once XMP is
enabled (then re-run `tools/membw/membw --threads 16 --mib 2048` and the step traces).

Harness on b550: `~/work/kexp/steptrace.sh` (speed), `evalkl.sh` (KL, reference `/mnt/disk01/kexp/eval4-aff0.logits`),
`codecheck.sh` + `codegate.py` (coding bench fixtures; `-Werror` relaxed on retry), `humaneval.sh` (+ `hesum.py`).
