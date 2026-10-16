# llama.cpp #29928 (GLM5-Next MTP) against the Strata GLM-5.3-Flash runner

Study of [ggml-org/llama.cpp#29928](https://github.com/ggml-org/llama.cpp/pull/29928) ("feat: add GLM5Next MTP,
optimize", merged 2026-10-07; it supersedes #27917) and an adoption plan for the Strata runner. It also covers
the runner-simulator extension that ranks the plan (`tools/sim/glm_runner.py`, section "MTP model" in
[GLM_RUNNER_SIM.md](GLM_RUNNER_SIM.md)). Receipts are in `docs/benchmarks/mtp_29928_20261016/`. Everything below
the study section is a **forecast**: no engine code changed and nothing new was measured.

## What the PR does

Three files, +173/-25.

- **`graph_mtp`** (`src/models/glm5-next.cpp`) is the NextN draft head: one block appended after the trunk.
  - Input: `eh_proj(concat(enorm(embed(token)), hnorm(h)))`, where h is the trunk's hidden state for the
    previous position.
  - Attention: `attn_norm`, then a DSA/MLA block (`build_dsa_layer`) with its own latent cache and indexer.
  - `crop_before_nextn`: keeps only the output rows before the FFN.
  - FFN: `ffn_norm`, then routed MoE plus the shared expert, then the residual.
  - Head: `shared_head_norm` (falls back to `output_norm`), then `shared_head_head` (falls back to `output`).
- **Loader.** `mtp_only` / `trunk_only` detection, so the draft head can come from a separate GGUF.
  ggml-org ships `mtp-GLM-5.3-Flash-Q4_0.gguf` next to the trunk quants. A follow-up will share that block
  across the 16 models that copy it.
- **`llama-memory-recurrent.cpp`.** Partial rollback of recurrent state (the KDA layers) uses per-token
  snapshots. The PR adds a refusal when a cell is shared by several sequences.
- **Not in the merged diff:** #27917's `index_share_for_mtp_iteration`. In that version the draft computes its
  indexer top-k once per chain and reuses it in later steps. The merged `graph_mtp` passes `prev_sel=nullptr`.
  #27917 says the GLM-5.3-Flash `config.json` sets this flag to true. No local file confirms that yet.

**Reported numbers.**

| Source | Setup | Result |
|---|---|---|
| #27917 | Q4 head, depth 5 | 3474/4664 drafts accepted (0.745), mean length 4.13 |
| #27917 | same | Acceptance per position, share of rounds: 0.924 / 0.813 / 0.600 / 0.441 / 0.348 |
| #27917 | same, as P(i \| earlier accepted) | 0.924 / 0.880 / 0.738 / 0.735 / 0.789 |
| #27917 | CPU-offloaded, depth 3 | +15-30% |
| #29928 | RTX PRO 6000, n=2 | about 2x decode at short context, +52% at 32K |
| #29928 | five prompts up to 32K, n=2 | Acceptance 71-81%, matching unsloth's implementation |
| #29928 | same GPU, MTP on | The draft context's VRAM cut usable context from 128K to 48K |

## What Strata already has

Our runner already implements every mechanism the PR adds:

- **NextN path.** `blk.45`, loaded into `m.draft_layers` (`include/strata/core/model.hpp:146-180`).
  `mtp_step` and `mtp_prime_batch` in `src/program/glm_decode.cpp` compute the same graph as `graph_mtp`.
- **Speculation loop.** `generate_mtp` drafts a chain, `verify()` checks the window in one batched pass (split
  into two groups by `STRATA_GLM_SPLIT_VERIFY`), and `restore_verified` rolls back without replay through
  per-token verify-history snapshots. That is the same idea as llama.cpp's `n_rs_seq`.
- **Draft experts.** `--mtp-experts=cpu|gpu` places them on the CPU or in a GPU draft bank.
- **Decode resync.** `STRATA_GLM_MTP_SYNC_STATE_ONLY=1` makes the resync state-only, which is the decode-time
  version of `crop_before_nextn`.

Measured acceptance on TR16 (cuda-hip, 8K holdout-code, REAP-224) is 0.931 / 0.784 / 0.75 per position, given
earlier positions were accepted. That is about 3.2 tokens per round at depth 3. Position 2 is about 10 pp below
llama.cpp's head.

Two differences may explain the gap:

1. **Pruned draft head.** Our head is the REAP-224 one. `GLM_PRUNE_224.md` drops the draft layer's 64
   highest-router-bias experts without calibration, while llama.cpp keeps all 288.
2. **Measurement setup.** The prompt mixes differ, and soft accept and lookup drafts change our bookkeeping.

## Gaps the PR exposes

1. **Priming does work nobody reads.**
   - Without `STRATA_GLM_MTP_BATCHED`, which is unset in every shipped config, `prepare_mtp` primes the
     draft with `sync_mtp(prompt, 0)`. That is one state-only draft step per prompt token, after the prefill.
     At the measured resync rate (`host.resync_ms / accepted` ≈ 0.45 ms per token) that is about **3.7 s
     for an 8K prompt** before the first token.
   - The batched path (`mtp_prime_batch`, `glm_decode.cpp:8455-8462`) runs the draft's attention output and
     a full MoE over every prompt row, and nothing reads the result. llama.cpp's `crop_before_nextn` avoids
     exactly this.
   - The priming time is not in the `PREFILL` tok/s the release docs quote. It is in `MTP_PRIME ms=`, which
     no saved trace has recorded.
2. **The draft indexer is recomputed every step.** The draft block scores the whole context with its indexer
   in every draft step (`mla_gpu` for layer 45). With a shared selection, only step 0 would pay for it.
3. **We can't load a different draft head.** The draft head always comes from the base GGUF, so the full
   288-expert Q4_0 head can't be tried without a repack.

## Adoption plan

Each item has a lever in `tools/sim/data/glm_levers.json` and a measurement to take before building it.

| # | Change | Kind | Lever | Forecast (sim) |
|---|---|---|---|---|
| C1 | Prime the draft in the prefill pass, state only. In `mtp_prime_batch`: `mla_gpu(..., state_only=true)`; drop `moe_gpu`, the saxpys and the attention output. Make batched priming the default. | Lossless: draft proposals identical, check with `--check-verify` | `mtp_prime_crop` / `mtp_prime_batched` | 8K prompt: 3.7 s → 0.09-0.12 s of priming. Time-to-first-token throughput 398 → 483 tok/s (cuda-hip), 424 → 523 (hip-cuda). Batched without the crop: about 0.5 s. |
| C2 | Indexer selection shared across the draft chain: draft steps 1..k-1 reuse step 0's top-k | Output lossless; acceptance may move either way | `mtp_index_share` | +0.7% at 8K, +2.2% at 32K (cuda-hip); grows with context |
| C3 | `--mtp-gguf=PATH`: load `blk.45.*` / `nextn.*` from a second GGUF, through the same overlay as the expert packs. A/B: shipped head vs ggml-org full 288-expert Q4_0 vs Q8_0. | Lossless | `mtp_q4_full_head` | +2.4% at 8K, if llama.cpp's acceptance curve carries over. The CPU draft bytes are 1.42x per expert. A GPU bank would need 4.1 GB = 10.7 slots per layer, a net loss on cuda-hip. |
| C4 | Verify window shape: unsplit `STRATA_GLM_SPLIT_VERIFY=0` at depth 3, and depth 4 as (2,3) | Lossless | `mtp_unsplit_4`, `mtp_window_2_3` | Unsplit (4,): +9.7% at 8K (cuda-hip), +20% (hip-cuda). Outside the fit, see below. (2,3): +2.1% / +6.8%. Depth 2 (1,2): −9% / −12%. |
| — | Not adopted: llama.cpp's separate draft context, and the shared-cell rollback guard | — | — | Our draft shares the process and the KV allocator. Multi-sequence serving has its own batch path (`glm_batch_serve.inc`). |

**Order.**

1. Measure C4's unsplit window first. It is one environment variable on the release decode config, and its
   forecast is the largest, but also the least trusted (see below).
2. Build C1. It is small, lossless, and removes seconds from every long prompt.
3. Build C3, whose A/B answers the 10 pp question at position 2.
4. Build C2 last, together with the 32K work.

The sim's best stack on cuda-hip at 8K, unsplit (4,) + Q4 head + index share, forecasts **39.5 tok/s**
against 35.0 for today's shape. Run `sweep --primary cuda --worker hip --only mtp_* ...` to reproduce it.

## What the forecasts rest on

- **Unsplit and wider windows are outside the calibration.**
  - Every record is split (2,2).
  - The sim prices other widths as the fitted width-2 cost times the raw microbench ratio (`width_ratio`).
  - Split verify was introduced on 2026-10-06 for the single-GPU, CPU-bound runner, where overlapping the two
    groups paid off. On the two-GPU runner the probes found decode bound by the per-slot GPU chain, which is
    why one wider group looks better.
  - It may not be: the unsplit path also loses the overlap between the GPU mixer and the CPU lane. Measure it.
- **Depth 4/5 has failed once.** On single-GPU TR16 on 2026-10-10 it measured worse than depth 3. The tail
  acceptance at positions 3-4 (0.75 / 0.70) is extrapolated, not measured.
- **The Q4 head's acceptance is imported.** It comes from a CPU-offloaded llama.cpp setup with an unknown prompt
  mix. It is a prior for the A/B, not a TR16 number.
- **Priming costs are estimates.** Serial priming uses the measured resync rate. Batched priming is one MLA
  mixer plus one average MoE layer from the prefill model. Crop assumes the state-only MLA pass costs 35% of a
  mixer (`model.mtp_prime_state_fraction`). Replace all three with `MTP_PRIME ms` measurements.

## Measurements to take

All on a quiet machine, ≥512 generated tokens, with `STRATA_GLM_STEP_TRACE=2` so that `records-from-bench`
picks up `MTP_POSITION` (records now carry `acceptance_by_position`).

1. Release decode config, `STRATA_GLM_SPLIT_VERIFY=0` vs `1`, at 1K and 8K, both orientations.
2. `MTP_PRIME ms` at 2K and 8K with and without `STRATA_GLM_MTP_BATCHED=1`. After C1, also with the crop.
3. After C3: `MTP_POSITION` with the shipped head, the ggml-org Q4_0 head and a Q8_0 head. Also `draft_ms`
   per step.
4. After C2: `draft_ms` per step and `MTP_POSITION` at 1K, 8K and 32K, with and without the share.
5. Depth 4 (2,3) on the two-GPU runner, for `MTP_POSITION` index 3.

Feed each run back with `records-from-bench --merge-into`, add an acceptance profile from its
`acceptance_by_position` if it differs, and re-run `calibrate` / `validate`.
