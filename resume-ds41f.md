# Resume DS41 two-node prefill optimization

Continue implementing the approved plan below. This file replaces the old chronological handoff. Read AGENTS.md first, inspect the working tree, and preserve unrelated work. Do not mistake the protocol foundation for a working distributed prefill implementation.

## Objective and authorization

Implement opt-in two-node DS41 whole-expert prefill, targeting **1,200+ tokens/s for one warm 8,192-token sequence** (at most 6.8267 seconds). Include router, Engram, staging, network, GPU work, reduction, and final logits in elapsed prefill time. Report cold setup, cache restoration, and true TTFT separately. A 16K prefix is secondary.

The user approved implementation and borrowing decode-cache VRAM during prefill, provided it is restored before decode. No new approximation is allowed: preserve weights, activation precision, routing and truncation. Measured FP32 implementation/reduction drift versus the current MMQ baseline is allowed only within the quality gates below.

The user chose a frozen repository C++ prefix: tokenize decoder source from commit `89e439c7` into exactly 8,192 input IDs. Freeze tokenizer identity, source hash and token-ID hash. Repository prose and a 16K prefix are secondary fixtures. This fixture and a fresh matched baseline still need creating; historical timings are not comparable qualification evidence.

Do not push. Prefill changes are uncommitted. No active goal was requested for this phase. Do not spawn subagents unless explicitly authorized by the user or applicable instructions.

## Current implementation: foundation only

New files (currently untracked):

- `include/strata/host/ds41_prefill_wire.h`
- `src/host/ds41_prefill_wire.c`
- `tools/ds41_prefill_wire_test.c`

`CMakeLists.txt` adds the implementation to `strata_host` and registers `ds41_prefill_wire_test` with label `ds41` under `STRATA_BUILD_TESTS`.

The C11 implementation provides:

- STD3 magic, version 1 and capability validation for Q8 transport, weighted FP32 replies, phase transitions and requested geometry.
- A 56-byte request with sequence number, operation, layer, absolute first token, token/route counts, hidden size, top-k, activation kind and payload byte count.
- BEGIN / EXECUTE / END / DECODE_RAW validation and a phase state machine. Decode is rejected during prefill; replay and sequence overflow are rejected; invalid requests leave state unchanged. The caller must commit a transition only after GPU work and transfers complete.
- Partial route counts, including zero, bounded by tokens times top-k. This is needed for expert partitions.
- CPU Q8 repack reference: native `[column_block][row]` to wire `[row][column_block]`, bitwise copies, excludes native column padding, clears padding and caller-supplied tail on unpack. Buffers must be disjoint.

Important discovery: the original planned universal FP16 d/s wire representation was incorrect for mixed formats. Actual MMQ has three metadata layouts. The wire now declares logical 128-value blocks of **16 metadata bytes plus 128 signed quant bytes**, retaining all original bits:

- DS4 (activation enum Q8_1): four FP16 scale/sum pairs; IQ1_S uses this.
- D4: four FP32 scales.
- D2S6: two FP16 scales and six FP16 sums; Q2_K uses this, with 64-value quantization scale groups.

Never convert D4 metadata to FP16 or requantize on the worker. Inspect the current ggml MMQ quantizer and type-to-layout mapping before GPU implementation. Root quantization must match the baseline for each layer's weight format, not a universal substitute.

**Not implemented:** decoder/worker STD3 negotiation, setup serialization, request transport, reply validation, runtime feature flag, GPU Q8 pack/unpack, route partitioning, whole-expert worker MMQ, cache lending/restoration, cost model, fixtures, profiling, distributed benchmarks. The worker still accepts legacy STD1/STD2 only. No 1,200+ tokens/s claim exists.

Checks completed for this foundation:

- CMake target and CTest pass in `build-glm` (CUDA-configured) and `build-hip-rocm101-tr16` (HIP-configured). These are CPU tests, not GPU execution or full backend builds.
- Strict C11 and C++20 compile/run with `-Wall -Wextra -Werror` pass.
- ASan/UBSan pass with `ASAN_OPTIONS=detect_leaks=0`; LeakSanitizer itself cannot run under the environment's ptrace setup. No leak-check claim.
- Tests cover missing capabilities, invalid version and bounds, partial/zero routes, 100 repeated phase cycles, replay, decode interlock, sequence exhaustion, exact repack bytes and padding, insufficient capacity without writes.
- `git diff --check -- CMakeLists.txt` passed.

Review remaining protocol edge cases before runtime use: wire byte order/ABI, message length arithmetic, route payload validation, reply shape/status and errors, peer disconnects, sequence acknowledgement, and consistency of accepted hidden geometry with 128-value repack blocks. Capability validation is only a helper until actually exchanged before weights transfer.

## Approved implementation sequence

1. Freeze fixtures and run fresh same-day baselines with the existing guarded path.
2. Add opt-in `STRATA_DS41_PREFILL_EXPERT_PARALLEL=1` via `glm_env.h` helpers. Preserve the default path and STD1/STD2 wire behavior.
3. Negotiate STD3/version/capabilities before any weights transfer. Keep existing whole-expert RAM ownership (approximately 48 GiB worker); do not redistribute weights per prompt.
4. Keep root's layer-major 8K FFN pass and tiled trunk. Quantize baseline activations once for the layer format; send canonical logical Q8 blocks with unchanged metadata and quant bytes. Worker repacks without requantization. Use bounded pinned staging buffers; no GPUDirect requirement.
5. Partition selected routes completely and disjointly. Exclude remote experts from root staging/producer readiness. Worker returns one router-weighted FP32 partial sum per token, rather than raw outputs for every route. Handle zero routes, tails, rcap splitting and absolute token offsets.
6. Borrow decode GPU arenas only after synchronizing outstanding work. Retain shared RAM weights, bound temporary buffers, restore caches before decode, and measure restoration and actual TTFT. Exercise repeated transitions.
7. Use a per-layer cost model to assign worker-owned experts to either GPU; experts unavailable on the worker stay local. Every route must execute exactly once.
8. Profile actual active DMA, exposed waits, wire time, per-layer active rows and GPU computation. Remeasure interconnect bandwidth. Do not call event intervals containing idle waits active DMA.
9. Only if still below target and IQ1_S products occupy at least 20% of exposed critical path: evaluate lossless GPU-side expansion, `main = 8 * scale * grid` (range +/-120), separate +/-scale delta correction, retain original FP16 base. Do not round a combined FP16 scale. At most two expanded buffers of 256 MiB each. Other formats retain ordinary MMQ. Adopt a geometry only with at least 10% chain improvement including expansion and a full-prefill improvement.

## Engine integration map

The actual DS41 engine is `src/program/ds41_decode.cpp`, not `glm_decode.cpp`.

- `include/strata/net/expert_tp.hpp`: existing setup/accept/weights/ready, STD1 row split and STD2 whole-expert ownership. Keep their layouts unchanged.
- `connect_own()` around line 4855 rejects streamed prompts, peers and incompatible routing. Relax only for a negotiated new mode; do not broaden legacy STD2 implicitly.
- Whole-owner setup around 4967 sends STD2 and appends owned IDs/profile counts. Worker receives setup near line 130; must reject incompatible STD3 before accepting weights.
- Decoder `Remote` near line 390 sends F32 activations and full top-k jobs; legacy raw replies are per selected route. Do not reuse its unchecked sizing for STD3.
- Long streamed FFN near line 6738 bypasses whole-owner acceleration. `stream_experts()` near 6212 groups routes by expert; route exclusion and producer readiness both need adjustment.
- Existing `PREFILL_HALVES` condition around 2990 keeps tiled trunk and one long FFN pass; preserve row0/skip/absolute KV indices. Prompt chunk selection is near 9318.
- `src/program/ds41_worker.cpp`: `use_mmq = !own && mq::built()` near 325 explicitly disables MMQ for STD2. Legacy maxT is 4096, rcap 8192; owned canonical scratch is only 64 tokens. Owned raw dispatch near 545 precedes ordinary MMQ near 714. Audit all capacities rather than changing one constant.
- Worker existing ordinary MMQ groups routes, double-buffers layer weights, quantizes, runs gate/up, SwiGLU, down quantization, down MMQ and scatter. Reuse appropriate pieces while preserving whole-expert ownership and bounded memory.
- `include/strata/prefill/moe_mmq.hpp`, `src/prefill/moe_mmq.cu`: activation padding to 512; native blocks are column-major over 128-value blocks. Product IDs require 128 readable initialized tail entries. `dst_rows` bounds diagnostics are pre-existing uncommitted work and must be retained.
- ggml quantizer reference: `build-tr16-goal-20261010/ggml-reference/ggml/src/ggml-cuda/quantize.cu`, and `mmq.cuh` for `mmq_get_q8_1_ds_layout` and `block_q8_1_mmq`.
- DS41 geometry: hidden 5120, FFN 2304, top-k 6, 40 layers, 384 experts, mixed quant formats.

## Validation and qualification

- Bitwise GPU Q8 repack against the CPU reference for all metadata layouts; complete/disjoint routing; weighted sums; zero routes; rcap and padding; malformed protocol; 4K/8K/16K inputs and repeated phase transitions.
- For each prompt, mean KL <= max(1e-4, 2 * baseline self-KL), and top-1 disagreement <= max(0.1%, 2 * baseline self-disagreement). Also report perplexity/accuracy. Do not loosen gates.
- Run the C++ parse_u64 oracle (400,532 cases, fatal UBSan) after approximately 8K context on baseline and candidate.
- Three paired baseline/candidate processes, each one cold and two warm passes. All six warm candidate passes aggregated must reach 1,200 tok/s; each process's warm median must also reach it. Scored runs have detailed profiling disabled. Prevent accidental prefix/KV reuse.
- Build affected HIP/CUDA backends consistently with existing bounds hardening, check unchanged defaults and decode handoff regression. Report unavailable SYCL checks explicitly where applicable.
- If the measured ceiling is below target, report target unmet; do not introduce lossy behavior.

## Safety and environment

- Use existing ignored local configuration for model, profile, remote connection and binary paths. Never add private endpoints or personal filesystem paths to source, reports or commits.
- Relevant local configuration is under `experiment/ds41f_tr16_decode/logs/`, including `model.local`, `expert-profile.local`, `worker-bin.local`, and `reference-root.local`.
- Local HIP build: `build-hip-rocm101-tr16`; CUDA build: `build-glm`. Alternate decode build helper: `experiment/ds41f_tr16_decode/build_alternate.py`.
- Existing remote alias: `b550`. Its current worker build uses Makefiles, not Ninja. Discover actual source/build locations through existing local config or remote read-only inspection. Protocol/library/program updates must be deployed consistently.
- Root RAM 128 GiB; worker RAM 64 GiB. The worker is shared. Respect current reservations and memory guards; no swap workaround or relaxed memory caps.
- Keep the local desktop GPU free-memory floor at **2.5 GiB**, and inspect recent GPU faults. Do not bypass guards.
- `tools/ds41f_prefill_guard.sh` is pre-existing untracked work. It needs deliberate forwarding of the new remote/prefill options; do not bypass its clean environment with arbitrary extra arguments.
- Never edit an executing shell file in place. Use atomic replacement and preserve mode. Never use `pkill -f`; stop exact owned PIDs only. Never glob-delete shared weight caches.
- Previous experiment processes were stopped and one specifically owned 48 GiB remote RAM cache was removed after ownership/no-use checks. A new experiment must refill it; inspect current state rather than assuming the worker is idle.

## Existing decode results and working-tree boundaries

Local commit `89e439c7` (`ds41: add opt-in exact IQ1_S and external-merge decode fast paths`) completed the previous decode phase; it was not pushed.

- Exact candidate: paired 24.9935 / 26.1361 tok/s, mean 25.5648; another four grid-on runs mean 25.02755. Grid and paired IQ1_S paths have frozen parity checks.
- Lossy trim 0.125 plus canonical budget 2304: 30.4605 / 30.5729, mean 30.5167 tok/s. Both modes passed the C++ oracle, but lossy six-prompt KL 0.0387–0.187 fails the earlier KL <= 0.08 numerical gate. Do not call that generally quality-qualified.
- Committed summaries/presets: `experiment/ds41f_tr16_decode/CANDIDATES.md` and `results_2026-10-29.json`.
- Historical single-node prefill around 645 tok/s used another fixture. Root transfer bandwidth around 12.2–13 GB/s and interconnect around 2.44 GB/s are context only, not fresh benchmark evidence.

Pre-existing tracked edits include `docs/DS41F_PLAN.md`, kernel headers/CUDA files, parity tests, prefill GEMM/MMQ files, `src/program/glm_decode.cpp`, pack/quantization tools, and memory-bandwidth tools. There are also many untracked experiment trees. Preserve these; do not reset, broadly stage, or overwrite them. In particular, the prefill MMQ bounds hardening is necessary existing work. Inspect exact diffs before building or committing. This replacement handoff is intentionally updated, but the older detailed evidence remains in its original files.

Next concrete work: finish protocol framing/reply validation and wire it into decoder/worker setup, implement GPU pack/unpack against the CPU oracle, then add guarded worker whole-expert MMQ dispatch and root route exclusion before attempting performance qualification. Do not stop at another documentation-only or protocol-only milestone when resuming implementation.
