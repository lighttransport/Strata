# GLM-5.3-Flash two-GPU runner simulator (`tools/sim/glm_runner.py`)

A calibrated model of one decode verify step and one 8K prefill of the mixed two-GPU runner (CUDA primary +
HIP expert worker, or the reverse) on TR16: RTX 5060 Ti 16 GB (PCIe 3 x8), RX 9070 XT 16 GB (PCIe 3 x16),
Threadripper 1950X with 15 expert threads over two NUMA nodes. It was built in October 2026 to rank the
engineering levers for the 50 tok/s decode / 900 tok/s prefill targets before writing engine code; the
measured outcome of each lever is in [GLM_TR16_RUNNER_PLAN.md](GLM_TR16_RUNNER_PLAN.md). The older
single-GPU simulator for the same model is [GLM_SIM.md](GLM_SIM.md) (`tools/sim/glm_sim.py`); this one
shares its hardware tables and the generic DAG scheduler in `tools/sim/heterogeneous.py`.

Every number comes from `tools/sim/data/glm_kernels_tr16.json` (measured kernel and transfer constants plus
the fitted ones), `tools/sim/data/hw_tr16_runner.json` (link, RAM and NVMe rates) and
`tools/sim/data/glm_runner_params_tr16.json` (the fit). It predicts speed only; it makes no quality claim.

## What it models

A decode step is a dependency graph scheduled on these resources: the primary GPU's compute stream, the
worker GPU's compute stream, one serial CPU lane (the expert pool) with RAM bandwidth charged per NUMA node,
the two PCIe links, and the worker's host service thread. Per MoE layer and verify group the graph has:

- the mixer segment on the primary (KDA or MLA at the group width and context, dense projections, norms,
  router, publication to the CPU mailbox);
- the CPU slot: fixed per-slot overhead plus the routed experts' bytes at the lane's bandwidth;
- the worker request: submit latency, the worker's fixed cost plus per-route cost, the reply over its link;
- the resident experts on the primary and the combine that the next mixer of the same group waits on.

The two verify groups alternate per layer, as the engine enqueues them (group A runs ahead through the dense
layers). Fixed per-step work is the head (embedding, dense layers 0-2), the tail (output norm, logits readback,
sampling), the MTP draft chain (three draft steps on the primary with CPU or GPU draft experts) and the
resynchronization. Expert residency on each GPU comes from recorded traces (for calibration) or from a cache
policy (LRU, LFU with halving, or the engine's own C11 admission rule through `tools/sim/cache_policy.py`).

Prefill at 8192 tokens: per layer the mixer runs on the primary, activations cross to the worker (F32 or F16,
with the memcpy stages priced as host work), expert groups of 16 stream through the staging ring to the
owning device, dequantize-plus-GEMM per group, the worker returns rows in chunks or one partial, and the next
layer's groups prestage during the mixer. The topology B variant (a layer-range pipeline across the two
cards, with residual handoff through host memory) is also modelled; it was built and measured at 12.5-17.6 tok/s on
this desktop (plan section 11) and is kept opt-in for two-socket, same-vendor servers.

## Command line

`PYTHONPATH=tools python3 tools/glm_runner_sim.py <subcommand>`:

| subcommand | what it does |
|---|---|
| `calibrate` | coordinate descent over the free constants (per-slot overhead, worker request and round trip, publish and combine, MLA context term, memcpy rate, ack cost) against the fit records; writes the params file with source hashes |
| `validate` | holdout gate: decode rows within 10%, prefill within 15%, each STEP_TRACE scope within 15%; exit 1 on a miss |
| `decode`, `prefill` | one prediction with the scopes (head, CPU service, exposed wait, tail, draft, resync, other) defined like the engine's `STEP_TRACE` |
| `sweep` | greedy forward selection over the levers in `tools/sim/data/glm_levers.json` (lossless by default, `--allow-lossy` adds the rest); prints the ranked path with cumulative tok/s and whether the targets are reached |
| `records-from-bench` | turns an instrumented `tools/glm_mixed_bench.py` run (`STRATA_GLM_STEP_TRACE=2`) into fit or holdout records |
| `manifest` | emits a `strata.heterogeneous.v1` manifest that `glm_sim.py heterogeneous` reproduces, used by the tests |
| `probe-plan` | lists the measurements still missing for the table, with the commands that take them |

Useful `sweep` options: `--primary cuda|hip`, `--worker`, `--context`, `--shares cpu=,worker=,primary=` (the
measured hit shares of a run), `--accepted` (tokens per verify step), `--draft-bank cpu|primary|worker`,
`--fraction` (prefill expert share on the primary), `--exclude lever,lever` for levers already built or
removed, `--target-decode`, `--target-prefill`.

Tests: `PYTHONPATH=tools python3 -m unittest tools.test_glm_runner_sim` (no GPU or model; scope conservation,
one worker request per slot, calibration recovers planted constants, validate refuses a holdout miss, sweep
is monotone and excludes lossy levers by default, prefill F16 halves the exchange bytes).

## Two nodes

`multinode` puts the worker GPU on a second host behind a network link (`--links gbe ib-qdr ib 10gbe` or
`rtt_us,gbps,exposed_us`; presets in `tools/sim/hw.py` `LINKS`) and prints decode and prefill per link next to the
primary alone and the one-host pair, searching the worker's expert slots, an optional CPU lane on the worker node
(`--cpu-shares`) and the prefill fraction. `--anchor-decode` / `--anchor-prefill` scale every row to a measured
primary-alone number. The configuration and the TR16 + B550 results are in [GLM_MULTINODE.md](GLM_MULTINODE.md).
`split-model` prices a model larger than one node's RAM (`--expert-scale`, `--node-a-weights-gb`): one node with
enough RAM against two nodes where the bytes that do not fit live on node B and its CPU serves them
(`two_node_hardware(..., gpu_on_b=False)`, per-layer shares from the coverage priors).

## Calibration record

Fit on the cuda-primary and hip-primary 8K traces, the single-card 8K rows and the 8K prefill timelines;
held out: 2K rows and the chat prompt. Mean |log error| after the fit 0.024; 8K decode and 8K prefill
holdouts within tolerance, 2K prefill rows miss by 16-63% (an unmodelled per-layer cost in the mixed 2K
path; 2K is reported, not targeted). Re-anchored on the release decode configuration's own traces it
predicted 39.9 tok/s at 8K and 43.9 at 1K against 40.0 and 42.0 measured.

## What held up and what did not

The simulator was right about the regime: decode is bound by the primary GPU's per-slot chain plus the
worker protocol, not by CPU bytes, and the layer-range pipeline cannot beat the interleaved mixed runner.
It was right about the GPU draft bank (+7%) and the role swap to a CUDA primary (+5%). It overestimated
the shared projection read (serializing the two groups costs more than the saved weight reads), the
resident side stream (-60% measured) and the fused router/publish (+0 at 8K), and it had no term for the
GPU tier's expert promotions (12 x 9.1 MB uploads per step), which turned out to be the one 8K lever that
paid. Lever-by-lever forecasts carry errors of several tok/s in both directions; use the sweep to choose
what to measure next, not as a result.
