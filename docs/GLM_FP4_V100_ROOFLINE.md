# Native FP4 simulator and roofline gaps on two V100 GPUs

The 2026-10-18 native FP4 measurements are now an explicit workload in
`tools/sim/glm_fp4.py`, available through `tools/glm_runner_sim.py native-fp4`.
It records the benchmark and estimates conditional resource ceilings. It does
not fit throughput constants to these three repeats or replace the existing
GGUF/RPC simulator profiles.

The measured machine had two Tesla V100-PCIE-32GB GPUs, two Xeon Gold 6240
sockets, 32 CPU workers and a 181 GiB allocation. The checkpoint's 42 files
occupied 177,463,841,024 bytes and stayed unchanged. Results concern the frozen
8192-token prompt, 512 output tokens, warm resident weights and one request at
a time. Decode timing covers 511 advances after the prefill token.

| Recorded configuration | Decode tok/s | Prefill tok/s |
| --- | ---: | ---: |
| Preserved W4A16 reference, 35 workers | 5.26 | 393.80 |
| Q8 CPU, bounded route dropping, greedy | 19.08 | 394.74 |
| Q8 CPU, bounded route dropping, MTP and suffix lookup | 31.02 | 390.70 |

The final row used two full discarded warmups, executable/library page-cache
preparation during the second warmup, a discarded one-token request and three
timed requests. All timed major faults and sampled swap were zero. The three
output sequences matched. Coding passed 18/20 tasks versus 19/20 for the
preserved baseline; the policy was frozen before that check. Lossy routing and
CPU arithmetic remain opt-in. See the [benchmark report](GLM_FP4_V100.md) and
[frozen measurements](glm_fp4_v100_decode20_measurement.json).

## What changed in the simulator

The [native profile](../tools/sim/data/glm_fp4_v100.json) stores qualified
observations as `fit: false`, timed-request work, MTP phase timers, compressed
expert extents and a checkpoint-header matrix census. Source reports and the
selected binary/log hashes are recorded and checked. This is a workload import,
not independent validation of a fitted timing model.

- Native FP4 has 288 experts per MoE layer. REAP-224 or Q23 byte tables give
  the wrong traffic. One expert has 25,165,824 elements, 12,582,912 FP4 code
  bytes and a 14,155,776-byte expanded GPU code/scale allocation.
- `uploaded_bytes` counts expanded GPU allocations, not PCIe wire bytes.
  Upload-count deltas and compressed source extents supply the transfer proxy.
  Repeated shutdown reports are deduplicated.
- Q8 preserves FP4 weight codes. Borrowed sparse scales, fused dispatch and
  spinning workers affect CPU decode. Prefill retains W4A16 and all routes.
- A 50% routing-mass budget does not imply half the expert work disappears.
  Work comes from CPU groups and evaluated rows, including rejected verification
  rows. The cumulative 38.57% discarded mass is not a compute multiplier.
- Zero `cpu_expert_ms` fields mean native phase timers were disabled, not that
  CPU work was free. Graphs and smaller-reserve controls measured slower; no
  speedup is assigned to them. No coding quality or KL is predicted.

## Resource assumptions

V100 PCIe specifications supply an optimistic **900 GB/s HBM bandwidth and
112 TFLOP/s tensor peak per GPU**, distinct from V100S specifications. These
are component peaks, not achieved FP4 inference rates.
[NVIDIA specifications](https://www.nvidia.com/en-gb/data-center/tesla-v100/)

The CPU's maximum DDR4-2933 rate and six channels per socket give
`2 × 6 × 2933 × 8 / 1000 = 281.568 GB/s` across both sockets. Installed DIMM
speed, channel population and sustained expert-stream bandwidth were not
measured. The 80/120/180 GB/s rows below are sensitivity assumptions, not
measured or fitted bandwidths.
[Intel specifications](https://www.intel.com/content/www/us/en/products/sku/192443/intel-xeon-gold-6240-processor-24-75m-cache-2-60-ghz/specifications.html)

The registered-RAM copy probe measured approximately 12.31/12.34 GB/s per GPU
and a 24.615 GB/s median aggregate with concurrent 256 MiB copies. It does not
establish inference DMA throughput or overlap. Each GPU path is modeled
separately: the busier primary cannot use both links' combined bandwidth.

## Decode roofline and gap

The request at the median decode rate evaluated 625 main-model rows in 146
MTP/lookup rounds, delivering 511 advances, or 3.50 advances per round. It ran
27,590 CPU expert groups. Reading each group's original FP4 codes once costs
347.16 GB per request, or **0.679 GB per delivered advance**.

The streaming roof is `bandwidth / 0.679 GB`. A loose primary-HBM floor for
mandatory FFN/head reads is also included. Independent resource floors overlap
with `max`, rather than being added. The CPU stream dominates these scenarios:

| Assumed CPU stream GB/s | Conditional code-stream ceiling tok/s | Measured / ceiling |
| ---: | ---: | ---: |
| 80 | 117.8 | 26.3% |
| 120 | 176.6 | 17.6% |
| 180 | 264.9 | 11.7% |
| 281.568, specification scenario | 414.4 | 7.5% |

These deliberately loose memory roofs assume no cross-call LLC retention and
omit scale metadata, activation traffic, integer unpack/dot instructions, most
GPU traffic, synchronization and recurrence. Percentages are elapsed-rate
ratios, not measured memory utilization. **177–265 tok/s is not an achievable
speed forecast.** A tighter kernel/DAG model needs phase measurements that this
uninstrumented run does not provide.

The phase timers give a firmer optimization constraint:

| Median-rate request phase | Seconds | Share of decode |
| --- | ---: | ---: |
| Verification | 14.3281 | 87.0% |
| Draft | 1.2807 | 7.8% |
| Resynchronization | 0.5372 | 3.3% |
| Preparation and remaining work | 0.3275 | 2.0% |
| Total | 16.4735 | 100% |

Removing **all** non-verification time while keeping verification unchanged
would yield only **35.66 tok/s**. At unchanged non-verification cost, 50 tok/s
requires verification to improve **1.77×**; a 2× verification improvement would
give about 54.89 tok/s. These are Amdahl calculations, not measured variants.
The next useful measurement splits target verification into CPU/GPU work and
per-layer waits, including CPU unpack/quantization, fixed GPU products and
resident experts. Improving only draft/resync runtime cannot close the
memory-roof gap. Changes to acceptance or lookup yield need new work counters.

## Prefill roofline and gap

The discarded one-token request after two full warmups caused 5,175 primary
and 3,965 secondary expert uploads. Its counters include a small internal MTP
step and 56 CPU groups. This is a **same-prompt traffic proxy**, not an exact
wire trace of timed prefill phases.

Expanded GPU allocation traffic was 73.26/56.13 GB. Applying checkpoint source
extents gives estimated **69.62/53.34 GB over PCIe**. The mean uses the
checkpoint's expert-size distribution, not the unrecorded upload-weighted
distribution. Minimum/maximum expert source sizes bound that assumption.

| Resource | Optimistic seconds per 8192-token prompt |
| --- | ---: |
| Primary H2D | 5.66 |
| Secondary H2D | 4.32 |
| Aggregate H2D | 5.00 |
| Ideal projection compute, including device imbalance | 1.78 |

The primary transfer path limits this proxy to **1,448.2 tok/s**, with a
**1,376.3–1,450.2 tok/s** interval from source-size bounds. Summing both links
without accounting for expert ownership would overstate the ceiling. The
projection-only compute ceiling is about 4,607 tok/s and omits attention,
KDA recurrence, mHC, dequantization, launches and data dependencies.

Measured prefill was **20.9674 seconds / 390.70 tok/s**, about **27.0%** of the
central traffic roof, leaving **15.31 seconds above that optimistic floor**.
Those seconds are not all removable overhead: omitted computation and
dependencies must also execute. A trace is needed before assigning this gap
to PCIe, dequantization, GEMMs or synchronization.

The 800 tok/s target needs **2.05×** the measured prefill rate. Under the same
proxy it requires about **6.80 GB/s on the primary link**, below its copy-probe
rate. This model does not rule the target out, but it does not demonstrate an
implementation that achieves it. The next experiment should measure compressed
wire bytes and copy/compute overlap per layer, then compare grouped projections,
dequantization reuse and fewer synchronization points. CPU Q8 and route dropping
do not accelerate the unchanged prefill expert path.

## Reproduce and validate

No GPU, model download or network access is needed for the checked-in report:

```sh
python3 tools/glm_runner_sim.py native-fp4 \
  --output docs/glm_fp4_v100_roofline.json
python3 tools/glm_runner_sim.py native-fp4 --cpu-gbps 120 \
  --target-decode 50 --target-prefill 800 --output tmp/fp4-roof-120.json
PYTHONPATH=tools python3 -m unittest \
  tools.test_glm_fp4_sim tools.test_glm_runner_sim tools.test_sim
```

The [machine-readable analysis](glm_fp4_v100_roofline.json) records resource
floors, timing remainders, sensitivity rows, limitations and source/config
hashes. Eight new accounting tests cover units, overlap, counter deduplication,
compressed transfers, qualified inputs, phase accounting, provenance and CLI
selection. The combined 131-test run passed with one existing routing-trace
test skipped because local traces were unavailable. No new GPU inference,
held-out quality run or hardware calibration was performed for this analysis.
