# GLM DFlash2 on V100 (experimental)

The opt-in `dflash2` decoder mode uses the published
[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
drafter. It is separate from the qualified MTP preset, which measured 32.10
decode / 520.89 prefill tok/s on two V100 PCIe 32 GB GPUs and a dual Xeon Gold
6240 host with 181 GiB RAM. The fastest repeated DFlash2 trial measured
38.56 decode / 515.68 prefill tok/s, but failed all eight reused coding checks.
Its route-drop cap of 0.75 is rejected. A less aggressive DFlash2 trial passed
eight checks at 26.84 decode tok/s. The MTP preset remains selected, and the
50 tok/s target remains unmet.

The pinned checkpoint revision is
`bf582e4eacc1810f76656d1811693ff6c6737d2a`. Its safetensors payload is
2,342,169,800 bytes. The worker converts BF16 draft weights to FP16 in device
memory for Volta. The mixed-precision path uses a small cuBLAS helper with FP16 operands and
FP32 accumulation/output, plus ordinary PyTorch attention, without
FlashAttention, Triton, or BF16 instructions. Checkpoint weights are not
modified on disk or included in this repository. The checkpoint's separate
[research/evaluation license](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2#license)
applies. The reference implementation's [MIT notice](../third_party/licenses/dflash.txt)
is included separately.

## Execution

Install a CUDA PyTorch build supporting SM70, NumPy, safetensors and Ninja in a separate
Python environment. The initial V100 environment uses PyTorch 2.5.1+cu121 and
safetensors 0.6.2. Build the helper on the GPU node with the CUDA toolkit and a
GCC version supported by PyTorch:

```bash
python tools/glm_dflash2_build.py --build-dir build-dflash2-local
```

Optional packed-half GEMMs reuse the existing Volta skinny core. Build them with
`--skinny --build-dir build-dflash2-skinny-local` and add
`--skinny-library build-dflash2-skinny-local/strata_dflash2_skinny.so` to the
worker arguments. Add `--cuda-graph` to replay the five-layer forward pass for
complete 2,048-row windows; shorter windows use eager execution. Graph inputs
receive the current committed K/V, embeddings and absolute position on every
replay. Neither option changes the checkpoint on disk.

The target's KDA mixers can also use verification graphs. Add `--decode-graphs`
and set `STRATA_GLM_VERIFY_GRAPHS=1` with
`STRATA_GLM_VERIFY_GRAPH_MAX_TOKENS=8` to cover complete DFlash2 windows. The
maximum defaults to four rows. This is separate from the draft worker's
`--cuda-graph` option.

`STRATA_GLM_DFLASH_GPU_TOP16=1` optionally selects the sixteen finite, allowed
head candidates per slot on the GPU. This reduces the head readback from
4,336,640 bytes to 896 bytes per neural draft. Ties retain the lowest token ID,
and selected logits retain their original FP32 bits. The default CPU selection
remains available for exact candidate and generated-output comparisons.

Download the checkpoint revision above to a local directory.
Create the worker configuration inside the ignored build directory, with paths
appropriate to the machine:

```json
{
  "argv": [
    "python3", "tools/glm_dflash2_worker.py",
    "--model", "models/glm-flash-dflash2",
    "--device", "cuda:1",
    "--gemm-f32", "build-dflash2-local/strata_dflash2_f32.so"
  ],
  "timeout_ms": 1000
}
```

For a resident benchmark with frozen prompt IDs:

```bash
python tools/glm_v100_bench.py MODEL_DIR PROMPT_IDS \
  --prompt-ids --decoder build/strata-glm-decode --fp4-kernel sm70 \
  --context 12288 --prompt-tokens 8192 --generated-tokens 128 \
  --prefill-batch 8192 --threads 32 --gpu-devices 0,1 \
  --gpu-budget-mib 32768 --speculative dflash2 --draft-depth 7 \
  --dflash2-worker-config build-dflash2-local/worker.json \
  --full-warmups 1 --repetitions 1 --output build-dflash2-local/result.json
```

The one-shot decoder accepts `--speculative=dflash2` and
`--dflash2-worker-config=build-dflash2-local/worker.json`. Existing modes and presets do not
start the worker. Do not combine DFlash2 with MTP, split verification, deferred
experts, independent sequence batching, or the GGUF expert cache.

## Feature and verification contract

The engine captures the mean of the four mHC residual streams after layers
5, 14, 24, 33 and 42, before the final output normalization. These are the
checkpoint's five trained target features. The worker projects concatenated
features once, caches each draft layer's context K/V, and retains at most the
trained 2,048-row sliding window. Only committed target rows are appended;
proposals never enter this context cache. A new request resets the window.

Each draft uses the target's raw embedding for its anchor and seven mask slots.
The five draft layers implement bidirectional sliding attention, Q/K RMSNorm,
rotary positions, SwiGLU, and the checkpoint's two-tap dynamic convolutions.
The engine applies its existing compressed output head to the seven draft
hidden rows and returns sixteen candidates per slot. The checkpoint's
predecessor/successor selector walks a coherent path through those candidates.
No second full embedding table or output head is resident in the worker.

Two proposal-only ablations are available on the worker: `--round-bf16` emulates
BF16 activation rounding using FP32 storage, and `--selector-scale` adjusts the
transition scores (default 1, range 0–2, zero tests unary candidates). These
change draft proposals, while target verification and its configured acceptance
policy remain responsible for emitted tokens. They are not selected presets.

The target verifies anchor plus proposals in a window of at most eight rows.
Only the matching greedy prefix is accepted; target state is restored to its
committed length after a rejection. This acceptance rule does not use the
MTP soft-acceptance margin. A separate opt-in
`STRATA_GLM_DFLASH_ACCEPT_MARGIN` accepts 0–3 logit regret and defaults to zero.
Its bounded mode requires a top-eight candidate, protects all three real EOS
IDs, and blocks repeated four-grams in the last 128 generated tokens.
`STRATA_GLM_DFLASH_ACCEPT_MAX_SOFT` permits 1–7 soft accepts per window and
defaults to three; this limit is relevant only with a positive margin.
`STRATA_GLM_DFLASH_LOOKUP_DEPTH=1..7` optionally copies a matching prompt/output
suffix before using the neural drafter (default 0, off).
`STRATA_GLM_DFLASH_LOOKUP_MIN_MATCH=3..32` controls the required suffix length
(default 24). Copied proposals require exact target-greedy acceptance, even
when the separate neural acceptance margin is positive. Verification history
covers the larger window, and a long sequence of copied windows re-primes the
worker from committed features before its next neural proposal.
Batched target arithmetic and any enabled route
dropping still apply, so this experimental FP4 mode has no claim of matching
an unmodified BF16 target or sequential greedy output.

The bounded local protocol uses JSON control headers and little-endian FP32
feature/embedding/hidden frames. Invalid shapes, nonfinite values, missing
selector responses and timeouts fail the request; they do not silently switch
to another draft algorithm. The worker starts before target VRAM admission.
The worker materializes full-context workspaces before target cache admission.
Target feature history uses 80 KiB of host memory per input/verified position;
the device capture buffer holds at most 2,048 rows. The initial 4 GiB handshake bound covers its PyTorch reservation at startup;
it is not a measurement of total CUDA process memory or a runtime allocation cap.

## Validation

CPU tests cover independent scalar convolution, selector predecessor dependence,
float RMS reductions, incremental versus recomputed context K/V, rejected draft
cache isolation, request resets, empty deltas and truncated frames. Unix-socket
tests cover coalesced binary frames and preserve the existing protocol-1 tests.
The native decoder builds with CUDA SM70 and HIP; this DFlash2 worker is CUDA only.
The FP16-only real-target smoke run failed with activation overflow. A synthetic
real-weight diagnostic found a first-layer down-projection output of about
330,763, beyond FP16 range; disabling reduced-precision accumulation did not
solve it. Keep residuals and GEMM outputs in FP32. The revised small FP32 CPU
backbone agrees with the upstream implementation within 2.4e-7 absolute error.
These are numerical checks, not model-quality or speed qualifications.
The real-weight graph probes matched eager outputs exactly for the tested
initial window, three-row update, short reset and full reset at position 8,192.
This is not a full real-weight BF16 activation or target-feature parity check.
The standalone GPU test covers seven GEMM shapes, values above FP16 output
range, a nondefault stream and graph cache/position resets:

```bash
python tools/test_glm_dflash2_cuda.py \
  --library build-dflash2-local/strata_dflash2_f32.so \
  --skinny-library build-dflash2-skinny-local/strata_dflash2_skinny.so --graph
```

On V100, `glm_top16_test` passed 240 comparisons against an independent CPU stable sort,
including masked IDs, ties, signed zero, subnormals, nonfinite values, partial
blocks and buffer guards. Three graph replays with changed inputs also passed;
32-case sanitizer runs reported zero memory errors and zero race hazards. The
selector averaged 0.128 ms over twenty GPU-only iterations after two warmups.
These are kernel measurements, not model throughput. Build this target with the
CUDA or HIP backend.

The first CPU/GPU selector comparison produced identical 512 token IDs, but the
GPU timing had four draft-worker major faults and failed qualification. It is
retained as output-parity evidence. Its apparent 29.10 versus 27.82 tok/s supplies
no qualified speedup claim. Repeated trials add bounded target and child library
preparation during a discarded warmup; residency gates remain enabled.

Numerical tests, compute-sanitizer memcheck and racecheck passed on V100.

The wider target-graph checker passed 64 capture checks on a real 128-input
request: every width from one through eight, with and without retained history,
three warm/capture/replay passes and two history allocations. Every logit and
the next logits after retained-prefix rollback matched direct launches bit for
bit. The default path also preserved all 128 frozen reference IDs with zero
major faults. These checks establish output behavior for this workload; they
do not measure decode throughput.

Three instrumented 128-input / 24-output MTP depth-seven requests exercised
eight-row target graphs with zero reported memory errors. Their cold outputs
differed, so the benchmark rejected repeat qualification and their timings are
excluded. An instrumented DFlash2 attempt lost the worker connection before
timed requests and supplied no complete memory result. Subsequent stages wait
for earlier GPU compute contexts to retire before starting.

## Measurements and remaining gap

[The measurement record](glm_fp4_v100_dflash2_measurement.json) preserves source,
binary, configuration and output hashes, individual requests, failures and phase
timings. The host RAM clock was not measured and GPU clocks were unlocked.
All model timings below use the same frozen 8,192 input IDs. Full trials use two
complete warmups, a discarded two-output request and three 512-output requests;
decode throughput counts 511 advances. Initialization and shutdown are excluded.
The RAM loads used identical payloads: 42 indexed files totaling
177,463,841,024 bytes, verified by their per-file SHA-256 manifests.
The trials use the environment from the
[KDA/MTP preset](../configs/experimental/glm53f-fp4-dual-v100-kda.json), with
DFlash2 acceptance set separately as shown below. Unless explicitly labeled,
decode route dropping remains bounded at 0.5 with at most seven routes dropped
and retained weights renormalized; prefill uses all routes.

| Mode | Decode tok/s | Prefill tok/s | Reused coding checks | Scope |
| --- | ---: | ---: | ---: | --- |
| MTP depth 3, early control | 32.44 | 519.98 | 8/8 | Full trial, median of three |
| DFlash2 depth 3, route-drop cap 0.5 | 26.84 | 514.77 | 8/8 | Full trial, median of three |
| MTP depth 3, packed/graph comparison control | 32.08 | 520.94 | 8/8 | Full trial, median of three |
| DFlash2 packed/graph depth 3, cap 0.75, exact suffix lookup | 38.56 | 515.68 | 0/8 | Rejected; full trial, median of three |
| MTP depth 3, wider-window comparison control | 32.65 | 520.58 | Same-build 8/8 panel reused | Full trial, median of three |
| DFlash2 packed/graph depth 7, cap 0.75, exact suffix lookup | 35.99 | 512.46 | 1/8 | Rejected; full trial, median of three |
| DFlash2 packed/graph depth 5, cap 0.75, exact suffix lookup | 34.47 | 514.65 | 0/8 | Rejected; full trial, median of three |
| DFlash2 depth 7, greedy acceptance | 12.81 | 503.10 | Unscored | One 128-output screen; worker residency not separately sampled |
| DFlash2 depth 7, margin 3, at most seven soft accepts | 21.61 | 503.26 | Unscored | One 128-output screen; target and worker residency checked |

The repeated trials exited successfully, had no timed major faults or observed
swap, and emitted no real EOS IDs in the fixed-length timings. Neural bounded
acceptance uses margin three and at most the configured draft depth in soft
accepts; suffix proposals require exact greedy acceptance. The coding panel uses
roughly 4,096 input tokens, real EOS stopping and a 1,536-output cap. Failures
include incomplete answers and missing compilable code. The fast depth-three
trial repeated 40.86% of its output four-grams, versus 8.25% for its MTP control.
That performance prompt supplies no evidence of useful answer throughput.
The small coding checks do not establish general quality equivalence.
These higher-cap policies failed selection before fresh testing, so the fresh twenty
holdouts remain unscored.

The paired 128-output screens below use two complete warmups and one timed
request. All four emitted identical 128 IDs. They measure the same bounded
depth-three policy and are not 512-output qualifications.

| Draft backend | Decode tok/s |
| --- | ---: |
| cuBLAS, eager | 29.96 |
| Packed FP16, eager | 30.19 |
| cuBLAS, CUDA graph | 31.55 |
| Packed FP16, CUDA graph | 31.60 |

On a separate real-weight probe with synthetic features, graph replay reduced
the median draft-plus-selector time from 12.15 to 9.11 ms with cuBLAS, and from
11.61 to 8.10 ms with packed half. Those probes exclude the target head,
verification and transport. They cannot supply a model throughput estimate.

Several independent changes were tested: FP32 GEMM outputs to fix overflow,
merged projections and grouped attention, software BF16 rounding, selector
transition scaling, shorter verification windows, bounded acceptance, and an
optional packed-half SM70 kernel. Software rounding did not improve strict
acceptance. Some selector screens faulted pages in the draft worker during timing
and are retained as failed qualifications. A short full-route diagnostic accepted
18 of 73 proposals, but faulted pages and cannot establish a speed result or
attribute the acceptance gap to route dropping alone.

Native cache-affinity screens add a selection bonus to GPU-resident experts,
while retaining their original sigmoid weights. The bonus applies only during
target decode; prefill and MTP draft routing keep their existing behavior.
`STRATA_GLM_FP4_ROUTE_AFFINITY` defaults to zero. The following same-build screens
used two full warmups and one timed 512-output request, with no timed major faults,
swap or EOS IDs. They are screening observations; candidate quality and repeated
timing qualification are still pending.

| Mode and affinity bonus | Decode tok/s | CPU expert groups | Verify ms per round |
| --- | ---: | ---: | ---: |
| MTP, zero bonus control | 28.76 | 27,590 | 107.58 |
| MTP, 0.025 | 31.95 | 20,455 | 105.23 |
| MTP, 0.10 | 26.91 | 14,229 | 110.10 |
| MTP, 0.15 | 34.14 | 9,700 | 98.36 |
| DFlash2 depth 7, 0.05 | 28.35 | 25,840 | 125.41 |

Reducing CPU groups does not remove the verification bottleneck. The 0.10 MTP
screen halved those groups but took longer per verification round. Changes in
GPU work, routing and accepted proposals must be measured together. The control's
output matched the frozen 512 IDs; its timed request created 333 CPU scale-cache
entries. The last two reference requests created none. These single timings do
not establish a gain over the repeated reference. On the later same-build pair,
secondary-only bonuses of 0.025, 0.05 and 0.10 measured 28.80, 28.95 and 29.65
tok/s against a 28.74 control. Its 0.15 trial had six major faults and failed
timing qualification. Neither experiment changes a selected preset.

The simulator keeps the selected 32.10 / 520.89 measurements and resource
capacities unchanged. It adds measured speculative phases as separate scenarios:

- In the strict depth-seven screen, target verification took 8,337.53 ms of
  9,912.09 ms and only 62 of 435 proposals were accepted. Removing all other work
  while keeping that proposal stream would yield at most 15.23 tok/s.
- In the median full depth-three DFlash2 request, removing all nonverification
  work gives 34.36 tok/s. Reaching 50 with its other work fixed needs verification
  to run 2.46 times faster. This is a phase-deletion calculation, not a hardware
  roof or an estimate for a different acceptance stream.
- An explicitly conditional perfect-acceptance calculation for the strict screen
  gives 48.97 tok/s if cost per verified row and draft call stay fixed. Acceptance,
  routing and window cost change together, so this is neither a forecast nor a
  speed qualification.
- In the 0.15 affinity screen, deleting all nonverification work would yield
  39.06 tok/s for that proposal stream. Reaching 50 with other work fixed requires
  1.57 times faster verification. Candidate quality remains unscored.

The selected MTP path still needs 1.56 times its measured decode rate for 50,
1.54 times its prefill rate for 800, and 1.92 times for 1,000. The simulator's
173.4 decode and 1,436.7 prefill tok/s streaming ceilings omit compute,
reconstruction and dependencies; they do not predict attainable throughput.
The frozen 5.26 tok/s reference remains unchanged.
