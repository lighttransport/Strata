# GLM DFlash2 on V100 (experimental)

The opt-in `dflash2` decoder mode uses the published
[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
drafter. It is separate from the qualified MTP preset, which measured 32.10
decode / 520.89 prefill tok/s on two V100 PCIe 32 GB GPUs and a dual Xeon Gold
6240 host with 181 GiB RAM. A later experimental MTP configuration measured
40.26 decode / 519.34 prefill tok/s and passed eight reused long and eight
reused natural-length coding checks; Fresh20 remain unscored.
A grouped target core later measured 41.24 decode / 519.34 prefill tok/s and
passed the same reused panels in separate functional validation. It missed
the predeclared speed-selection threshold, so the reference remains unchanged.
The fastest repeated DFlash2 trial measured
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

The optional [bridge-calibration tools](GLM_DFLASH2_CALIBRATION.md) export checked
head operands and train a small, separate FC adapter. Calibration loss alone
does not establish a speed or quality gain; no calibrated preset is selected.
The first rank 8 adapter raised agreement from 28/56 to 31/56 labels on
synthetic evaluation windows. Its depth-three screen measured 30.35 / 32.02 decode tok/s for
base / calibrated workers after three warmups and one timed 512-output request
per side. Both passed residency checks; repeated gain and coding quality
remain unqualified. The selected preset stays unchanged.
The corresponding depth-seven pair measured 23.95 / 24.97 tok/s. The base
depth-seven request accepted 364 draft tokens versus 343 at depth three, while
verification time rose from 14.19 to 18.92 seconds. The longer block reduced
rounds from 168 to 147 but increased total decode time.

A later rank 32 FC adapter measured 30.06 / 33.39 decode tok/s for base /
calibrated workers on the same hardware, after three full warmups and one timed
512-output request per side. Both passed residency checks. This 11.05% screen
gain has no repeated timing or coding qualification.
For that measured proposal stream, removing all work outside verification
would reach only 39.53 tok/s. Reaching 50 tok/s with the other phases held fixed
requires a 1.65× verification speedup. These are conditional phase calculations,
not forecasts for a changed drafter or routing policy.

The combined CPU-scale/KDA-projection candidate still failed residency checks
after executable and library pages were pinned during discarded warmup.
Its raw median was 37.59 tok/s, with timed target major faults `[0, 39, 1]`;
the control measured 36.04 tok/s with none. The candidate is unqualified.
These counters do not identify the faulting mapping. Selected presets and
the original 5.26 tok/s reference remain unchanged.

A separate externally sampled confidence-depth case captured 466 user major
faults across 40 target tasks, with no lost samples: 464 addresses mapped to the
decoder executable and two to cuBLAS. Most occurred during discarded warmups;
the case's timed request recorded zero major faults. Its instrumented timing
is excluded. These samples do not identify the backing file of faults in other
cases. An optional RAM executable is being tested as a mitigation.

A subsequent matched pair used the RAM executable on both sides, with three
full warmups and three timed 512-output requests per side. Enabling one-token
CPU scale decoding and batched KDA Q/K/V projections measured 36.11 / 37.46
decode tok/s for control / candidate, a 3.74% gain. Prefill measured 520.63 /
520.39 tok/s. All six requests had zero timed target major faults and matched
the earlier mild-policy 512-token reference. Both sides also passed eight
reused long and eight reused natural-length coding checks. Fresh20 remain
unscored and the selected preset stays unchanged. This comparison measures the
two kernel options with RAM execution enabled; it does not isolate a speedup
from RAM execution itself.
For the candidate's median request, removing every other phase would reach
43.85 tok/s. With those phases held fixed, 50 tok/s still requires a 1.42×
verification speedup. This calculation keeps the measured proposal stream.

Earlier route readback measured **37.78 / 40.26 decode tok/s** for control /
candidate on the same hardware, a 6.54% gain. Both used one-token CPU scale
decoding, batched KDA projections, RAM execution and the same mild MTP policy,
with three full warmups and three timed 512-output requests per side. Prefill
measured 519.18 / 519.34 tok/s. All six streams matched the mild reference,
with zero timed major faults and swap; both sides passed eight reused long and
eight reused natural-length coding checks. Fresh20 remain unscored.

`STRATA_GLM_FP4_EARLY_READBACK=1` copies narrow routed inputs and coefficients
before the shared expert FFN. CPU experts wait for that copy's event while the
shared FFN runs; the final sum remains after the shared FFN on its original
stream. The option is off by default and changes no arithmetic or route policy.
CUDA and HIP builds and six host checks per backend passed; native FP4 GPU
execution was tested on CUDA only. With the option off and on, both original
128-token output checks were identical. The opt-in
[`early-readback profile`](../configs/experimental/glm53f-fp4-dual-v100-early-readback.json)
records the measured Linux settings. Selected presets remain unchanged.

Both modes used 134 verification rounds and 20,455 CPU expert groups per timed
request. For the candidate's median request, removing all nonverification work
would reach only 47.23 tok/s; reaching 50 with the other phases fixed requires
1.30 times faster verification. Its measured gaps are 1.24 times for 50 decode,
1.54 times for 800 prefill and 1.93 times for 1,000 prefill. These are conditional
phase calculations and measured gaps, with no fitted hardware capacity.

A larger graph prototype captured the native KDA/router prefix and shared FFN,
keeping the readback event outside capture. Mixed widths 1–4, history
reallocation and retained-prefix rollback passed bitwise logit checks. The
matched model pair measured 39.85 / 39.38 decode and 519.62 / 518.81 prefill
tok/s, with three full warmups and three timed 512-output requests per side.
All six output streams matched the mild reference, with zero timed faults or
swap. The 1.19% regression failed the speed gate; the prototype remains private
and its coding panels were skipped.

A separate KDA component probe compared the existing serial column layouts
with ordered row partitions. All 144 state/output/history checks matched
bitwise. For 64 heads and four tokens without snapshots, serial-128 and
ordered-four medians were 75.62 and 31.52 microseconds. The ordered kernel in
that build did not support rollback snapshots. These grouped CUDA-event
samples include host enqueue gaps and supply no full-model speed claim.

Ordered KDA row partitions now save speculative rollback snapshots directly
from the recurrence. The opt-in combination is `STRATA_GLM_KDA_ROW_PARTS=4`
and `STRATA_GLM_KDA_ORDERED_ROWS=1`; the default remains one row partition.
This supports full rollback history. Compact replay history still requires
the serial kernel.

On the same V100, 564 bitwise checks covered complete state, outputs, every
retained snapshot, padding, extreme inputs and rollback, including 24 graph
replays. CUDA memcheck and racecheck were clean. The initial snapshot build passed
mixed widths 1–4 with and without history, history reallocation and rollback
logit checks; its original 128-token default output also matched. CUDA and HIP
builds and six host checks per backend passed; HIP GPU execution was not tested.
With four tokens, 64 heads and rollback snapshots, the separate component
medians in the cleaned build were 116.93 microseconds for serial-128 and
46.34 for ordered-four. All 180 component comparisons matched bitwise. These grouped event samples
include host enqueue gaps. The cleaned build repeated the model parity checks.
Compiler resource reporting found 255 registers and nonzero spills for the
serial kernel, versus 95 registers and no spills for ordered-four; these are
static compiler properties, not measured runtime traffic.

The matched model pair measured **40.51 / 40.48 decode** and **520.45 / 518.71
prefill tok/s**, after three full warmups and three timed 512-output requests
per side. All six streams matched the mild reference, with zero timed major
faults or swap. The 0.07% lower decode median supplies no model gain; coding
panels were skipped and selected profiles remain unchanged. The component
speedup does not predict full-model throughput.

A separate private FP4 kernel prototype omitted output reductions for unused
rows. It passed 96 bitwise cases, 36 graph replays and clean CUDA memcheck and
racecheck. With identical weight/input/output storage and five alternating
128-call samples per variant, its geometric mean speed ratio was 0.999901
across eight expert-shape/width cases. It failed the 3% component gate; no model
timing or coding checks were started, and the original FP4 kernel is unchanged.

Two further private prototypes kept the expert order and arithmetic intact.
Uploading one pinned route-index table per GPU/layer measured **39.60 / 40.12
decode** and **518.96 / 519.27 prefill tok/s**, with three full warmups and three
timed 512-output requests per side. All six streams matched the mild reference,
with zero timed major faults or swap. Its 1.32% decode gain fell below the 3%
selection gate and the 40.26 reference, so coding checks were skipped.

A shared-memory address swizzle passed 96 bitwise cases, 36 graph replays,
clean CUDA memcheck/racecheck, the original 128-output check and real-model
width/history/rollback checks. Its component geometric mean speed ratio was
0.999586 across eight expert-shape/width cases, using the same allocations and
five alternating 128-call samples. It failed the component gate; no model
throughput or coding trial was selected. The scalar address-bank analysis was
not a measurement of GPU memory transactions. Neither prototype changed a
selected profile or a simulator capacity.

Resident GPU expert grouping showed a larger component improvement. The first
private version passed bitwise and sanitizer checks but had a 0.925120 speed
ratio. A revision using descriptor references and the original runtime loop
dimensions reached **1.535068×** across twelve group/width cases. Both versions
passed 44 bitwise cases, 44 graph checks and clean CUDA memcheck/racecheck.
The compiler reported an 80-byte stack frame for the initial grouped gate/up
kernel and none for the revision; this is a static property, not measured
runtime traffic or proof of the cause of the speed difference.

The revised MTP model pair measured **39.96 / 41.33 decode** and **518.28 / 518.17
prefill tok/s**, with three full warmups and three timed 512-output requests per
side. All six streams matched the mild reference, with zero timed major faults
or swap. Its 3.43% paired decode gain missed the predeclared threshold of 3%
above the historical 40.26 reference, so coding panels were skipped. The
timing did not select a preset. Default 128-output and real-model
width/history/rollback checks passed,
as did CUDA and fresh HIP builds with six host tests each. HIP GPU execution
was not tested. An initial launcher budget typo failed before model startup;
the corrected launcher used the same binary, and the raw failure is retained.

Coalescing the reconstructed FP16 operand stores reached a **1.665286×**
component speed ratio on the two expert matrix shapes. It passed 35 bitwise
cases, 14 graph checks, CUDA memcheck/racecheck and both original 128-output
model checks. The cold default-off check had 30 major faults and supplies no
throughput measurement; the counters do not identify their backing file.
The matched MTP model pair measured **519.19 / 520.63 prefill tok/s** and
**39.53 / 39.82 decode tok/s**, after three full warmups and three timed
512-output requests per side. All six streams matched the mild reference,
with zero timed major faults or swap. The 0.28% prefill gain failed the gate;
coding was skipped. HIP rebuilt with six host checks and an identical binary
to the preceding HIP build, with no HIP GPU execution. The prototype remains
private; the component improvement does not establish material model speed.

Counting all suitable GPU arena holes for decode cache admission, while keeping
the scratch reserves, measured **39.68 / 37.59 MTP decode** and **519.55 / 495.80
prefill tok/s**. Each side used three full warmups and three timed 512-output
requests, with zero timed major faults or swap. The candidate outputs changed
under the existing affinity/drop policy, but were repeatable and passed length,
EOS and repetition checks. CPU expert groups fell from 20,455 to 20,138 per
request, while speculative rounds rose from 134 to 137. These counters do not
establish the cause of the slowdown. Coding was skipped after the 5.28% decode
regression; the option stays disabled in the experimental profile.

A follow-up kernel folds the grouped experts' scatter calls into one
launch, preserving each token's expert order and separate FP32 multiply/add
rounding. It passed 88 bitwise cases and 88 graph checks, including nonzero
initial sums, with clean CUDA memcheck/racecheck. With grouped products active
in both variants, its component geometric mean speed ratio was **1.053437×**
across twelve group/width cases. The original 128-output and real-model
width/history/rollback checks passed on both GPUs. CUDA and HIP builds and six
host checks per backend passed; HIP GPU execution was not tested.

The subsequent matched MTP pair measured **39.85 / 41.24 decode** and
**519.94 / 519.34 prefill tok/s**, with three full warmups and three timed
512-output requests per side. All six streams matched the mild reference,
with zero timed major faults or swap. The 3.50% paired gain missed the
predeclared absolute threshold of 3% above 40.26, so its conditional coding
panels were skipped. An earlier wrapper failed before either model started
because an idle-check output filename already existed; the corrected wrapper
used the same frozen binary and settings. Both records are retained.
For the candidate's median request, removing all nonverification work would
reach 48.37 tok/s. Reaching 50 with other phases fixed requires 1.26 times
faster verification. These are conditional calculations for the measured
proposal stream, not hardware rooflines or forecasts.

On that same grouped core, increasing MTP depth and the soft acceptance cap
from three to four measured **41.79 / 40.47 decode** and **519.59 / 520.57
prefill tok/s**. Both sides used three full warmups and three timed 512-output
requests, with zero timed major faults or swap. The candidate outputs changed
but were repeatable and passed length, EOS and repetition checks. The 3.14%
decode regression failed the speed gate; coding was skipped. The depth-three
control's 41.79 timing and the separate pair's 41.24 timing use the same core
configuration; neither establishes a new recommended preset. These are MTP
results. Separate functional validation subsequently passed all eight long and
eight natural-length coding checks on both sides of the 41.24 pair. This does
not change the original failed speed-selection gate or select a new preset.
The experimental reference remains 40.26 tok/s, and Fresh20 remain unscored.

A compact CPU scale lookup prototype passed 1,920 bitwise checks on native
AVX512 VNNI, with the same count on each AVX2 and scalar fallback. With 32
workers, scalar lookup reached a **1.052376×** component speed ratio across
the gate/up and down matrix shapes. Each shape used five cyclic timing samples
and sixteen rotating 4 MiB code buffers, with shared activation storage across
variants. Single-worker ratios were 1.024051× and 1.026193×. A four-lane gather
variant regressed to a 0.531222× geometric mean ratio at 32 workers and was
rejected. These are CPU product timings, not model throughput. The full-model
scalar experiment includes the original zero-scale sentinel and charges its
padded tables against the existing CPU cache budget.

## Execution

The native CUDA executor has three optional narrow decode switches:

- `STRATA_GLM_FP4_ROUTE_ID_BATCH=1` uploads a layer's route IDs together.
- `STRATA_GLM_FP4_RESIDENT_GROUPED=1` groups resident NVFP4 expert products.
- `STRATA_GLM_FP4_GROUPED_SCATTER=1` combines their ordered scatter calls.

All are off by default. Grouping requires the SM70 kernel, device routing,
GPU reduction and peer access, with more than one resident expert and at most
eight input rows. Other cases use the existing path. It allocates less than
3 MiB of device scratch and 5,376 bytes of pinned metadata per participating
GPU; separate route batching uses another 256 pinned bytes. It preserves the
original MMA chains, K partitions and FP32 expert addition order. These switches
do not change a selected profile. The component and model measurements above
used an earlier private version; the source also drains pending copies on
exceptions when grouping is enabled without separate route batching.
The current source passed CUDA and HIP builds and six host tests per backend.
Its CUDA kernel passed 88 bitwise cases, 88 graph checks and clean memcheck and
racecheck. The disabled default matched all 128 frozen output IDs. Real-model
widths 1–4, both history modes, graph replay, rollback and reallocation matched
bitwise with separate route batching disabled and enabled. HIP GPU execution
was not tested; the rebuilt HIP binary was unchanged.

On Linux, `STRATA_GLM_RAM_EXECUTABLE=1` copies the decoder executable into a
sealed RAM file and executes it before engine initialization, preserving its
PID, arguments and transport descriptors. The copy is capped at 128 MiB; the
tested CUDA image was 82,545,320 bytes and used the same 181 GiB allocation.
The option is off by default. CUDA and HIP builds and six host tests per build
passed. Both the disk and RAM CUDA paths matched all 128 frozen default token
IDs. HIP model execution was not tested. These are build and output checks;
paired full-model timing is separate.

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

The optional worker flags `--warm-protocol --warm-incremental` materialize
binary export, selector work and committed-prefix append widths one through
eight before readiness, including the anchor when all proposals are accepted.
They require `--gemm-f32` and restore the committed
cache, positions and hidden state afterward. Both are off by default.
On V100, 32 CPU tests and real-weight graph-output equality checks passed.
One 19-operation socket smoke with the rank 32 FC adapter used 3,046 MiB;
incremental append-and-draft calls took 9.75–10.12 ms and every operation met its
deadline. This is worker validation, not full-model throughput or evidence
that later paging cannot recur.

`--warm-first-prepare` optionally repeats protocol and append warmup after the
first real context arrives, after target loading. It requires both warmup
flags above and uses the existing 30-second prepare allowance. Draft and
selector deadlines remain one second. The warmup restores the committed
context and leaves no draft output; short contexts retain eager execution.
All 41 CPU tests passed. Real-weight V100 checks preserved subsequent output
bits at context lengths 8 and 2,048 with an existing captured graph. A separate
19-operation socket check passed with 3,046 MiB at readiness, a 282.14 ms
prepare and 9.74–10.33 ms incremental calls. The depth-seven native control /
candidate screen measured 24.08 / 24.17 decode and 503.17 / 502.90 prefill
tok/s after three full warmups and one timed 512-output request per side.
Both passed the existing deadlines and residency checks and matched all 512
reference IDs. The 0.36% difference missed the speed gate. Since the control
also completed, this pair does not establish that intermittent paging is fixed.

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

`STRATA_GLM_DFLASH_ADAPTIVE_DEPTH=1` optionally trims uncertain proposal tails
before target verification. `STRATA_GLM_DFLASH_MIN_DEPTH` sets a minimum of one
to seven proposals (default one); `STRATA_GLM_DFLASH_DEPTH_GAP` sets a finite
logit gap from zero to 32 (default one). After that minimum, the engine stops
before the first selected token whose unary logit is less than the strongest
other candidate plus the gap. Equality retains the token. Short request tails
can contain fewer proposals than the minimum.

The seven-slot draft and its token choices stay unchanged; the policy uses
candidates already copied to the host. Each retained proposal still receives
target verification. The C11 rule lives in `glm_draft.h` and has GPU-free tests.
The option is off by default. CUDA and HIP builds and four host tests passed.
With new options off, a two-V100 run matched all 128 generated token IDs against
the frozen default reference. Its cold timing was not qualified.

With the rank 32 FC adapter, the fixed-seven and adaptive minimum-two cases
measured 22.70 and 28.93 decode tok/s after three full warmups and one timed
512-output request each. Both passed residency checks, but neither beat the
earlier calibrated fixed-three screen at 33.39 tok/s. The new fixed-three
control measured a raw 34.00 tok/s and failed qualification with 119 timed
target major faults; all 512 IDs still matched the earlier fixed-three result.
The minimum-one case was externally profiled and is excluded from qualified
timing. Minimum three failed before timing: its first incremental worker call
took 1.80 seconds and recorded 206 major faults, exceeding its one-second
deadline. No adaptive preset is selected, and repeated timing and coding
qualification remain unestablished.

The retry with RAM execution and append warmup completed cleanly. Fixed three
measured 34.31 decode / 504.80 prefill tok/s; adaptive minimum three measured
31.27 / 501.61 tok/s. Each used three full warmups and one timed 512-output
request. The fixed-three output matched the earlier 512-token reference, and
the adaptive case was 8.87% slower than its matched control. It remains
unselected.

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

`--selector-lookahead D` optionally scores whole candidate paths using the
learned transition codebooks and discounted future scores. `D` ranges from
zero to one; zero is the default and keeps the existing sequential selector.
The option uses the same seven slots and sixteen head candidates per slot.

With the frozen rank 32 FC adapter, discount 0.25 raised mean correct strict
teacher-prefix length from 2.16 to 2.69 tokens across 32 independent capture
windows. First-token agreement rose from 23/32 to 24/32. Zero-discount paths
matched serving selection on every base and FC-adapter window. These are
calibration proxies. The matched depth-three native screen measured 34.79 /
31.50 decode tok/s for control / lookahead after three full warmups and one
timed 512-output request per side. Both passed residency checks; lookahead was
9.46% slower. Both depth-seven cases failed before timing on worker deadlines.
No lookahead preset is selected, and coding quality remains unscored.
All 37 CPU tests passed, along with real-weight V100 selector/graph checks and
a 19-operation socket check of the enabled option. The default `Draft` class
stays unchanged; no target or checkpoint weights are updated by this option.

A broader calibration trained seven rank eight residual adapters on 7,833
samples from 64 synthetic prompts, selecting step 896 by the independent
32-window prefix check. Mean correct prefix rose from 2.22 to 2.38 tokens.
Its depth-seven native screen measured 24.88 / 24.33 decode and
503.14 / 504.32 prefill tok/s for control / adapter, after three full warmups
and one timed 512-output request per side. Both passed residency checks; the
candidate was 2.21% slower. At depth three, the adapter measured 31.01 decode
tok/s, but the control failed residency with 179 worker major faults. No
repeated timing or coding panel qualified this calibration, and no preset changes.

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
The same 64 checks also passed with MLA graphs enabled.

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
| DFlash2 depth 7, native graphs up to four rows | 24.87 | 514.23 | Unscored | Full trial, median of three |
| DFlash2 depth 7, native graphs up to eight rows | 24.94 | 514.53 | Unscored | Full trial, median of three |
| MTP depth 3, affinity comparison control | 32.05 | 520.11 | 8/8 long and 8/8 short | Full trial, median of three |
| MTP depth 3, routing affinity 0.025 on both GPUs | 36.33 | 519.76 | 8/8 long and 8/8 short | Full trial, median of three; fresh holdout unscored |
| MTP depth 3, affinity 0.025 plus native verification graphs | 35.55 | 519.73 | Graph combination unscored | Full trial, median of three; timing IDs unchanged |
| MTP depth 7, soft-accept cap 3, affinity 0.025 and exact CPU dispatch | 23.14 | 522.33 | Unscored | One 512-output screen after two full warmups |

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

The four-row and eight-row native graph trials used the same binary, draft
worker and 0.5 route-drop policy. All six 512-ID outputs matched, with zero
timed major faults or swap. The 0.25% decode difference supplies no meaningful
gain with unlocked clocks. The four-row control used 38,392 CPU expert groups
and 140 verification rounds per timed request, with no CPU scale-cache misses;
verification took 18.23 seconds of its 20.55-second median decode request.
Adding MLA graphs also preserved the outputs and measured 24.92 tok/s, supplying
no gain. The selected MTP preset remains unchanged.

The separate MTP graph comparison measured 32.13 / 520.11 tok/s with direct
launches and 32.87 / 519.97 with four-row verification graphs. Both used three
timed requests after two warmups and preserved the frozen 512 IDs. The 2.28%
decode difference falls below the campaign's 3% selection threshold.

A 10 GiB CPU scale-cache control measured 32.13 / 520.37 tok/s. Enabling expanded
quartet scales at that limit exited during the second warmup, before any timed
request. Its first warmup retained 7.69 GB of scale metadata; peak process RSS
was 174.4 GiB. The exit cause was not captured, and the later allocation-level
memory snapshot reported zero OOM kills. This experiment supplies no throughput
result and does not change the 181 GiB allocation or the selected preset.

The opt-in `STRATA_GLM_FP4_CPU_Q8_BIAS=1` path instead changes VNNI integer
operands: unsigned FP4 integers include an offset of 12, canceled by a
precomputed signed activation-sum correction. The existing Q8 quantization,
floating-point factors and reduction order stay the same. CPU bitwise checks
passed on the Xeon's VNNI path and forced AVX2/scalar fallbacks, including widths
1–8, row tiles, borrowed and owned scales, sparse exceptions, extreme signed
bytes and output guards. CUDA and HIP decoder builds passed; HIP GPU execution
was not tested. The native GLM path is excluded from the SYCL build. With the
option off, the original 128 output IDs also matched exactly.

Its CPU-only microbenchmark used 32 pinned workers, a rotating 64 MiB code set,
16 discarded calls and 64 timed calls in each of three samples. Median
one-token gate/up product time changed from 497.50 to 501.75 microseconds and
down-product time from 573.06 to 641.86, so it failed the gate for a full-model
trial. Eight-token products improved from 999.41 to 544.90 and from 1,000.69 to
594.10 microseconds. Activation quantization and model execution are excluded;
these component results supply no model throughput claim. The option remains
off and no selected profile uses it.

The later `CPU_SCALE_LOCAL` experiment decodes at most 256 temporary bytes per
output row instead of retaining expanded tables. Its VNNI and forced
AVX2/scalar checks passed for owned, borrowed and cached scales, including
protected pages at the code-span end and four tensor-base alignments. Default
128-token output parity passed. The subsequent `CPU_Q8_POLICY` experiment
combines the exact kernels by projection shape and token width; both actual
expert dimensions passed bitwise checks at widths 1–8. Both options remain
off in the selected profiles. Neither uniform CPU variant passed its complete
microbenchmark gate for a model comparison.

CPU benchmark grouping and storage require care. The first driver woke its
32-worker pool for every matrix product; the amortized driver runs 16 products
per wake, retaining 16 discarded and 64 timed products per sample. The pool-only
wake cost was about 10 microseconds, which does not explain all the grouping
difference. Later, every variant was given identical activation and scale
storage. With that control, the apparent eight-token gain did not persist:
the per-width policy measured 448.17 versus 462.20 microseconds for gate/up and
569.92 versus 508.46 for down. One-token products measured 32.96 versus 38.42
and 36.10 versus 41.50. These are component medians, not model predictions.
The original measurements are retained with their driver hashes. One copied
metadata sentence incorrectly named kernel version 89; a separate hash-audited
correction identifies version 91 without changing the raw record.

The later depth-seven screen with a soft-accept cap of five measured 24.17
decode / 522.34 prefill tok/s. Fixed global admission modes two and three,
combined with affinity 0.025 and the CPU policy, did not supply a faster
candidate: mode three measured 28.59 / 521.61 in one timed request; mode two
failed residency qualification. These screens used two full warmups and do
not establish repeated gains. No selected setting changed.

The subsequent same-build CPU-policy pair used three full warmups and three
timed 512-output requests per variant. Control measured 36.43 decode / 519.74
prefill tok/s; the exact per-width CPU policy measured 35.45 / 520.94, a 2.69%
decode regression. All six output streams matched the mild-affinity stream.
The policy stays off.

The instrumented 128-output run recorded 2,105 ms of target CPU setup and
1,073 ms of fused CPU products. Its timed request also created 359 CPU scale
metadata entries. This is not a steady-state setup measurement. Profiling
synchronizes GPU streams, and CPU/GPU phase times overlap; these totals must
not be added or used to fit a throughput estimate.

A separate instrumented 512-output request with the CPU policy enabled,
after three full warmups, created
no new CPU metadata entries. It recorded 271 ms of CPU setup, 4,147 ms of
fused CPU products and 4,857 ms of routing/input preparation and coordinator
waiting. The latter includes host routing work and stream synchronization;
it is not an isolated copy or dense-matrix timer. CPU products and the work
before routed products remain optimization targets. These instrumented sums
do not replace the uninstrumented throughput measurements.
Of its 20,455 CPU expert groups, 17,841 used one routed token and only one
used eight; these counters include request preparation and decode.

`STRATA_GLM_FP4_EXPERT_FUSION=1` adds an experimental narrow GPU kernel. It
shares activation loads between gate and up, keeps each projection's two MMA
accumulator chains and K reduction, then applies the existing SwiGLU expression
and FP16 rounding. It replaces four launches with one for those two products
and their pointwise operations. Prefill and other matrix formats keep their
existing path. CUDA and HIP decoder builds passed locally; the native FP4
path is excluded from HIP. On both a GeForce RTX 5060 Ti and a Tesla V100,
128 bitwise output cases across widths 1–8 and 16 graph replays with changed
inputs passed. V100 memory checking reported zero errors and race checking
reported zero hazards.

The corrected V100 component benchmark used a 2048-by-4096 gate/up pair,
16 warm calls and three samples of 128 timed calls, with ordinary launches.
The paired weight and scale working set was 9 MiB.

| Routed width | Separate products and pointwise operations, µs | Fused, µs |
| --- | ---: | ---: |
| 1 | 26.09 | 20.78 |
| 4 | 33.87 | 22.76 |
| 8 | 48.43 | 29.86 |

These component gains did not improve the model run. On two V100 32GB GPUs,
the same-build pair with affinity 0.025 measured **36.88 decode / 520.21
prefill tok/s** with fusion off and **36.31 / 520.33** with it on. Each used
three full warmups and three timed 512-output requests. Both passed residency
checks and all six token streams were identical. The 1.54% decode regression
keeps fusion off; the kernel remains an opt-in experiment. No extra coding
panel was run for this slower combination. The default 128-token parity run
matched the original IDs and exited cleanly, but its 65 cold major faults
exclude its timing. The corrected test harness reused that parity and full
pair only after checking that its decoder binary was byte-identical.
The latest control still needs 1.36 times its decode rate to reach 50 tok/s,
1.54 times its prefill rate to reach 800, and 1.92 times to reach 1,000.
These are gaps from measured rates, not forecasts or revised hardware roofs.

The depth-seven MTP screen explains why the wide-product result was a poor
guide to model throughput. Only **5 of 31,014 CPU expert groups** used eight
routed tokens; 25,446 used one and 4,088 used two. These counters include
request preparation and decode. Verification took 18.96 seconds of the
22.09-second decode interval, with 393 of 821 proposals accepted over 118
rounds. For that same proposal stream, deleting all other work would give
26.95 tok/s; reaching 50 with the other work fixed requires about 2.67 times
faster verification. This single screen supplies no quality or repeated-speed
qualification. Narrow expert work and admission remain the useful targets.

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

The later repeated affinity pair measured 32.05 / 520.11 tok/s for the control
and 36.33 / 519.76 for the 0.025 candidate: 13.36% higher median decode throughput
on the same source-65 binary. Each passed all eight reused long coding checks
and all eight reused natural-length checks. The stronger 0.15 candidate passed
only seven long checks and was rejected. The mild candidate changed the timed
output IDs; this is a lossy routing policy. All three requests in each mode
repeated their respective outputs, with zero timed major faults or observed
swap. Their first timed requests created 333 and 305 CPU scale-cache entries;
the next two created none. The individual candidate timings were 32.30, 36.83
and 36.33 tok/s. These reused panels do not establish general quality, and the
fresh twenty-case holdout remains unscored. The selected preset stays unchanged.

A later calibration pair on the same two V100s used three full warmups and
three timed 512-output requests per side. The depth-three, margin-three control
measured 36.17 decode / 520.34 prefill tok/s. Depth four, margin four and a
0.60 CPU route-drop cap measured 41.29 / 521.29, a 14.15% decode gain. All timed
requests completed without major faults, swap or EOS IDs. This faster policy
passed only seven of eight reused long coding checks: one answer repeated a
paragraph until the fixed output limit. Its natural-length panel was skipped.
The control passed all eight long and eight natural-length checks. The faster
policy is rejected as a usable preset; the fresh twenty remain unscored.

A lower-depth fallback kept depth three and used margin four with the 0.60
route-drop cap. Its matched pair measured 36.39 / 39.41 decode and
520.02 / 519.50 prefill tok/s, with three full warmups and three timed
512-output requests per side. This 8.29% decode gain passed the timing gate.
The candidate then passed only two of eight reused long coding checks; its
natural-length panel was skipped. The control passed all eight long and eight
natural-length checks. This policy is rejected as a usable preset. The selected
preset and unscored fresh holdout stay unchanged.

An exact one-token CPU scale-decoding option was also tested with three full
warmups and three timed 512-output requests per side. Control and candidate
measured 36.41 / 36.84 decode and 520.38 / 521.00 prefill tok/s. All six output
streams were identical. The 1.18% decode difference missed the 3% gate, so the
option stays off and no additional coding-quality panel was queued. The CPU
component result did not translate into the same gain for the whole model.

Combining this CPU option with batched KDA projections produced raw medians of
36.74 / 38.43 decode and 520.09 / 520.03 prefill tok/s. The candidate's third
timed request had 42 target-process major faults, so the pair failed residency
qualification. Its raw median does not establish a repeated gain. The fault
counter does not identify which backing file faulted. A controlled repeat
measured 36.40 / 37.96 decode and 519.97 / 519.54 prefill tok/s, but the
candidate's third timed request again had major faults, this time seven.
Both modes completed their discarded library-page preparation. The repeated
pair also fails residency qualification; neither raw candidate median is a
qualified gain.

The named fixed-projection profile separates target prefill from cached MTP
preparation and decode. KDA query, key, value and output projections were the
largest fixed narrow GPU groups. Their timing intervals include input conversion
and host enqueue gaps and overlap broader GPU product timers. These instrumented
intervals do not replace model throughput or fit resource capacities. The
separate CPU route-drop screens at caps 0.55 and 0.60 measured 36.12 and 36.24
decode tok/s; reducing CPU groups also reduced proposal acceptance.

The opt-in KDA projection batch uses one launch for query, key and value,
retaining their existing K reduction and resident compressed weights. On V100,
the four-row component measured 221.87 / 173.02 microseconds for separate /
batched products at N=8,192 and K=4,096: sixteen warmups, three samples of
64 calls. Forty-eight bitwise cases and twelve changed-input CUDA graph replays
passed, with zero Compute Sanitizer memory errors or race hazards. CUDA and HIP
decoder builds passed; the native FP4 implementation is excluded from HIP.
The original default 128-token output remained identical. The matched full-model
pair measured 36.44 / 37.41 decode and 519.95 / 519.62 prefill tok/s, with
three full warmups and three timed 512-output requests per side. All six output
streams matched the earlier mild policy. The 2.65% decode gain missed the 3%
gate; the option stays off and no additional coding panel was queued. The
component speedup must not be applied to model throughput.

The candidate used 20,455 CPU expert groups and 134 verification rounds per
timed request, versus 27,590 groups and 146 rounds for its control. In its
median request, verification took 12,093.4 ms of 14,067.0 ms. Removing all
nonverification work for that same proposal stream would give 42.25 tok/s;
reaching 50 with the other work fixed requires 1.47 times faster verification.
The measured throughput still needs a 1.38 times decode gain and a 1.54 times
prefill gain to reach 50 / 800, or 1.92 times prefill to reach 1,000. These are
gaps and conditional phase calculations, not forecasts for a new policy.

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
