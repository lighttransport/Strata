# GLM GPU fault diagnostics

`tools/glm_guarded_run.py` runs one GLM benchmark in its own UUID-named user
service. It limits RAM to 60 GiB with no swap and records the boot ID, executable,
config and prompt hashes. It stops on a new kernel fault, a lost PCI identity or
a timeout, and cleans up only the service and launcher it created.

Each run writes `<output>.tasks.jsonl` with UTC and local timestamps and task states:
preflight, model cache cleared, launch, running, passed or failed. Entries are
flushed and synced to disk. Five-second running entries record raw log sizes,
GPU busy counters, VRAM use, power, temperature and clock readings where the
kernel exposes them. Sensor values retain their sysfs units: bytes for memory,
microwatts for power, millidegrees Celsius for temperature and Hz for clocks.
Raw logs and token/evaluation output are also synced at those checkpoints; their
last lines are included in the journal. Missing counters are omitted.

For more detail, set these opt-in environment variables in the benchmark config:

```json
{
  "STRATA_GLM_TASK_TRACE": "/tmp/glm-prefill.engine.jsonl",
  "STRATA_GLM_TASK_TRACE_SYNC": "1"
}
```

The engine trace appends and syncs each marker. It records Unix milliseconds,
context position, token count, layer and phase. Main-program markers cover model
loading, GPU setup, cache preparation and the streamed-evaluation and graph
checks. Batch markers cover prefill and decode. GPU prefill also records each
layer's start, mixer completion and FFN completion.

With `STRATA_GLM_TASK_TRACE_SYNC=1`, the prefill mixer and FFN markers follow
successful completion of the primary GPU stream. Without it, they mark host
enqueue progress; the batch GPU-completion marker still follows synchronization.
Copy workers and other streams can remain active, so these boundaries narrow a
failure region rather than identifying one kernel. Decode does not add phase
synchronization because its GPU mailbox can depend on concurrent CPU work.

Tracing and disk sync change timing and scheduling. Use traced runs for fault
diagnosis, and separate runs without engine tracing for performance measurements.
The guard's task journal stays enabled in both. With engine tracing unset, the
engine opens no trace file and adds no GPU synchronization.

Use a fresh output prefix and trace path for each configuration. After a reboot,
match the task's boot ID to the kernel journal and inspect the final persisted
marker. An absent completion or final result means interrupted, never passed.
Even synced files can lose recent writes during a device or power failure.
Do not interpret a silent power-off as proof of a GPU or RAM fault.

Read the last task and phase from a campaign directory without starting GPU work:

```sh
python3 tools/glm_diagnostic_status.py /tmp/glm-campaign
```

This distinguishes completed failures from runs left unfinished across a reboot,
and reports malformed log lines while retaining the last readable phase.

## Opt-in REAP50 staged prefetch

The original staged prefetch setup accepts 288 experts. REAP50 has 144 experts,
so it previously stopped at setup with "staged prefetch requires GGUF expert
groups and no direct upload". `STRATA_GLM_REAP_PREFETCH=1` enables that setup for
144-expert GGUF models only. It requires staged prefetch and rejects direct
upload. The new flag defaults off; the existing 288-expert path keeps its pool
size and the same transfers.

```json
{
  "STRATA_GLM_REAP_PREFETCH": "1",
  "STRATA_GLM_STAGE_PREFETCH": "1",
  "STRATA_GLM_PREFETCH_GROUPS": "9",
  "STRATA_GLM_PREFETCH_REUSE_DEVICE": "1"
}
```

The REAP pool is capped at nine groups of sixteen experts. Each transfer checks
the group and pool index against the tensor geometry before reading source
rows. Copy and completion events retain the existing protection for the two
shared pinned buffers and reused GPU slots. This changes upload scheduling,
not weight values or the GPU product kernels.

On 2026-10-11, X570S AERO G / Ryzen 3950X ECO / 64 GiB DDR4-2666 / RX 9070 XT
16 GB / ROCm 10.1, a traced 2K-input, 32-output-token BF16 WMMA diagnostic
passed. Output tokens and all saved final logits matched the non-prefetched
control byte-for-byte. The pool occupied 1,134.14 MiB with no extra host buffers.
HIP and CUDA builds passed; CUDA execution was not tested. Long-prompt timing
and coding qualification are separate checks, recorded in the
[recovery benchmark](benchmarks/glm_x570s_2666_eco_recovery_20261011.json).

## Match one-token arithmetic during speculative verification

Two existing one-token optimizations use different rounding from their batched
fallbacks: `STRATA_GLM_HC_FUSED=1` applies the norm scale after mHC projection,
and `STRATA_GLM_HEADS_BLOCKS=1` changes the Q8 head reduction order. With both
enabled, the ROCm 10.1 full-model verification check failed against sequential
decode. This was a software mismatch, with no recorded hardware fault.

These opt-in counterparts reuse the existing one-token calls for every small
batch row, without changing their kernel arithmetic:

```json
{
  "STRATA_GLM_HC_FUSED_BATCH": "1",
  "STRATA_GLM_HEADS_BLOCKS_BATCH": "1"
}
```

The head dispatcher checks that token strides do not overlap. Its production
query/value tests matched individual blocked-head calls bit-for-bit at widths
2, 3, 4, 5 and 8, including padded output guards. On the hardware above, the
combined options passed eight-row sequential-logit equality, retained-prefix
and next-token rollback checks, and a 64-token lookup output comparison with
an accepted proposal. HIP execution passed; CUDA is compile-only.

Both counterparts default off. Ordinary one-token calls keep their existing
arithmetic. Internal lookup and MTP reject an enabled one-token optimization
without its matching batch counterpart. External drafting takes the existing
sequential verification fallback in that case. Other batching requirements,
including canonical combining and routing/deferral constraints, still apply.

`STRATA_GLM_CHECK_VERIFY_WIDTH=1..8` selects a smaller window for the opt-in
verification diagnostic; its default remains eight. This changes the check,
not inference settings, and does not weaken equality or rollback requirements.

## Prefill repeatability and matched MTP controls

`STRATA_GLM_CHECK_PREFILL_REPEAT=1` compares each timed prefill's full final-token
logits with the first prefill, including the untimed warmup when present. It
stops on any difference and prints the trial and maximum absolute difference.
With engine tracing enabled, it also persists `prefill_repeat_mismatch`.
The flag defaults off and adds no GPU work; it compares logits already copied
for generation. This is a diagnostic check, not a relaxed quality threshold.

`--mtp-greedy-control` prepares the MTP draft and rollback history, then generates
with ordinary greedy target decoding. It requires `--speculative=mtp` and cannot
be combined with lookup, a decode-mode sweep or corpus evaluation. The low-memory
benchmark config exposes it as `mtp_greedy_control: true` with `speculative:
"none"`. Use the same draft depth, target settings and cache fingerprint for
both arms of a performance comparison.

In the October 11–12 recovery campaign, one depth-1 pair and its repeat matched
384 output tokens and measured about 13.5% faster than the prepared greedy
control. Other runs selected a different cache and changed output. Three fresh
prepared greedy repeats reproduced a different prefill distribution (about
9.6% relative L2 from the matched warm control), including at depth 1. Thus the
initial discrepancy can arise before speculative verification. Its cause is
unresolved; the MTP timings remain provisional and do not qualify a default or
a coding recommendation. No hardware fault was recorded in these runs.

A later traced repeatability check compared the untimed first prefill and two
timed repeats and passed bitwise equality. Thus repetition alone has not been
established as the cause of the cross-process difference. Preserve both kinds
of evidence; a passing repeat check does not resolve the earlier discrepancy.
