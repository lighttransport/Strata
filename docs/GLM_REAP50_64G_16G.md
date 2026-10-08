# GLM REAP-50 with 60 GiB RAM and 2 GiB free VRAM

The improved REAP-50 Q23 recipe measures **20.40 tokens/s median decode at 1K**
and **21.13 decode / 405.44 warm prefill at 8K**, on the Threadripper 1950X and
RTX 5060 Ti 16 GB. Its RAM peak is 55.36 GiB and it leaves at least 2,344 MiB
of GPU memory free. All checked outputs and final logits match the previous
REAP recipe bit for bit. It passes the warm 20 decode / 300 prefill targets
at 8K; 1K prefill remains 102.66 tokens/s. Cold throughput is lower and the
model's existing quality regression remains, so the profiles stay experimental.

[Latest measured records](glm53_flash_reap50_v10_measurement.json) contain commands,
input and executable hashes, memory peaks, trial timings and output comparisons.
[Initial v9 records](glm53_flash_reap50_64g_16g_measurement.json) retain the
earlier measurements and quality smoke.
Saved profiles: [short prompts](../configs/glm53f-reap50-q23-64g-16g-experimental-short.json)
and [long prompts](../configs/glm53f-reap50-q23-64g-16g-experimental-long.json).

The later [full/128 GB and REAP/64 GB experiments](GLM_DUAL_128G_64G.md)
measure 23.97 tokens/s REAP decode at 8K with NUMA placement, Q8 MLA/mHC and
CPU prefetch.
Those additional precision settings have passed a matched tuning comparison
and remain experimental. The v10 profiles documented here remain available.

## Machine, model and limits

Measured October 7, 2026 UTC on a Threadripper 1950X, 16 physical cores, two NUMA
nodes, 128 GB DDR4 and RTX 5060 Ti 16 GB. A Linux cgroup limits the engine and
its file cache to **60 GiB**, with swap disabled. The GPU guard limits total use
to 14 GiB and requires **at least 2,048 MiB actually free**, including desktop
use and NVIDIA's separate reservation in its accounting. Both model files are
on `/mnt/nvme01`; the full model's Q23 pack was on `/mnt/nvme02` in the preceding
comparison. These are measurements of this machine and storage arrangement.

REAP-50 has 144 routed experts per layer, against 288 in the full model, and
still selects eight per token. Its Q4_K_M GGUF is 99,330,938,048 bytes (99.33 GB).
The existing Q23 expert sidecar is **55,094,298,560 bytes, or 51.31 GiB**. Dense
weights and draft experts still come from the Q4_K_M file. The sidecar makes
most of the working set fit the RAM allowance; pruning does not halve the
eight-expert computation for each token. The earlier native Q4_K_M experiment
read more bytes per selected expert and reached about 12.5 tokens/s with MTP.

The initial v9 decoder was the same frozen executable used for the
[unbiased 64 GB experiment](GLM_64G_16G_SAFE.md), with 15 workers, a 12,288-token
context, canonical Q23 arithmetic, CPU-draft MTP depth two, split verification,
and adaptive GPU caching. GPU and RAM routing biases are zero. The requested
4,800 MiB GPU expert cache allocated **4,291.62 MiB, 494 experts**; dense GPU
weights occupied 4,806.35 MiB. Compact staging uses two 140 MiB slots.

Staged prefetch was omitted because the existing prefetch initialization only
supports the 288-expert geometry. Other prefill candidate settings, including
FP16 MLA and four-part KDA, were retained. No engine source changes were needed.

## Initial v9 throughput

Each process began with only this model's clean file-cache pages cleared. The
inventory verified zero residual cached model bytes before entering the cgroup.
Three-trial cases run an untimed prefill warmup, then three timed prefills and
three decode repetitions. The one-trial cold case has no prefill warmup.

| Workload | Prefill tokens/s | Decode tokens/s | Minimum sampled free VRAM |
| --- | ---: | ---: | ---: |
| 1,024 prompt / 256 output, warm, three trials | 102.937 / 102.800 / 102.888 | 14.855 / 16.661 / 16.841 | 2,346 MiB |
| 8,192 prompt / 128 output, warm, three trials | 410.181 / 409.851 / 409.644 | 13.123 / 18.669 / 19.235 | 2,358 MiB |
| 8,192 prompt / 128 output, cold, one trial | 109.120 | 15.766 | 2,356 MiB |

The medians are **102.89 prefill / 16.66 decode** at 1K and
**409.85 prefill / 18.67 decode** at 8K. The first decode repetition is slower
while the expert cache warms. All output IDs matched across repetitions for
each workload, with no stop IDs in the timed output. Initialization, draft
preparation, GPU cache filling and checkpoint allocation are excluded from
decode throughput.

All four runs, including the quality smoke, passed the memory guard with zero
swap, no OOM and no sustained external CPU interference. Performance-case RAM
peaks were 59.47 GiB at 1K and 60.00 GiB at 8K. Whole-process SSD reads were
57.75 GiB at 1K and 58.63 GiB at warm 8K, including loading and all repetitions.
The previous full-model cold 8K process read 172.63 GiB and decoded at 2.30
tokens/s. Warmup scopes differ; the REAP cold case provides the same 8K/128
prompt/output workload for comparison.

The 410 result is warm throughput. The cold 8K prefill takes 75.07 seconds and
does not meet 300 tokens/s. Once resident, expert work remains the main decode
cost: the final warm 8K repetition reports 4.75 seconds in CPU expert flow out
of 6.60 seconds of decode, with approximately 68.9 GB/s of accounted expert
bytes. These counters describe the current implementation, not an isolated
hardware bandwidth test.

## Output-preserving v10 changes

The selected recipe uses **MTP depth one** and releases clean CPU source pages
for dense weights that already have persistent GPU copies. Expert weights,
embeddings, routing scores and numerical kernels retain their previous behavior.
The isolated executable links the same frozen libraries as v9; its source,
archive and executable hashes are in
`build-q2-v10-reap50/build-manifest-release.json`.

Depth one proposes fewer drafts. At 8K it proposed 72 and accepted 55, against
112 proposed / 71 accepted at depth two for the same 128-token answer. The final
warm repetition accounted for 297.62 GB of CPU expert bytes, against 327.40 GB
at depth two. Verification history shrinks from 291.21 to 145.61 MiB, allowing
the GPU expert tier to grow to **4,421.94 MiB, 509 experts**. Together these
changes improved the plain depth-one median to 20.82 tokens/s at 8K and 20.42
at 1K, before releasing dense source pages.

`STRATA_GLM_RELEASE_GPU_DENSE_HOST=1` opts into the host-page release. It waits
for GPU work before advising away fully covered source pages, using the existing
GGUF page-release helper. CPU-read embeddings and routed experts are excluded.
The mapping and file contents remain valid; any later source read can fault the
same bytes back in. Main and draft source tensors totaled 5,040.35 and 104.63
MiB advised, respectively; these are source sizes, not a direct measurement of
bytes freed. Both profiles omit whole-model warming and locking. The measured
memory benefit is specific to this Linux GGUF configuration.

| Selected v10 workload | Prefill tokens/s | Decode tokens/s | Peak cgroup RAM | Minimum free VRAM |
| --- | ---: | ---: | ---: | ---: |
| 1,024 / 256, warm median of three | 102.662 | 20.397 | 54.41 GiB | 2,344 MiB |
| 8,192 / 128, warm median of three | 405.440 | 21.134 | 55.36 GiB | 2,354 MiB |
| 8,192 / 128, cold, one trial | 109.032 | 17.539 | 54.92 GiB | 2,356 MiB |

The warm 1K decode trials were 18.578 / 20.397 / 20.450 tokens/s; the 8K trials
were 17.666 / 21.134 / 21.354. Depth one with dense-page release improved the
median by 22.4% at 1K and 13.2% at 8K relative to v9. All runs had zero swap,
no OOM and no sustained external CPU interference. Releasing dense source
pages lowered the 8K RAM peak from 60.00 to 55.36 GiB. Its first decode repeat
improved from 14.715 to 17.666 tokens/s against plain depth one; warm medians
changed little. The first decode pass still misses 20 tokens/s.

A second opt-in feature, `STRATA_GLM_DECODE_CACHE_RECENT=1`, lets a fixed-size
GPU tier use `--decode-cache-window` (or the server's `decode_cache_window`)
like the automatic tier. The default window is 256 tokens; zero ranks the whole
prompt. The existing fixed tier used the whole prompt even with a window set.
The 256-token case reached 21.076 tokens/s at 8K, against plain depth one's
20.816, with exact output and final-logit equality. This small timing difference
does not establish a reliable benefit; recent-window ranking is available for
further experiments but is omitted from the selected profiles. The memory
benchmark now forwards the optional window setting to the decoder.

Across six v10 cases, **all 2,816 generated token IDs and all six final logit
arrays** (154,880 values each) matched the corresponding v9 reference exactly.
This verifies the checked workloads, including the cold case. It does not
replace full task or prefill quality qualification. Neither new engine option
is enabled by default. Ten CPU-only memory-guard tests and `git diff --check`
passed. The 35 tokens/s target remains unmet.

## Quality

Current ordinary teacher-forced decode was compared with the saved unbiased
canonical Q23 reference on 512 tuning targets, half code and half prose.

| Category | Reference perplexity | REAP-50 Q23 perplexity | Change |
| --- | ---: | ---: | ---: |
| Code | 2.98494 | 3.12177 | +4.58% |
| Prose | 10.69069 | 11.82354 | +10.60% |
| Combined | 5.64899 | 6.07539 | +7.55% |

Both categories fail the additional 1% allowance. KL was 0.5491 nats/token and
top-token agreement was 76.37%. This is a tuning smoke, not a full held-out or
HumanEval qualification. It does not qualify the prefill precision settings.

The earlier 32,768-target evaluation scored 11.115 perplexity against 8.566
for the original UD-Q2_K_XL (+29.76%), with top-token agreement 69.82%. That
used an earlier expert arithmetic path and a different reference from the
current smoke. Both results measure **REAP pruning plus Q23 conversion and
the changed dense quantization together**; they do not isolate pure REAP
pruning quality. The current pack therefore remains an explicit size/speed
tradeoff rather than a quality-qualified replacement.

## Reproduction

The updated profiles point to the isolated executable
`build-q2-v10-reap50/strata-glm-decode-release`. Its archive hashes and build
commands are recorded in `build-manifest-release.json` in that directory.

```sh
build-exl3-venv/bin/python tools/glm_low_memory_bench.py \
  configs/glm53f-reap50-q23-64g-16g-experimental-long.json \
  build-q2-v4/single-8192.ids --output build-q2-v10-reap50/reproduction-warm \
  --ram-gib 60 --gpu-capacity-mib 16384 --gpu-used-limit-mib 14336 \
  --tokens 128 --trials 3
```

Use a new output prefix. For the cold case use `--trials 1`. The guard requires
Linux user systemd and NVIDIA's `nvidia-smi`; it measures the physical 16 GB
GPU and simulates the smaller RAM allowance on the installed 128 GB host.
