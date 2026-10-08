# GLM with 32 GB RAM and 8 GB VRAM

The tested full-expert configuration missed both targets. A 128-token coding
prefix followed by 64 output tokens measured **0.30 tokens/s decode** and
**0.56 tokens/s prefill**. A shorter check with continuous total-GPU accounting
measured **0.33 decode** and **0.61 prefill**. These are single cold-cache trials,
not a release qualification or a measurement on a physical 8 GB card.

[Measured records](glm53_flash_32g_8g_measurement.json) retain the commands,
source-file sizes, prompt and executable hashes, timing scopes and memory checks.
[The experimental configuration](../configs/glm53f-q2-32g-8g-experimental.json)
keeps all routed experts, canonical Q23 arithmetic and unbiased routing. It uses
legacy prefill with four rows and a 4,096 MiB dense-weight cache. It does not meet
the requested 20 decode / 300 prefill tokens/s and is not a recommended fast profile.

## Hardware and limits

Measured October 7, 2026 UTC on a Threadripper 1950X, 16 physical cores, 128 GB
DDR4, and RTX 5060 Ti 16 GB (16,311 MiB usable). The engine used 15 workers plus
the host, context 2,048 and a Q23 expert sidecar. The original model was on
`/mnt/nvme01`, the expert pack on `/mnt/nvme02`, both SUNEAST Gen4 G70 2TB NVMe
drives. These results depend on this CPU, GPU, storage and their links.

A separate Linux cgroup enforced **28 GiB for the engine and its file cache**,
with swap disabled; the remaining nominal 4 GiB represents the OS allowance on
a 32 GB PC. Every source shard and the expert pack had zero cached bytes after
the targeted cache reset. No global cache reset was used.

The GPU capacity proxy is 8,192 MiB. The guard counts reported memory use,
including library allocations and desktop clients, separately accounts for
NVIDIA's reserved memory, and leaves 512 MiB free. Compute speed and bandwidth remain those
of the installed 16 GB card. The configuration has not been tested on an 8 GB
board or at a longer context.

| Case | Prefill tokens/s | Decode tokens/s | Memory observations |
| --- | ---: | ---: | --- |
| 128 prompt / 64 output, legacy prefill | 0.56182 | 0.295304 | 28.0 GiB cgroup peak; zero swap; 4,740 MiB sampled CUDA process peak |
| 32 prompt / 8 output, final capacity check | 0.60520 | 0.328085 | 28.0 GiB cgroup peak; zero swap; 6,001 MiB sampled total GPU peak |

Neither run recorded sustained external CPU interference. The longer case used
process-based GPU accounting and reported about 5,966 MiB total device usage at
its phase ends. The subsequent short check continuously sampled total GPU usage;
its 6,001 MiB reported-use peak plus 404 MiB of NVIDIA reserved memory and the
512 MiB free reserve fits the 8,192 MiB proxy. The reserved-memory correction was
added during the subsequent 64 GB test; it does not change this case's outcome.

Decode timings exclude the first token obtained from prefill, startup and
preparation. The 64-output case times 63 decode steps. Prefill includes the SSD
misses during prompt processing. There was no whole-model warmup or untimed
prefill. SSD reads over the complete longer process were **275.98 GiB**; its
prefill recorded **833,439 major page faults**. The short check read 63.58 GiB.

The routed experts occupy **111,188,901,888 bytes (103.55 GiB)** across the pack
and original-format exception layers. The engine cannot keep that collection
resident under the RAM limit. In this configuration, demand paging and streaming
dense weights dominate; the previous results with enough RAM do not transfer to
this memory budget.

## Fast-prefill diagnostics that did not qualify

The ordinary GPU prefill setup failed its allocation fit check even with a
64-row batch and small context. A 64 MiB scratch arena also failed: the 32-row
MLA gather alone requested 128 MiB.

Two opt-in allocation changes were implemented for the experiment:

- `STRATA_GLM_COMPACT_STAGING=1` sizes the two staging slots from the effective
  artifact after overlays. For this pack, each slot fell from 208 to 167 MiB,
  saving 82 MiB. Projection bytes and kernels are unchanged.
- `STRATA_GLM_BUDGET_RESERVE_MIB=512` lowers the logical allowance for allocations
  outside the engine's tracker; the default remains 1,024 MiB. Actual total GPU
  usage and physical headroom still have to pass the guard.

That trial tracked 7,301 MiB of engine allocations, but its CUDA process reached
8,014 MiB before accounting for desktop allocations or the physical reserve.
It therefore **does not fit the final 8 GB proxy**. Its 0.98 decode / 0.72 prefill
numbers are diagnostics only.

A 384 MiB arena, 64-row prefill and MTP depth two produced 1.05 prefill and
0.73 decode tokens/s on the 128/64 case. It also exceeds the final capacity
definition. Its monitor encountered an exit-time `/proc` read race after all
64 tokens were emitted, so only its raw timings are retained. The monitor now
handles that race and saves intermediate telemetry.

The corrected guard rejected the oversized fast-prefill profile in 3.16 seconds,
terminated its own child and recorded no throughput result. No diagnostic profile
was promoted. The legacy profile adds no channel pruning or residency routing
bonus; separate perplexity and task-level quality qualification was not run
after the throughput targets failed.

## Reproduction

Linux, a user systemd manager and NVIDIA's `nvidia-smi` are required. Use Strata's
Python environment, which supplies the tokenizer/server dependencies. Only one
GLM decoder should be active; the cache reset refuses to proceed if one is found.

The frozen measured executable is
`build-q2-v7-lowmem/strata-glm-decode-measured`; its isolated build hashes and
commands are in `build-q2-v7-lowmem/build-measured-manifest.json`. For a fresh
build, build `strata-glm-decode` normally and change the configuration's `exe`.
The profile is experimental; it is not selected automatically by setup.

```sh
build-exl3-venv/bin/python tools/glm_low_memory_bench.py \
  configs/glm53f-q2-32g-8g-experimental.json \
  build-q2-v7-lowmem/prompt-128.ids \
  --output build-q2-v7-lowmem/reproduction --tokens 64 --trials 1 --single
```

The output prefix must be new. The runner creates a temporary 28 GiB, no-swap
systemd unit and records raw logs, output IDs, final logits, the launch manifest,
memory events, per-process SSD reads, interference and GPU samples. Its default
three repetitions include an untimed prefill warmup from the engine; use one
trial to reproduce the cold-cache measurements above. Do not interpret an
unconstrained run on a 128 GB host as a 32 GB measurement.

`tools/test_glm_low_memory_bench.py` originally passed seven checks covering timing parsing,
GPU capacity accounting, selective file inventory, `mincore` and cgroup records.
The isolated decoder build and `git diff --check` passed.
