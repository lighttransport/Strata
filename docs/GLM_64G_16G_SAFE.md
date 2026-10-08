# GLM with 64 GB RAM and 16 GB VRAM, with safety margins

The unbiased configuration fits the requested margins but still misses the
20 decode / 300 prefill tokens/s targets. On the tested machine it measured
**2.36 decode / 13.48 prefill** with a 1K prompt, and **2.30 decode / 104.04 prefill**
with an 8K prompt. Both runs used a hard 60 GiB RAM allowance and left at least
2 GiB of actual GPU memory free.

The earlier resident-routing recipe was also tested. Its warm decode reached
5.52–5.98 tokens/s, but its prose perplexity exceeded the additional 1% allowance.
It was not promoted. [Measured records](glm53_flash_64g_16g_safe_measurement.json)
contain the commands, source and prompt identities, memory samples and quality
results. The saved unbiased profiles are
[short prompts](../configs/glm53f-q23-v8-64g-16g-safe-short.json) and
[long prompts](../configs/glm53f-q23-v8-64g-16g-safe-long.json).

## Machine and budget

Measured October 7, 2026 UTC on a Threadripper 1950X, 16 physical cores, two NUMA
nodes, 128 GB DDR4 and RTX 5060 Ti 16 GB. The RAM capacity was simulated by a
Linux cgroup; the GPU was the physical 16 GB card. Original GGUF shards were on
`/mnt/nvme01`, the Q23 sidecar on `/mnt/nvme02`. Performance depends on this
CPU, GPU and storage arrangement.

- RAM: **60 GiB for the engine, its pinned buffers and file cache**, with swap
  disabled. The nominal remaining 4 GiB represents the OS allowance on a 64 GB PC.
- GPU: at most **14,336 MiB total use**, including the engine, libraries, desktop
  clients and NVIDIA's separately reserved memory; at least **2,048 MiB actually free**.

The card reports 16,311 MiB total and 404 MiB reserved by NVIDIA, leaving about
15,907 MiB available to CUDA. The actual 2 GiB free requirement is therefore the
tighter bound: reported use excluding the reserved portion must stay at or below
13,859 MiB. Including the reserved portion, that is 14,263 MiB, below nominal
14 GiB. The guard measures free memory directly; it does not subtract the 2 GiB
margin twice from the configured 14 GiB allocation budget.

Each run cleared only this model's clean cache pages before entering the cgroup.
The inventory reported zero cached bytes for all four shards and the expert pack.
The retained measurements had zero swap and no sustained external CPU interference.
Their RAM peaks were 60 GiB; the lowest sampled free GPU memory among the accepted
performance cases was **2,338 MiB**.

## Throughput

Context allocation was 12,288 tokens, with 15 workers plus the host. All cases
used the canonical Q23 expert pack and CPU-draft MTP depth two. The GPU expert
cache requested 4,800 MiB; physical headroom and runtime reservations limited the
actual allocation to **3,692.19 MiB, 425 experts**. Routing affinity to GPU experts
was zero.

| Profile / workload | Prefill tokens/s | Decode tokens/s | Minimum sampled free VRAM |
| --- | ---: | ---: | ---: |
| Unbiased, 1,024 prompt / 256 output, one cold trial | 13.478 | 2.35965 | 2,347 MiB |
| Unbiased, 8,192 prompt / 128 output, one cold trial | 104.035 | 2.29564 | 2,352 MiB |
| RAM residency margin 0.10, 1,024 / 256, three trials | 12.0987 / 11.9296 / 11.9496 | 4.09643 / 5.51971 / 5.98312 | 2,338 MiB |

The resident case had an initial 48,000 MiB expert-set preload and an untimed
prefill warmup, then three timed prefills and three decode repetitions. Its median
prefill was 11.9496 tokens/s and median decode 5.51971. Its output changed between
repetitions as residency evolved. No generated EOS or stop IDs were found in the
timed outputs. This is not a comparison of identical warmup scopes with the cold
unbiased cases.

Decode excludes the first token obtained from prefill, initialization, MTP
preparation, cache filling and benchmark checkpoint allocation. Prefill includes
SSD misses during prompt processing. The 8K unbiased run spent **76.55 of 78.74
seconds in staging** and read **172.63 GiB from storage over the complete process**.
Increasing prompt length amortized the weight reads, but did not reach 300 tokens/s.

The routed weight collection is 111,188,901,888 bytes, or 103.55 GiB. It remains
larger than the RAM allowance and the available GPU expert tier. These runs do
not reproduce the earlier 15.1 tokens/s resident result: that used different
coding fixtures and a smaller GPU free-memory reserve. The measurements above
use the current 1K/8K workload and the full 2 GiB margin.

## Why the earlier 20–24 tokens/s does not carry over

The earlier selected v5 1K/1K run measured 22.14 / 23.10 / 22.57 decode
tokens/s after warming the weights, on the full 128 GB host. Its recorded peak
RSS was **114.24 GiB**, with no 60 GiB cgroup limit. It also used GPU routing
affinity 0.10 and about 4.79 GiB of GPU experts. These are different conditions
from the unbiased 60 GiB runs above, with 3.61 GiB of GPU experts and affinity
zero. The main capacity problem is the 103.55 GiB routed weight collection:
the 60 GiB allowance cannot retain it, so decoding repeatedly reads evicted
experts from storage. The older unbiased 60 GiB experiment already measured
2.4 tokens/s, close to the current 2.30–2.36.

There was also an older 60 GiB result near 23 tokens/s that forced routing into
a frozen 48,000 MiB expert set. Its held-out top-token agreement with the
original model was only 79.8%; it changed which experts the model used. That
result is not a quality-preserving 20 tokens/s configuration. These historical
measurements used different workloads and GPU reserves, so they do not isolate
the individual costs of each setting. See
[single-stream measurements](GLM_SINGLE_OPTIMIZATION.md) and the
[64 GB mode comparison](GLM_Q2_DECODE_REDESIGN.md#64-gb-host).

The subsequent [REAP-50 measurement](GLM_REAP50_64G_16G.md) uses the same
margins. Its improved 51.31 GiB Q23 sidecar recipe reaches 21.13 tokens/s median
decode and 405.44 tokens/s warm prefill at 8K, with a 55.36 GiB RAM peak.
Checked outputs match its earlier recipe exactly, but the REAP quality smoke
still fails both category allowances. Cold 8K throughput is 17.54 decode /
109.03 prefill tokens/s. It remains an experimental size/speed tradeoff.

## Quality and allocation controls

Ordinary teacher-forced evaluation used the frozen 512-target tuning smoke
sample, with 256 code and 256 prose targets, against the saved unbiased reference.
It is not a full held-out or task-level qualification.

| RAM routing margin | Code perplexity change | Prose perplexity change | Separate 1% category gate |
| ---: | ---: | ---: | --- |
| 0.10 | -1.66% | +2.26% | Failed |
| 0.05 | -2.83% | +3.02% | Failed |

The combined averages were only +0.28% and +0.05%, respectively; they would hide
the prose regressions. Both residency biases remain diagnostic. HumanEval and
larger qualification were not run after the throughput and smoke gates failed.

The unbiased control matched **all 79,298,560 logits bit for bit**, with top-token
agreement 100% and unchanged category perplexity. This verifies ordinary decode
against the canonical reference at the new memory budget. The existing FP16 MLA
and four-part KDA prefill candidate still needs its separate full quality checks;
this decode control does not qualify those prefill precision settings.

The first 1K trial used the old large prefill buffers and briefly fell 2 MiB below
the requested GPU margin. It is excluded from accepted results. Resizing the
short-prompt workspace and reducing prefetched groups from twelve to eight
retained **all 256 output IDs and all 154,880 final logits bit for bit**. The safe
short profile uses batch 1,024 and a 1,024 MiB scratch cap; the long profile uses
batch 8,192 and a 2,048 MiB cap. Both use eight prefetched groups and compact
167 MiB staging slots.

## Reproduction

The experimental profiles point to the frozen measured decoder in
`build-q2-v8-64g/`; its archive hashes and build commands are in
`build-manifest.json`. With a fresh normal build, change the profile's `exe` to
that `strata-glm-decode`. Neither profile is selected automatically by setup.

```sh
build-exl3-venv/bin/python tools/glm_low_memory_bench.py \
  configs/glm53f-q23-v8-64g-16g-safe-short.json \
  build-q2-v4/single-1024.ids --output build-q2-v8-64g/reproduction-short \
  --ram-gib 60 --gpu-capacity-mib 16384 --gpu-used-limit-mib 14336 \
  --tokens 256 --trials 1

build-exl3-venv/bin/python tools/glm_low_memory_bench.py \
  configs/glm53f-q23-v8-64g-16g-safe-long.json \
  build-q2-v4/single-8192.ids --output build-q2-v8-64g/reproduction-long \
  --ram-gib 60 --gpu-capacity-mib 16384 --gpu-used-limit-mib 14336 \
  --tokens 128 --trials 1
```

Output prefixes must be new. The tool requires Linux user systemd and NVIDIA's
`nvidia-smi`, records actual free/reserved GPU memory and enforces the RAM cgroup.
Its new `--eval-corpus` and `--eval-reference` options run the same guarded
ordinary quality evaluation. Ten CPU-only helper tests and `git diff --check`
passed.
