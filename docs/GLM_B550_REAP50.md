# REAP50 on B550 (RX 9070 XT)

For the consolidated hardware summary, release scope and reproduction commands,
see [the experimental release candidate](GLM_RELEASE_CANDIDATE.md).

This experimental deployment uses B550's Ryzen 9 3950X, one NUMA node, 64 GB-class
system RAM and RX 9070 XT (gfx1201). The process has a hard 60 GiB memory cgroup,
no swap, and a watchdog requiring at least 2048 MiB of free physical VRAM.
The GPU reports 16304 MiB usable VRAM. This is an AMD measurement, not an
extrapolation from the earlier NVIDIA/dual-NUMA tests.

## Current optimized profile: both targets met

After the host owner reset GPU clocks to automatic mode, the final profile
measured **10.21 tok/s single-sequence decode** and **230.06 tok/s warm prefill**.
This uses 2048 prompt tokens, prefill batch 2048, 128 generated tokens and three
trials. Decode times 127 steps per trial, without speculation. All three trials
exceeded 10 tok/s decode and 200 tok/s warm prefill:

| Trial | Warm prefill tok/s | Single decode tok/s |
| --- | ---: | ---: |
| 1 | 229.469 | 10.0797 |
| 2 | 230.361 | 10.2559 |
| 3 | 230.060 | 10.2121 |

The benchmark cleared model cache before creating a fresh 60 GiB cgroup, then
performed an untimed prefill before the measured repeats. These are **warm
prefill rates**, not cold SSD-loading rates or promises for shorter prompts.
Peak cgroup RAM was 52.90 GiB; minimum free global VRAM was 3002 MiB. There were
no swap, OOM, memory-limit hits or guard rejections. The whole run, including
loading and the untimed cold prefill, took 142.75 seconds.

Changes in the current configuration:

- Fix GPU-local staging affinity when PCIe `numa_node=-1`. Previously all four
  staging workers inherited the coordinator's CPU 0 mask; they now use CPUs
  0, 1, 2 and 3 within the original allowed CPU allocation. The affinity-only
  change preserved generated-token and final-logit hashes exactly.
- Use `GLIBC_TUNABLES=glibc.cpu.x86_non_temporal_threshold=1048576` in the B550
  process environment. For expert-sized copies, the pinned-memory microbenchmark
  improved from 9.12 to 14.08 GB/s with four workers and verified identical bytes
  after upload/download. This setting is specific to the tested Linux/glibc host.
- Use the engine's existing `f16-batched` expert-prefill backend. MMQ with automatic
  GPU clocks measured 78 tok/s at 1024 tokens; affinity and streaming-copy changes
  reached about 107 tok/s there, or 134 tok/s at 2048 tokens. The FP16 path provides
  the larger improvement. These latter tuning cases used cached-file diagnostics;
  the final rates above come from the clean, isolated qualification.
- Use 12 CPU expert workers and six native tasks per worker, with a 6144 MiB decode
  cache. The original attention precision and KDA settings are retained. Additional
  attention experiments were not selected.

The quantized model files and decode arithmetic are unchanged. **FP16 prefill is
not bit-exact to MMQ.** A 2048-target teacher-forced comparison over four prose/code
documents measured perplexity 9.0374 → 9.1036 (+0.73%), mean KL 0.0351 and 91.85%
top-token agreement. This is a numerical smoke test, not full task-quality
qualification. The previous MMQ settings remain available in
`configs/glm53f-reap50-q23-b550-60g-9070xt-mmq-reference.json`.

The service also enables `warm_expert_pages`: before announcing readiness, the
wrapper reads the 126 main expert projections into its own memory cgroup with an
8 MiB buffer. It refuses this warmup unless at least 4 GiB of the configured RAM
allowance remains beyond the expert bytes. No model tensor is changed.

Warmup took 47.78 seconds and readiness 63.05 seconds in the service test. Total
service cgroup memory peaked at **55.86 GiB**, with no swap or memory-limit hits;
peak global GPU usage was 13301 MiB (3003 MiB free). The first real chat returned
`hello` with normal stop in 15.61 seconds, versus 55.48 seconds without warmup.
Its short 61-token output measured 9.9 tok/s, illustrating startup overhead and
prompt-dependent variation around the longer benchmark's 10.21 tok/s median.
The startup disk-read cost is moved before readiness, not eliminated.

The guarded service was verified at `http://127.0.0.1:8080` on B550.
The default launcher selects the optimized profile. Detailed measurements:
`docs/benchmarks/glm_b550_reap50_optimized_20261008.json`.
The sections below retain the earlier manual-clock results for comparison.

## Code and model

Upstream Strata 0.1.40.3 (`d5ea7133741e67743c0e886bb426c0ce8d69cf6c`) was merged
into the GLM branch as `036c926e6db07e8f66d6b125b3139ef282dad327` before the HIP
integration. Local GLM work is still uncommitted; the merge commit alone does not
include these deployment changes. Existing local and B550 edits were backed up.

TR16's `/mnt/nvme01/models/glm53f/reap-50/reap50-q23.gguf` contains the processed
REAP50 experts, rather than a complete model. Its source is
`GLM-5.3-Flash-REAP50-Q4_K_M.gguf` in the same directory. The standalone assembly
copies all 126 processed expert projections plus the source's remaining tensors,
without requantization. It validates the original sidecar fingerprint and expert
checksums. `strata.expert_pack.assembled` preserves the engine's converted-expert
execution path; it does not require the original base on the deployment host.

The assembly is 63,205,406,688 bytes (58.87 GiB), SHA-256
`c3b0cfb144556bd28e265d8255d48eeafce3d127aba5b9b5dd4e326322a34aa9`.
B550 destination:
`/mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf`.
`/mnt/disk1/models` resolves to `/mnt/disk01/models`; `/mmt/disk1/models` does not exist.
The old Q3_K_M model is retained.

Build the byte-preserving assembler with CMake's `strata-glm-assemble` target, or:

```sh
c++ -O2 -std=c++20 -Iinclude tools/glm_assemble.cpp -o strata-glm-assemble
./strata-glm-assemble ORIGINAL.gguf EXPERTS.gguf OUTPUT.gguf
python3 tools/test_glm_assemble.py
```

The output retains the base's original metadata, including its original quantizer
classification, and adds the processed expert provenance. Tensor directory types
and the assembly filename identify the effective mixed Q2_K/Q3_K representation.
The assembler refuses existing output/partial files. A failed `.partial` is not a
usable model and is deliberately retained for diagnosis.

## B550 commands

```sh
cd ~/work/Strata
tools/glm_b550_60g.sh start
tools/glm_b550_60g.sh status
tools/glm_b550_60g.sh stop
```

The API binds only `127.0.0.1:8080`. Configuration:
`configs/glm53f-reap50-q23-b550-60g-9070xt-experimental.json`.
The launcher uses the existing ROCm installation and `.venv-glm-hip`.
Telemetry is `build-hip-glm-v11/server-memory.json`.
The GLM engine currently accepts greedy generation (`temperature: 0`).

## Existing Q3 baseline

At 1024 prompt tokens, prefill batch 1024 and 64 generated tokens, the original
Q3_K_M model measured **10.35 tok/s cold prefill and 1.90 tok/s decode**. The decoder
completed cleanly with no swap or OOM. Its process read 83.09 GiB from disk and the
memory cgroup reached 60 GiB. Main routed expert weights alone occupy 67.18 GiB,
so this model pages from the SSD under the requested limit. Peak global GPU usage
was 12505 MiB, leaving 3799 MiB free. These numbers do not meet the requested
20–30+ tok/s decode target.

## Processed Q23 results (2026-10-08)

The transferred file's full SHA-256 matches TR16. Both runs completed cleanly,
with swap disabled, no OOM and no memory-limit hits. Context capacity was 4096,
prompt length 1024, prefill batch 1024, CPU threads 15, and decode cache 6144 MiB.
Single-sequence decode generated 64 tokens; the timed portion covers 63 steps.

| Run | Prefill tok/s | Decode tok/s | Peak cgroup RAM | Minimum free VRAM |
| --- | ---: | ---: | ---: | ---: |
| Cold, one trial | 12.87 | 4.39 | 52.01 GiB | 2863 MiB |
| Warm, median of two trials | 18.87 | 4.46 | 51.99 GiB | 2863 MiB |

For the warm run, the engine performs an untimed prefill before its two measured
prefills. Both measured prefills had zero major page faults. The last ten seconds
of warm decode read no additional disk data. The model now fits RAM; these rates
remain below the requested 20–30+ tok/s decode and 300+ tok/s prefill targets.

**Power-policy limitation:** during active warm prefill, sysfs reported GPU
utilization 91%, manual power mode, and memory clock 96 MHz (listed maximum
1258 MHz). This is a likely contributor to the low rates. These historical results predate the automatic-mode comparison above; they
are measurements of that earlier configuration, not RX 9070 XT peak performance.
Changing that root-owned setting requires the host owner to run:

```sh
echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
```

A separate 16-token/32-output diagnostic (not a qualified throughput result)
reported 232.6 ms/step: about 82.5 ms CPU routed work and 150.1 ms in the remaining
host/GPU wait intervals. This is a host timeline, not a GPU-kernel profile.
The diagnostic does not establish how much automatic power mode will recover.

Full metrics and model provenance:
`docs/benchmarks/glm_b550_reap50_q23_20261008.json`.
Raw logs, manifests and telemetry are under `build-hip-glm-v11/reap-q23-*` on
B550, with copies under `build-b550-v11/raw/` on TR16.

Validation passed: four assembly/rejection tests, six HIP GLM kernel tests,
five GLM API tests, and eleven memory-guard tests. The assembler checked all
126 expert payload hashes. This is not a broad model-quality evaluation.
The temporary 58.9 GiB assembly on TR16 was removed after destination checksum
verification; its original Q4 base and Q23 sidecar remain unchanged.


## Service verification

The guarded service was verified on B550 at `http://127.0.0.1:8080`, model ID
`glm53f-reap50-q23-b550-rx9070xt`. A real chat request returned content `hello`
with `finish_reason: stop`; the corrected GLM role-token stops prevent generation
from continuing into a fabricated user turn. A 96-token budget initially expired
during reasoning; a 320-token budget allowed completion in 159 generated tokens.
The completed smoke request took 45.59 seconds with the existing manual GPU mode.
Service telemetry during the smoke test peaked at 41.29 GiB summed RSS and
13353 MiB global VRAM usage, leaving 2951 MiB free; no memory-limit hits or OOM.

## Longer chat context screen

The [16K–128K document-chat report](GLM_LONG_CHAT_QA.md) records the 90/10 input/output-budget ladder and a multi-turn follow-up. All tested capacities fit with a 3 GiB expert cache, but the 64K answer and follow-up confused two different metrics. At 128K, the 117955-token input measured 173.4 tok/s prefill and 9.2 tok/s decode. These are separate from the 2K warm throughput result above.

### LAN web UI (2026-10-09)

`tools/glm_b550_60g.sh start-lan` binds the normal Q23 service to
`0.0.0.0:8080`, retaining the 60 GiB RAM and 2 GiB free-VRAM guards. It requires
`build-hip-glm-v11/server-api.env` containing `STRATA_API_KEY=...`; keep this
local file mode 600 and out of source control. The launcher loads it through
systemd's `EnvironmentFile`, without placing the key in command arguments.
The guarded launcher refuses a non-loopback host without a key.

Open `http://192.168.100.34:8080` on B550's LAN and enter the key under
**About → Settings → API key**. Retrieve it over SSH:

```sh
ssh b550 'cat ~/work/Strata/build-hip-glm-v11/server-api.env'
```

`start` still binds loopback; use `stop` before switching modes. If the key file
exists, authentication remains enabled in either mode. No firewall rules were
changed.

### Live web throughput

The chat composer shows **PP** (prompt processing) and **TG** (token generation)
in tok/s, refreshed from `/metrics` every second. GLM emits cumulative prefill
milliseconds and tok/s with each completed `PP` chunk; the first number appears
after the first chunk completes. This PP timing excludes request reset and
post-prefill GPU expert-cache preparation. TG uses the existing rolling decode
rate while generating. Once idle, both counters show the last request's rates;
the Monitor retains its detailed charts. Hard-refresh an already-open page after
deploying the updated assets.
