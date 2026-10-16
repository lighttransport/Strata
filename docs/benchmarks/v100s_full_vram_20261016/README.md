# V100s VRAM preallocation and coding checks, 2026-10-16 JST

VRAM preallocation now runs on both 32 GB cards with useful expert caches
above the old limits. The measured full-memory configuration is **slower than
the smaller-cache training control and does not pass the coding-quality gate**.
It remains an explicit experimental profile. Production defaults are unchanged.

Hardware: **v100s node**, two Tesla V100 PCIe 32 GB GPUs, two Xeon Gold 6240
sockets, 36 physical cores / 72 logical CPUs, four NUMA domains, and a 160 GiB
RAM allocation. RAM speed was unavailable. The Release build uses CUDA
12.9.86, GNU 12.2.1 and sm70. The inference code is
`f8bf1ce79b8b5422a5bfd0a227fe222362be99cf`.

The model is GLM-5.3-Flash REAP-224 Q23, 95,538,419,872 bytes, SHA-256
`3ef5726ae4a9768fa8311453927674401f3958a8c3884effddac92fc41262003`.
The expert prior's SHA-256 is
`8162269a14b84d1057402278d49a25b4e4407f0e1a24e9154f483d162a73a7c2`.

## Memory

The successful configuration sets `STRATA_GLM_VRAM_PREALLOC=1`, with a
**512 MiB** driver/library margin on each card and a 512 MiB tier runtime
reserve. Both GPU budgets are 32,768 MiB. Requested expert caches are
30,720 MiB primary and 32,000 MiB worker; admission leaves space for the
model's other allocations.

| Measurement, MiB | GPU 0, primary | GPU 1, worker |
| --- | ---: | ---: |
| Advertised NVML capacity | 32,768 | 32,768 |
| CUDA usable capacity | 32,494 | 32,494 |
| Preallocated pool | 31,616 | 31,616 |
| Populated expert cache | 22,952.4 | 29,488.4 |
| Peak sampled global use | 32,112 | 32,280 |
| Minimum sampled free memory | 383 | 215 |

Peak sampled use is **98.8% / 99.3% of CUDA usable capacity**. Pool reservation
and populated expert weights are different measurements. Dense weights, MTP
weights, state, transport and workspaces also use VRAM; some pool space remains
available for runtime allocations and cache placement. The worker's cache
admission budget was 30,676.2 MiB, with 83 slots per layer before exclusions
for primary ownership.

The capacity probe sampled NVML once per second. Its timings are excluded
from throughput claims. A 256 MiB library margin failed during worker decode
in both the initial probe and the following training attempt. Those frozen
failures remain in [results.json](results.json); they are not accepted timings.
The 512 MiB probe and all three longer timing cases completed successfully.
[memory.json](memory.json) contains the numeric observations and raw-evidence
hashes.

## Throughput

All rows below used the same allocation, binaries, model and socket placement.
The full-memory configuration changes preallocation, cache sizes, GPU budgets,
runtime reserve and active-only warmup. This comparison does not isolate one
of those changes.

| Configuration and prompt | Decode, tok/s | Prefill, tok/s |
| --- | ---: | ---: |
| Conservative cache, training code | 49.4581 | 854.583 |
| Full memory, training code | 44.4578 | 848.062 |
| Full memory, held-out code | 42.6132 | 847.903 |
| Full memory, held-out chat | 44.1962 | 846.669 |

These are medians at 8,192 prompt tokens and 512 generated tokens. Each case
has a prefill warmup followed by three timed prefills, then one complete
512-token decode warmup and three timed decode repeats from the same prefill
state. Decode rates count tokens after the initial token. Staging, checksum
verification, initialization and initial cache setup are untimed.

Affinity is **0.08**, MTP depth **3**, repeat guard **3**, soft acceptance
margin/top-k/max-soft **6/8/3**, draft top-k **4**, prefetch groups **14**, and
the prefill split **0.5**. Primary CPUs are 0-34 with 34 workers; worker CPU
is 35. Expert row halves are placed by physical socket, interleaving within
each socket's two NUMA domains.

The conservative caches are 2,048 / 28,672 MiB, with 30,720 MiB budgets and
preallocation disabled. All 16 conservative/full-memory warmup and timed
decode streams completed and passed the early-stop and consecutive-cycle
checks. Every timed prefill matched its warmup's last-prompt logits and route
counts. Observed inference-process swap was zero; their reported PSS excludes
the separate resident owner.

Adaptive cache learning continues between decode repeats. Token streams can
change, and the stricter identical-output timing gate remains unmet. The full
training rates grew from 42.4404 to 44.4578 to 45.0317 tok/s across the three
timed repeats. These measurements are not a proof of steady-state throughput
or a precise speed ratio. The full-memory configuration misses the 50 tok/s
decode target and exceeds the 700 tok/s prefill target.

## Coding checks

All **376 samples** were generated with frozen configurations. No generated
code was executed on the GPU node. Python and C++ tests ran locally in
bubblewrap namespaces without access to home, model files or networking.

| Check | Affinity 0.08, MTP 3 | Affinity 0, no MTP |
| --- | ---: | ---: |
| Chat HumanEval, one sample per task | **141/164 (85.98%)** | **149/164 (90.85%)** |
| C++ tasks overall | **21/24** | **20/24** |
| C++ mandatory tasks | 4/4 | 2/4 |
| C++ held-out tasks | 17/20 | 18/20 |
| Original parser, 400,532-case oracle | Pass | Fail |

HumanEval uses the pinned 164-task dataset, a chat prompt requesting the
complete function and imports, low reasoning, temperature zero, all assistant
turn stop IDs, and a 2,048-token generation cap. This is a chat protocol,
not the original completion-only protocol. Seven fast-profile samples and
four control samples reached the cap. The dataset SHA-256 is
`b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`.

The C++ suite has three mandatory repair contracts, twenty held-out contracts,
and the original parser fixture with exactly 4,096 input tokens. The other
contracts use 1,024 input tokens. All have a 2,048-token generation cap.
The independent compiled oracles check boundaries and seeded random inputs.
The original parser additionally compiles with `-Wall -Wextra -Werror` and
checks 400,532 cases. The local judge uses GNU g++ 13.3.0 and Python 3.12.3.

The fast profile failed `parse_i64`, `parse_hex` and `rotate_right` (an
incomplete code block). The control failed `checked_slice`, `mul_i64` and
`parse_hex` with incomplete code blocks, and its original parser failed the
numeric oracle. Manual review found its extra remainder condition rejects
valid values near `UINT64_MAX`. The fast original parser passed the numeric
oracle and manual checks for `noexcept`, no allocations, ASCII bytes, defined
unsigned arithmetic and unchanged output on failure.

The fast profile has eleven HumanEval task regressions and three gains versus
the control, a net loss of eight passes. Its new C++ held-out failures are
`parse_i64` and `rotate_right`; its gains are `checked_slice`, `mul_i64` and
the original parser. All 164 HumanEval and 24 C++ paired input-token hashes
match. Per-task outcomes, output hashes and manual reviews are in
[quality.json](quality.json).

The required floor is 152/164 HumanEval, at most one pass lost against the
paired control, all mandatory C++ tasks passing, and no new held-out C++
failures. **The fast profile fails this gate. Both profiles are below the
HumanEval floor.** The control retains dense Q4 and experimental FP16 prefill
formats. It also has more cache space available without MTP. This comparison
does not qualify those formats against a full-precision or unpruned model.

Generation used two renewable three-hour allocations. The greedy HumanEval
engine restarted before task 155 after the planned checkpoint; its first
155 samples and last nine samples therefore have different cache histories.
The other three sets each used one persistent engine. [generation.json](generation.json)
records the exact sample grouping and invocation hashes. Settings and completed
samples were not changed after inspecting held-out results.

## Reproduction and release scope

[profiles.json](profiles.json) contains complete inference environments and
arguments for the timing and quality configurations. Timing children also set
`GLIBC_TUNABLES=glibc.cpu.x86_non_temporal_threshold=1048576`; quality profiles
inherit the allocation environment without explicitly setting this tunable.
Quality generation wall times are not throughput benchmarks.

Use the [previous resident-owner CLI recipe](../v100s_opt_20261016/README.md#reproduction)
with `timing_full` from this report, a 30,720 MiB primary cache, 32,000 MiB
worker cache and 32,768 MiB budgets on both cards. Keep the new profile's
512 MiB preallocation margin, 512 MiB runtime reserve, affinity 0.08 and
repeat guard 3. The 8K prompts and expert prior remain repository-relative.
CPU masks must follow the actual allowed cpuset, socket and PCI topology.
Load this report's `profiles.json` entry and read its `socket_row_plan` path;
the placement-plan file remains in the previous report's directory.

The public resident copier performed two complete 95 GB copies, full SHA-256
checks, locking and sealing, with ten sampled panels verified on their intended
sockets in each allocation. This closes the older report's full-copy validation
limit. Use `--warm-weights` for the CLI; `--lock-weights` would create a private
COW copy. For serving, keep `lock_weights=false`; the owner pins the shared
model and `STRATA_GLM_WARM_ACTIVE_ONLY=1` performs unlocked warmup.

The [experimental API profile](../../../configs/experimental/glm53f-q23-reap224-v100s-full-vram-aff008.json)
is explicitly unqualified. It uses the public mixed decoder adapter and binds
to `127.0.0.1`. Activate the project Python environment and run from the
repository root, inside a suitable allocation with sufficient RAM and memlock:

```sh
PYTHONPATH=tools numactl --interleave=all python3 - <<'PY'
import json, os, subprocess, sys, tempfile
from pathlib import Path
from glm_resident_model import ResidentModel

cfg = json.loads(Path('configs/experimental/glm53f-q23-reap224-v100s-full-vram-aff008.json').read_text())
plan = json.loads(Path(cfg['socket_row_plan']).read_text())
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
Path('tmp').mkdir(exist_ok=True)
with ResidentModel(cfg['model'], cfg['model_sha256'], max_bytes=128 * 1024**3,
                   socket_row_plan=plan) as resident:
    cfg['model'] = str(resident.path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir='tmp') as f:
        json.dump(cfg, f)
        f.flush()
        subprocess.run([sys.executable, '-m', 'serve.server', '--engine', 'glm',
                        '--config', f.name], env=env, check=True)
PY
```

The measured quality wrapper keeps the original GGUF path for tokenizer and
resume identity, verifies the current owner's model SHA, size and seals, then
substitutes its live memfd path only for inference. It invokes the public
`tools/glm_mixed_decode.py` adapter with the recorded arguments. The public
[HumanEval generator](../../../tools/glm_q2_humaneval.py),
[isolated grader](../../../tools/glm_humaneval_grade.py) and
[C++ contracts/oracles](../../../tools/glm_coding_quality.py) define the tests;
connection settings and frozen raw outputs remain private.

[validation.json](validation.json) records the 17 source hashes, binary and
pipeline identities, model placement, compiler versions and shutdown checks.
Thirty-two focused unit tests passed. Final-source CUDA and HIP compilation
passed; CUDA sm70 execution ran on the V100s. With new options off, 16 greedy
tokens and both complete 154,880-float logit arrays match the frozen default
reference byte for byte. The experimental API profile passed configuration
validation with mocked weights/engine and matched the frozen inference
arguments and environment. A GPU HTTP frontend smoke test was not run.

HIP GPU execution, Windows execution and full-precision model quality were
not tested. The ordinary numeric cache ceiling remains 12 GiB; the larger
limit requires explicit preallocation. The GPU session finished with zero
active jobs and both controllers stopped. No default was changed to this
slower, unqualified profile.
