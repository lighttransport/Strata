# Reproduce the v100s node experiment

Run from the repository root. The commands assume a Linux allocation with two
V100 32 GB GPUs, 36 physical cores, enough locked-memory allowance and 160 GiB RAM.
Do not run another GPU benchmark concurrently.

## Models

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Full-288 assembled Q23 | 120875172704 | `8e280fd86714b606211c285552e77cde7b8c31ac0e9483df85c72ab1174a50e7` |
| REAP-224 assembled Q23 | 95538419872 | `3ef5726ae4a9768fa8311453927674401f3958a8c3884effddac92fc41262003` |
| Pruning plan | 34959 | `d5e4086e684a4a3b48d76161e67f2322613b38af2eeaaccbb03f9dd28d762711` |
| Expert prior | 102563 | `8162269a14b84d1057402278d49a25b4e4407f0e1a24e9154f483d162a73a7c2` |

The exact plan/prior are in `tools/sim/data/v100s_release/`. Derive the pruned
model without requantization:

```bash
build-glm-v100/strata-glm-prune-experts \
  models/glm53f/assembled/GLM-5.3-Flash-Q23-assembled.gguf \
  tools/sim/data/v100s_release/plan-keep224.txt \
  models/glm53f/assembled/GLM-5.3-Flash-Q23-REAP224-assembled.gguf
```

The existing numeric fixtures in `docs/fixtures/glm_tr16_20261009/prompts/8k/`
have 8,192 IDs each. The timings use `train-code.ids`; `holdout-chat.ids` is the
held-out prompt. Their exact hashes are recorded per case in `results.json`. IDs are passed
directly to the engine, so tokenizer/package changes cannot change the prompt.

## Shared RAM release benchmark

Build `strata-glm-decode`, `strata-glm-gpu-worker` and the pruner in Release with
CUDA architecture 70. The measured build uses CUDA 12.9 and the ggml dependency
revision in `build.json`. For exact timing-build reproduction, build the C++
engine from a separate source checkout at `83f3cac9`; keep the benchmark helper
and configuration from this report checkout. The later `bba75e56` native-stage
repair has separate correctness evidence and was not used for the timing table.
Set the desired depth/cache/prompt in the snippet.
Keep MTP-2 groups `(1,2)` and MTP-3 groups `(2,2)` distinct in the simulator.

```bash
numactl --interleave=all python3 - <<'PY'
import json, os, pathlib, subprocess, sys
sys.path.insert(0, 'tools')
from glm_resident_model import ResidentModel

config = json.loads(pathlib.Path('configs/experimental/glm53f-q23-reap224-v100s-aff014.json').read_text())
env = {k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
env.update(config['env'])
physical = {}
for cpu in sorted(os.sched_getaffinity(0)):
    topology = pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    key = tuple(int((topology/name).read_text()) for name in ('physical_package_id', 'core_id'))
    physical.setdefault(key, cpu)
cores = sorted(physical.values())
assert len(cores) == 36
prompt = pathlib.Path('docs/fixtures/glm_tr16_20261009/prompts/8k/train-code.ids').resolve()
model = pathlib.Path(config['model']).resolve()
with ResidentModel(model, '3ef5726ae4a9768fa8311453927674401f3958a8c3884effddac92fc41262003', max_bytes=136*1024**3) as resident:
    command = [sys.executable, 'tools/glm_mixed_run.py',
        '--primary', 'build-glm-v100/strata-glm-decode',
        '--worker', 'build-glm-v100/strata-glm-gpu-worker',
        '--model', str(resident.path), '--primary-device', '0', '--worker-device', '1',
        '--combine', 'device', '--primary-cpus', ','.join(map(str, cores[:-1])),
        '--worker-cpus', str(cores[-1]), '--worker-cache-mib', '14336',
        '--worker-gpu-budget-mib', '30720', '--', '@'+str(prompt), '512', '4096', '34',
        '--context=12288', '--prefill-batch=8192', '--gpu-budget-mib=30720',
        '--gpu-devices=0', '--cpu-affinity=numa', '--decode-experts=cpu',
        '--bench=3', '--decode-bench=4', '--decode-cache-mib=10240',
        '--prefill-experts=f16-batched', '--prefill-expert-cache-mib=0', '--decode-graphs',
        '--speculative=mtp', '--draft-depth=3', '--mtp-experts=gpu', '--warm-weights']
    with open('v100s-tokens.txt', 'w') as tokens, open('v100s-timing.log', 'w') as log:
        subprocess.run(command, env=env, stdout=tokens, stderr=log, check=True)
PY
```

Treat the first decode as warmup; use the median of the remaining three. Preserve
raw tokens and timing logs, their hashes and every failed attempt. Inspect early
EOS and cycles before qualifying timing. For MTP-2 change only `--draft-depth=2`.
Cache variants use explicit primary sizes up to 12,288 MiB and a 24,576 MiB worker
cache. Automatic primary cache sizing is unsupported for MTP in this engine.

For diagnostic scopes, set `STRATA_GLM_STEP_TRACE=2`, `STRATA_GLM_MTP_TIMING=1`,
`STRATA_GLM_MIXED_PROFILE_DECODE=1`; shorten the run to 256 tokens, two prefill
measurements and two decode trials (one warm, one timed). Add `--routing-trace=...`
and `STRATA_GLM_ROUTING_EVENTS=1` for routing evidence. Do not use an instrumented
sample as the clean throughput headline.

## Component probes

The GPU transfer probe is `tools/glm_transfer_probe.cu`: compile with
`nvcc -O3 -arch=sm_70` and run `h2d 0.5`, `d2h 0.5` and `kernel 0.5` on each
allocation-visible GPU. Each command emits a warmup plus five timed samples.

`real-expert-probe.cpp` is a CUDA-only diagnostic, linked against the matching
build's `strata_kernels_cpu`, `strata_host`, `ggml-cpu`, `ggml-base`, CUDA runtime
and pthread libraries with C++20/O3. Run `MODEL WORKERS DEVICE` under
`numactl --interleave=all` with the recorded CPU masks. The source hash in
`component-measurements.json` is the exact source compiled on the node.

`tools/glm_mixed_worker_loopback.py` supplies the worker RPC measurements. Use
300 requests after 50 warmups, CPU 0 / memory domain 0 for the client, and worker
NUMA 0 or 2 for GPU 0 or 1. The original measurement uses default worker flags.
The release-environment probes add `--widths 1 2 4`, then `--shared-experts`, to
measure distinct and shared experts separately. Record zero activations and
round-trip scope; do not call the result an isolated GPU-kernel time.
