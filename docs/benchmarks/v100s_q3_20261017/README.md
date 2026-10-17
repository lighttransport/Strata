# Full 288-expert GLM UD-Q3_K_XL on the v100s node

Quality and memory measurements began on 2026-10-16 UTC. This run uses the
original full 288-expert GLM-5.3-Flash UD-Q3_K_XL weights. The merged GGUF is
147,535,921,504 bytes (137.4 GiB, 147.5 GB decimal), leaving 22.6 GiB of a
160 GiB allocation before runtime workspace. Its SHA-256 is
`7800e21bc527db7ae6eaea36a190711c3c500109c2e5f5904017f1d2d9089b5a`.

The label is a mixed quantization: most routed tensors are IQ3_XXS or IQ4_XS;
selected Q3_K, Q4_K and Q6_K tensors and the original dense formats remain.
There is no additional expert pruning or offline requantization.
[model.json](model.json) records all four shard checksums, the tensor census
and the byte-preserving join. All shard SHA-256 values, 1,412 tensor
descriptors and three payload samples per tensor were checked. The complete
merged SHA-256 was then verified in each sealed resident RAM copy.

## Paired correctness results

| Profile | HumanEval | C++ |
| --- | ---: | ---: |
| Greedy, affinity 0, no speculation | 154/164 (93.90%) | 24/24 |
| Affinity 0.08, MTP depth 3 | 153/164 (93.29%) | 21/24 |

Token-cap hits: greedy: HumanEval 0, C++ 0; fast: HumanEval 1, C++ 0.

These are one sample per task, chat format, low reasoning, temperature
zero and at most 2,048 generated tokens. The two profiles use identical input
tokens. The `greedy` profile has affinity 0 and no speculation; `fast` has
affinity 0.08, MTP depth 3, soft margin 6, top-k 8, maximum three soft accepts
and repeat guard 3. Affinity and soft MTP acceptance can change the selected
tokens. Both use native expert arithmetic, original dense
formats and FP32 decode exchange. The presence-based `STRATA_GLM_MLA_F16`
switch is absent. MTP changes the cache capacity admitted on the primary GPU.

HumanEval includes all 164 tasks from the pinned dataset, SHA-256
`b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`.
The authoritative dataset identity is also recorded in [validation.json](validation.json).
Generation uses [glm_q2_humaneval.py](../../../tools/glm_q2_humaneval.py);
grading uses [glm_humaneval_grade.py](../../../tools/glm_humaneval_grade.py).
Each completion is tested in an isolated process with a five-second limit.

The C++ set contains three published training contracts, twenty held-out
contracts from [glm_coding_quality.py](../../../tools/glm_coding_quality.py),
and the original 4K `parse_u64` contract. The first 23 inputs have 1,024 tokens;
the original contract has 4,096. The original parser oracle exercises 400,532
cases. Both parser outputs in each profile receive a manual contract review,
including `noexcept`, allocation, ASCII handling, overflow and preservation
of the output value on failure. C++17 programs are compiled with `-O2`;
the original parser also uses `-Wall -Wextra -Werror`. Compilation has a
60-second wall limit and execution a 20-second limit, with a 3 GiB address
space limit. The grader rejects a deliberately incorrect parser in its self-test.

Generated code executes locally in Bubblewrap user, process, filesystem and
network namespaces, with read-only toolchain/oracles, a temporary `/tmp`, and
no model, home directory or network access. Raw generated outputs and test
diagnostics stay in immutable private snapshots. [quality.json](quality.json)
contains per-task outcomes, output/input/code hashes, truncation counts and
the paired comparison. Profiles were frozen before held-out generation;
there was no tuning in response to held-out scores.

The fast-profile gate requires at least 152/164 HumanEval passes, at most one
pass lost against this new greedy baseline, all four mandatory C++ tasks,
no new held-out C++ failures and a passing manual original-parser review.
The affinity/MTP profile fails the requirement for no new held-out C++
failures. It passes both HumanEval criteria, all four mandatory C++ cases
and the original parser's manual review. Use the greedy profile as the Q3
quality reference. HumanEval has four new failures and three new passes
under affinity/MTP; task identities are in `quality.json`.

The fast C++ failures are `parse_i64` and `parse_hex` (oracle failures), and
`replace_all` (two complete C++ code blocks rather than the required one).
The frozen extractor labels that last result an incomplete-code-block
failure; it was 385 tokens and did not hit the cap. No block was selected
or regraded after inspection.

The [earlier full-VRAM Q23 REAP224 run](../v100s_full_vram_20261016/README.md)
scored 149/164 HumanEval and 20/24 C++
without speculation, and 141/164 and 21/24 with affinity 0.08/MTP. Those are
context, not a quantization-only comparison: this run also changes 224 experts
to the original 288, retains the original dense formats and uses FP32 decode
exchange. No full-precision model quality comparison was run.

## Hardware, placement and memory

The v100s node has two Tesla V100 PCIe 32 GB GPUs and two Xeon Gold 6240 CPUs,
36 physical cores/72 logical CPUs, with four SNC NUMA nodes grouped into two
physical sockets. Each allocation requests 72 logical CPUs, 160 GiB RAM and
both GPUs for at most three hours. RAM speed was unavailable. Compiler,
driver, architecture and allocation details are recorded in
[validation.json](validation.json).

The public [resident copier](../../../tools/glm_resident_model.py) keeps one
sealed, locked shared RAM model per allocation, with a 140 GiB admission
budget. [socket-row-plan.json](socket-row-plan.json) binds 129 routed projection
tensors to this exact model hash and size. The two row halves are first-touched
on their owning physical sockets, interleaving within each socket's SNC nodes.
Ten sampled panels per resident copy were verified on the intended sockets.
The primary uses allowed logical CPUs 0-34 and the worker uses 35 on this node;
these masks must be adapted to the actual cpuset and PCI/NUMA topology elsewhere.
Use 34 CPU workers plus the caller, and keep the two-GPU split at one half.

Both GPUs use `STRATA_GLM_VRAM_PREALLOC=1` with a 512 MiB margin and 32,768 MiB
budgets. Requested decode caches are 30,720 MiB primary and 32,000 MiB worker.
The engine reserves 31,616 MiB pools per card; attention, staging and MTP
workspace reduce the bytes available for cached experts. Reservation is not
the number of useful cached weight bytes. [memory.json](memory.json) records
actual populated cache sizes, sampled NVML peaks, free memory and host memory
for every allocation.

Observed GPU peaks were 32,356 MiB used on GPU 0 (minimum 139 MiB free) and
32,484 MiB on GPU 1 (minimum 11 MiB free). NVML nominal memory is 32,768 MiB
per card; CUDA exposed about 32,495 MiB usable. Peak sampled cgroup usage was
152.48 GiB, including reclaimable file cache; peak owner RSS was 137.46 GiB.
No process swap was observed. The recorded cgroup counter has its own
reported limit; 160 GiB is the requested Slurm allocation. These checks
cover the recorded quality requests and 4K preflights, with inputs up to
4,096 tokens.

Generation resumes from completed samples across allocation expiry. Model
identity and placement are reverified on every allocation; caches restart.
[generation.json](generation.json) records normalized allocation labels and
the engine starts with hashes of their raw invocation records. The only
operational owner change during generation adds post-quality RAM retention
for release validation. Inference binaries, runner, inputs and profile
settings remain frozen throughout the quality suites.

## Reproduction and validation scope

The [greedy Q3 API profile](../../../configs/experimental/glm53f-ud-q3-k-xl-v100s-greedy.json)
and [affinity 0.08/MTP3 API profile](../../../configs/experimental/glm53f-ud-q3-k-xl-v100s-aff008-mtp3.json)
are explicit experiments. They do not change installer or server defaults.
Their functional environments and positional arguments match the measured
quality profiles; only model and binary paths are normalized. Both passed
configuration checks with mocked weights and engine. Use the recorded
quality gate when choosing a profile.

The measured CUDA build uses these options:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DSTRATA_ENABLE_HIP=OFF -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build --target strata-glm-decode strata-glm-gpu-worker -j12
```

The shard join preserves the first shard's raw metadata entries except `split.*`,
writes tensor descriptors in shard/offset order, aligns the header to 32 bytes,
and concatenates the original aligned payload regions without changing them.
The frozen placement plan requires the exact merged hash in `model.json`.
Connection settings and the orchestration used to access storage are private.

Run inside a suitable allocation after placing the merged weights under
`models/`. Activate the project Python environment and use this shared-owner
recipe from the repository root; set `profile` to the selected experimental
configuration:

```python
import json, os, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, 'tools')
from glm_resident_model import ResidentModel

profile = 'configs/experimental/glm53f-ud-q3-k-xl-v100s-greedy.json'
cfg = json.loads(Path(profile).read_text())
plan = json.loads(Path(cfg['socket_row_plan']).read_text())
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
Path('tmp').mkdir(exist_ok=True)
with ResidentModel(cfg['model'], cfg['model_sha256'],
                   max_bytes=cfg['resident_max_bytes'], socket_row_plan=plan) as resident:
    cfg['model'] = str(resident.path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir='tmp') as f:
        json.dump(cfg, f)
        f.flush()
        subprocess.run([sys.executable, '-m', 'serve.server', '--engine', 'glm',
                        '--config', f.name], env=env, check=True)
```

Start the recipe with NUMA interleaving for memory outside the socket row plan.
Keep `lock_weights=false`: the owner pins shared pages, while the decoder's
private locking mode would create another model copy. The profile uses the
public mixed decoder adapter and binds the server to `127.0.0.1`.

The engine change raises the explicitly requested mapped/shared NUMA weight
budget ceiling to 1 TiB; copied arenas retain their 128 GiB ceiling and default
budgets are unchanged. It also admits the existing tensor-prefill dequantizers
for supported retained formats, and links the optional CUDA K-quant library
to the canonical C11 environment helpers. Three ordinary inference runs
(previous decoder, measured decoder with K-quant enabled, measured decoder
without that optional switch) produce identical 16 tokens and complete
prefill/final 154,880-float logit arrays for a frozen 64-token input. A separate
local download and byte comparison verifies these arrays independently.

All 376 quality samples ran the measured `bb9295c9` decoder. A subsequent
startup-only guard requires compiled base prefill helpers for the newly
admitted tensor path. CUDA `mmq::built()` is true, so this guard does not
change its arithmetic. The final source compiled for CUDA sm80 and HIP
gfx1201. The separate final CUDA sm70 binary passed ordinary
64-input/16-output token and full-logit parity on the V100, plus both
4K/128-token full-memory training probes using the unchanged measured GPU
worker. Both probes matched all 128 frozen tokens. The full quality suites
were not rerun on that startup-guard binary. Exact source, compiler and
binary identities are in `validation.json`. The C11 names, environment and
expert-pack tests also passed.

All inference setup repairs occurred before held-out generation and remain
recorded in [validation.json](validation.json): filesystem page invalidation
on an older disk-backed reference, prompt delimiter validation, retained Q6_K
prefill admission, an optional library's missing C11 symbol, a stale
pruning-only flag, and an insufficient manual attention scratch cap. None
is a qualified numerical or quality result. The successful profiles use
the engine's computed attention workspace rather than the failed manual cap.
The local HumanEval grading launcher initially omitted its required dataset
argument and exited before grading. It was corrected to supply the pinned
snapshot dataset; generated samples, graders and oracles were unchanged.
The post-quality memory-probe launcher initially selected the system Python
and failed before GPU inference. Its successful ordinary parity receipt and
raw failure were preserved; a project-Python wrapper resumed the same binary
and sealed model for the probes.

HIP GPU execution, Windows execution, a GPU HTTP frontend smoke test and
full-precision quality were not tested. SYCL does not build this GLM decoder.
Quality generation wall times include prefill and cache initialization and
are not decode throughput measurements. The previous 50 tok/s decode and
700+ tok/s prefill targets have not been requalified on these Q3 weights.

The experiment finished with zero queued or running jobs for its pinned
session name and all three session/experiment/post-validation controllers
stopped. The final shutdown receipt and immutable raw-snapshot archive hash
are recorded in `validation.json`. Connection settings, raw logs, generated
code, datasets, weights and binaries are excluded from this report.
