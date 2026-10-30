# Native FP4 across a CPU host and one CUDA GPU

This experimental, opt-in text generation path keeps the checkpoint's FP4 codes
and scales. A CPU host owns a fixed prefix of whole experts in every main MoE
layer. The decoder host owns the remaining experts, embeddings, fixed matrices
and the native MTP layer. Only the decoder's selected NVIDIA GPU is visible.
Vision and API serving are outside this launcher.

`tools/glm_fp4_layout.py` reads the checkpoint headers and chooses ownership from
both hosts' available memory. Each host keeps at least 6 GiB outside its resident
weight partition. The launcher also leaves 1 GiB for worker startup metadata.
Admission includes cgroup limits; a plan that cannot fit fails.
A plan is a capacity check, not evidence of successful inference or throughput.

With `gpu_fixed: true`, compressed sources for fixed matrices retained on the GPU
use disk-backed files on the decoder host. Their clean pages are released after
synchronized GPU upload. Routed experts and embeddings remain in sealed RAM
files; the MTP expert layer stays on the decoder host. This avoids keeping a
second host-RAM copy of GPU-resident fixed weights. The remote parent must be on
a disk filesystem with sufficient free space, not tmpfs. The CPU worker must
have enough memlock allowance to lock its partition. The decoder uses locked
weights when its allowance permits; otherwise it requires a host with no swap
so sealed RAM weights cannot be paged out. The launcher checks these conditions
before transferring weights.

## Configure and run

Build `strata-glm-fp4-cpu-worker` on the CPU host and a CUDA
`strata-glm-decode` on the GPU host. Build on the destination when its libc is
older than the build host's. The decoder capability probe
`strata-glm-decode --fp4-distributed-capability` must print `1`.
Install `requirements-exl3.txt` in a Python environment on the launcher host.
Both hosts need Linux sealed memfd support and the configured ucomm transport.

Copy `configs/experimental/glm53f-fp4-two-node.example.json` into ignored
`priv/` and set the model, binaries, trusted SSH alias and IB endpoints there.
Do not commit connection settings. For startup SSH over IB, set
`ssh_data_address` and `ssh_host_key_alias` to the address and the already trusted
SSH host-key identity. Host-key verification remains enabled.

```sh
python tools/glm_fp4_two_node.py --config priv/glm-fp4.json \
  --output tmp/glm-fp4-plan --plan-only
python tools/glm_fp4_two_node.py --config priv/glm-fp4.json \
  --output tmp/glm-fp4-smoke --prompt 'Reply with just OK.' \
  --tokens 2 --no-mtp --check-remote
python tools/glm_fp4_two_node.py --config priv/glm-fp4.json \
  --output tmp/glm-fp4-mtp --prompt 'Explain virtual memory.' --tokens 32
```

Each output directory must be new. MTP is enabled unless `--no-mtp` is given;
acceptance uses the native exact verifier with zero acceptance margin.
`--check-remote` recomputes remote CPU outputs locally and requires bit equality;
it is expensive and is intended for validation. The dedicated profile disables
Q8 conversion, routing approximation and asynchronous weight modes. It selects
one idle RTX 5060 Ti by UUID and forces AVX2 CPU arithmetic on both hosts.
The experimental reserve settings apply only to this explicit profile.

Startup copies compact exact-byte partitions over SSH, verifies hashes with an
independent readback, and seals RAM files against modification. Temporary local
decoder staging is released before allocating the CPU worker's partition.
Runtime expert RPC and bounded prefill weight fetches use ucomm. Protocols check
ownership, sequence, dimensions and transfer bounds. Transport errors terminate
the run; they do not silently substitute local experts or approximate results.
The launcher cleans up only its own processes and staging. It never removes
older model staging or unrelated files to satisfy admission.

`decoder.stderr`, `worker.stderr` and numeric `decoder.stdout` retain run evidence;
`summary.json` is written only after decoder and worker shutdown succeed.
Keep these local until reviewed for private paths and connection details.
`--prompt-file` reads a UTF-8 prompt from a file. `--trials=4` restores the same
prefill state for four decode trials and requires identical output tokens. Treat
trial zero as warmup when reporting the remaining three timings. Phase fault
counters are enabled in the launcher profile.

Long-context quality, exact MTP agreement and sustained performance require
hardware runs; successful builds and synthetic protocol tests do not establish
those results.

## Reuse staging for validation

`--suite-file` accepts a JSON list of up to 16 requests. Each request has
`prompt` (or a `prompt_file` relative to the suite file), `tokens`, `mtp`,
`trials` and `check_remote`. For example:

```json
[
  {"prompt": "Explain virtual memory.", "tokens": 32, "mtp": false, "trials": 4},
  {"prompt": "Explain virtual memory.", "tokens": 32, "mtp": true, "trials": 4}
]
```

Each request starts a fresh decoder and worker connection while retaining the
same sealed weight partitions. Logs mark request boundaries. For matching plain/MTP requests, the plain baseline prepares the same draft
allocation using `--mtp-greedy-control` and then uses ordinary greedy decode.
This keeps CPU/GPU expert placement comparable: reserving draft memory otherwise
changes which experts run on the GPU. The launcher checks every output token
against prior matching requests and trial prefixes, and stops at the first
difference. A failure ends the batch and cleans up its staging. This does not reuse
KV state between different prompts. Use a dedicated machine for long batches;
startup admission cannot reserve RAM against unrelated jobs that start later.

For short decode tuning, the launcher accepts `cpu_compact_cache: true` in its
local configuration. This enables lossless borrowed CSF32 scales, per-row scale
decode and spinning CPU pools on both hosts. The CPU worker also prewarms its
bounded 2 GiB expert cache before reporting ready and retains it between sessions.
The option is off by default. It does not enable Q8 activations. Full-model speed
for this option is still under measurement.

The CPU component benchmark accepts `STRATA_GLM_FP4_BENCH_Q8=1` to compare
experimental Q8 activations with W4A16 on the same checkpoint experts. It reports
relative L2 and maximum absolute expert-output error for synthetic sinusoidal
inputs. These are component errors, not language-model quality measurements.
The worker protocol continues to use W4A16; Q8 is benchmark-only here.
