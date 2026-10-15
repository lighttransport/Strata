# Running and benchmarking the GLM-5.3-Flash runner on a new machine

This guide is for bringing the GLM runner up on hardware that is not one of the measured machines in
[GLM53F_RELEASE.md](GLM53F_RELEASE.md): what to measure about the machine first, which weights and
profile to start from, how to get a clean first run, how to tune, and how to benchmark and gate quality
so the numbers can be compared with the published ones. It assumes Linux, a CUDA or ROCm GPU, and a
build of `strata-glm-decode` (and `strata-glm-gpu-worker` for two cards) as described in the release notes.

## 1. Know the machine

The runner's speed is set by four numbers; write them down before tuning.

| what | how to get it | why it matters |
|---|---|---|
| RAM bandwidth per NUMA node | `tools/glm_memory_bandwidth.cpp` (build with `g++ -O2 -fopenmp`), or `numactl -H` for the node layout | the routed experts stream from RAM on every token; TR16 reads ~70 GB/s over two nodes, a single-node Ryzen with DDR4-2666 ~35 GB/s |
| VRAM free after the desktop | `nvidia-smi` / `rocm-smi --showmeminfo vram` with your normal desktop running | decides the dense cache, the prefill scratch and the GPU-resident expert tier; leave 1-2 GB free |
| PCIe link to the GPU | `nvidia-smi -q | grep -A2 "Link Width"`, `lspci -vv` | expert uploads and prefill streaming are link-bound: Gen3 x8 is ~7 GB/s, x16 ~14 GB/s |
| CPU cores and SMT | `lscpu` | the expert pool uses one thread per physical core minus one for the coordinator (15 on a 16-core part) |

A quick link and copy probe: `nvcc -O2 tools/glm_transfer_probe.cu -o probe && ./probe h2d` (or
`hipcc -DSTRATA_PROBE_HIP`), which prints GB/s for host-to-device copies and, with `register`, whether
the driver can pin shared mappings.

## 2. Pick weights and a starting profile

| RAM | weights (Hugging Face `lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts`) | start from |
|---|---|---|
| 128 GB | `GLM-5.3-Flash-Q23-assembled.gguf` (120.9 GB) or `reap224/...` (95.5 GB) | `configs/experimental/glm53f-q23-reap224-tr16-tuned.json` |
| 64 GB | `reap50/GLM-5.3-Flash-REAP50-Q23-assembled.gguf` (63.2 GB) | the B550 profile in [GLM_B550_REAP50.md](GLM_B550_REAP50.md); for an 8 GB card the RTX 3070 presets (zen5 branch) |
| 32 GB | nothing fits with acceptable speed ([GLM_32G_8G.md](GLM_32G_8G.md)) | |

The profile is a JSON file with the decoder arguments and an `env` block of `STRATA_GLM_*` switches. The
fields you change first:

| field | meaning | rule of thumb |
|---|---|---|
| `threads` | CPU expert threads | physical cores minus one |
| `cpu_affinity` | `numa` pins threads per node (two-node machines), `auto` on one node | |
| `gpu_budget_mib` | total VRAM the runner may use | free VRAM minus 1024-2048 MiB |
| `dense_cache_mib` | VRAM for the dense/attention weights | 4096 for the Q4 dense set; smaller cards stream them (`STRATA_GLM_PREFILL_DENSE_MIB`, zen5 branch) |
| `decode_cache_mib` | GPU-resident expert tier | what is left after dense cache and prefill scratch; more residents = faster decode |
| `prefill_batch` | prompt tokens per GPU pass | 8192 with 16 GB, 2048-4096 with 8 GB |
| `context` | token capacity | prompt plus the longest answer plus the draft window |
| `speculative`, `draft_depth`, `mtp_experts` | MTP drafting; `gpu` draft experts need `STRATA_GLM_MTP_GPU_CACHE=1` beside a decode cache | depth 3 on TR16, 1 on slower CPUs |
| `env.STRATA_GLM_ROUTE_AFFINITY` | lossy: prefer GPU-resident experts within this logit margin | 0 for bit-exact; 0.08-0.14 passed the gate on TR16, 0.20 did not |
| `env.STRATA_GLM_EXPERT_PRIOR` | expert prior file for the tier (`coverage-*.json` next to the weights) | always set it for the pruned models |

Lossy switches (affinity, soft MTP acceptance `STRATA_GLM_MTP_ACCEPT_*`, `STRATA_GLM_MIXED_ADAPT_CACHE=0`,
F16 attention in prefill) change outputs; everything else is byte-identical to the defaults. Keep a copy
of the strict profile (affinity 0, `MTP_ACCEPT_MARGIN=4`, `MTP_ACCEPT_TOP_K=6`) as your quality baseline.

## 3. First run

Run the decoder once with a short prompt to see the memory plan and catch a refusal early:

```sh
numactl --interleave=all build-glm/strata-glm-decode models/<model>.gguf 1,2,3 8 4096 15 \
  --prefill-batch=256 --gpu-budget-mib=<budget> --decode-cache-mib=<tier> --decode-graphs 2>&1 | grep -E "GPU|DECODE_CACHE|refus|requires|error"
```

Then the service with your profile:

```sh
python3 -m serve.server --engine glm --config my-profile.json
```

Memory discipline, which every published number used and which the benchmark tools enforce:

- no swap for the process: run under `systemd-run --user --scope -p MemorySwapMax=0` (or a cgroup with a
  memory limit below RAM, e.g. 56-60 GiB on a 64 GB box);
- never force kernel memory compaction while the model is mapped;
- a VRAM reserve: `STRATA_GLM_GPU_RESERVE_MIB=1024` and `STRATA_GLM_BUDGET_RESERVE_MIB=1024`;
- a quiet machine for timing: no builds, downloads or other GPU jobs (`tools/glm_q2_run_guard.py` checks
  for interference and refuses or records it).

## 4. Tune, one knob at a time

Order of expected return on a single GPU, each checked with a 3-trial run (section 5):

1. `decode_cache_mib` as large as VRAM allows (the tier learns the prompt's hot experts; `STRATA_GLM_TIER_ADAPT=1`,
   `STRATA_GLM_TIER_INFLIGHT=4` bounds the upload traffic per step).
2. `threads` and `cpu_affinity` (SMT threads do not help; one node vs two matters).
3. `prefill_batch` up to what the prefill scratch allows (`PREFILL_SCRATCH ... peak_MiB` in the log).
4. MTP depth 1 → 3 (`mtp_experts: gpu` with `STRATA_GLM_MTP_GPU_CACHE=1` when VRAM allows the draft bank).
5. Only then the lossy switches, each followed by the quality gate.

For two GPUs see section 7.

## 5. Benchmark protocol

Use the fixture prompts so results compare with the tables: `docs/fixtures/glm_tr16_20261009/prompts/{2k,8k}/`
(`holdout-code.ids`, `holdout-chat.ids`, `holdout-doc.ids`; the `train-*` prompts were used for tuning).
Report prompt length, generated tokens, trials and the medians; one trial is not a measurement.

Single GPU (guarded, warm-up excluded, three timed trials of 512 tokens):

```sh
python3 tools/glm_tr16_calibrate.py --config my-profile.json --decoder build-glm/strata-glm-decode \
  --prompt docs/fixtures/glm_tr16_20261009/prompts/8k/holdout-code.ids --output runs/mybox-8k \
  --cases baseline --tokens 512 --trials 3
```

It writes `measurement.json` with prefill and decode tok/s per trial, page faults, swap, VRAM minimum and the
hashes of the binary, profile and prompt. Run the 2K prompt as well; long-context decode is usually slower
(MLA attention and the larger KV state).

What the log lines mean: `PREFILL trial=.. tok_s=` is the prompt rate (first trial warm, loading excluded);
`SPECULATIVE ... accepted=.. tok_s=` is decode with the draft acceptance; `STEP_TRACE` (with
`STRATA_GLM_STEP_TRACE=1`) splits a step into head, CPU service, exposed GPU wait and tail, which tells
you whether the CPU lane or the GPU chain is the bottleneck; `tier_swaps` counts expert promotions.

Stream health: `tools/glm_mixed_bench.py` (also usable with one GPU, `--cases cuda` or `hip`) reports the
distinct 4-gram fraction and repeat-guard counts of every generated stream; a looping output is a quality
failure even when the speed is good.

## 6. Quality gate

Speed with a lossy switch is only reportable with the gate it passed. The agreed gate for the published
TR16 numbers:

1. **Corpus**: `tools/glm_tr16_quality.py --config <strict profile> --output runs/q-baseline` once, then
   `--config <candidate> --baseline runs/q-baseline/result.json --quality-gate tr16-35`: perplexity within
   +6%, KL within +0.08, top-1 agreement within 3 pp of the strict run on the 35-track corpus. About 10
   minutes on TR16.
2. **HumanEval**: `tools/glm_q2_humaneval.py <model> HumanEval.jsonl.gz --output runs/he --decoder ... `
   with the profile's settings; the floor is 152/164. About two hours. The `--limit 30` subset is a smoke
   test, not a score.
3. **Loops**: the 8K stream-health numbers above on a 512-token run.

Bit-exact changes (new kernels, scheduling) need none of this: compare generated token streams and
`--dump-logits` hashes before and after.

## 7. Two GPUs

The mixed runner is one primary process (the model) plus one expert-worker process on the other card,
joined by a shared memfd and a socket; `tools/glm_mixed_run.py` launches both and
`tools/glm_mixed_bench.py` benchmarks them:

```sh
python3 tools/glm_mixed_bench.py --config my-two-gpu-profile.json --cuda-build build-glm --hip-build build-hip \
  --model models/<model>.gguf --prior models/<coverage>.json --combine device \
  --cases cuda-hip hip-cuda --fractions 0.5 --tokens 512 --trials 3 --repetitions 1 \
  --prompts docs/fixtures/glm_tr16_20261009/prompts/8k/holdout-code.ids --output runs/mybox-two-gpu
```

Decide the orientation from the measurements (on TR16 the CUDA primary decodes faster, the HIP primary
prefills faster), then the prefill expert split with `tools/glm_mixed_fraction_calibrate.py --case hip-cuda
--groups <experts/16> ...`, which hill-climbs over whole expert groups. The worker cache (`--worker-cache-mib`)
should take most of the second card. Start from `configs/experimental/glm53f-reap224-tr16-two-gpu-decode.json`
and `-prefill.json`; the switches that mattered on TR16 are explained in
[GLM_TR16_RUNNER_PLAN.md](GLM_TR16_RUNNER_PLAN.md) sections 12-15. On an NVIDIA worker, registration of the
exchange buffer requires anonymous memory (the launcher uses a memfd); on a file-backed mapping CUDA
refuses it.

Before building anything new for a pair of cards, the simulator can rank the levers from a calibrated
kernel table: [GLM_RUNNER_SIM.md](GLM_RUNNER_SIM.md). Its `probe-plan` lists the measurements a new
machine needs (per-slot kernel decomposition, worker round trip, link bandwidth) and `sweep` prints the
expected path. Treat its forecasts as an ordering, not a result.

## 8. Record what you measured

Keep for every reported number: the profile JSON, the binary and source hashes (the tools record them), the
prompt and its hash, trials and medians, the memory guards in force, and the quality gate result or "not
gated". The published records follow `docs/benchmarks/*.json`; a short doc with the table and the
limitations (what was not measured) makes the result usable by the next person.
