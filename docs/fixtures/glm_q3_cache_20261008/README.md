# Q3 cache study evidence

[Report](../../GLM_Q3_CACHE_STUDY.md).
`code` and `document` contain actual benchmark configs, prompt/output IDs,
384-token generation routes and raw measurements. No EOS/role-stop appeared in
either generation; the replayer validates 383 complete 42-layer decode steps.
`inventory.json` records each expert's three GGUF byte spans.

From the repository root, reproduce the replay without a GPU or model download:

```sh
python3 tools/test_glm_cache_replay.py
python3 tools/glm_cache_replay.py \
  docs/fixtures/glm_q3_cache_20261008/inventory.json \
  docs/fixtures/glm_q3_cache_20261008/code.routes.csv \
  --prompt docs/fixtures/glm_q3_cache_20261008/code.ids \
  --output-ids docs/fixtures/glm_q3_cache_20261008/code-run.stdout \
  --output /tmp/code-replay.json
# Substitute document for code for the second trace.
# For mixed replay, copy the fixture directory to a writable output directory:
cp -a docs/fixtures/glm_q3_cache_20261008 /tmp/q3-cache-replay
python3 /tmp/q3-cache-replay/mixed_replay.py /tmp/q3-cache-replay
```

To repeat inference on B550, stop other inference and ensure storage is not busy.
The configuration paths assume its existing ROCm build and model installation.

```sh
mkdir -p build-hip-glm-v11/q3-cache-study
cp docs/fixtures/glm_q3_cache_20261008/code.json build-hip-glm-v11/q3-cache-study/
cp docs/fixtures/glm_q3_cache_20261008/document.json build-hip-glm-v11/q3-cache-study/
.venv-glm-hip/bin/python docs/fixtures/glm_q3_cache_20261008/prepare.py
tools/glm_b550_60g.sh stop
.venv-glm-hip/bin/python tools/glm_low_memory_bench.py \
  build-hip-glm-v11/q3-cache-study/code.json \
  build-hip-glm-v11/q3-cache-study/code.ids \
  --output build-hip-glm-v11/q3-cache-study/code-repeat \
  --ram-gib 60 --tokens 384 --trials 1 --timeout 1200 --single \
  --gpu-capacity-mib 16304 --gpu-used-limit-mib 14256
# Repeat with the document config/prompt and a fresh output prefix.
# Existing routes.csv will be overwritten by a rerun; preserve original evidence.
```

The read-only SSD probe uses this exact model's offsets, not arbitrary files:

```sh
c++ -O2 -std=c++20 -pthread tools/glm_expert_read_bench.cpp \
  -o build-hip-glm-v11/q3-cache-study/expert-read-bench
cp docs/fixtures/glm_q3_cache_20261008/ssd-spans.txt build-hip-glm-v11/q3-cache-study/
# prepare.py above creates the inventory used by the runner.
.venv-glm-hip/bin/python docs/fixtures/glm_q3_cache_20261008/run_ssd.py
tools/glm_b550_60g.sh start
```

The original SSD probe ran under a 2 GiB cgroup with swap disabled and without
inference. It reads 384 experts (4.270 GiB aligned) per pass, with 1/4/8/16
workers twice. `ssd-plan.json` records the deterministic selection. Process
physical-read bytes equaled aligned requested bytes in every pass.

The raw `clean` flag is not a no-thrashing assertion; inspect memory-limit,
refault and major-fault counters. Replay misses are modeled, not actual reads
from an implemented exclusive cache. Static/hybrid preloading and LRU warmup
costs differ; see the report before comparing their evaluation-window numbers.
