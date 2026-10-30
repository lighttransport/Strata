# DeepSeek-V4.1-Flash on a GPU node + a CPU-only worker (the "reversed layout")

The decoder (`strata-glm-decode`, CUDA) runs on the node with the GPU (B550, RTX 5060 Ti 16 GB, 64 GB RAM); `strata-ds41-cpu-worker` runs on a node
with no GPU (TR16, Threadripper 1950X, 125 GB RAM) and owns every expert the GPU node does not keep. The nodes talk over InfiniBand (`ucomm`, IB port 1,
raw RDMA 3.94 GB/s each way). Whole experts are the unit (STD2, `include/strata/net/expert_tp.hpp`): the lanes of both nodes run the same canonical kernels,
so a route's output is bit-identical wherever it runs. Measurements, design notes and the history are in `docs/DS41F_PLAN.md` (sections 8l-8n).

Everything described here is opt-in (environment variables / worker options); the defaults are unchanged.

## Setup

```
# TR16:  cmake --build build-glm --target strata-ds41-cpu-worker strata-ds41-feed
# B550:  make -C <build> strata-glm-decode
experiment/ds41f_tr16_decode/rev_node.sh feed NAME      # NO_DSPARK=1 KEEP_MIB=34000: dense tensors + the hottest 34 GB of experts into the B550's /dev/shm (50 s over IB)
```

The sparse shards in `/dev/shm` hold the kept experts only. **Never read the others through the mapping**: a shared mapping of a tmpfs hole allocates its page, and
a stray full read fills `/dev/shm` (the decoder is then killed). Streamed prefill reads the owned experts from the worker instead (below); timing probes use `STRATA_DS41_STAGE_SKIP=1`.
Run the worker fresh for each decoder run: a persistent worker is ~30x slower from its second connection.

## Batched decode (`STRATA_DS41_SESSIONS=S`)

S independent sequences decode together, one token each per pass (plain decoding; no drafter). Preset: `experiment/ds41f_tr16_decode/rev_sessions_best.env`;
runners `cfg16.sh` (adaptive tier), `settle.sh` / `dual.sh` (static tier), `sess.sh` (base), `sweep.sh`. Aggregate tok/s at 48 sessions, 120-token prompts: 53-58
(3.4x the best single stream of 16.6, 4.9x plain single-stream decode; the 4x target of 66 is not met). Teacher-forced accuracy against decoding each session alone
(`STRATA_DS41_SESSION_CHECK=n STRATA_DS41_SESSION_TF=1`): top-1 97.4 %, mean KL 0.008.

| variable | meaning |
|---|---|
| `STRATA_DS41_SESSIONS`, `_SESSION_LEN` | session count; each session's prompt length (slices of the prompt) |
| `STRATA_DS41_CANON_TOKENS=N` | the canonical GPU tier takes passes of up to N tokens (default 8, max 64) |
| `STRATA_DS41_DENSE_MMQ=1 STRATA_MMQ_KQ=1` | dense Q6_K/Q5_K trunk matrices of a session pass through int8 MMQ |
| `STRATA_DS41_CANON_STREAM=q` (<= 32) `STRATA_DS41_CANON_PIN_MIB=a` `STRATA_DS41_CANON_PIN_OWNED_MIB=b` | a host-pinned second expert level (the next-hottest kept experts, and worker-owned ones fetched once over the fetch link) that the idle GPU streams over PCIe each pass; needs `STRATA_DS41_REMOTE_OWN_SKIP_MIB` = the kept MiB |
| `STRATA_DS41_CANON_TRAFFIC=trace`, `_CANON_SETTLE=trace` | rank the tier by a `STRATA_DS41_ROUTES` trace of the workload; seed the adaptive tier from it, swap once and freeze (a static tier) |
| `STRATA_DS41_DUAL=1` | a second decoder in the process sharing weights, tier and pinned level, ports +10 (worker: `--root2 --engram-port2`); measured no faster than one decoder: the worker is saturated |
| `STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_MERGE_VECTOR=1` | merge route outputs in place; spread over the lane threads for wide passes (`STRATA_DS41_MERGE_PAR=0` turns that off) |
| `STRATA_IQ1S_GRID8=1 STRATA_IQ1S_PAIR=1 STRATA_IQ1S_VIDX=1` | IQ1_S row kernels: pre-multiplied grid, paired gate/up, grid indices built in vector registers (bit-identical) |
| `STRATA_DS41_SYNC_SPIN=1` | spin on GPU syncs (the worker's request-to-reply time falls ~7 %) |
| `STRATA_DS41_LANE_STATS=1`, `_LANE_CPU_OFFSET=k` | per-thread barrier/work and merge timing at exit; start a second lane k cores in |

The sessions summary prints aggregate rates, routes per layer (GPU tier / worker / B550 lane), lane busy time and the per-layer worker timing.

## Prefill (GPU-streamed, the worker's experts over IB)

A prompt chunk of `STRATA_DS41_STREAM_MIN` (1024) tokens or more computes every routed expert on the GPU, streaming them through pinned host slots. On the reversed layout
the worker's experts come from `--stream-port`:

```
worker:  strata-ds41-cpu-worker ... --stream-port=29574 [--stream-cpu=N]      STRATA_DS41_STREAM_RING (24), STRATA_DS41_STREAM_COPIERS (4), STRATA_DS41_WORKER_THP=1
decoder: STRATA_DS41_OWN_STREAM=HOST:29574  STRATA_DS41_PROMPT_BATCH=12288 STRATA_DS41_PREFILL_HALVES=1 STRATA_DS41_PREFILL_TILE=2048 STRATA_DS41_ATTN_MMA=1
         STRATA_DS41_STAGE_HOST_SLOTS=24 [STRATA_DS41_OWN_STREAM_DEPTH=2] [STRATA_DS41_KV=f16 STRATA_DS41_INDEX_GEMM=1 STRATA_DS41_INDEX_GEMM_MIN=64]
```

`experiment/ds41f_tr16_decode/pf1.sh NAME ENV=VAL...` runs it end to end (a fresh worker; `STREAM=0` gives the CPU-lane prefill as the reference); `pf0.sh` times the GPU / PCIe ceiling without
the worker. Measured: 37 tok/s with the CPU lanes, 151 (4000-token prompt) and 317-344 (12,000-token prompt, one layer-major pass) with the stream; the generated tokens match the CPU-lane prefill.
The 700 tok/s goal is not met: GPU work is ~1.7-2.0 ms/token (expert MMQ, output projection, indexer, attention) and the 79.6 GB of owned experts per pass arrive at ~3.2 of the link's 3.94 GB/s.

* `STRATA_DS41_ATTN_MMA=1` is a FlashAttention-style prefill attention on tensor cores (FP16 Q / K / V / P, FP32 accumulation; relative error 3e-4 against the tiled FP32 kernel), 4.5-4.9x faster. It rounds the
  attention inputs to FP16; the KL of the next-token logits against the FP32 path has not been measured on real routing.
* A pass is limited to ~15.5K tokens by VRAM (`PREFILL_HALVES` keeps the attention-local buffers at the tile size).
* The worker sends each expert from a small registered staging ring: reading scattered experts out of the registered 74 GB of rows runs the HCA's translation cache dry (1.19 GB/s for 30 GB; 3.8 GB/s from a small region).

## Tests and benchmarks

| | |
|---|---|
| `python3 -m unittest test_sim_ds41f` (in `tools/`) | simulator tests, including the plain-sessions and CPU-worker prefill models (`glm_ds41f_sim.py rev-sessions`, `prefill-cpuworker`) |
| `iq_avx2_parity` (`build-glm`) | the IQ row kernels (incl. `STRATA_IQ1S_VIDX=1 STRATA_IQ1S_GRID8=1`) against the reference: 0 failures expected |
| `iq1s_w1a8_bench --bench` | row-kernel throughput; checks the vector-index kernel bit for bit against the baseline decode |
| `experiment/ds41f_tr16_decode/attnbench/attn_bench.cu` | the tensor-core attention against the tiled kernel (error + time; exit 1 above 2e-3); build line in its header |
| `experiment/ds41f_tr16_decode/ucbench/stream_pattern.c` | the expert-stream message pattern over `ucomm` alone (region size, THP, message size) |
