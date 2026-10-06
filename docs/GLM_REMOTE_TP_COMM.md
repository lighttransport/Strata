# GLM remote expert TP: communication cost

This note covers the network cost of cross-node expert tensor parallelism in `strata-glm-decode`
(`--remote-tp=HOST:PORT`, worker `strata-glm-tp-worker`, transport `tools/ucomm`). It measures the
communication only; compute balance between the nodes is a separate question.

## What crosses the network

Every CPU-routed expert is split by FFN rows. The decoder keeps rows `[0, split)` and the matching down
columns; the worker computes the rest. Per MoE layer and decode token there is one round trip:

| Direction | Contents | Bytes (hidden 4096, top-8) |
|---|---|---|
| request | header, 8 × (expert, token, weight), activation quantized to the gate/up dot type | ~4.8 KB |
| reply | per token: fp32 scale + 4096 fp16 partial sums (`--remote-tp-reply=f16`, the default) | 8,196 B |
| reply, fp32 | 4096 fp32 partial sums (`--remote-tp-reply=f32`) | 16,384 B |

GLM-5.3-Flash has 42 MoE layers, so a decode token makes **42 sequential round trips** of about 12.8 KB.
They cannot be pipelined across layers: each layer's input depends on the previous layer's output.

## Requirement for a given decode rate

| Target | Time per token | Largest round trip if comm used the whole token | Bandwidth needed |
|---|---|---|---|
| 10 tok/s | 100 ms | 2.38 ms | 5.4 MB/s |
| 20 tok/s | 50 ms | 1.19 ms | 10.8 MB/s |
| 50 tok/s | 20 ms | 0.48 ms | 27 MB/s |

Bandwidth is never the limit; the count of sequential round trips is. A practical target is to keep comm
near 10% of the token time: about **120 µs per round trip at 20 tok/s**.

## Measured (1 GbE)

tr16 (Threadripper 1950X) ↔ b550 (Ryzen 9 3950X), 1 GbE, MTU 1500, UDP backend, `UCOMM_UDP_MTU=1400`.
b550 was running other heavy jobs (load average ~37) during all runs. `ping` round trip: 0.28 ms.

Round trips in isolation (`ucomm_bench -p REQ:REP -n 3000`):

| Pattern | Mean | p50 | p99 |
|---|---|---|---|
| 64 B → 64 B | 222 µs | 219 µs | 304 µs |
| 4800 B → 8192 B (f16 reply) | 401 µs | 400 µs | 460 µs |
| 4800 B → 16384 B (f32 reply) | 484 µs | 485 µs | 551 µs |
| 64 B → 64 B, `UCOMM_SPIN_US=2000` | 202 µs | 201 µs | 253 µs |
| 4800 B → 8192 B, `UCOMM_SPIN_US=2000` | 388 µs | 381 µs | 485 µs |

About 200 µs of every round trip is a fixed floor (NIC interrupt handling and wakeups on the busy b550). The
rest is wire time: 12.8 KB takes ~103 µs at 1 Gb/s. Busy-polling removes only ~20 µs.

In the decoder (8-token prompt, 63 decode steps, worker `--null-compute` so only communication is measured,
3 trials each):

| Reply | Exposed wait per layer | Exposed per token (×42) |
|---|---|---|
| f32 | 0.097 ms | 4.1 ms |
| f16 | 0.080 ms | 3.4 ms |
| f16, `UCOMM_SPIN_US=300` | 0.069 ms | 2.9 ms |

The exposed wait is much smaller than the round trip because the request is sent before the decoder starts
its own half of the experts (~1.7 ms per layer measured on tr16), so most of the round trip is hidden. That
holds only while the remote half plus the round trip finishes before the local half.

## Measured (InfiniBand)

tr16 ↔ b550 over Mellanox ConnectX-3 (mlx4, one port each, subnet manager on b550; IPoIB addresses 10.10.10.2 and
10.10.10.1 for the bootstrap), `UCOMM_BACKEND=ib` (verbs, RC queue pairs), 2026-10-06, while b550 also ran other builds.

| Measurement (`ucomm_bench -b ib`) | Result |
|---|---|
| latency, 8 B / 1 KiB / 8 KiB (half round trip) | 3.5 / 4.7 / 13.1 µs |
| bandwidth, 64 KiB / 1 MiB / 64 MiB messages | 1.55 / 1.65 / 1.65 GB/s |
| allreduce f32, 16 MiB, 2 ranks | 16.0 ms (1.05 GB/s algbw) |
| round trip 64 B → 64 B (`-p 64:64 -n 20000`) | mean 7.3 µs, p50 6.8, p99 16.5 |
| round trip 4800 B → 8192 B (f16 reply) | mean 20.9 µs, p50 20.1, p99 31.4 |
| round trip 4800 B → 16384 B (f32 reply) | mean 26.5 µs, p50 25.6, p99 39.2 |

The decode-shaped round trip is 19× shorter than on 1 GbE (20.9 vs 401 µs): 42 layers cost about 0.9 ms per
token even with no overlap. The f32 reply costs 5.6 µs more per round trip, so on IB it is a cheap way to remove
the fp16 rounding. `ib_write_lat` (perftest) reports 1.35 µs typical one-way on the same link, so most of ucomm's
latency is its own protocol and polling, not the wire.

## Two-node runner with real weights

The worker no longer needs dummy weights or a model file. At startup the decoder streams the worker's rows of every
expert over the link (`--remote-tp-weights=send`, the default); the worker host needs no persistent disk at all
when the worker binary is also copied to its tmpfs (`SHIP_WORKER`, below).

- **Memory check first.** After the setup the worker answers with what its rows need and what it has
  (MemAvailable minus `--reserve-gib`, default 2, and with `--weights-dir` that directory's free space). If they do
  not fit, the decoder stops before any bytes move and names the `--remote-tp-share` that would fit.
- **Where the rows live.** By default in the worker's anonymous memory, gone when it exits. With
  `--weights-dir=/dev/shm` they go to a tmpfs file named after the decoder's fingerprint of the rows (the pack's
  payload checksums when present, the geometry and 16 sampled 4 KiB blocks per tensor, the split). The next worker
  given the same rows maps the file (populated, so the first decode does not fault it in) and the decoder sends
  nothing (`weights=kept`). One file is kept per directory; a different setup replaces it.
- **Local rows.** The decoder keeps gate/up rows `[0, keep)` and down columns `[0, keep)` of every expert in
  node-owned memory, as if the expert were `keep` wide; GPU uploads (prefill, the GPU tier) assemble whole experts
  from those rows and the worker's part read from the file.
- **Split granularity.** `keep` is a multiple of 512 rows (owned rows are split across the two NUMA nodes in 256-row
  chunks), so with GLM's 2048-row experts the local share is 0.25, 0.5 or 0.75.
- **Step pipeline.** The mailbox path (`STRATA_GLM_STEP_PIPELINE=1`, MTP, split verify, the GPU tier) carries the
  remote work: the request leaves before the local rows start, the reply is added to the local sum.
- **Buffers registered once.** ucomm registers a rendezvous buffer per message unless it lies in a registered region;
  on ConnectX-3 a registration is a firmware command. Registering the request and reply buffers once on both sides
  cut the decoder's wait per layer from 0.198 ms to 0.023 ms. The transfer registers the worker's rows one layer
  at a time (the memlock limit is ~8 GB on b550).

```sh
# tr16 (decoder) + b550 (worker) over InfiniBand; the worker binary is copied to b550's /dev/shm
SHIP_WORKER=build-glm/strata-glm-tp-worker WORKER_ENV=STRATA_POOL_SPIN_US=20000 \
tools/glm_remote_tp.sh b550 /dev/shm/strata-glm-tp-worker 10.10.10.2:29611 --weights-dir=/dev/shm -- \
  build-glm/strata-glm-decode MODEL @prompt.ids 256 4096 15 --expert-pack=PACK --cpu-affinity=numa \
  --decode-graphs --remote-tp-share=0.75 --remote-tp-reply=f32
# serving: the config's "env" takes STRATA_GLM_REMOTE_TP=10.10.10.2:PORT, STRATA_GLM_REMOTE_TP_SHARE,
# STRATA_GLM_REMOTE_TP_REPLY; start the worker on the other host separately.
```

### Measured (REAP-50 Q2_K/Q3_K pack)

Correctness, teacher-forced on 3 sequences of the evaluation corpus (1,536 predictions), against the single-node
logits:

| Run | KL | top-1 agreement | ppl |
|---|---|---|---|
| single node again (control) | 0 | 100% | 4.156 |
| two nodes, 75% local, f32 reply | 0.0187 | 95.2% | 4.136 |
| two nodes, 75% local, f16 reply | 0.0226 | 94.7% | 4.161 |

`STRATA_GLM_REMOTE_TP_CHECK=1` recomputes every split layer's experts whole from the file and compares: the largest
difference was 4.0e-7 of the layer's largest value over 2,646 layer calls (all 42 layers). The split is exact up to
float summation order. The KL is the model's sensitivity to that order: a 2-bit model with int8 activations turns a
1e-7 change into flipped roundings that propagate. The f16 reply's ~1000× larger error raises the KL only to
0.0226.

Transfer: 12.8 GiB (25% of each expert) in 15.2 s into anonymous memory and 19.0 s into /dev/shm; 25.7 GiB (50%)
in 47.6 s. The decoder reads the worker's part from the pack on disk (down columns touch every page of each down
tensor), which bounds it well below the link's 1.65 GB/s. A kept run sends nothing.

Decode speed, prime fixture (53-token prompt, up to 256 tokens, 3 trials, first trial included), tr16 decoder with
15 threads + b550 worker with 16 threads, `--remote-tp-reply=f32`, worker `STRATA_POOL_SPIN_US=20000`.
Pairs ran back to back so both sides saw the same machine; other users' jobs kept tr16's load average at 6-13
during this set, which costs the single-node runs more (they use all of tr16's cores for experts):

| Mode | single node | two nodes, 75% local |
|---|---|---|
| ordinary | 10.48-10.90 | 15.98-16.16 |
| MTP depth 2 + split verify | 14.06-14.66 | 15.68-16.15 |
| GPU tier 3584 MiB (240/241 slots), ordinary | 11.71-12.63 | 16.97-17.13 |
| GPU tier (251/301 slots) + MTP 2 + split verify | 16.06-16.34 | 15.79-16.12 |

Earlier unpaired runs (machine load not recorded), before the buffer registration fix (wait 0.2 ms per layer), trials
2-3: MTP 2 + split verify 15.00-15.10 single vs 18.17-18.28 two nodes; GPU tier + MTP 2 + split 18.57-18.74 vs
19.35-19.36. The GPU
tier's slot count depends on the GPU memory free at startup (other GPU users), so tier rows compare loosely.

Per layer (two nodes, GPU tier, ordinary): the local 75% takes ~0.75 ms, b550's 25% ~0.73 ms of compute, and the
decoder waits 0.025 ms after its own part. b550's expert rows run at ~26 GB/s whatever the thread count (8-24
threads; memory bound), so a 50% split makes b550 the long pole (1.50 ms per request, 9.4-9.6 tok/s ordinary).

Prefill is the weak spot: a GPU upload assembles whole experts from the local rows and the worker's part read from
the pack, so the first prefill reads that part from disk (53 tokens: 40.7 s vs 9.3 s single node). Later prefills
find it in the page cache; the decode path never touches it.

Not covered yet: the split granularity (512 rows) allows no share between 0.5 and 0.75 (a 256-row step needs the
fused hidden quantization off); the worker computes in phases (no layer dataflow); no prefill work is sent to the
worker.

## Comm-only ceilings

| Link | Round trip per layer | Comm per token | Share of 50 ms | Comm-only ceiling |
|---|---|---|---|---|
| 1 GbE, full round trip (measured) | 0.40 ms | 16.8 ms | 34% | ~60 tok/s |
| 1 GbE, exposed after overlap (measured) | 0.069 ms | 2.9 ms | 6% | ~345 tok/s |
| 2.5 GbE, same latency floor (estimate) | ~0.34 ms | ~14 ms | ~29% | ~70 tok/s |
| 2.5 GbE, tuned NIC (estimate) | ~0.10 ms | ~4 ms | ~8% | ~240 tok/s |
| 10 GbE (estimate) | ~0.05–0.08 ms | 2–3 ms | ~5% | ~400 tok/s |
| IB, ConnectX-3 (measured round trip, f16 reply) | 0.021 ms | 0.88 ms | 1.8% | ~1100 tok/s |

Rows marked estimate are not measured. 2.5 GbE only shortens the wire part (103 → 41 µs); the ~200 µs floor
stays unless the NIC is tuned (interrupt coalescing off, busy polling) or replaced by RDMA.

## Conclusions

- For communication alone, 1 GbE already supports 20 tok/s; 2.5 GbE adds headroom but no step change.
- The cost is latency × 42, not bandwidth. Lowering the per-round-trip floor matters more than link speed.
- The fp16 reply halves the reply bytes and cut the exposed wait by 18%; the busy-poll option by a further 14%.
- Once the worker computes for real, each layer costs max(local half, remote half + round trip). Balance the
  halves with `--remote-tp-share` so the round trip stays hidden. Measured with real weights (above): on IB a
  75/25 split between tr16 and b550 leaves a 0.025 ms wait per layer.

## Reproduce

```sh
# round trips (both hosts built from tools/ucomm)
b550$ build/ucomm_bench -r 1 -w 2 -a 192.168.100.22:29700 -p 4800:8192 -n 3000
tr16$ build/ucomm_bench -r 0 -w 2 -a 192.168.100.22:29700 -p 4800:8192 -n 3000

# decoder with communication only
b550$ ./strata-glm-tp-worker --root 192.168.100.22:29600 --pool-gib=1 --null-compute
tr16$ build-glm/strata-glm-decode MODEL 154822,154824,154828,154841,9703,220,108714,100461 64 4096 15 \
        --prefill-batch=2048 --context=8192 --gpu-budget-mib=12288 --decode-experts=cpu --decode-bench=3 \
        --cpu-affinity=auto --warm-weights --remote-tp=192.168.100.22:29600 \
        --remote-tp-weights=dummy   # dummy rows: nothing is streamed; prints REMOTE_TP wait_ms_per_call
```
