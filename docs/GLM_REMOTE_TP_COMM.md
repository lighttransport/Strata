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

## Comm-only ceilings

| Link | Round trip per layer | Comm per token | Share of 50 ms | Comm-only ceiling |
|---|---|---|---|---|
| 1 GbE, full round trip (measured) | 0.40 ms | 16.8 ms | 34% | ~60 tok/s |
| 1 GbE, exposed after overlap (measured) | 0.069 ms | 2.9 ms | 6% | ~345 tok/s |
| 2.5 GbE, same latency floor (estimate) | ~0.34 ms | ~14 ms | ~29% | ~70 tok/s |
| 2.5 GbE, tuned NIC (estimate) | ~0.10 ms | ~4 ms | ~8% | ~240 tok/s |
| 10 GbE (estimate) | ~0.05–0.08 ms | 2–3 ms | ~5% | ~400 tok/s |
| IB QDR, ConnectX-3 (estimate) | ~0.01 ms | ~0.4 ms | ~1% | — |

Rows marked estimate are not measured. 2.5 GbE only shortens the wire part (103 → 41 µs); the ~200 µs floor
stays unless the NIC is tuned (interrupt coalescing off, busy polling) or replaced by RDMA.

## Conclusions

- For communication alone, 1 GbE already supports 20 tok/s; 2.5 GbE adds headroom but no step change.
- The cost is latency × 42, not bandwidth. Lowering the per-round-trip floor matters more than link speed.
- The fp16 reply halves the reply bytes and cut the exposed wait by 18%; the busy-poll option by a further 14%.
- Once the worker computes for real, each layer costs max(local half, remote half + round trip). Balance the
  halves with `--remote-tp-share` so the round trip stays hidden.

## Reproduce

```sh
# round trips (both hosts built from tools/ucomm)
b550$ build/ucomm_bench -r 1 -w 2 -a 192.168.100.22:29700 -p 4800:8192 -n 3000
tr16$ build/ucomm_bench -r 0 -w 2 -a 192.168.100.22:29700 -p 4800:8192 -n 3000

# decoder with communication only
b550$ ./strata-glm-tp-worker --root 192.168.100.22:29600 --pool-gib=1 --null-compute
tr16$ build-glm/strata-glm-decode MODEL 154822,154824,154828,154841,9703,220,108714,100461 64 4096 15 \
        --prefill-batch=2048 --context=8192 --gpu-budget-mib=12288 --decode-experts=cpu --decode-bench=3 \
        --cpu-affinity=auto --warm-weights --remote-tp=192.168.100.22:29600   # prints REMOTE_TP wait_ms_per_call
```
