# ucomm

A small C11 communication library for running LLM inference across 2–4 machines on a local network
(up to 8 ranks). It has two backends:

- **ib**: InfiniBand / RoCE verbs, aimed at used ConnectX-3 (QDR, mlx4) cards. `libibverbs.so.1` is loaded at
  run time with `dlopen`/`dlsym` (the cuew approach), so building needs no verbs headers or libraries, and the
  binary runs on machines without them.
- **udp**: reliable UDP for machines without an HCA. Picked automatically when any rank lacks an active HCA port.

The only build dependency is libc (`-ldl` on older glibc).

## Build and test

```sh
make                 # build/libucomm.a, build/libucomm.so, build/test_loopback, build/ucomm_bench
make test            # 2, 3 and 4 ranks on 127.0.0.1, plus a run with 5% induced packet loss
```

CMake works too (`add_subdirectory(tools/ucomm)`, target `ucomm`).

## API

```c
#include "ucomm.h"

ucomm_config cfg;
ucomm_config_default(&cfg);          /* reads UCOMM_RANK / UCOMM_WORLD / UCOMM_ROOT / ... */
cfg.rank = rank; cfg.world = 2; cfg.root_addr = "192.168.100.22:29500";
ucomm_t *c;
ucomm_init(&cfg, &c);                /* rank 0 listens on root_addr for a short TCP bootstrap */

ucomm_send(c, peer, tag, buf, len);  /* blocking */
ucomm_recv(c, peer, tag, buf, cap, &nbytes);

ucomm_req_t *r[2];                   /* nonblocking; wait/test release the request */
ucomm_irecv(c, left,  tag, rbuf, n, &r[0]);
ucomm_isend(c, right, tag, sbuf, n, &r[1]);
ucomm_waitall(2, r);

ucomm_allreduce_sum(c, x, count, UCOMM_BF16);  /* f32 / f16 / bf16 / i32, in place, ring algorithm */
ucomm_allgather(c, mine, all, bytes_per_rank);
ucomm_bcast(c, buf, len, root);
ucomm_barrier(c);

ucomm_mr_t *mr; ucomm_mr_reg(c, big_buf, size, &mr);  /* IB: register long-lived buffers once */
ucomm_finalize(c);                   /* drains traffic, then a barrier */
```

Rules: one context per process, driven from one thread; messages between two ranks with the same tag match
in order; user tags must be below `UCOMM_TAG_RESERVED`; a send buffer must stay unchanged until its request
completes; a too-small receive buffer gives `UCOMM_ERR_TRUNC` with the full size in `nbytes`.

## Environment

| Variable | Meaning |
|---|---|
| `UCOMM_RANK`, `UCOMM_WORLD`, `UCOMM_ROOT` | defaults for the config fields |
| `UCOMM_BACKEND` | `auto` (default), `ib`, `udp` |
| `UCOMM_IB_DEV`, `UCOMM_IB_PORT`, `UCOMM_IB_GID_INDEX` | HCA choice; GID index only matters for RoCE |
| `UCOMM_IBVERBS_LIB` | path to libibverbs if it is not `libibverbs.so.1` on the loader path |
| `UCOMM_UDP_MTU` | UDP payload per datagram (default 1400; use ~8900 on a jumbo-frame network) |
| `UCOMM_UDP_DROP` | test hook: drop this fraction of outgoing datagrams |
| `UCOMM_TIMEOUT_MS` | bootstrap and wait timeout (default 60000, <=0 waits forever) |
| `UCOMM_VERBOSE` | log backend choice, peers and retransmit counts |

## How it works

- **Bootstrap** (`ucomm_boot.c`): TCP star around rank 0. Ranks exchange which backends they have, then the
  backend's addresses (UDP ip:port, or IB LID/GID/QP numbers/credit-word rkeys). It is also the finalize
  barrier.
- **udp** (`ucomm_udp.c`): messages are cut into fragments with a per-peer sequence number. The receiver
  delivers in order and answers with a cumulative ack plus a 64-bit SACK bitmap (piggybacked on data when
  there is reverse traffic). The sender keeps an AIMD window capped by the receiver's socket buffer,
  retransmits on RTO (from a smoothed RTT) and on SACK holes. Sends up to 16 KiB are copied and complete
  at once; larger ones are sent straight from the user buffer and complete when fully acknowledged.
- **ib** (`ucomm_ib.c`, `ibvew.h`): one RC QP per peer and one CQ, busy-polled. Messages up to 8 KiB go
  eagerly into 64 pre-posted receive slots; larger ones send an RTS and the receiver RDMA-READs straight
  into the user buffer in 1 MiB chunks, then sends FIN. The receiver RDMA-WRITEs its count of reposted slots
  into a credit word on the sender, so there are never receiver-not-ready retries. `ibvew.h` is a hand-written
  copy of the verbs ABI subset used; its struct offsets were checked against rdma-core's `verbs.h`
  (Ubuntu 24.04 `libibverbs-dev`), and the loader was checked against that release's `libibverbs.so.1`.

## Measured

UDP backend, tr16 (rank 0) ↔ b550 (rank 1), 1 GbE, MTU 1500, default `UCOMM_UDP_MTU=1400`, `ucomm_bench`.
`ping` round trip on this link is 0.28 ms.

| Test | Result |
|---|---|
| latency, 8 B / 1 KiB / 8 KiB (half round trip) | 110 / 122 / 218 µs |
| bandwidth, 1 KiB messages | 100 MB/s |
| bandwidth, 256 KiB – 64 MiB messages | 109–113 MB/s (1 GbE line rate is ~117 MB/s of UDP payload) |
| allreduce f32, 16 KiB, 2 ranks | 444 µs |
| allreduce f32, 16 MiB, 2 ranks | 240 ms |
| allreduce f32, 16 KiB / 16 MiB, 4 ranks (2 per host) | 1.5 ms / 599 ms |

The **ib** backend builds and its probe and loader were tested (it falls back to udp when no HCA port is
active), but it has **not been run on an HCA yet**: neither machine has one. Validate it with `make test` and
`ucomm_bench -b ib` once ConnectX-3 cards are installed.
