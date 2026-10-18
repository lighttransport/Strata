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
cfg.rank = rank; cfg.world = 2; cfg.root_addr = "192.0.2.1:29500";
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

UDP backend, node-a (rank 0) ↔ node-b (rank 1), 1 GbE, MTU 1500, default `UCOMM_UDP_MTU=1400`, `ucomm_bench`.
`ping` round trip on this link is 0.28 ms.

| Test | Result |
|---|---|
| latency, 8 B / 1 KiB / 8 KiB (half round trip) | 110 / 122 / 218 µs |
| bandwidth, 1 KiB messages | 100 MB/s |
| bandwidth, 256 KiB – 64 MiB messages | 109–113 MB/s (1 GbE line rate is ~117 MB/s of UDP payload) |
| allreduce f32, 16 KiB, 2 ranks | 444 µs |
| allreduce f32, 16 MiB, 2 ranks | 240 ms |
| allreduce f32, 16 KiB / 16 MiB, 4 ranks (2 per host) | 1.5 ms / 599 ms |

The **ib** backend has since run on ConnectX-3 QDR between node-a and node-b; its numbers are in
`docs/GLM_REMOTE_TP_COMM.md` (and the file transfer rates below).

## File transfer: `ucomm_file` and `ucfile`

`include/ucomm_file.h` moves a byte range of an open file between two ranks of a context:

```c
ucomm_file_opts o; ucomm_file_opts_default(&o);   /* 16 MiB chunks, 4 in flight */
o.verify = UCOMM_FILE_VERIFY_CRC32C;                /* optional: per-chunk CRC32C, failed chunks re-sent (3 rounds) */
ucomm_file_send(c, peer, tag, fd, offset, length, &o, &stats);        /* tag and tag + 1 are used */
ucomm_file_recv(c, peer, tag, fd, 0, length, &o, have_bitmap, &stats); /* have_bitmap: chunks already held (resume) */
```

Chunks travel in registered buffers, so on **ib** the receiver RDMA-reads straight into them. There is no compression
on any path. The receiver sends its bitmap first, so a resumed transfer moves only the missing chunks. The CRC32C uses
the SSE4.2 instruction when available.

`ucfile` (`build/ucfile`, Python wrapper `python/ucfile.py`) runs jobs on top of it:

```sh
ucfile agent --root /data [--root /dev/shm/x] [--name N]   # on each node: UDP discovery + TCP control on port 29610
ucfile discover                                             # broadcast on every interface (Ethernet and IPoIB)
ucfile run job.json [--dry-run] [-v]                        # plan, preflight, transfer
ucfile df|ls|stat|rm|mkdir NODE [PATH...]                   # NODE: agent name or host[:port]
```

```json
{"self": "nodeA", "verify": "crc32c", "chunk_mib": 16, "backend": "auto", "reserve_gib": 4,
 "nodes": {"nodeA": {}, "nodeB": {"addr": "auto"}, "nodeC": {"addr": "192.0.2.3"}},
 "transfers": [
   {"src": "nodeA:/models/a.gguf", "dst": "nodeB:/data/a.gguf"},
   {"src": "nodeA:/models/a.gguf", "dst": "nodeC:/data/a.tail", "offset": 1048576, "length": 4194304},
   {"src": "nodeA:/models/b.gguf",
    "shard": {"to": ["nodeB", "nodeC"], "weights": [2, 1], "dst": "{node}:/data/b.{index}",
              "manifest": "/models/b.manifest.json"}}
 ]}
```

| Field | Meaning |
|---|---|
| `self` | the node this coordinator runs on (default: the hostname); it is rank 0 and the bootstrap root |
| `nodes.N.addr` | `auto` (discovery by name), `host` or `host:port` |
| `backend` | `auto` (ib when every rank has an active HCA port, else udp), `ib`, `udp` |
| `verify` | `none` or `crc32c` |
| `overwrite` | replace existing destinations (default false: the preflight refuses) |
| `reserve_gib` | free space left on each destination filesystem after the job (default 4) |
| `root_addr`, `data_port` | bootstrap address (default: the local address that routes to the agents), port (agent port + 1) |
| `discover`, `discover_ms` | extra discovery targets (`addr` or `addr:port`), wait |
| `chunk_mib`, `depth`, `timeout_ms`, `keep_mtime` | transfer tuning |

How it works:
- **Discovery:** agents answer a UDP probe with their name, addresses, whether an HCA port is active, and their roots'
  free space. A node that answers on several interfaces is reached through IPoIB.
- **Preflight:** the coordinator checks that each source exists, that no destination exists (unless `overwrite`), and
  that every destination filesystem holds its bytes plus `reserve_gib`. A resumable `.part` counts as allocated.
- **Execution:** the agents get the expanded job, and every node runs one ucomm rank over the same item list. A node
  that is both source and destination copies locally (`copy_file_range`).
- **Writes:**
  - Data goes to `DEST.part`, preallocated with `posix_fallocate`.
  - `DEST.part.ucstate` holds the chunk bitmap and is synced every 32 chunks.
  - A finished file is fsynced, renamed into place, and given the source's mtime.
  - A rerun of the same job resumes from the state when the source size, mtime and range are unchanged.
- **Agent jobs:** each job runs in its own process. If the coordinator disconnects, the agent stops the job at once and
  keeps its resume state.
- **Paths:** an agent touches only paths under its `--root` directories (no `..` escapes), and `rm` removes files or
  empty directories, never a root.
- **Shards:** contiguous 1 MiB-aligned ranges by weight. The coordinator writes the manifest (node, path, offset,
  length per shard) when every shard arrived. `ucfile.py reassemble MANIFEST OUT` joins local shards back.

Measured 2026-10-17, node-a → node-b, ConnectX-3 QDR, 1 GiB from tmpfs to tmpfs:

| Backend | Verify | GB/s |
|---|---|---:|
| ib | none | 2.44 |
| ib | crc32c | 1.65 (the CRC runs in the transfer loop on both sides) |
| udp over 1 GbE | none | 0.11 (line rate) |

To node-b's disk, which was 99% full and under load, ib ran at 0.12 GB/s: the destination disk sets the rate there.
`make test` adds `test_file`: whole ranges, CRC32C, resume and sub-ranges, also over udp with 2% induced loss.
