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
make test-relay      # relay/API/error-path tests without running the full suite
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

Rules: one rank per context, driven from one thread; independent contexts may coexist for relaying.
Messages between two ranks with the same tag match
in order; user tags must be below `UCOMM_TAG_RESERVED`; a send buffer must stay unchanged until its request
completes; a too-small receive buffer gives `UCOMM_ERR_TRUNC` with the full size in `nbytes`.

## Environment

| Variable | Meaning |
|---|---|
| `UCOMM_RANK`, `UCOMM_WORLD`, `UCOMM_ROOT` | defaults for the config fields |
| `UCOMM_BACKEND` | `auto` (default), `ib`, `udp` |
| `UCOMM_IB_DEV`, `UCOMM_IB_PORT`, `UCOMM_IB_GID_INDEX` | HCA choice; GID index only matters for RoCE |
| `UCOMM_IB_INLINE` | opt-in maximum inline SEND/credit bytes, including the 40-byte message header; 0 (default) disables it; 64 was tested on ConnectX-3 |
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

## Buffered verbs relay between separate fabrics

`include/ucomm_relay.h` adds an opt-in application relay. The middle host opens two independent two-rank
contexts, one per HCA port. Each endpoint is rank 1 of its own context; the relay is rank 0 of both.
Each fabric still needs its own subnet manager. No IB switch, IP forwarding, or cross-subnet IPoIB route
is needed for payload transport. TCP connectivity to both bootstrap ports is required; Ethernet may be used.

```text
left endpoint -- fabric A -- relay port 2 | registered RAM | relay port 1 -- fabric B -- right endpoint
```

The relay has a bounded ring of `min(depth, message_count)` slots in one buffer pool, registered once
with each context. A zero-message relay allocates no payload buffers. It receives a complete
message, forwards its actual length in receive order, and reuses that slot only when the outgoing send
completes. Both contexts are polled on every progress call, so simultaneous forward/reverse transfers
can progress. For large messages, each hop uses the existing rendezvous RDMA READ protocol: the payload
lands in relay memory and is read from that same memory on the second hop, without a CPU payload copy.
Small eager messages still use the existing copies. The CPU handles completions and posting work.

```c
#include "ucomm_relay.h"

/* a and b are independently initialized contexts, on different local ports. */
ucomm_relay_t *forward;
ucomm_relay_create(a, 1, source_tag, b, 1, destination_tag,
                   max_message_bytes, 3, message_count, &forward);
int done = 0;
while (!done) {
    /* Check the status, enforce a deadline, and progress a reverse relay here if used. */
    ucomm_status status = ucomm_relay_progress(forward, &done);
    if (status != UCOMM_OK) { /* terminate the communicating session */ break; }
}
/* Destroy only after successful completion (or before any progress call). */
if (done) ucomm_relay_destroy(forward);
```

Use a separate relay object per direction/tag stream. The caller supplies matching source/destination
peers, tags, maximum message size, and finite message count; this API does not discover arbitrary tags or
rewrite QPs, addresses, memory keys, reads, writes, or atomics. Existing applications must explicitly use
the relay protocol. `ucfile` and collectives are not automatically relayed by this addition.

Relay slots apply backpressure to buffer reuse. Keep endpoint send windows bounded as well: the existing
backend can queue unexpected eager messages outside this pool. The relay API itself has no timeout and
the caller must enforce a session deadline. On a transport or truncation error, end the session; there is
no pending-request cancellation API. Destroy refuses to release buffers with outstanding requests because
verbs may still reference them. An errored relay can be destroyed if it has no outstanding requests;
otherwise terminate the communicating process/session. Do not finalize either context while a relay using
it is active. `ucomm_test` now reports progress errors with `done=0`, preserving the request; `ucomm_waitall`
also preserves the current and remaining request pointers on a timeout or progress error.

### Build and exercise three hosts

Build the same sources on all hosts with `make -C tools/ucomm`. The standalone `build/ucomm_relay` tool
forces IB by default and never silently falls back to UDP. It first runs 8-byte ping-pongs (10 warmups,
1000 measured exchanges by default, independently controlled by `-l`), then a pipelined bandwidth phase
and a terminal acknowledgement from the destination. Pool allocation, registration and initialization
finish on all roles before the bandwidth timer starts. RTT output includes p50 and p99; `-J` emits JSON
lines for automation. Use matching `-l` values on all roles as well as the other protocol settings.

In three terminals, use the same roots, size, count, depth and verification options. Here `relay-host`
is a configured hostname reachable by both endpoints; replace it with your own bootstrap hostname.

On the middle host (left cable on port 2, right cable on port 1):

```sh
UCOMM_VERBOSE=1 tools/ucomm/build/ucomm_relay -r relay \
  -a relay-host:29681 -z relay-host:29682 -p 2 -q 1 \
  -s 1048576 -n 512 -d 3 -V
```

On the left endpoint (its cable on port 1):

```sh
UCOMM_VERBOSE=1 tools/ucomm/build/ucomm_relay -r left \
  -a relay-host:29681 -z relay-host:29682 -p 1 \
  -s 1048576 -n 512 -d 3 -V
```

On the right endpoint (its cable on port 1):

```sh
UCOMM_VERBOSE=1 tools/ucomm/build/ucomm_relay -r right \
  -a relay-host:29681 -z relay-host:29682 -p 1 \
  -s 1048576 -n 512 -d 3 -V
```

`-V` verifies every bulk payload byte and, for messages at least 8 bytes long, its sequence number.
Remove `-V` on all roles to time bulk transfer without the receiver's verification loop. Add `-R` on
all three roles to reverse bandwidth direction. The latency phase always originates on the left.
`-b udp` on all roles exercises the same relay logic without verbs for testing. `UCOMM_IB_DEV` selects
the local HCA if more than one exists. `UCOMM_TIMEOUT_MS` controls bootstrap/wait timeouts and the
relay tool's overall deadline per phase; choose a value appropriate to the entire message count.

Check `ulimit -l` before increasing buffer sizes. Endpoint registered bulk memory is `size * depth`,
plus transport buffers. During bulk transfer the relay tool allocates a forward pool of
`size * min(depth, messages)` bytes and an 8-byte acknowledgement pool, plus both contexts' transport
buffers. Each pool is registered once per context. Registration failure commonly indicates
insufficient locked-memory allowance. Large existing regression tests can exceed an 8 MiB allowance.

### Validation and initial measurements

Measured 2026-10-18 with three ConnectX-3 adapters, native IB, both links reporting 40 Gb/s (4X QDR),
GCC `-O2`, one relay progress thread, no CPU pinning. Relay CPU: Ryzen 9 3950X; right CPU: Threadripper
1950X. Linux kernels: left/relay 7.0.0-38-generic, right 6.8.0-147-generic. These are single runs, not
hardware peak claims. Both relay contexts logged backend `ib`; Ethernet addresses were used only for
bootstrap. The library's pre-existing wire format and transport implementation were unchanged.

| Test | Result |
|---|---:|
| Left → right, 512 × 1 MiB, depth 3, verification off | 810.7 MB/s (6.485 Gbit/s) |
| Right → left, same settings | 874.6 MB/s (6.997 Gbit/s) |
| 8-byte RTT, 512 measured exchanges, forward run: min / mean / max | 12.159 / 15.187 / 160.143 µs |
| 8-byte RTT, reverse run: min / mean / max | 12.069 / 16.676 / 927.208 µs |
| Full-byte and sequence verification, 32 × 1,048,579 bytes, depth 3, left → right | passed; 733.6 MB/s |
| Full-byte and sequence verification, same settings, right → left | passed; 660.5 MB/s |
| Eager payload, 2 × 1 byte, depth 8 | passed |

Half RTT is approximately 7.6–8.3 µs in these runs; it is not a directly measured one-way latency.
These initial timings predate the ready handshake and include endpoint setup skew and the final
acknowledgement. The repeatable measurements below use the handshake. The software relay stages
every message in host memory; it does not create end-to-end one-sided RDMA semantics.

Validation commands:

```sh
make -C tools/ucomm
UCOMM_BACKEND=udp make -C tools/ucomm test
cmake -S tools/ucomm -B /tmp/ucomm-cmake
cmake --build /tmp/ucomm-cmake -j4
ctest --test-dir /tmp/ucomm-cmake --output-on-failure
```

The UDP suite covers the existing point-to-point/collective/file tests plus three-process relay tests:
depth 1, pipeline wraparound, eager boundary sizes, odd-sized large payloads, count below depth,
reverse traffic and induced loss. `test_relay_api` additionally checks zero/variable-length forwarding,
message order, zero message count, truncation and sticky errors, invalid peers/tags/depth, byte accounting,
and refusal to free a busy relay. `test_progress` injects registration/progress failures and checks request
lifetime, pool rollback, and completed/empty fast paths. The runner test checks structured output, metadata,
hashes and alternating repeat order. An unqualified `make test` on an active HCA selects IB automatically; the existing 16/64 MiB
tests failed under the left host's 8 MiB locked-memory limit. Explicit UDP regression tests passed.
The test runner uses Python 3; the library and executable build do not require Python.

### Repeated measurements and the PCIe ×1 limit

`python/relay_bench.py` runs the three roles locally or over SSH, alternates direction order between
repeats, validates JSON results and byte counts, and saves per-role logs and a compact `results.json`.
It records source/binary hashes, CPU/kernel, locked-memory limits, negotiated PCIe width/speed, and IB
link rates. Reports do not embed connection settings. SSH targets use quoted argument lists and each
remote process has a bounded lifetime via GNU `timeout`. Local and remote hosts need Python 3 for
metadata collection; the runner does not build or copy binaries.

Keep actual connection settings outside the repository or in an ignored `.local` file. Example config:

```json
{
  "roots": ["relay-host:29681", "relay-host:29682"],
  "left": {"exe": "/path/to/ucomm_relay", "port": 1, "ib_dev": "mlx4_0"},
  "relay": {"host": "relay-host", "exe": "/path/to/ucomm_relay", "left_port": 2, "right_port": 1},
  "right": {"host": "right-host", "exe": "/path/to/ucomm_relay", "port": 1}
}
```

Omit `host` for local execution, and use an absolute executable path or a path relative to the remote
login directory; `~` is not expanded. `ib_dev` is optional; specify it on hosts with multiple HCAs.
Run from the repository root, choosing a new results directory each time:

```sh
python3 tools/ucomm/python/relay_bench.py --config /path/to/relay-config.json \
  --output /tmp/relay-x1-results --size 1048576 --depth 3 \
  --messages 512 --latency-iters 1000 --repeats 3
```

Use `--verify --messages 32 --repeats 1` for a separate byte/sequence verification run in both directions.
The runner's `--timeout` is an overall session bound; keep it long enough for bootstrap and both phases.
Partial results and raw logs are retained after failure; review error logs before sharing, as diagnostics
may contain private connection settings. Preserve this baseline and repeat the same settings after a lane
width change, recording the newly negotiated PCIe width rather than assuming a BIOS setting took effect.

Measured on the same three-host setup on 2026-10-18, still using PCIe 3.0 ×1 at the left endpoint and
PCIe 3.0 ×8 at the relay/right endpoint. Both active IB cables negotiated 40 Gb/s QDR. One progress
thread, GCC `-O2`, no affinity or clock changes, 1 MiB messages, depth 3, 512 messages per transfer,
three repeats per direction, verification off in timed bulk runs:

| Direction | Minimum / median / maximum MB/s | Median Gbit/s |
|---|---:|---:|
| Left → relay → right | 819.45 / 819.54 / 819.66 | 6.556 |
| Right → relay → left | 880.51 / 880.55 / 880.59 | 7.044 |

Across the six latency runs (1000 measured 8-byte exchanges each), RTT p50 ranged from 13.29 to
14.37 µs and p99 from 24.91 to 29.97 µs. Separate full-byte/sequence checks passed both directions
with 32 odd-sized 1,048,579-byte messages. Address/undefined-behavior sanitizer checks passed on the
loopback/API/error-path tests; they were not used for hardware performance measurements.

For context, direct-link `ib_write_bw` at 64 KiB measured 6.85 Gbit/s left → relay and 7.17 Gbit/s
relay → left; the right ↔ relay link measured 31.52–31.53 Gbit/s. `ib_read_bw` measured 7.16 and
31.53 Gbit/s respectively. These perftest 6.20 runs used one RC QP, 2048-byte IB MTU, five-second
duration with one-second margins. The relay is close to the measured left-link ceiling; the ×1 PCIe
connection, rather than QDR negotiation, limits these end-to-end results. The small changes from the
initial single-run numbers are not evidence of a transport speedup.

Compact evidence: [repeated results](../../docs/benchmarks/ucomm_relay_x1_20261018/results.json) and
[byte-verified results](../../docs/benchmarks/ucomm_relay_x1_20261018/verified.json). Historical source
and binary hashes are preserved as measured. RAM timings were not collected. CUDA/HIP/SYCL engine
backends, automatic failover, transparent `ucfile` relaying and PCIe ×4 operation were not tested here.
The existing IB/UDP/bootstrap wire implementations are unchanged; the shared core changes affect error
propagation and request lifetime, with regression coverage for the successful paths.

### Opt-in inline sends and paired polling (PCIe ×4)

The inline path avoids fetching a small SEND or credit payload from host RAM after posting the work
request. Set `UCOMM_IB_INLINE=64` to request inline capacity when creating QPs; 8-byte messages plus
the existing 40-byte header fit this limit. Larger messages keep the existing eager/rendezvous protocol.
The provider must support the requested capacity; unsupported requests can fail initialization. Values
must be 0..256 bytes. The default is zero, preserving the previous posting flags, QP capacities, wire
format, READ/FIN rendezvous and credit protocol. No change to `ibvew` or verbs headers is required.

`ucomm_relay_progress_pair(forward, reverse, &forward_done, &reverse_done)` is an additional opt-in API
for rings using the same two contexts in opposite directions. It polls each context once and then
examines completed requests without polling again for each pending slot. It preserves both directions'
order, bounded pool, error propagation, and buffer lifetime. The existing single-ring progress API
continues to use its previous behavior. The tool selects paired polling with `-P`; the repeat runner
selects it with `--paired`.

For the runner, optional `cpu` integers in each role's local config pin only that benchmark process
using `taskset`; they do not change system-wide affinity or clocks. Optional `host_key_alias` selects
an existing SSH known-host identity when reaching a machine by an alternate address. Connection
settings remain in the external config. `--source-root` records hashes from a preserved source snapshot
when benchmarking a separately built baseline; the private path is omitted from the result summary.

```sh
python3 tools/ucomm/python/relay_bench.py --config /path/to/relay-config.json \
  --output /tmp/relay-inline-results --size 1048576 --depth 4 \
  --messages 1024 --latency-iters 10000 --repeats 3 --ib-inline 64 --paired
```

The direct-link asymmetric benchmark now allocates/registers only the larger of the request/reply
sizes. This permits small latency tests under an 8 MiB locked-memory limit; the original full benchmark
still uses its 64 MiB buffer. Invalid `-p` shapes and nonpositive iteration counts are rejected before
bootstrap. On two endpoints in the same direct fabric, select their connected ports and the same root:

```sh
UCOMM_IB_INLINE=64 UCOMM_IB_PORT=2 tools/ucomm/build/ucomm_bench \
  -r 0 -w 2 -a root-host:29811 -b ib -p 8:8 -n 10000
UCOMM_IB_INLINE=64 UCOMM_IB_PORT=2 tools/ucomm/build/ucomm_bench \
  -r 1 -w 2 -a root-host:29811 -b ib -p 8:8 -n 10000
```

Matched measurements on 2026-10-19: same ConnectX-3/QDR hardware as above, left endpoint now PCIe
3.0 ×4, relay/right ×8, firmware 2.42.5000, one progress thread, GCC `-O2`. Each role was pinned to
CPU 2. All relay arms used 1 MiB messages, depth 4, 1024 messages, 10000 measured 8-byte round trips
after 10 warmups, and three repeats in each bandwidth direction. Arms ran baseline, inline-only, then
inline+paired; system-wide clocks, IRQ placement and background load were not controlled. This is a
same-day comparison against the clean committed `f5d7b3a3` sources, not a comparison against the old ×1
hardware numbers. Bulk verification was disabled for the timed runs and tested separately.

| Arm | Median of six RTT p50 values | Median of six RTT p99 values | Forward / reverse median Gbit/s |
|---|---:|---:|---:|
| Committed baseline | 11.855 µs | 23.325 µs | 24.872 / 25.281 |
| Inline 64, existing polling | 6.955 µs | 18.580 µs | 24.875 / 25.626 |
| Inline 64, paired polling | 5.850 µs | 16.155 µs | 24.873 / 26.485 |

On the newly connected direct fabric between the same endpoints, the retained inline implementation
measured **3.0 µs RTT p50** in each of three CPU-matched 10000-exchange runs (1000 warmups). Thus the
paired relay median is **1.95× direct**, approximately half the previous relay latency. These are round
trips, not independently measured one-way latencies. Direct benchmark output has 0.1 µs precision.

The reverse direction exceeds 25 Gbit/s; forward is approximately 24.9 Gbit/s, **0.5% below a strict
25.0 Gbit/s target**. Forward rates were 24.870–24.874 Gbit/s. Reverse rates varied more under the
uncontrolled load: 25.374–26.646 Gbit/s with paired polling. Do not infer a bandwidth speedup from those
reverse medians. Deeper/larger buffers and exploratory WRITE/read-pipeline variants did not establish
a useful gain; only inline sends and paired polling are retained. No global default was changed.

Validation: full UDP regression suite; native IB variable/zero-length, zero-capacity truncation and
file CRC/resume tests; native three-host byte/sequence checks in both directions at 1, 8152, 8153 and
1048579 bytes (129 messages for the small/eager-boundary cases, exercising credit rollover); and six
ASan/UBSan CTest checks including both polling paths and benchmark argument handling. RoCE and other
HCA families were not exercised. The default bootstrap/control header layouts and payload matching
are unchanged; defaults and both opt-in modes were covered by the regression tests.

Evidence: [baseline](../../docs/benchmarks/ucomm_relay_inline_20261019/baseline.json),
[inline](../../docs/benchmarks/ucomm_relay_inline_20261019/inline.json),
[paired](../../docs/benchmarks/ucomm_relay_inline_20261019/paired.json),
[direct](../../docs/benchmarks/ucomm_relay_inline_20261019/direct.json), and
[native integrity checks](../../docs/benchmarks/ucomm_relay_inline_20261019/verified.json).
