# GLM-5.3-Flash on two nodes: TR16 + RX 9070 XT and B550 + RTX 5060 Ti (simulated, 2026-10-16)

Can the second GPU sit in another machine instead of the same one? This note gives the configuration, the
simulator's prediction for 1GbE and InfiniBand QDR, and the engine work it would take. Nothing here is measured on
two nodes yet; the engine cannot run it today.

## Configuration

| | node A: TR16 (primary) | node B: B550 (worker) |
|---|---|---|
| CPU, RAM | Threadripper 1950X, 125 GB, 2 NUMA nodes | Ryzen 9 3950X, 62 GB DDR4-2133 (29 GB/s) |
| GPU | RX 9070 XT 16 GB, PCIe 3 x16 (14.3 GB/s measured) | RTX 5060 Ti 16 GB, PCIe 4 x8 (~13 GB/s, estimate), headless |
| role | holds REAP-224 (95.5 GB) in RAM; mixers, dense weights, MTP, GPU-resident experts, the CPU expert lane (15 threads) | GPU expert worker: serves its resident experts per decode slot; optional prefill expert groups; optional CPU lane |
| model data | the whole model | only the worker's expert share (its cache fills from local RAM, not over the link) |

The reverse orientation is not possible: the model does not fit in 62 GB. With the 5060 Ti gone, TR16's display
moves to the 9070 XT (about 2 GB of VRAM, fewer primary expert slots) unless TR16 runs headless.

Per decode verify step (2 groups x 42 MoE layers) the primary sends one request per slot, about 36 KB (two tokens'
activations plus routes), and the worker returns its expert rows, 16 KB each. Prefill sends each layer's
activations (F16, 64 MB at 8K tokens) to the worker and gets one partial back.

## The link (measured 2026-10-16)

perftest between TR16 (ConnectX-3 on PCIe 3 x8, `ibp66s0`) and the B550 (ConnectX-3 on PCIe 3 x4, `ibp4s0`), port
rate 40 (4x QDR):

| test | result |
|---|---|
| `ib_write_bw`, one way | 26.36 Gb/s = 3.30 GB/s from 4 KB messages up |
| `ib_write_bw -b`, both ways at once | 48.0 Gb/s peak, 43.3 average (about 2.7 GB/s each way; the B550's x4 slot is the likely limit) |
| `ib_send_lat`, one way | 1.3 us up to 128 B; 5.2 us at 4 KB, 6.4 us at 8 KB, 14.7 us at 32 KB, 24.7 us at 64 KB |

Through the engine's own transport (`tools/ucomm/build/ucomm_bench -b ib`, RC queue pairs, IPoIB bootstrap on
10.10.10.x; TR16 rank 0, load average 2.5):

| `ucomm_bench` | 2026-10-16, full QDR | 2026-10-06, degraded link |
|---|---|---|
| latency 8 B / 1 KiB / 8 KiB (half round trip) | 3.3 / 4.1 / 10.0 us | 3.5 / 4.7 / 13.1 us |
| bandwidth, 64 KiB / 1 MiB / 64 MiB messages | 3.24 / 3.26 / 3.26 GB/s | 1.55 / 1.65 / 1.65 GB/s |
| round trip 64 B -> 64 B | mean 6.7 us, p50 6.3, p99 13.3 | mean 7.3 us |
| round trip 4800 B -> 8192 B (decode-shaped) | mean 15.7 us, p50 15.2, p99 20.5 | mean 20.9 us |
| round trip 4800 B -> 16384 B | mean 18.2 us, p50 17.8, p99 21.4 | mean 26.5 us |
| round trip 36864 B -> 49152 B (two-node worker request) | mean 42.3 us, p50 42.0, p99 47.1 | - |
| allreduce f32 16 MiB, 2 ranks | 10.7 ms (1.56 GB/s algbw) | 16.0 ms |

The simulator's `ib-qdr` preset uses the ucomm numbers: rtt 15.7 us, 3.26 GB/s. The runner model charges each
message rtt/2 plus its bytes at that rate, which gives 44.3 us for the 36 KB -> 48 KB round trip (measured 42.3).
perftest exchanged its connection data through an SSH port forward (the B550 does not accept port 18515 on the LAN);
TR16's IPoIB profile `ib0` was re-pointed to the renamed interface (`nmcli con modify ib0 connection.interface-name
ibp66s0`).

## Predicted speed (8K context, REAP-224)

`PYTHONPATH=tools python3 tools/glm_runner_sim.py multinode --anchor-decode 34.5 --anchor-prefill 615`

The simulator's base state is lossless routing; "anchored" scales each row by measured / simulated for the
RX 9070 XT alone (34.5 tok/s decode with the release profile, 615 tok/s prefill). Decode picks the best of "no
worker" and 11-36 worker expert slots per layer (33 = the measured 12 GB worker cache); prefill picks the best
fraction of expert groups on the primary (1.00 = worker not used).

| configuration | link | decode, simulated | decode, anchored | prefill, anchored |
|---|---|---:|---:|---:|
| TR16 + RX 9070 XT alone | - | 25.1 | 34.5 | 615 |
| TR16, both GPUs in one host (reference) | PCIe | 28.4 | 39.0 | 683 |
| TR16 + 9070 XT, B550 + 5060 Ti | 1GbE | 25.1 (worker not used; with it 20.0) | 34.5 | 615 (worker not used) |
| same | InfiniBand QDR (measured: 3.26 GB/s, 15.7 us ucomm round trip) | 28.4 | 39.0 | 633 (fraction 0.90) |
| same | InfiniBand as measured today (1.65 GB/s, 21 us) | 28.4 | 39.0 | 615 (worker not used) |
| same | 10GbE (estimate) | 28.3 | 38.9 | 615 (worker not used) |
| same, plus a CPU expert lane on the B550 | IB QDR | 28.4 | 39.0 | - |

What the model says:

- **InfiniBand QDR costs nothing in decode.** The request and reply (tens of microseconds each) hide in the slack of
  the primary GPU's per-slot chain; the step takes 112.8 ms as on one host. The worker on another node is worth
  the same +13% (simulated) as in the same host.
- **1GbE loses.** Each of the 84 worker requests per step waits about 1 ms (0.4 ms round trip, 36 KB out and
  ~48 KB back at 117 MB/s), adding about 50 ms to a 113 ms step: 20.0 tok/s, slower than
  the 9070 XT alone. Over 1GbE the only useful configuration is not to use the second node for decode.
- **Prefill gains little over any link.** Shipping 64 MB of activations each way per layer costs 40 ms on QDR and
  more than a second on 1GbE; only QDR shows a small gain (+3%), and the measured one-host gain (870 vs 615 tok/s)
  is larger than this model predicts for one host (683), so the remote prefill number is the least certain.
- **A CPU expert lane on the B550 does not help** in this regime: each slot sends only ~2.3 experts (21 MB) to the CPU
  lane, whose time is fixed per-slot overhead, and the step is bound by the primary GPU's chain.

Uncertainty: the simulator's lever forecasts have been off by several tok/s in both directions
([GLM_RUNNER_SIM.md](GLM_RUNNER_SIM.md)). The measured one-host two-GPU gain at release settings (CUDA primary,
lossy routing: 43.6 vs 34.5, +26%) is larger than the simulated lossless HIP-primary gain (+13%), so the anchored
QDR row may be low. The 5060 Ti's PCIe rate on the B550 and the B550's memcpy rate are estimates.

## A model that does not fit one node (160 GB)

`PYTHONPATH=tools python3 tools/glm_runner_sim.py split-model --anchor-decode 34.5 --anchor-sim 25.1`

Example model: REAP-224 with experts 1.75 times the Q23 size (about Q4_K): 150 GB of experts plus 10 GB of dense
weights. TR16 holds about 112 GB of weights, so 32% of the expert bytes (48 GB) must live on the B550 (62 GB). It is
compared with the same machine given 192 GB of RAM. GPU expert slots shrink by the same 1.75 (same VRAM). Decode is
anchored to the measured RX 9070 XT alone on REAP-224 Q23 (34.5 / 25.1 simulated); prefill is simulated.

| configuration | link | decode, simulated | decode, anchored | prefill (8K), simulated |
|---|---|---:|---:|---:|
| one node with 192 GB, both GPUs | - | 18.9 | 25.9 | 641 |
| one node with 192 GB, RX 9070 XT only | - | 13.0 | 17.9 | 533 |
| two nodes, coldest 32% of each layer's experts on the B550 (its CPU serves them) | IB QDR | 20.8 | 28.5 | see below |
| same, node-B share balanced to its memory bandwidth | IB QDR | 21.6 | 29.6 | |
| two nodes, every expert's rows split 68/32 (remote expert TP, exists in the engine for CPU workers) | IB QDR | 21.3 | 29.2 | |
| two nodes, coldest 32% on the B550 | 1GbE | 15.7 | 21.6 | |
| two nodes, rows split 68/32 | 1GbE | 13.9 | 19.1 | |
| two nodes, RTX 5060 Ti on the B550, its experts streamed locally | IB QDR | | | 607 |
| same | 1GbE | | | 98 |

- **On InfiniBand the split model is not slower in decode; it is about 10% faster** than one 192 GB node, because the
  B550 adds its own memory bandwidth (26 GB/s) to TR16's 66 GB/s for the CPU-served experts, and the link latency
  hides in the step. This holds while the B550's share of the CPU-lane traffic stays near its bandwidth share: the
  measured routing counts put the coldest 32% of experts at 5-23% of a layer's routes (mean 10%); a layer where they
  carry 23% makes the B550 the long pole (14.0 tok/s if every layer were like that).
- **On 1GbE decode loses about 17%** (each step waits ~1 ms per remote request, 84 per step).
- **Prefill needs a GPU on the B550.** With the 5060 Ti there, the B550's 48 GB of experts stream into it locally and
  only activations cross: 607 against 641 tok/s on one node (-5%). Without a GPU on the B550, its experts would have
  to cross the link every 8K chunk (48 GB: about 15 s on QDR, minutes on 1GbE; not simulated).
- RAM: the B550 has room for the 48 GB of cold experts, but little for copies of hot experts for a 5060 Ti cache
  (~7-13 GB): the GPU-on-B550 layout needs a smaller worker cache or more RAM.

## What the engine needs

The mixed GPU worker (`strata-glm-gpu-worker`) exchanges requests through a shared memfd and a Unix socket, so it
runs only on the primary's host. Running it on node B needs:

1. the worker protocol over `ucomm` (TCP, or the InfiniBand backend `UCOMM_BACKEND=ib` used by the CPU expert TP in
   `include/strata/net/expert_tp.hpp`), with the activation request and the row reply as messages;
2. the worker loading its expert share from a local copy of the model (B550: 62 GB RAM; the worker needs ~12-14 GB
   of rows for its cache, plus room for ownership refresh);
3. ownership refresh (the worker's cache adaption) reading from node B's own RAM, not node A's;
4. for prefill, the activation exchange over the link (or prefill on the primary alone, as the model suggests).

Worth building only with InfiniBand (or faster); the expected gain equals the one-host two-GPU gain.

## Simulator

`two_node_hardware()` in `tools/sim/glm_runner.py` derives the two-node manifest from the one-host one: the
worker's PCIe moves to node B, every path between node A and the worker crosses `net.ab` / `net.ba` with
rtt/2 + per-message overhead as a latency stage, the worker streams prefill groups from node B's RAM, and an
optional CPU lane on node B (`Topology.worker_cpu_share`) serves part of the cold experts. Link presets are
`tools/sim/hw.py` `LINKS`: `gbe` (measured), `ib` (measured, degraded), `ib-qdr` and `10gbe` (estimates), or
`rtt_us,gbps,exposed_us`. With no link the model is unchanged (validate output identical). Tests:
`PYTHONPATH=.:tools python3 -m unittest tools.test_glm_runner_sim` (`TwoNodeTests`).
