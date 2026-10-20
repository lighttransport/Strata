# Two-layer NVFP4 checkpoint results

Measured on RX 9070 XT (gfx1201), 2026-10-19, with the miner running.
Only the six dense MLP projections from layers 0 and 1 were downloaded and
tested, plus their scales: **162 MiB of tensor payload**, no complete shards.
Pinned source and per-tensor checksums are in [selection.json](results/20261019/selection.json).

All **36 real-weight cases** passed across six kernel modes (two activation
types × three batch sizes × six matrices). Every output matched the predecoded
baseline numerically and met the FP64 error bound, including after timed rounds.
All **four chained MLP smoke cases** also passed. The download range guard passed
five unit tests; the HIP build was warning-free, with BF16/F16 WMMA in generated
assembly. Earlier experiment source and evidence hashes remain unchanged.

These results validate actual checkpoint storage/scales and local kernel arithmetic.
They do not validate complete transformer blocks, generation quality, perplexity,
or fidelity to the published model's NVFP4 activation path. Activations are synthetic;
only the chained smoke cases feed gate/up-derived intermediates into down.

## Matched 64x64 tile comparison, M=512

Median milliseconds (minimum–maximum across five rounds of 20 launches).
Gate/up have N=12288,K=4096; down has N=4096,K=12288. Miner interference can
make ranges wide; small median differences are not reliable speedup evidence.

| Layer | Projection | Activation | Packed NVFP4 | Predecoded |
|---|---|---|---:|---:|
| 0 | gate | BF16 | 4.804 (4.220–7.588) | 3.228 (3.166–4.967) |
| 0 | gate | FP16 | 4.771 (4.304–5.420) | 3.220 (3.193–4.350) |
| 0 | up | BF16 | 4.478 (4.223–5.652) | 4.298 (3.169–4.928) |
| 0 | up | FP16 | 5.023 (4.251–5.568) | 3.196 (3.142–4.970) |
| 0 | down | BF16 | 4.680 (4.595–4.725) | 3.730 (3.440–3.815) |
| 0 | down | FP16 | 4.646 (4.598–4.721) | 3.730 (3.462–3.820) |
| 1 | gate | BF16 | 4.826 (4.268–5.668) | 3.224 (3.135–4.990) |
| 1 | gate | FP16 | 4.674 (4.261–5.593) | 4.419 (3.139–5.054) |
| 1 | up | BF16 | 5.148 (4.280–5.729) | 3.219 (3.188–3.536) |
| 1 | up | FP16 | 4.912 (4.249–5.737) | 3.557 (3.165–4.996) |
| 1 | down | BF16 | 4.684 (4.639–4.746) | 3.513 (3.420–3.800) |
| 1 | down | FP16 | 4.614 (4.564–4.692) | 3.756 (3.424–3.787) |

Full M=1,16,512 results for all tiles and separate weight expansion are in the
[1260-row CSV](results/20261019/timings.csv). Predecoded weights keep local scales
folded into 16-bit values, with the checkpoint global scale applied afterward,
exactly as the packed path does. Expansion cost is excluded from GEMM timing.
No library baseline was run. Buffers remain resident; this is not streaming or
end-to-end latency. M=1 is not a specialized GEMV benchmark.

Hardware/toolchain: RX 9070 XT, Threadripper 1950X, 131869116 KiB OS MemTotal,
RAM speed unmeasured; HIP 7.16.26385, clang 24, Linux 7.0.0-38-generic.
[Before](results/20261019/before.txt) and [after](results/20261019/after.txt)
telemetry record the GPU load and clocks. No settings were changed.

[Validation log](results/20261019/validation-and-timings.txt),
[manifest and hashes](results/20261019/manifest.json), and
[reproduction and arithmetic details](README.md).
