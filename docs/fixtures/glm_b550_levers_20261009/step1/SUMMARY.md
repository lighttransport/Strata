# B550 step 1 measurements

| Run | Mean tok/s | Neighbour BASE15 tok/s | Ratio | Tier MiB | Decode VRAM used MiB (min–max) | IDs matching BASE15 / 512, trials 2–3 | Start/end load |
|---|---:|---:|---:|---:|---:|---|---|
| step1-b15360-01-base15 | 11.230 | — | — | 7957.750 | 15392–15392 | Ref: 512,512 | 0.00/12.00 |
| step1-b15360-02-aff06 | 17.072 | 11.233 | 1.520 | 7957.750 | 15392–15392 | L: 9,9; R: 9,7 | 0.17/10.96 |
| step1-b15360-03-base15 | 11.235 | — | — | 7957.750 | 15392–15392 | Ref: 11,11 | 0.14/13.08 |
| step1-b15360-04-aff08 | 18.186 | 11.865 | 1.533 | 7957.750 | 15392–15392 | L: 3,1; R: 6,2 | 0.13/10.35 |
| step1-b15360-05-base15 | 12.494 | — | — | 7957.750 | 15392–15392 | Ref: 12,12 | 0.13/11.66 |
| step1-b15360-06-mtp1 | 11.138 | 11.980 | 0.930 | 7714.500 | 15540–15540 | L: 16,16; R: 22,22 | 0.13/12.15 |
| step1-b15360-07-base15 | 11.467 | — | — | 7957.750 | 15392–15392 | Ref: 11,11 | 0.13/12.43 |
| step1-b15360-08-aff08-mtp1 | 17.308 | 11.452 | 1.511 | 7714.500 | 15540–15540 | L: 14,4; R: 7,13 | 0.14/10.96 |
| step1-b15360-09-base15 | 11.437 | — | — | 7957.750 | 15392–15392 | Ref: 4,4 | 0.14/12.13 |

L/R compares the lever to its two neighbouring controls, trial by trial. BASE rows compare to the first BASE15 at the same budget. First differing indices are zero-based and retained for all three trials in each record. End load includes the benchmark. VRAM is total ROCm usage, including driver and other clients, sampled during confirmed decode.

Selected allocation budget: 15360 MiB. Parameter SHA256: `6786b52fe952e5ec3b86af363964c35d80570ba1a3f8e62573e1cb5c563ad9b6`.

> **Correction (tr16 session, 2026-10-09):** the MTP1 identity failure below is not an MTP bug. The batched FP16 prefill expert path (`prefill_experts: f16-batched`) is nondeterministic across processes; with `mmq` prefill, BASE and MTP1 are bit-identical. See `../determinism/README.md`.

**BUG: MTP1 without affinity failed the exact BASE15 token-identity requirement:** step1-b15360-06-mtp1. BASE-control reproducibility is recorded separately; this does not isolate the root cause.

## Token differences

| Lever | First differing index L, trials 1/2/3 | First differing index R, trials 1/2/3 | Matching positions L / 512 | Matching positions R / 512 |
|---|---|---|---|---|
| aff06 | 1,9,6 | 1,6,6 | 3,9,9 | 1,9,7 |
| aff08 | 1,1,1 | 1,1,1 | 8,3,1 | 11,6,2 |
| mtp1 | 12,12,12 | 18,18,18 | 16,16,16 | 22,22,22 |
| aff08-mtp1 | 1,4,1 | 4,1,11 | 3,14,4 | 7,7,13 |
