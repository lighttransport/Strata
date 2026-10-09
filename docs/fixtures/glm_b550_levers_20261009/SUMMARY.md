# B550 lever measurements — 2026-10-09

| Lever | Kind | Measured tok/s | Neighbour BASE tok/s | Ratio | IDs equal to BASE | Actual slots / MiB | Start / end load | Start / end / loaded peak MCLK MHz | Status |
|---|---|---|---|---|---|---|---|---|---|
| BASE | decode | 10.7157 | — | — | — | 707 / 6142.06 | 0.00 / 12.16 | 772.0 / 96.0 / 1258 | complete |
| mtp1 | decode | 10.5800 | 10.7681 | 0.983 | no | 707 / 6142.06 | 0.21 / 12.32 | 772.0 / 96.0 / 1258 | complete |
| mtp2 | decode | 10.3447 | 10.8072 | 0.957 | no | 707 / 6142.06 | 0.13 / 12.45 | 772.0 / 96.0 / 1258 | complete |
| mtp3 | decode | 8.3249 | 10.8227 | 0.769 | no | 707 / 6142.06 | 0.14 / 13.71 | 772.0 / 96.0 / 1258 | complete |
| mtp2-unsplit | decode | 8.2960 | 10.7302 | 0.773 | no | 707 / 6142.06 | 0.13 / 13.71 | 772.0 / 96.0 / 1258 | complete |
| mtp2-aff05 | decode | 12.0534 | — | — | — | 707 / 6142.06 | 0.14 / 11.64 | 772.0 / 96.0 / 1258 | complete |
| mtp2-aff10 | decode | 15.6094 | — | — | — | 707 / 6142.06 | 0.12 / 12.72 | 772.0 / 96.0 / 1258 | complete |
| mtp3-margin2 | decode | 8.8679 | 10.5763 | 0.838 | no | 707 / 6142.06 | 0.13 / 13.54 | 772.0 / 96.0 / 1258 | complete |
| static-tier | decode | 10.1458 | 10.5180 | 0.965 | no | 707 / 6142.06 | 0.29 / 12.27 | 772.0 / 96.0 / 1258 | complete |
| cache3072 | decode | 9.7130 | 10.7329 | 0.905 | no | 353 / 3066.69 | 0.14 / 13.42 | 772.0 / 96.0 / 1258 | complete |
| largest-admitted-cache | decode | — | 10.5612 | — | — | — | 0.14 / 0.19 | 772.0 / 96.0 / — | failed |

Rates are means of trials 2–3. Each ratio uses the mean of the two neighbouring matched BASE runs. Token equality uses all three 512-token streams and both BASE neighbours. Prefill has no decoder steps; its one emitted token per trial was already computed by prefill. End load includes the benchmark itself. A 96 MHz clock after exit is an idle reading; loaded peaks are reported separately.

Successful runs: 21; failed attempts: 2. Failed attempts are retained separately from importable records, with null tok/s. See each manifest, result, log, token file and record for exact flags and diagnostics.
