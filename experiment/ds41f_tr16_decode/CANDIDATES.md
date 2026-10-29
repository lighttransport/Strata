# Measured decode candidates

Measured on the existing TR1950X / RX 9070 XT plus B550 / RTX 5060 Ti setup over IB, single sequence,
c_osd 1,000-token prompt and 128 generated tokens, 14 host threads and 16 worker threads. RAM configuration
was unchanged; its speed was not re-read. These are workload-specific results on a shared desktop.

| Preset | Independent runs (tok/s) | Mean |
|---|---:|---:|
| `candidate_exact.env`, canonical cache requested 3328 MiB | 24.9935 / 26.1361 | **25.5648** |
| `candidate30_low_loss.env`, trim 0.125, cache requested 2304 MiB | 30.4605 / 30.5729 | **30.5167** |

All four grid-enabled exact runs, including the earlier A/B controls, average **25.02755 tok/s**
(range 23.8575–26.1361). This reaches a 25 tok/s mean, not a 25 tok/s floor. Both lossy timing runs
also generated the exact reference's tokens on this prompt; the nonzero logit drift below makes them lossy.

The new opt-in `STRATA_IQ1S_GRID8=1` prepares a 16 KiB byte-times-eight grid on first CPU dispatch and
keeps each row's integer sums and float accumulation order. It defaults off. Worker CPU time fell from
1.580/1.576 to 1.435/1.434 ms/request in two A/B pairs. Direct/vector merge is also opt-in. The grid's
disabled and enabled paths both matched every frozen logit at 1,000 positions each on c_osd and c_llama.
The smaller 2304 MiB cache also matched both frozen references byte for byte before trimming was enabled.

## C++ task gate

Both exact and trim-0.125 candidates passed the strict C++17 `parse_u64` task: **400,532 oracle cases**,
warnings as errors, fatal UBSan, and networkless/resource-limited execution. Each generated 512 tokens
from the same 183-token prompt. The exact candidate's output IDs matched the old exact baseline; the
lossy candidate generated a different program that passed the oracle. This is one coding task, not a broad coding benchmark.

Both task runs used a 2304 MiB canonical cache. Single task timings were 15.7042 tok/s exact and
21.2254 lossy; these are separate workload timings, not the 25/30 tok/s measurements above or a repeated
performance A/B. An earlier exact task attempt with the larger cache stopped at the unchanged 2.5 GiB
physical-VRAM floor. It has no speed/quality qualification. The smaller-cache exact retry passed with
at least 3.54 GiB free. No OOM or GPU fault was reported.

## Six-prompt quality, trim 0.125

Fresh target-only captures, 1,000 positions each, against the frozen exact reference:

| Prompt | Mean KL | Top-1 agreement | Next-token accuracy change | Perplexity change |
|---|---:|---:|---:|---:|
| c_osd | 0.0995 | 90.5% | -1.9 pp | +2.98% |
| c_imgui | 0.0387 | 98.3% | -0.1 pp | +1.55% |
| c_llama | 0.0513 | 96.9% | -0.5 pp | +1.79% |
| c_ff | 0.0553 | 95.9% | +0.0 pp | +1.46% |
| t_mujoco | 0.1870 | 84.1% | -1.0 pp | -0.97% |
| t_doc2 | 0.0808 | 87.4% | -0.6 pp | +2.16% |

The requested C++ task gate passes. The older mean-KL <=0.08 / C++ agreement >=97% limits still fail.
Perplexity changed by -0.97% to +2.98%, with next-token accuracy down at most 1.9 pp on these six prompts;
broader coding-quality or semantic equivalence is not established. The previous trim-0.15 candidate
(`candidate30.env`) measured 30.7496/31.1793 tok/s and passed the same task, but had greater measured drift
(KL up to 0.3063 and perplexity up to +8.59%). Prefer the measured lower-trim preset when using this task gate.

Trimming is decided for the logical CPU-expert set before canonical offload. Changing the canonical
cache alone does not change trim eligibility here. The prior larger-cache trim-0.125 c_osd capture and
the current smaller-cache capture are byte-identical. Cache sizes are nevertheless fixed within the
reported speed/task/quality configurations for reproducibility.

## Reproduce

Rebuild both nodes with the new CPU code and set the grid switch in both environments. For the current
session's preserved alternate binaries, use the guarded production runner with a fresh run name:

```bash
cd experiment/ds41f_tr16_decode
unset STRATA_DS41_DECODE_TIMING
export STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-grid8-hip
export DS41_WORKER_BIN=/tmp/strata-ds41-worker-grid8
export DS41_TRIAL_ENV_FILE=logs/best2n_no_timing.env
export W_ENV=STRATA_IQ1S_GRID8=1
bash production_trial.sh NEW_EXACT_RUN $(cat candidate_exact.env)
bash production_trial.sh NEW_LOSSY_RUN $(cat candidate30_low_loss.env)
```

`DECODE_TIMING` is presence-enabled: setting it to zero still profiles. The alternate environment snapshot
omits it. Model/profile/connection paths stay in ignored local configuration. The worker's temporary 48 GiB
cache is released after the experiments; another run will refill it.

`build_alternate.py --build-dir BUILD --out-dir OUTPUT` builds the candidate CPU/lane objects and links
an alternate decoder from an existing Ninja backend build. It reproduced the measured HIP binary's hash.
The worker was built separately with the same CPU source. Existing other backend objects are reused;
these are not clean builds. Full binary, source, configuration and log hashes are in
[results_2026-10-29.json](results_2026-10-29.json).

Validation: CPU bitwise parity (widths 1–8, clamp/separate up, direct dots and row guards), 96 merge geometries,
HIP/CUDA decoder links, HIP decoder plus CUDA worker model execution, four grid off/on frozen-logit captures,
two smaller-cache frozen-logit captures, and both C++ task gates. Separate CUDA decoder execution, SYCL,
Windows and Intel AVX-VNNI execution were not tested. Default-path output was checked against the frozen
exact references; no feature default was changed.

Prefetch 0/4096, worker 8/32 and drafter-Q8 gave no qualified improvement. N4/N6 and higher-trim arms were
left as saved plans after these speed means and task gates were reached; they are not measured failures.

Commit preparation changed the paired-IQ1_S environment read to the C11 strict-one helper (the same
accepted value), and removed machine paths from the guarded runners. The historical measurement hashes
remain unchanged. The helper migration is separately compile/parity checked; speed measurements were not rerun.
