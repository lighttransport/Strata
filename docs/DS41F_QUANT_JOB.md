# DeepSeek V4.1 Flash: quantizing on the xeon+tesla GPU node, streamable and resumable

Written 2026-10-18. Companion of docs/DS41F_PLAN.md. Nothing here has run on the GPU node yet; everything was exercised on
synthetic checkpoints on TR16 (`tools/test_ds41f_pack.py`), and the budget numbers are measured quantizer rates.

## The node and the constraint

Slurm partition `gpu1` (the gemm repository's `cuda/tools/slurm-gpu` example config): 36 Xeon Gold 6240 cores, 2x Tesla V100 32 GB,
memory limit 160 GiB as observed (181 GB per the user; pass `--mem` accordingly), **180-minute jobs**, shared storage at
about **100 MB/s**. Earlier GLM runs there stalled on shared-filesystem rereads and had to lock private copies in RAM
(`bench/results/2026-10-14-glm-v100/README.md`). The quantizer must therefore read the source once, sequentially, and
survive the end of any allocation.

## Budget (measured on TR16, 3 threads, synthetic 2304 x 5120 experts, `strata-ds41f-pack experts`)

| Formats | weights per core-second | full 543.6B experts | on 36 cores |
|---|---:|---:|---:|
| IQ1_S / IQ1_S / IQ1_S (imatrix) | 2.83 M | 53 core-hours | ~1.5 h |
| IQ2_XXS / IQ2_XXS / Q2_K (imatrix) | 2.38 M | 63 core-hours | ~1.8 h |
| Q2_K / Q2_K / Q3_K (uncalibrated) | 4.94 M | 30 core-hours | ~0.85 h |

Reading the 288.8 GB MXFP4 source at 100 MB/s is 48 min and overlaps the quantization (one reader thread, 36 workers);
writing a 106-197 GB pack is 18-33 min. A pack fits one job only when nothing stalls, so the job is built to stop
cleanly and continue: **one job for Q2_K/Q3_K, one to two for IQ1_S.** Xeon per-core speed is assumed equal to TR16's;
the first job measures it (`PACK layer=L seconds=` lines).

## How it works

```
tools/ds41f_fetch.py      index.json -> shard table (header peek over HTTP), skips the 2 Engram-only shards (~203 GB),
                          resumable range downloads, header SHA-256 verified, fetch-manifest.json
strata-ds41f-pack experts MODEL_DIR --chunk-dir=DIR --gate=.. --up=.. --down=.. (--imatrix=F | --uncalibrated)
                          [--layers=0-39] [--threads=36] [--deadline=EPOCH]
                          per layer: reader thread streams the 384 experts in source order (w1/w3/w2 + scales, 56 MB
                          each) -> workers dequantize MXFP4 exactly and quantize all three projections -> three
                          blk.L.ffn_{gate,up,down}_exps.weight.chunk.partial appended in expert order -> fdatasync,
                          posix_fadvise(DONTNEED), rename, *.chunk.json {type, shape, bytes, FNV hash, nmse, keep}
                          plan.json: formats, keep list, imatrix sha256, per-shard header sha256, source fingerprint,
                          layer range; a rerun refuses a different plan, skips finished layers, deletes stray partials,
                          and exits 3 ("more to do") when the next layer would not finish before --deadline
strata-ds41f-pack assemble DIR OUT.gguf [--check]
                          concatenates the chunks into the sidecar GGUF (hashes re-checked while copying); --check only
                          lists missing layers (exit 3) so the job driver knows when the range is complete
tools/ds41f_quant_job.sh  sbatch script: deadline = Slurm EndTime - 10 min, runs fetch (optional) and pack, requeues
                          itself with --dependency=afterany on exit 3 until `assemble --check` passes
tools/ds41f_transfer.sh   rsync --partial --inplace from the login node in N streams with a retry loop, logs the rate,
                          then assemble + verify on this machine
```

Identity: a pack binds to the checkpoint through shard basenames, sizes and header bytes, not mtimes or paths, so a
re-downloaded or copied checkpoint still verifies (`strata-ds41f-pack verify [MODEL_DIR] PACK`).

## Commands

```sh
# on the login node or inside the job (DS41F_FETCH=1): the non-Engram shards, ~307 GB
build-exl3-venv/bin/python tools/ds41f_fetch.py --out $WORK/ds41f-1017 --plan-only      # shard table and bytes
build-exl3-venv/bin/python tools/ds41f_fetch.py --out $WORK/ds41f-1017

# bring-up pack (no imatrix needed), one job
DS41F_CKPT=$WORK/ds41f-1017 DS41F_CHUNKS=$WORK/experts-q23.d \
DS41F_FORMATS="--gate=q2_k --up=q2_k --down=q3_k --uncalibrated" sbatch tools/ds41f_quant_job.sh

# production packs after the A64FX imatrix job (docs/DS41F_PLAN.md section 5)
DS41F_CKPT=$WORK/ds41f-1017 DS41F_CHUNKS=$WORK/experts-iq1s.d \
DS41F_FORMATS="--gate=iq1_s --up=iq1_s --down=iq1_s --imatrix=$WORK/ds41f-imatrix.gguf" sbatch tools/ds41f_quant_job.sh
DS41F_CHUNKS=$WORK/experts-iq2xxs-q2k.d DS41F_FORMATS="--gate=iq2_xxs --up=iq2_xxs --down=q2_k --imatrix=..." sbatch tools/ds41f_quant_job.sh

# progress / completion
build-ds41f/strata-ds41f-pack assemble $WORK/experts-iq1s.d --check

# to TR16 (chunks, then the single file here); B550 gets its share with ucfile over InfiniBand
tools/ds41f_transfer.sh LOGIN:$WORK/experts-iq1s.d /models/ds41f/experts-iq1s.d /models/ds41f/experts-iq1s.gguf --streams=4   # LOGIN: your ssh alias of the login node

# trunk (small, 4-8 GB; any machine that holds the shards)
build-ds41f/strata-ds41f-pack trunk $WORK/ds41f-1017 trunk-q4k.gguf --trunk=q4_k --head=q4_k
```

## Space and time per machine

| Where | What | Size |
|---|---|---:|
| GPU node storage | non-Engram shards | ~307 GB |
| | chunk directories, per pack | 106 (IQ1_S) / 119 (IQ1_M) / 153 (2.25 bpw) / 197 (Q2_K/Q3_K) GB |
| TR16 | Engram 203 + IQ1_S 106 + trunk 4.5 (+ DSpark 8) | ~320 GB (free today: 196 + 142 GB on two NVMes) |
| B550 | its share of a 2.25 bpw pack (~43 GB) | B550 has 13 GB free today |
| Xeon node (if it also runs the model) | Engram 203 + 153 + 8.4 | ~372 GB |

Transfer GPU node -> TR16: rate unmeasured. At the 11 MB/s the site uplink showed for HF uploads, 106 GB is ~2.7 h per
stream; the script runs 4 streams and resumes. The first run writes the measured MB/s into `<dest>.transfer.log`.

## Failure modes

- Job ends mid-layer: the `.partial` files are deleted on the next run; the layer is redone (~1 min of work lost).
- Plan mismatch (different formats, keep list, imatrix or shards): refused; use another chunk directory.
- A shard changed (re-downloaded 1017 over V4.1-Flash): header sha256 differs, refused. The fetcher verifies each
  shard's header against `fetch-manifest.json` before the pack tool sees it.
- Disk full: the tool refuses to start a layer without the layer's bytes + 1 GiB free.
- `--max-jobs` (20) stops the requeue chain; `assemble --check` says what is missing.

## Unmeasured, to be filled by the first jobs

The node's storage sequential rate and random 4 KiB latency, the HF -> GPU node download rate (whether compute nodes
reach the internet at all, else fetch on the login node), the Xeon's quantizer rate per core, the GPU node -> TR16 rate.

## On the A64FX cluster (12 nodes, PJM batch jobs of at most 6 h)

The checkpoint (DeepSeek-V4.1-Flash-1017, 48 shards, 510.3 GB) is on the shared filesystem, so there is no fetch step.
Every node runs one `strata-ds41f-pack experts` process over its own layers into **one shared chunk directory per pack**
(each layer has exactly one owner; plan.json is created atomically on the head node with `--plan-only`, each process only
clears stale partials of its own layers), and rank 11 also packs the trunk. Jobs are self-contained and resumable:
resubmit the same command and finished layers are skipped. Site values (group, resource group, LLIO cache volume, paths)
go on the `pjsub` command line and live only in the untracked private notes.

1. **Build and publish** (on a compute node; `tools/a64fx_probe/build_ds41f_pack.sh`): ggml static, OpenMP off, with FCC
   in clang mode (the `disable_autovec.h` workaround of `ggml_cmake_args.sh` for ggml's C++ files); the pack tool as
   C++17. Give it the same ggml tree as the x86 builds (`GGML_SRC`) and pin it with `DS41F_GGML_QUANTS_SHA256`
   (quantizer output depends on `ggml-quants.c`). The binary and the job scripts go to a content-addressed directory
   `strata-ds41f-pack-<sha12>/` that is never edited; jobs check `DS41F_PACK_SHA256` before use.
2. **Probe** (`tools/ds41f_pjm_probe.sh`, 12 nodes, ~1.5 h): builds (or takes `DS41F_SNAP`), then rank r packs real layer
   r into node-local storage with one setting per rank (q23, IQ1_S, IQ2_XXS/Q2_K at 47/48/24 threads, IQ1_M, Q4_K x3 as
   the trunk stand-in; IQ ranks use `tools/ds41f_synth_imatrix.py`'s constant imatrix), copies the chunks to the shared FS
   (write rate), checks the next rank's copies (cross-node coherence) and prints `PROBE_DECISION` lines: threads,
   per-layer seconds seeds, packs per job, trunk threading, shared vs per-rank chunk directories. Its canary file is what
   `DS41F_CANARY` of the pack jobs checks.
3. **Pack jobs** (`tools/ds41f_pjm_pack.sh` + `ds41f_pjm_pack_rank.sh`): `DS41F_PACKS=q23:q2_k/q2_k/q3_k` and
   `DS41F_TRUNK=q4_k/q8_0` first (no imatrix); later `iq1s:iq1_s/iq1_s/iq1_s+iq2:iq2_xxs/iq2_xxs/q2_k+iq1m:iq1_m/iq1_m/iq1_m`
   with `DS41F_IMATRIX`. Ranks 0-3 own 4 layers, ranks 4-11 own 3. The job ends with `SENTINEL DS41F_JOB_END
   state=complete|resubmit|failed`; per-rank status in `$DS41F_OUT/status/<job>/`, an nmse summary per pack.
4. **imatrix** (gemm `a64fx/ds41f/pjsub_ds41f_imatrix_12n.sh`, `ds41f_run --imatrix-out`): stages the model per job,
   validates the hooks (64 tokens sequential vs batched), runs 32,768-token prompt files from `tools/ds41f_calib_corpus.py`
   until the deadline (~330-520k tokens per job at the batched prefill rate), and merges every finished launch into
   `imatrix.gguf` (`ds41f_imatrix_merge`: llama.cpp layout, counts `{1, 384}`, which the pack tool accepts).
5. **Transfer** the IQ packs to the destination node with `tools/ds41f_transfer.sh` as above; the 197 GB q23 pack stays on the cluster.

Tested locally: `tools/test_ds41f_pack.py` (incl. 4 processes racing on one chunk directory, `--plan-only`, imatrix count
shapes, deadline seed, threaded trunk) and `tools/test_ds41f_pjm.py` (the batch and probe scripts with a fake `mpiexec`:
40 layers x 2 packs + trunk complete, resubmission exits early, a deadline-bound job asks for resubmission, a stale
binary is refused). Unmeasured until the probe: the A64FX quantize rate, shared-FS write rate and coherence, the trunk time.
