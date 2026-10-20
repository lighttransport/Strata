# strata-pfetch / strata-pput

Cooperative staging between a shared filesystem and node-local disks, for clusters where one read stream per node is
slow (tens of MB/s) and every file sits on one storage target.

- `strata-pfetch` (C11 + MPI): runs a manifest of byte copies (`pf_plan.h` documents the format) on all ranks of one
  mpiexec. Source files are cut into aligned blocks; each block is read once, by one rank, with K reader threads per
  rank spread over different files, and its bytes are sent (MPI point-to-point, CRC32C per message) to every rank whose
  destinations need them. Destinations are sparse full-size files or atomic files (`.pfpart`, renamed when complete,
  optional `.sha256` sidecar). Also `--measure` (filesystem, MPI and local-disk rates) and `--plan-dump`.
- `strata-pput`: the same engine built without MPI (one rank), plus `--write=LIST`: K streams copying files into
  `.pfpart`, renamed in the order the list's `after=` fields give (e.g. a chunk before its json).

Build: `make` (`MPICC=mpicc` by default; `mpifcc -Nclang` natively on A64FX, `mpifccpx -Nclang` cross; also built by
`tools/a64fx_probe/build_ds41f_pack.sh`). Tests: `make test` (needs `mpirun`).

Used by the DS4.1F jobs: `tools/ds41f_pfetch_plan.py` + `tools/ds41f_pjm_pack.sh` (a sparse node-local mirror of the
checkpoint per rank), gemm `a64fx/ds41f/stage_plan.py` + `pjsub_ds41f_imatrix_12n.sh` (the per-rank staging), and the
probe `tools/ds41f_pjm_probe.sh` (`--measure`). See docs/DS41F_QUANT_JOB.md.
