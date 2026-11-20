# DS4.1 CPU prefill

The standalone runner uses one MPI rank per A64FX node. It is separate from the
CUDA, HIP and SYCL engines. Build on Linux with the MPI compiler wrappers and the
uTofu SDK available:

```sh
UTOFU_DIR=/path/to/utofu-tests bash tools/a64fx_probe/build_ds41_runner.sh
```

`UTOFU_DIR` supplies `tp_allreduce.h`. Compiler names and flags can be supplied
through `DS41_CC`, `DS41_CXX`, `DS41_CFLAGS` and `DS41_CXXFLAGS`. The build writes
only to its output directory (the optional first argument).
Load the compiler's runtime-library environment before building and running.

## Accepted configuration

The October 31, 2026 run used twelve A64FX nodes, one rank per node, 46 compute
threads per rank, one communication CPU and eight readers sharing the remaining
I/O CPU. The model was the 40-layer DS4.1 Flash IQ2_XS build. HBM clock speed was
not recorded. The prompt timing includes prompt-cache preparation and prefill;
model loading is excluded.

| Prompt | Owner index off, ms/token | Owner index on, ms/token |
| --- | --- | --- |
| 8,192 tokens | 3.724, 3.749 | 3.735, 3.743 |
| 16,384 tokens | 3.896, 3.896 | 3.779, 3.797 |

Orders were off/on, then on/off. These are about 267 tokens/s at 8K and 264
tokens/s at 16K with the owner index enabled. They are whole-configuration
measurements: only rank zero's repack retention was recorded in this run, so
the 16K difference is not an isolated operator speedup.

The measured source fingerprint was
`89d275209685486d55bf324abcee9331de498fba42c8e650d5c299781997ce44`,
with kernel fingerprint
`1458276aed3121371ef2ff5ed8616edf93158f227be3e3fd8900484fb6a37d4c`
and transport fingerprint
`ee4938a0bb8bad4d39b97cd81d75276c6b59881333dc1269a094ab42a390f6c9`.
The cleaned runner retains this implementation's arithmetic and primary
communication path; unused copy-worker and communication-request experiments
were removed. Later workspace, expanded-weight and layer/group scheduling
experiments are not included.

## Running

Supply a model, staged Engram shards, token IDs and an image directory through
local configuration. The measured prefill settings were:

```sh
export OMP_NUM_THREADS=46 OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=active
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
export STRATA_RUNNER_COMM=utofu STRATA_RUNNER_COMM_CORE=-1
export STRATA_DS41_XPF=1 STRATA_DS41_IQ2P=1 STRATA_DS41_Q5P=1
export STRATA_DS41_XROT=1 STRATA_DS41_KNT=8 STRATA_DS41_MERGE=11391
export STRATA_DS41_ENTRY_SHARD=1 STRATA_DS41_ES_OUTB=1
export STRATA_DS41_ES_QK=1 STRATA_DS41_ES_XQ=1
export STRATA_DS41_EMB_AG=1 STRATA_DS41_EG_ADVISE=1 STRATA_DS41_ES_OASP=1
export STRATA_DS41_PASS_MAX=512 STRATA_DS41_INDEX_ROWS=1
export STRATA_DS41_IQ2P_TILE16=1 STRATA_DS41_PREFILL_FIBERS=1
export STRATA_DS41_PREFILL_EG_CACHE=1
export STRATA_DS41_PREFILL_EG_ASYNC=1 STRATA_DS41_PREFILL_EG_ASYNC_THREADS=8
export STRATA_DS41_ES_INDEX=1 STRATA_DS41_ES_INDEX_QK=1
export STRATA_RUNNER_A2A_PAR=1
export STRATA_DS41_ARBENCH=1
mpiexec -np 12 build/ds41-cpu/ds41_cpu_runner "$MODEL" \
    --engram-dir "$ENGRAM_DIR" --image-dir "$IMAGE_DIR" \
    --eval "$PROMPT_IDS" --tokens 8192 --context 16384 \
    --prompt-pass 512 --generate 0
```

Ensure the launcher forwards these variables to every rank. The asynchronous
reader path checks close binding and disjoint compute, communication and I/O
CPUs before loading the model. Image files are allocation-local and must match
the model, rank count and layer count.

All features remain explicit switches. With no `--prompt-pass`, the ordinary
evaluation/decode path remains available. Reader verification is enabled by
`STRATA_DS41_PREFILL_EG_CACHE_VERIFY=1`; owner-score verification by
`STRATA_DS41_ES_INDEX_VERIFY=1`.

## Validation

The measured configuration passed both Engram gather paths with independent
reader-byte checks, the owner-score verifier, both prompt lengths, 1,024 prompt
tokens plus 512 decoded tokens, and fresh B=128 prefill plus 32 decode steps with
the owner index both off and on. Prompt references were:

| Tokens | Argmax hash |
| --- | --- |
| 512 | `c56c177f1095e5fb` |
| 8,192 | `df6d16b15b783325` |
| 16,384 | `6d889c3e0a43e14c` |

KL was `0.399234`, perplexity `2.03586`; the 512-token continuation fingerprint
was `c3191d32` (MD5 prefix). Each B=128 run matched all 128 baseline sequence
hashes. These checks describe the measured implementation; cleanup build and
regression results must be recorded separately from historical timings.

The cleanup passed a native macOS runner object build (with a Linux affinity
compatibility shim), the scalar kernel parity probe, IQ2 repack and tile tests
under AddressSanitizer/UndefinedBehaviorSanitizer, row-reader tests under those
sanitizers and ThreadSanitizer, and the host environment, tensor-name and expert
pack tests. This does not validate the SVE or MPI/uTofu builds. A fresh A64FX
build and model run of the cleaned source remain outstanding; the timing table
above has not been remeasured after cleanup.
