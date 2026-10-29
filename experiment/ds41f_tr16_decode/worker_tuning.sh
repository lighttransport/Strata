#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${DS41_EXPERIMENT_DEADLINE:?}"
export STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-vector-hip
export DS41_TRIAL_ENV_FILE=logs/best2n_no_timing.env
export STRATA_DS41_MERGE_VECTOR=1
unset STRATA_DS41_DECODE_TIMING
qualified() {
 python3 - "$1" <<'PY'
import pathlib,re,sys
r=pathlib.Path('logs');n=sys.argv[1];s=(r/f'{n}.log').read_text()
assert 'guard=none' in (r/f'{n}.guard').read_text()
assert 'decode_gpu_ms_per_pass' not in s
assert re.search(r'^tokens: .+$',s,re.M)[0]==re.search(r'^tokens: .+$',(r/'base_greedy_osd.log').read_text(),re.M)[0]
PY
}
for repeat in 1 2; do
 for prefetch in 2048 0 4096; do
  (( $(date +%s) + 180 < DS41_EXPERIMENT_DEADLINE )) || exit 0
  name=t36_worker_pf${prefetch}_r$repeat
  W_ENV="STRATA_IQ_PREFETCH=$prefetch" bash production_trial.sh "$name" STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_MERGE_VECTOR=1
  qualified "$name"
 done
done
for repeat in 1 2; do
 (( $(date +%s) + 180 < DS41_EXPERIMENT_DEADLINE )) || exit 0
 name=t36_worker32_r$repeat
 W_ARGS='--threads=32 --weights-dir=/dev/shm --reserve-gib=3 --gpu-reserve-mib=1024' bash production_trial.sh "$name" STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_MERGE_VECTOR=1
 qualified "$name"
done
