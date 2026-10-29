#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${DS41_EXPERIMENT_DEADLINE:?}"
# Do not overlap the CPU microbenchmark with timing runs.
while pgrep -x iq-tile-bench >/dev/null; do sleep 5; done
bash worker_tuning.sh
export STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-vector-hip
export DS41_TRIAL_ENV_FILE=logs/best2n_no_timing.env DS41_GATE_ENV_FILE=logs/best2n_no_timing.env
export STRATA_DS41_MERGE_VECTOR=1
unset STRATA_DS41_DECODE_TIMING
for trim in 0.14 0.135 0.1375; do
 label=${trim/./}
 for repeat in 1 2; do
  (( $(date +%s) + 180 < DS41_EXPERIMENT_DEADLINE )) || exit 0
  bash production_trial.sh "t37_trim${label}_r$repeat" STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_CPU_TRIM=$trim
 done
 if python3 - "$label" <<'PY'
import sys,pathlib,re
r=pathlib.Path('logs');label=sys.argv[1]
for rep in (1,2):
 n=f't37_trim{label}_r{rep}'
 assert 'guard=none' in (r/f'{n}.guard').read_text()
 s=(r/f'{n}.log').read_text()
 assert 'decode_gpu_ms_per_pass' not in s
 if float(re.search(r'decode_tok_s=([0-9.]+)',s)[1])<30:sys.exit(1)
PY
 then
  (( $(date +%s) + 180 < DS41_EXPERIMENT_DEADLINE )) || exit 0
  bash cpp_gate/production_run.sh "t37_cpp_trim$label" STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_CPU_TRIM=$trim
  bash quality_2n.sh "t37_trim$label" STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_CPU_TRIM=$trim
 fi
done
