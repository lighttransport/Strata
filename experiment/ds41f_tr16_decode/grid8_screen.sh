#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${DS41_EXPERIMENT_DEADLINE:?}" "${DS41_PREVIOUS_SCREEN_PID:?}"
while kill -0 "$DS41_PREVIOUS_SCREEN_PID" 2>/dev/null; do sleep 10; done
check_time() { (( $(date +%s) + 180 < DS41_EXPERIMENT_DEADLINE )); }
export STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-grid8-hip DS41_WORKER_BIN=/tmp/strata-ds41-worker-grid8
export STRATA_DS41_MERGE_VECTOR=1
export DS41_TRIAL_ENV_FILE=logs/best2n_no_timing.env DS41_GATE_ENV_FILE=logs/best2n_no_timing.env
unset STRATA_DS41_DECODE_TIMING
for mode in 0 1; do
 check_time || exit 0
 export STRATA_IQ1S_GRID8=$mode W_ENV=STRATA_IQ1S_GRID8=$mode
 DS41_MERGE_GATE_PREFIX=t38_grid8_${mode}_gate bash merge_gate.sh
 # Captures use best2n's profiling; throughput trials explicitly omit it.
 unset STRATA_DS41_DECODE_TIMING
done
for repeat in 1 2; do
 for mode in 0 1; do
  check_time || exit 0
  export STRATA_IQ1S_GRID8=$mode W_ENV=STRATA_IQ1S_GRID8=$mode
  bash production_trial.sh "t38_grid8_${mode}_r$repeat" STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=$mode
  python3 - "$mode" "$repeat" <<'PY'
import pathlib,re,sys
r=pathlib.Path('logs');name=f't38_grid8_{sys.argv[1]}_r{sys.argv[2]}'
s=(r/f'{name}.log').read_text()
assert 'guard=none' in (r/f'{name}.guard').read_text()
assert 'decode_gpu_ms_per_pass' not in s
assert re.search(r'^tokens: .+$',s,re.M)[0]==re.search(r'^tokens: .+$',(r/'base_greedy_osd.log').read_text(),re.M)[0]
PY
 done
done
check_time || exit 0
export STRATA_IQ1S_GRID8=1 W_ENV=STRATA_IQ1S_GRID8=1
bash cpp_gate/production_run.sh t38_cpp_grid8 STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=1
python3 - <<'PY'
import pathlib,re,json
r=pathlib.Path('logs')
assert re.search(r'^tokens: .+$',(r/'t31_cpp_exact.log').read_text(),re.M)[0]==re.search(r'^tokens: .+$',(r/'t38_cpp_grid8.log').read_text(),re.M)[0]
assert json.loads(pathlib.Path('cpp_gate/t38_cpp_grid8.grade.json').read_text())['passed']
PY
