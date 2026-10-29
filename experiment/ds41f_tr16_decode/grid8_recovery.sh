#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${DS41_EXPERIMENT_DEADLINE:?}"
(( $(date +%s) + 180 < DS41_EXPERIMENT_DEADLINE )) || exit 0
export STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-grid8-hip DS41_WORKER_BIN=/tmp/strata-ds41-worker-grid8
export DS41_GATE_ENV_FILE=logs/best2n_no_timing.env DS41_GATE_CANON_MIB=2304
export STRATA_DS41_MERGE_VECTOR=1 STRATA_IQ1S_GRID8=1 W_ENV=STRATA_IQ1S_GRID8=1
unset STRATA_DS41_DECODE_TIMING
bash cpp_gate/production_run.sh t38_cpp_grid8_v2 STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=1
python3 - <<'PY'
import pathlib,json,re
r=pathlib.Path('logs')
assert json.loads(pathlib.Path('cpp_gate/t38_cpp_grid8_v2.grade.json').read_text())['passed']
assert re.search(r'^tokens: .+$',(r/'t31_cpp_exact.log').read_text(),re.M)[0]==re.search(r'^tokens: .+$',(r/'t38_cpp_grid8_v2.log').read_text(),re.M)[0]
print('Smaller-cache C++ task passes and all512IDs equal exact baseline')
PY
unset DS41_PREVIOUS_SCREEN_PID
bash final_screen.sh
