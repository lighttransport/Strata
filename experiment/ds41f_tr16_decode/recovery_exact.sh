#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${DS41_EXPERIMENT_DEADLINE:?}"
check_time() { (( $(date +%s) + 180 < DS41_EXPERIMENT_DEADLINE )); }
export STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-vector-hip
export DS41_TRIAL_ENV_FILE=logs/best2n_no_timing.env DS41_GATE_ENV_FILE=logs/best2n_no_timing.env
export STRATA_DS41_MERGE_VECTOR=1
unset STRATA_DS41_DECODE_TIMING
sed '/^STRATA_DS41_DECODE_TIMING=/d' best2n.env > logs/best2n_no_timing.env
qualified() {
  [[ -s logs/$1.log ]] || return 1
  grep -q 'guard=none' "logs/$1.guard" || return 1
  if grep -q 'decode_gpu_ms_per_pass' "logs/$1.log"; then return 1; fi
  [[ "$(grep -m1 '^tokens:' "logs/$1.log")" == "$(grep -m1 '^tokens:' logs/base_greedy_osd.log)" ]] || return 1
  return 0
}

for arm in h14_r1 h16_r1 h14_r2 h16_r2; do
  check_time || exit 0
  threads=14; [[ "$arm" == h16* ]] && threads=16
  DS41_DECODE_THREADS=$threads bash production_trial.sh "t35_$arm" STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_MERGE_VECTOR=1
  qualified "t35_$arm"
done
threads=$(python3 - <<'PY'
import pathlib,re,statistics,json
r=pathlib.Path('logs'); rates={}
for h in (14,16):
    rates[h]=[float(re.search(r'decode_tok_s=([0-9.]+)',(r/f't35_h{h}_r{n}.log').read_text())[1]) for n in (1,2)]
best=max(rates,key=lambda h:statistics.mean(rates[h]))
(r/'t35_selection.json').write_text(json.dumps({'rates':rates,'host_threads':best},indent=2)+'\n')
print(best)
PY
)
export DS41_DECODE_THREADS=$threads
for repeat in 1 2; do
  check_time || exit 0
  bash production_trial.sh "t35_validate_r$repeat" STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_MERGE_VECTOR=1
  qualified "t35_validate_r$repeat"
done
check_time || exit 0
bash cpp_gate/production_run.sh t35_cpp_exact STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_MERGE_VECTOR=1
python3 - <<'PY'
import pathlib,re,json
r=pathlib.Path('logs')
a=re.search(r'^tokens: .+$',(r/'t31_cpp_exact.log').read_text(),re.M)[0]
b=re.search(r'^tokens: .+$',(r/'t35_cpp_exact.log').read_text(),re.M)[0]
assert a==b,'C++ exact token IDs differ'
assert json.loads(pathlib.Path('cpp_gate/t35_cpp_exact.grade.json').read_text())['passed']
print('C++ task passes and all generated IDs equal exact baseline')
PY
