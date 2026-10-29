#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${DS41_EXPERIMENT_DEADLINE:?}"
if [[ -n ${DS41_PREVIOUS_SCREEN_PID:-} ]]; then
 while kill -0 "$DS41_PREVIOUS_SCREEN_PID" 2>/dev/null; do sleep 10; done
fi
check_time() { (( $(date +%s) + 180 < DS41_EXPERIMENT_DEADLINE )); }
python3 - <<'CHECK'
import pathlib,json,re
r=pathlib.Path('logs')
for mode in (0,1):
 for prompt in ('c_osd','c_llama'):
  assert 'byte-identical' in (r/f't38_grid8_{mode}_gate_{prompt}.parity').read_text()
assert json.loads(pathlib.Path('cpp_gate/t38_cpp_grid8_v2.grade.json').read_text())['passed']
assert re.search(r'^tokens: .+$',(r/'t31_cpp_exact.log').read_text(),re.M)[0]==re.search(r'^tokens: .+$',(r/'t38_cpp_grid8_v2.log').read_text(),re.M)[0]
CHECK
export STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-grid8-hip DS41_WORKER_BIN=/tmp/strata-ds41-worker-grid8
export STRATA_DS41_MERGE_VECTOR=1 DS41_DECODE_THREADS=14 DS41_GATE_CANON_MIB=2304
export DS41_TRIAL_ENV_FILE=logs/best2n_no_timing.env DS41_GATE_ENV_FILE=logs/best2n_no_timing.env
unset STRATA_DS41_DECODE_TIMING
# Read only completed, qualified grid controls. CPU time distinguishes a kernel
# effect from unrelated variation in the shared desktop's GPU and merge waits.
mode=$(python3 - <<'PY'
import pathlib,json,re,statistics
r=pathlib.Path('logs');v={}
for mode in (0,1):
 rows=[]
 for rep in (1,2):
  n=f't38_grid8_{mode}_r{rep}';s=(r/f'{n}.log').read_text()
  assert 'guard=none' in (r/f'{n}.guard').read_text()
  rows.append({'tok_s':float(re.search(r'decode_tok_s=([0-9.]+)',s)[1]),'worker_cpu_ms':float(re.search(r'cpu_lane=([0-9.]+)',(r/f'{n}.run').read_text())[1])})
 v[mode]=rows
mean=lambda m,k:statistics.mean(q[k] for q in v[m])
selected=int(mean(1,'worker_cpu_ms')<mean(0,'worker_cpu_ms') and mean(1,'tok_s')>=.99*mean(0,'tok_s'))
(r/'t39_grid_selection.json').write_text(json.dumps({'runs':v,'selected':selected,'scope':'research candidate; defaults unchanged'},indent=2)+'\n');print(selected)
PY
)
export STRATA_IQ1S_GRID8=$mode W_ENV=STRATA_IQ1S_GRID8=$mode
qualified() {
 python3 - "$1" <<'PY'
import pathlib,re,sys
r=pathlib.Path('logs');n=sys.argv[1];s=(r/f'{n}.log').read_text()
assert 'guard=none' in (r/f'{n}.guard').read_text()
assert 'decode_gpu_ms_per_pass' not in s
assert re.search(r'^tokens: .+$',s,re.M)[0]==re.search(r'^tokens: .+$',(r/'base_greedy_osd.log').read_text(),re.M)[0]
PY
}
source ./draft_q8_screen.sh
if [[ $q8 == 1 ]]; then
 export DS41_THREAD_TRIAL_PREFIX=t42
else
 export DS41_THREAD_TRIAL_PREFIX=t39
fi
for threads in 32; do
 check_time || exit 0
 export W_ARGS="--threads=$threads --weights-dir=/dev/shm --reserve-gib=3 --gpu-reserve-mib=1024"
 if [[ ! -e logs/${DS41_THREAD_TRIAL_PREFIX}_worker${threads}_r1.log ]]; then
  bash production_trial.sh "${DS41_THREAD_TRIAL_PREFIX}_worker${threads}_r1" STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=$mode
 fi
 qualified "${DS41_THREAD_TRIAL_PREFIX}_worker${threads}_r1"
 # Repeat only an arm whose CPU kernel and decode can plausibly beat the control.
 if python3 - "$threads" "$mode" "$DS41_THREAD_TRIAL_PREFIX" "$q8" <<'PY'
import pathlib,re,statistics,sys
r=pathlib.Path('logs');t,m,prefix,q8=sys.argv[1:];n=f'{prefix}_worker{t}_r1'
rate=float(re.search(r'decode_tok_s=([0-9.]+)',(r/f'{n}.log').read_text())[1]);cpu=float(re.search(r'cpu_lane=([0-9.]+)',(r/f'{n}.run').read_text())[1])
oldcpu=statistics.mean(float(re.search(r'cpu_lane=([0-9.]+)',(r/(f't38_grid8_{m}_r{i}.run' if q8=='0' else f't41_draftq8_1_r{i}.run')).read_text())[1]) for i in (1,2))
oldrate=statistics.mean(float(re.search(r'decode_tok_s=([0-9.]+)',(r/(f't38_grid8_{m}_r{i}.log' if q8=='0' else f't41_draftq8_1_r{i}.log')).read_text())[1]) for i in (1,2))
raise SystemExit(0 if cpu<oldcpu and rate>.99*oldrate else 1)
PY
 then
  check_time || exit 0
  bash production_trial.sh "${DS41_THREAD_TRIAL_PREFIX}_worker${threads}_r2" STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=$mode
  qualified "${DS41_THREAD_TRIAL_PREFIX}_worker${threads}_r2"
 fi
done
threads=$(python3 - "$mode" "$DS41_THREAD_TRIAL_PREFIX" "$q8" <<'PY'
import pathlib,re,statistics,json,sys
r=pathlib.Path('logs');m,prefix,q8=sys.argv[1:];rows={16:[]}
for rep in (1,2):
 n=f't38_grid8_{m}_r{rep}' if q8=='0' else f't41_draftq8_1_r{rep}';rows[16].append((float(re.search(r'decode_tok_s=([0-9.]+)',(r/f'{n}.log').read_text())[1]),float(re.search(r'cpu_lane=([0-9.]+)',(r/f'{n}.run').read_text())[1])))
for t in (8,32):
 if (r/f'{prefix}_worker{t}_r2.log').exists():
  rows[t]=[(float(re.search(r'decode_tok_s=([0-9.]+)',(r/f'{prefix}_worker{t}_r{i}.log').read_text())[1]),float(re.search(r'cpu_lane=([0-9.]+)',(r/f'{prefix}_worker{t}_r{i}.run').read_text())[1])) for i in (1,2)]
mean=lambda t,k:statistics.mean(v[k] for v in rows[t])
valid=[16]+[t for t in rows if t!=16 and mean(t,0)>1.01*mean(16,0) and mean(t,1)<mean(16,1)]
best=max(valid,key=lambda t:mean(t,0));(r/'t39_worker_selection.json').write_text(json.dumps({'runs':rows,'threads':best},indent=2)+'\n');print(best)
PY
)
export W_ARGS="--threads=$threads --weights-dir=/dev/shm --reserve-gib=3 --gpu-reserve-mib=1024" DS41_GATE_WORKER_ARGS="--threads=$threads --weights-dir=/dev/shm --reserve-gib=3 --gpu-reserve-mib=1024"
for rep in 1 2; do
 check_time || exit 0
 bash production_trial.sh "t39_validate_r$rep" STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=$mode
 qualified "t39_validate_r$rep"
done
if [[ $threads != 16 ]]; then
 check_time || exit 0
 DS41_GATE_CANON_MIB=3328 DS41_MERGE_GATE_PREFIX=t39_worker_gate bash merge_gate.sh
 check_time || exit 0
 bash cpp_gate/production_run.sh t39_cpp_exact STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=$mode
 python3 - <<'CHECK'
import pathlib,json,re
r=pathlib.Path('logs')
assert json.loads(pathlib.Path('cpp_gate/t39_cpp_exact.grade.json').read_text())['passed']
assert re.search(r'^tokens: .+$',(r/'t31_cpp_exact.log').read_text(),re.M)[0]==re.search(r'^tokens: .+$',(r/'t39_cpp_exact.log').read_text(),re.M)[0]
CHECK
fi
check_time || exit 0
DS41_GATE_CANON_MIB=2304 DS41_MERGE_GATE_PREFIX=t40_cache_gate bash merge_gate.sh
# Lower route trimming, including draft widths that were slower on the untrimmed
# CPU-bound path. All settings remain lossy until separately graded and measured.
for arm in n5_0125 n4_0125 n6_0125 n5_0135 n5_014; do
 n=${arm%%_*};n=${n#n};label=${arm#*_}
 case $label in 0125) trim=0.125;; 0135) trim=0.135;; 014) trim=0.14;; esac
 for rep in 1 2; do
  check_time || exit 0
  bash production_trial.sh "t40_${arm}_r$rep" STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=$mode STRATA_DS41_CPU_TRIM=$trim STRATA_DS41_DSPARK_N=$n STRATA_DS41_CANON_MIB=2304
 done
 if python3 - "$arm" <<'PY'
import pathlib,re,sys
r=pathlib.Path('logs')
for i in (1,2):
 n=f't40_{sys.argv[1]}_r{i}';s=(r/f'{n}.log').read_text();assert 'guard=none' in (r/f'{n}.guard').read_text()
 assert 'decode_gpu_ms_per_pass' not in s
 if float(re.search(r'decode_tok_s=([0-9.]+)',s)[1])<30:sys.exit(1)
PY
 then
  check_time || exit 0
  bash cpp_gate/production_run.sh "t40_cpp_$arm" STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=$mode STRATA_DS41_CPU_TRIM=$trim STRATA_DS41_DSPARK_N=$n STRATA_DS41_CANON_MIB=2304
  if ! python3 - "$arm" <<'CHECK'
import pathlib,json,sys
raise SystemExit(0 if json.loads(pathlib.Path(f'cpp_gate/t40_cpp_{sys.argv[1]}.grade.json').read_text())['passed'] else 1)
CHECK
  then
   echo "$arm C++ task failed; candidate not qualified" >&2
   continue
  fi
  # Target-only quality has no draft depth, so evaluate each trim only once.
  if [[ ! -e logs/quality_t40_trim${label}.txt ]]; then
   bash quality_2n.sh "t40_trim$label" STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=$mode STRATA_DS41_CPU_TRIM=$trim STRATA_DS41_CANON_MIB=2304
  fi
 fi
done
