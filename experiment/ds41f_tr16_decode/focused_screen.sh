#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${DS41_EXPERIMENT_DEADLINE:?}"
check_time() { (( $(date +%s) + 180 < DS41_EXPERIMENT_DEADLINE )); }
# An isolated small kernel benchmark before loading the next model run.
check_time || exit 0
python3 - <<'VRAM'
from pathlib import Path
import re, subprocess
source=Path('../../tools/ds41f_decode_guard.sh').read_text()
pattern=re.search(r"readonly KERNEL_BAD_PATTERN='([^']+)'", source)[1]
journal=subprocess.run(['journalctl','--boot=all','--since','24 hours ago','--no-pager','-o','cat'],capture_output=True,text=True,check=True).stdout
assert not re.search(pattern,journal), 'kernel benchmark blocked by recent fault/pressure event'
free=[]
for device in Path('/sys/class/drm').glob('card[0-9]*/device'):
    total=device/'mem_info_vram_total'; used=device/'mem_info_vram_used'
    if total.exists() and used.exists():
        free.append(int(total.read_text())-int(used.read_text()))
assert free and min(free)>=3*1024**3, 'kernel benchmark requires at least 3 GiB free VRAM'
VRAM
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 \
  env LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib timeout 60 /tmp/ds41-kbench-cr4 \
  > logs/t33_kbench_cr4.txt 2>&1
export STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-merge-direct
# Qualify the register-merge implementation before timing it.
check_time || exit 0
STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-vector-hip DS41_DECODE_THREADS=14 \
  DS41_MERGE_GATE_PREFIX=t33_vector_gate STRATA_DS41_MERGE_VECTOR=1 bash merge_gate.sh
for arm in off_r1 on_r1 off_r2 on_r2; do
  check_time || exit 0
  vector=0
  [[ "$arm" == on* ]] && vector=1
  STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-vector-hip \
    ./trial.sh "t33_vector_$arm" STRATA_DS41_MERGE_DIRECT=1 "STRATA_DS41_MERGE_VECTOR=$vector"
done
# Carry the register path forward only after qualified exact A/B measurements.
if python3 - <<'SELECT'
import json,pathlib,re,statistics
r=pathlib.Path('logs'); rates={}; expected=re.search(r'^tokens: .+$',(r/'base_greedy_osd.log').read_text(),re.M)[0]
for mode in ('off','on'):
    rates[mode]=[]
    for repeat in (1,2):
        name=f't33_vector_{mode}_r{repeat}'
        text=(r/f'{name}.log').read_text()
        assert 'guard=none' in (r/f'{name}.guard').read_text()
        assert re.search(r'^tokens: .+$',text,re.M)[0]==expected
        rates[mode].append(float(re.search(r'decode_tok_s=([0-9.]+)',text)[1]))
selected=(statistics.mean(rates['on'])>1.01*statistics.mean(rates['off']) and min(rates['on'])>=min(rates['off']))
(r/'t33_vector_selection.json').write_text(json.dumps({'rates':rates,'selected':selected},indent=2)+'\n')
raise SystemExit(0 if selected else 1)
SELECT
then
  export STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-vector-hip STRATA_DS41_MERGE_VECTOR=1
else
  export STRATA_DS41_MERGE_VECTOR=0
fi
# Remove GPU event instrumentation for a production-speed exact control.
for repeat in 1 2; do
  check_time || exit 0
  ./trial.sh "t33_no_timing_r$repeat" STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_DECODE_TIMING=0
done
# Recheck the local lane balance after removing merge work and profiling overhead.
for repeat in 1 2; do
  check_time || exit 0
  DS41_DECODE_THREADS=16 ./trial.sh "t33_host16_r$repeat" \
    STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_DECODE_TIMING=0
done
for q4 in 0 1; do
  for repeat in 1 2; do
    check_time || exit 0
    ./trial.sh "t33_merge_trim125_q4${q4}_r$repeat" STRATA_DS41_MERGE_DIRECT=1 \
      STRATA_DS41_CPU_TRIM=0.125 "STRATA_DS41_DENSE_Q4K=$q4"
  done
  check_time || exit 0
  bash cpp_gate/run.sh "t33_cpp_trim125_q4$q4" STRATA_DS41_MERGE_DIRECT=1 \
    STRATA_DS41_CPU_TRIM=0.125 "STRATA_DS41_DENSE_Q4K=$q4"
  bash quality_2n.sh "t33_trim125_q4$q4" STRATA_DS41_MERGE_DIRECT=1 \
    STRATA_DS41_CPU_TRIM=0.125 "STRATA_DS41_DENSE_Q4K=$q4"
done
# Pure placement/prefetch changes remain exact: test the worker while the host stays on the proven merge path.
for prefetch in 0 4096; do
  for repeat in 1 2; do
    check_time || exit 0
    W_ENV="STRATA_IQ_PREFETCH=$prefetch" ./trial.sh "t33_worker_pf${prefetch}_r$repeat" STRATA_DS41_MERGE_DIRECT=1
  done
done
# Full SMT coverage avoids the asymmetric core load of the earlier 24-thread screen.
for repeat in 1 2; do
  check_time || exit 0
  W_ARGS='--threads=32 --weights-dir=/dev/shm --reserve-gib=3 --gpu-reserve-mib=1024' \
    ./trial.sh "t33_worker32_r$repeat" STRATA_DS41_MERGE_DIRECT=1
done
