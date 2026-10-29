#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${DS41_EXPERIMENT_DEADLINE:?}"
check_time() { (( $(date +%s) + 180 < DS41_EXPERIMENT_DEADLINE )); }
# Existing exact one-row launch geometry; compare against the depth-5 controls above.
for repeat in 1 2; do
  check_time || exit 0
  STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-merge-direct \
    ./trial.sh "t31_dense_rows1_r$repeat" STRATA_DS41_MERGE_DIRECT=1 \
    STRATA_DS41_DSPARK_EXTEND=1 STRATA_DS41_DSPARK_N=5 STRATA_NO_MMVQ_ROWS2=1
done
for trim in 0.05 0.10 0.125 0.15; do
  tag=${trim/./p}
  for repeat in 1 2; do
    check_time || { echo 'experiment deadline reached'; exit 0; }
    ./trial.sh "t32_trim_${tag}_r${repeat}" "STRATA_DS41_CPU_TRIM=$trim"
  done
  # A speed screen is not a qualification. Grade every arm reaching the target.
  if python3 - "$tag" <<'PY'
import pathlib,re,sys
rates=[]
for n in (1,2):
    text=pathlib.Path(f'logs/t32_trim_{sys.argv[1]}_r{n}.log').read_text()
    rates.append(float(re.search(r'decode_tok_s=([0-9.]+)',text)[1]))
sys.exit(0 if min(rates)>=30 else 1)
PY
  then
    check_time || exit 0
    bash cpp_gate/run.sh "t32_cpp_trim_$tag" "STRATA_DS41_CPU_TRIM=$trim"
    bash quality_2n.sh "t32_trim_$tag" "STRATA_DS41_CPU_TRIM=$trim"
  fi
done
# Dense trunk compression is separately opt-in; retain the coding gate on each fast arm.
for trim in 0 0.10 0.15; do
  tag=${trim/./p}
  for repeat in 1 2; do
    check_time || exit 0
    ./trial.sh "t32_q4_trim_${tag}_r${repeat}" STRATA_DS41_DENSE_Q4K=1 "STRATA_DS41_CPU_TRIM=$trim"
  done
  if python3 - "$tag" <<'PY'
import pathlib,re,sys
rates=[float(re.search(r'decode_tok_s=([0-9.]+)',pathlib.Path(f'logs/t32_q4_trim_{sys.argv[1]}_r{n}.log').read_text())[1]) for n in (1,2)]
sys.exit(0 if min(rates)>=30 else 1)
PY
  then
    check_time || exit 0
    bash cpp_gate/run.sh "t32_cpp_q4_trim_$tag" STRATA_DS41_DENSE_Q4K=1 "STRATA_DS41_CPU_TRIM=$trim"
    bash quality_2n.sh "t32_q4_trim_$tag" STRATA_DS41_DENSE_Q4K=1 "STRATA_DS41_CPU_TRIM=$trim"
  fi
done
