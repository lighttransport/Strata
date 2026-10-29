#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${DS41_EXPERIMENT_DEADLINE:?}"
export STRATA_DS41_DECODE_BIN=/tmp/strata-glm-decode-vector-hip
mkdir -p logs
sed '/^STRATA_DS41_DECODE_TIMING=/d' best2n.env > logs/best2n_no_timing.env
sha256sum best2n.env logs/best2n_no_timing.env > logs/t34_config_hashes.txt
export DS41_TRIAL_ENV_FILE=logs/best2n_no_timing.env
unset STRATA_DS41_DECODE_TIMING
for arm in h14_r1 h16_r1 h14_r2 h16_r2; do
  (( $(date +%s) + 180 < DS41_EXPERIMENT_DEADLINE )) || exit 0
  threads=14
  [[ "$arm" == h16* ]] && threads=16
  DS41_DECODE_THREADS=$threads bash production_trial.sh "t34_$arm" STRATA_DS41_MERGE_DIRECT=1 STRATA_DS41_MERGE_VECTOR=1
  if grep -q 'decode_gpu_ms_per_pass' "logs/t34_$arm.log"; then
    echo 'profiling unexpectedly active' >&2; exit 1
  fi
done
