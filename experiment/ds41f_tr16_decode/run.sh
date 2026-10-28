#!/usr/bin/env bash
# usage: run.sh NAME PROMPT_IDS OUTPUT_TOKENS [ENV=VAL ...]   (all runs go through tools/ds41f_decode_guard.sh)
# Base = exact canonical-CPU config from resume-ds41f.md; ENV=VAL args override. Logs: $E/logs/NAME.*
set -u
E=$(cd "$(dirname "$0")" && pwd); R=/mnt/nvme02/work/Strata
name=$1 prompt=$2 steps=$3; shift 3
mkdir -p "$E/logs"
export STRATA_DS41_DECODE_BIN=$R/build-hip-rocm101-tr16/strata-glm-decode LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
export DS41_DECODE_CONTEXT=${DS41_DECODE_CONTEXT:-1400} DS41_DECODE_THREADS=${DS41_DECODE_THREADS:-15}
export STRATA_DS41_VERIFY_EXACT=1 STRATA_DS41_TIER_PROMOTE=0 STRATA_DS41_GPU_EXPERT_MIB=0
export STRATA_DS41_EXPERT_PROFILE=/home/syoyo/.claude/jobs/c6566765/tmp/tr16/routes_ref1008.i32
export STRATA_DS41_LANE_SKIP_UNUSED=1 STRATA_DS41_LANE_GROUP_EXACT=1 STRATA_DS41_LANE_SLEEP=0 STRATA_IQ1S_PAIR=1
for kv in "$@"; do export "$kv"; done
export DS41_DECODE_LOG=$E/logs/$name.log DS41_DECODE_SAMPLES=$E/logs/$name.mem
M=/mnt/nvme02/models/ds41f-oct/iq1s/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf
dev=nvme1n1   # /mnt/nvme02 (model drive)
rd() { awk -v d=$dev '$3==d {print $6}' /proc/diskstats; }
s0=$(rd); t0=$(date +%s.%N)
( while sleep 2; do p=$(pgrep -x strata-glm-deco | head -1); [ -n "$p" ] && awk '{print $10}' /proc/$p/stat > "$E/logs/$name.majflt" 2>/dev/null; done ) & smp=$!
"$R/tools/ds41f_decode_guard.sh" "$name" "$M" "$prompt" "$steps" > "$E/logs/$name.guard" 2>&1; rc=$?
kill $smp 2>/dev/null; s1=$(rd); t1=$(date +%s.%N)
gb=$(awk -v a=$s0 -v b=$s1 'BEGIN{printf "%.2f", (b-a)*512/1e9}')
echo "== $name rc=$rc wall=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.0f", b-a}')s model_drive_read=${gb}GB majflt=$(cat $E/logs/$name.majflt 2>/dev/null)"
grep -hoE 'prompt_tok_s=[0-9.]+|decode_tok_s=[0-9.]+|tokens_per_step=[0-9.]+' "$E/logs/$name.log" | tr '\n' ' '; echo
grep -hE 'guard_reason|teacher_forced_logits|min_free|refus|error' "$E/logs/$name.guard" | tail -6
