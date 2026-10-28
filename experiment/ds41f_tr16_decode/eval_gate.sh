#!/usr/bin/env bash
# eval_gate.sh NAME [ENV=VAL ...]: teacher-forced chunk-5 captures of the six held-out prompts against the exact all-CPU
# references (made once, kept in /mnt/nvme01/tmp/ds41-refs), gate metrics per prompt into logs/gate_NAME.txt
set -u
E=$(cd "$(dirname "$0")" && pwd); cd "$E"; name=$1; shift
R=/mnt/nvme01/tmp/ds41-refs; out=logs/gate_$name.txt; : > "$out"
for p in c_osd c_imgui c_llama c_ff t_mujoco t_doc2; do
  if [[ ! -s $R/ref_$p.f32 ]]; then
    ./run.sh ref_$p eval/$p.txt 1 DS41_DECODE_SAVE_LOGITS=1 DS41_DECODE_QUALITY_CHUNK=5 > /dev/null
    f=$(grep -oE 'teacher_forced_logits=[^ ]+' logs/ref_$p.guard | tail -1 | cut -d= -f2)
    [[ -s $f ]] || { echo "$p: reference capture failed" | tee -a "$out"; continue; }
    mv "$f" "$R/ref_$p.f32"
  fi
  DS41_DECODE_TIMEOUT_SECONDS=1800 ./run.sh g_${name}_$p eval/$p.txt 1 DS41_DECODE_SAVE_LOGITS=1 DS41_DECODE_QUALITY_CHUNK=5 "$@" > /dev/null
  f=$(grep -oE 'teacher_forced_logits=[^ ]+' logs/g_${name}_$p.guard | tail -1 | cut -d= -f2)
  if [[ -s $f ]]; then echo "$p $(./cmp_gate $R/ref_$p.f32 "$f" eval/$p.txt)" | tee -a "$out"; rm -f "$f"
  else echo "$p: candidate capture failed ($(grep -oE 'guard=[a-z_]+' logs/g_${name}_$p.guard))" | tee -a "$out"; fi
done
