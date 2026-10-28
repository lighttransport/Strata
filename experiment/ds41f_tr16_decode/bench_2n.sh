#!/usr/bin/env bash
# single node vs TR16+B550 STD2 at several owned sizes; one line per run into logs/bench_2n.txt
set -u
E=$(cd "$(dirname "$0")" && pwd); cd "$E"
out=logs/bench_2n.txt; : > "$out"
S=/mnt/nvme01/tmp/claude-1000/-mnt-nvme02-work-Strata/f721d25a-111a-4892-8bfd-e15ef411b2ff/scratchpad
B=$(tr '\n' ' ' < best.env)
ref() { case $1 in c_osd) echo base_greedy_osd;; c_llama) echo base_greedy_llama;; esac; }
line() {   # name prompt
  local n=$1 p=$2
  local same=$( [ "$(grep '^tokens:' logs/$n.log | head -1)" = "$(grep '^tokens:' logs/$(ref $p).log | head -1)" ] && echo same || echo DIFF )
  echo "$n $p $(grep -oE 'decode_tok_s=[0-9.]+|tokens_per_step=[0-9.]+' logs/$n.log | head -2 | tr '\n' ' ') verify_s=$(grep -oE 'verify_s=[0-9.]+' logs/$n.log | head -1 | cut -d= -f2) lane=$(grep -E 'decode_gpu_ms' logs/$n.log | grep -oE 'lane [0-9.]+' | head -1) drive_GB=$(grep -oE 'model_drive_read=[0-9.]+' logs/$n.run 2>/dev/null) ids=$same" | tee -a "$out"
}
systemd-run --user --scope -q -u ds41-prewarm-bench $S/venv/bin/python -I prewarm_experts.py /mnt/nvme02/models/ds41f-oct/iq1s/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf > logs/bench_prewarm.txt 2>&1
for p in c_osd c_llama; do
  ./run.sh s1_$p eval/$p.txt 128 $B > logs/s1_$p.run 2>&1; line s1_$p $p
done
for mib in 8192 24576 40960; do
  for p in c_osd c_llama; do
    n=o${mib}_$p
    MODE=own W_BIN='~/work/strata-own-build/strata-ds41-worker' W_MAX=52G ./two_node.sh $n eval/$p.txt 128 \
      "--threads=15 --weights-dir=/dev/shm --reserve-gib=4 --gpu-reserve-mib=1024" -- STRATA_DS41_REMOTE_OWN_MIB=$mib $B > logs/$n.run 2>&1
    line $n $p
    echo "   worker: $(grep -E 'DS41_WORKER (done|gpu tier)' logs/$n.run | tr '\n' ' ' | cut -c1-260)" | tee -a "$out"
  done
done
ssh -o BatchMode=yes b550 'rm -f /dev/shm/strata-tp-*.weights'
echo BENCH_DONE | tee -a "$out"
