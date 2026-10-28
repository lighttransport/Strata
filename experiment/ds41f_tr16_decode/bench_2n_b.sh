#!/usr/bin/env bash
# exactness capture + load-share sweep (TR16 + B550 STD2, tier pre-fill, GPU prompt chunks); logs/bench_2n_b.txt
set -u
E=$(cd "$(dirname "$0")" && pwd); cd "$E"
out=logs/bench_2n_b.txt
B=$(tr '\n' ' ' < best.env)
W='--threads=15 --weights-dir=/dev/shm --reserve-gib=4 --gpu-reserve-mib=1024'
ref() { case $1 in c_osd) echo base_greedy_osd;; c_llama) echo base_greedy_llama;; esac; }
line() {
  local n=$1 p=$2
  local same=$( [ "$(grep '^tokens:' logs/$n.log | head -1)" = "$(grep '^tokens:' logs/$(ref $p).log | head -1)" ] && echo same || echo DIFF )
  echo "$n $(grep -oE 'decode_tok_s=[0-9.]+|prompt_tok_s=[0-9.]+' logs/$n.log | head -2 | tr '\n' ' ') wait=$(grep -oE 'wait_ms_per_call=[0-9.]+' logs/$n.log | tail -1) share=$(grep -oE 'load_share=[0-9.]+' logs/$n.log | head -1) lane=$(grep -E 'decode_gpu_ms' logs/$n.log | grep -oE 'lane [0-9.]+' | head -1) ids=$same | $(grep -oE 'tier_hit=[0-9.]+|decode_ms=[0-9.]+' logs/$n.run | tr '\n' ' ')" | tee -a "$out"
}
# 2. load-share sweep at 40 GiB
for v in 9 1; do
  for p in c_osd c_llama; do
    n=gmin${v}_$p
    MODE=own W_ENV="STRATA_DS41_WORKER_GPU_MIN_NT=$v" W_BIN='~/work/strata-own-build/strata-ds41-worker' W_MAX=52G ./two_node.sh $n eval/$p.txt 128 "$W" -- \
      STRATA_DS41_REMOTE_OWN_MIB=40960 $B > logs/$n.run 2>&1
    line $n $p
  done
done
ssh -o BatchMode=yes b550 'rm -f /dev/shm/strata-tp-*.weights'
echo BENCH_DONE | tee -a "$out"
