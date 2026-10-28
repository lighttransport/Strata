#!/usr/bin/env bash
# larger B550 shares (anonymous rows) + the decode-only split; logs/bench_2n_c.txt
set -u
E=$(cd "$(dirname "$0")" && pwd); cd "$E"
out=logs/bench_2n_d.txt; : > "$out"
B=$(tr '\n' ' ' < best.env)
W='--threads=15 --rows=anon --reserve-gib=3 --gpu-reserve-mib=1024'
ref() { case $1 in c_osd) echo base_greedy_osd;; c_llama) echo base_greedy_llama;; esac; }
for st in 2 0; do mib=53248
  for p in c_osd c_llama; do
    n=st${st}_$p
    ssh -o BatchMode=yes b550 "awk '/MemAvailable/ {printf \"b550 MemAvailable %.1f GiB\\n\", \$2/1048576}' /proc/meminfo" >> "$out"
    MODE=own W_BIN='~/work/strata-own-build/strata-ds41-worker' W_MAX=58G ./two_node.sh $n eval/$p.txt 128 "$W" -- STRATA_DS41_REMOTE_OWN_MIB=$mib $B STRATA_DS41_CANON_STREAM=$st > logs/$n.run 2>&1
    same=$( [ "$(grep '^tokens:' logs/$n.log | head -1)" = "$(grep '^tokens:' logs/$(ref $p).log | head -1)" ] && echo same || echo DIFF )
    {
      echo "$n $(grep -oE 'decode_tok_s=[0-9.]+|prompt_tok_s=[0-9.]+' logs/$n.log | head -2 | tr '\n' ' ') ids=$same guard=$(grep -oE 'guard=[a-z_]+' logs/$n.guard | tail -1) drive_GB=$(grep -oE 'model_drive_read=[0-9.]+' logs/$n.run | cut -d= -f2)"
      echo "   tr16: $(grep -E 'dspark steps' logs/$n.log | grep -oE 'draft_s=[0-9.]+|verify_s=[0-9.]+|verify_ms_by_width=[0-9.,]+' | tr '\n' ' ')"
      echo "   tr16 pass: $(grep -E 'decode_gpu_ms' logs/$n.log | cut -c1-420)"
      echo "   tr16 remote: $(grep -E 'remote_own_decode' logs/$n.log | cut -c1-250)"
      echo "   worker: $(grep -E 'DS41_WORKER (rows|decode_requests|error)' logs/$n.run | tr '\n' ' ' | cut -c1-400)"
    } | tee -a "$out"
  done
done
echo BENCH_DONE | tee -a "$out"
