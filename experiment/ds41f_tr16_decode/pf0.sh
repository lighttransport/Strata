#!/usr/bin/env bash
# pf0.sh NAME BATCH [ENV=VAL ...]: Stage 0(a) of the prefill plan: the B550 decoder alone (no worker, no remote experts: the owned experts are zero holes in the
# sparse tmpfs shards, so the results are garbage but the GPU / PCIe / copy timing of a streamed prefill is real). PROMPT_FILE (text) is copied to the B550.
cd "$(dirname "$0")"
source ./connection.sh
name=$1 batch=$2; shift 2
P=${PROMPT_FILE:-/mnt/nvme01/tmp/claude-1000/-mnt-nvme02-work-Strata/808a845c-499a-49b7-86f6-4b9195dd0fa8/scratchpad/long_prompt.txt}
D=/dev/shm/ds41rev; AUX=/dev/shm/ds41rev-aux
scp -q "$P" "$GPU_HOST":$AUX/pf_prompt.txt
first=$(basename "$(cat logs/model.local)")
ssh -o BatchMode=yes "$GPU_HOST" "pgrep -x strata-glm-deco >/dev/null" && { echo "a decoder is already running on the GPU node" >&2; exit 1; }
ssh -n -f -o BatchMode=yes "$GPU_HOST" "for i in \$(seq 1 600); do u=\$(df --output=used -BG /dev/shm | tail -1 | tr -dc 0-9); if [ \${u:-0} -gt 43 ]; then pkill -x strata-glm-deco; echo killed-shm > /tmp/pf0-watchdog; break; fi; pgrep -x strata-glm-deco >/dev/null || [ \$i -lt 30 ] || break; sleep 2; done" 2>/dev/null
ssh -o BatchMode=yes "$GPU_HOST" "cd /tmp && XDG_RUNTIME_DIR=/run/user/\$(id -u) systemd-run --user --scope -q -u pf0-$name -p MemoryMax=${D_MAX:-20G} -p MemorySwapMax=0 env \
  STRATA_DS41_DESKTOP_MIB=512 STRATA_DS41_PROMPT_BATCH=$batch STRATA_DS41_STREAM_MIN=1024 STRATA_DS41_DECODE_TIMING=1 STRATA_DS41_PREFILL_TIMING=1 STRATA_DS41_STAGE_SKIP=1 STRATA_DS41_PIN_GB=0 STRATA_DS41_PIN_EXPERTS=0 STRATA_DS41_POPULATE=0 STRATA_DS41_RAM_COPY=0 STRATA_DS41_TIER_PROMOTE=0 STRATA_DS41_GPU_EXPERT_MIB=0 $* \
  ${NSYS:-} $B_BIN/strata-glm-decode $D/$first @$AUX/pf_prompt.txt ${STEPS:-1} --context=${CTX:-18000} --threads=16 > /tmp/pf0-$name.log 2>&1"
rc=$?
ssh -o BatchMode=yes "$GPU_HOST" "df -h /dev/shm | tail -1"
ssh -o BatchMode=yes "$GPU_HOST" "cat /tmp/pf0-$name.log" > logs/pf0-$name.log
echo "== $name rc=$rc"; grep -hE "prompt=|prompt_tok_s|error|ds41:|refus" logs/pf0-$name.log | tail -4 | cut -c1-300
