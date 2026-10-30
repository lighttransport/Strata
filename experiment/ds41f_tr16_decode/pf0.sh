#!/usr/bin/env bash
# pf0.sh NAME BATCH [ENV=VAL ...]: the GPU / PCIe ceiling of a streamed prefill on the reversed layout: the B550 decoder alone (no worker, no remote experts). The experts
# the worker owns are zero holes in the sparse tmpfs shards and are never read (STRATA_DS41_STAGE_SKIP=1: the host slots keep whatever they held), so the generated
# tokens are garbage but the trunk / MMQ / PCIe timing is real. A mapped hole read would materialize its pages and fill /dev/shm, hence the watchdog below.
#   PROMPT_FILE  a token-id list, whitespace separated (default: the repository's eval id lists, ~12K tokens)   CTX  context (18000)   STEPS  tokens to generate (1)   D_MAX  memory cap of the scope (20G)
#   e.g. ./pf0.sh h12k 12288 STRATA_DS41_PREFILL_HALVES=1 STRATA_DS41_PREFILL_TILE=2048 STRATA_DS41_ATTN_MMA=1
cd "$(dirname "$0")"
source ./connection.sh
name=$1 batch=$2; shift 2
if [ -n "${PROMPT_FILE:-}" ]; then P=$PROMPT_FILE; else P=$(mktemp "${TMPDIR:-/tmp}/ds41_pf_prompt.XXXXXX"); trap 'rm -f "$P"' EXIT; for f in c_osd c_ff c_imgui c_llama t_doc2 t_mujoco c_osd c_ff c_imgui c_llama t_doc2 t_mujoco; do cat eval/$f.txt; echo; done > "$P"; fi
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
