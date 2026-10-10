#!/bin/bash
# Teacher-forced KL on the 4-sequence subset. Args: tag, mode (save|ref), extra env K=V ... ; AFF env sets affinity.
CAL=/home/syoyo/work/Strata-b550-calibration-15b787b5
ENVJSON=$CAL/docs/fixtures/glm_b550_levers_20261009/step1/step1-b15360-01-base15.config.json
tag=$1; mode=$2; shift 2; OUT=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010/kl-$tag
REF=${REF:-/mnt/disk01/kexp/eval4-aff0.logits}
pack_args=()
if [ -n "${EXPERT_PACK:-}" ]; then pack_args=(--expert-pack="$EXPERT_PACK"); fi
SETENV=$(python3 -c "
import json, sys; e = json.load(open('$ENVJSON'))['env']; e.pop('STRATA_GLM_STEP_TRACE', None)
for x in sys.argv[1:]:
    k, _, v = x.partition('=')
    if v == 'UNSET': e.pop(k, None)
    else: e[k] = v
print(' '.join('--setenv=%s=%s' % kv for kv in e.items()))" "$@")
if [ "$mode" = save ]; then EV="--eval-save-logits=$REF"; else EV="--eval-reference=$REF"; fi
while awk '{exit !($1 > 3)}' /proc/loadavg; do sleep 10; done
start=$(date +%s)
systemd-run --user --wait --pipe --unit=strata-kl-$tag-$$ --property=MemoryMax=60G --property=MemorySwapMax=0 --property=WorkingDirectory=$CAL $SETENV \
  numactl --interleave=all ${EXE:-/home/syoyo/work/Strata/build-hip/strata-glm-decode} /mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf \
  0 1 4096 12 --context=4096 --prefill-batch=2048 --gpu-budget-mib=${GPU_BUDGET_MIB:-15360} \
  --cpu-affinity=auto --decode-experts=cpu --prefill-experts=mmq --route-affinity=${AFF:-0} --decode-graphs --decode-cache-mib=12288 \
  --eval-corpus=${CORPUS:-/home/syoyo/work/kexp/eval4.ids} $EV "${pack_args[@]}" > $OUT.out 2> $OUT.err
echo "$tag exit $? $(( $(date +%s) - start )) s: $(tail -1 $OUT.out)"
