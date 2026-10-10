#!/bin/bash
# HumanEval pass@1 on the first LIMIT tasks with the step-1 env (+ overrides). Args: tag, extra env (K=V ...).
# AFF sets --route-affinity; LIMIT the task count (default 30).
CAL=/home/syoyo/work/Strata-b550-calibration-15b787b5
ENVJSON=$CAL/docs/fixtures/glm_b550_levers_20261009/step1/step1-b15360-01-base15.config.json
tag=$1; shift; OUT=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010/he-$tag; if [ -e "$OUT" ]; then echo "output already exists: $OUT" >&2; exit 1; fi
pack_args=()
if [ -n "${EXPERT_PACK:-}" ]; then pack_args=(--expert-pack "$EXPERT_PACK"); fi
eval "$(python3 -c "
import json, shlex, sys; e = json.load(open('$ENVJSON'))['env']; e.pop('STRATA_GLM_STEP_TRACE', None)
for x in sys.argv[1:]:
    k, _, v = x.partition('=')
    if v == 'UNSET': e.pop(k, None)
    else: e[k] = v
print(' '.join('export %s=%s' % (k, shlex.quote(v)) for k, v in e.items()))" "$@")"
while awk '{exit !($1 > 3)}' /proc/loadavg; do sleep 10; done
cd /home/syoyo/work/Strata
start=$(date +%s)
systemd-run --user --scope -q -p MemoryMax=60G -p MemorySwapMax=0 \
  python3 tools/humaneval_b550.py /mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf /mnt/disk01/kexp/HumanEval.jsonl.gz \
    --output $OUT --decoder build-hip/strata-glm-decode --limit ${LIMIT:-30} --offset ${OFFSET:-0} --tokens 1024 --speculative ${SPECULATIVE:-none} --draft-depth ${DRAFT_DEPTH:-1} \
    --gpu-budget-mib ${GPU_BUDGET_MIB:-15360} --decode-cache-mib 12288 --route-affinity ${AFF:-0} "${pack_args[@]}" > $OUT.log 2>&1
echo "$tag exit $? $(( $(date +%s) - start )) s: $(python3 -c "
import json; m = json.load(open('$OUT/measurement.json')); r = m['results']
print('pass@1 %.3f (%d/%d), mean tokens %.0f' % (m['pass_at_1'], sum(x['passed'] for x in r), len(r), sum(x['generated_tokens'] for x in r) / len(r)))" 2>&1)"
