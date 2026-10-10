#!/bin/bash
# Coding-coherence gate on b550: bench 3 fixtures with the step-1 env (+ overrides), then the oracle check.
# Args: tag, extra env (K=V | K=UNSET ...). AFF sets the route affinity (default 0.08).
CAL=/home/syoyo/work/Strata-b550-calibration-15b787b5
ENVJSON=$CAL/docs/fixtures/glm_b550_levers_20261009/step1/step1-b15360-01-base15.config.json
tag=$1; shift; OUT=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010/code-$tag; if [ -e "$OUT" ]; then echo "output already exists: $OUT" >&2; exit 1; fi
eval "$(python3 -c "
import json, shlex, sys; e = json.load(open('$ENVJSON'))['env']; e.pop('STRATA_GLM_STEP_TRACE', None)
for x in sys.argv[1:]:
    k, _, v = x.partition('=')
    if v == 'UNSET': e.pop(k, None)
    else: e[k] = v
print(' '.join('export %s=%s' % (k, shlex.quote(v)) for k, v in e.items()))" "$@")"
while awk '{exit !($1 > 3)}' /proc/loadavg; do sleep 10; done
cd /home/syoyo/work/Strata
systemd-run --user --scope -q -p MemoryMax=60G -p MemorySwapMax=0 \
  python3 tools/glm_q2_coding_bench.py /mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf \
    --decoder build-hip/strata-glm-decode --output $OUT --fixtures ${FIXTURES:-prime json_escape csv} \
    --repetitions 1 --allow-different-trials --no-warm-weights --tokens ${TOKENS:-512} \
    --decoder-flag=--gpu-budget-mib=15360 --decoder-flag=--prefill-experts=mmq --decoder-flag=--prefill-batch=2048 \
    --decoder-flag=--route-affinity=${AFF:-0.08} --decoder-flag=--decode-cache-mib=12288 --decoder-flag=--stop-ids=154820,154827 > $OUT.bench.log 2>&1
echo "bench exit $?"; grep -E "^(prime|json_escape|csv) " $OUT.bench.log
nice -n 15 python3 /home/syoyo/work/kexp/codegate.py $OUT ${FIXTURES:-prime json_escape csv} > $OUT.check.log 2>&1
echo "check exit $?"; tail -5 $OUT.check.log
