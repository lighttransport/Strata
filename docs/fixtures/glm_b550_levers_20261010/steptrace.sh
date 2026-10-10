#!/bin/bash
# One 64-token decode (step-1 env, mmq prefill), printing STEP_TRACE per step. Args: tag, extra env (K=V ...).
CAL=/home/syoyo/work/Strata-b550-calibration-15b787b5
ENVJSON=$CAL/docs/fixtures/glm_b550_levers_20261009/step1/step1-b15360-01-base15.config.json
tag=$1; shift; OUT=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010/st-$tag.err
pack_args=()
if [ -n "${EXPERT_PACK:-}" ]; then pack_args=(--expert-pack="$EXPERT_PACK"); fi
spec_args=()
if [ "${SPECULATIVE:-none}" = mtp ]; then spec_args=(--speculative=mtp --draft-depth="${DRAFT_DEPTH:-1}" --mtp-experts=cpu); fi
SETENV=$(python3 -c "
import json, sys; e = json.load(open('$ENVJSON'))['env']
for x in sys.argv[1:]:
    k, _, v = x.partition('=')
    if v == 'UNSET': e.pop(k, None)
    else: e[k] = v
print(' '.join('--setenv=%s=%s' % kv for kv in e.items()))" "$@")
while awk '{exit !($1 > 3)}' /proc/loadavg; do sleep 10; done
systemd-run --user --wait --pipe --unit=strata-st-$tag-$$ --property=MemoryMax=60G --property=MemorySwapMax=0 --property=WorkingDirectory=$CAL $SETENV \
  numactl --interleave=all ${EXE:-/home/syoyo/work/Strata/build-hip/strata-glm-decode} /mnt/disk01/models/glm53f-reap/GLM-5.3-Flash-REAP50-Q23-assembled.gguf \
  @/home/syoyo/work/Strata/build-hip-glm-v11/single-2048.ids 128 4096 ${THREADS:-12} --context=4096 --prefill-batch=2048 --gpu-budget-mib=${GPU_BUDGET_MIB:-15360} \
  --cpu-affinity=auto --decode-experts=cpu --prefill-experts=mmq --route-affinity=${AFF:-0.08} --decode-graphs --decode-cache-mib=12288 \
  "${pack_args[@]}" "${spec_args[@]}" > ${OUT%.err}.out 2> $OUT
python3 - "$tag" "$OUT" <<'PY'
import re, sys
t = open(sys.argv[2]).read()
st = re.findall(r"STEP_TRACE steps=(\d+) .*?head_ms=([\d.]+) cpu_ms=([\d.]+) between_ms=([\d.]+) tail_ms=([\d.]+)", t)
d = re.findall(r"^(?:DECODE steps=\d+|SPECULATIVE source=mtp .*?) ms=[\d.]+ tok_s=([\d.]+)", t, re.M)
if st:
    s, h, c, b, tl = map(float, st[-1])
    print(f"{sys.argv[1]:22s} tok/s {d[-1] if d else '?':>8s}  per token: cpu {c/s:5.1f}  between {b/s:5.1f}  head {h/s:4.1f}  tail {tl/s:4.1f} ms")
else:
    print(sys.argv[1], "no STEP_TRACE;", t.strip().splitlines()[-1] if t.strip() else "empty")
    raise SystemExit(1)
PY
