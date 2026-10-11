#!/bin/bash
# Sequential validation chain, relaunched after the Chrome GPU interference. Each step waits up to an hour for a quiet GPU.
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010; PY=build-exl3-venv/bin/python
log() { echo "$1 $(date +%T)" >> $D/stage3.out; }
# 1. mix2 + top-k 6 (guard): held-out 2K/8K 512x3 (decides whether the only in-gate candidate meets speed)
$PY -u $D/finalize.py --cases g7-mix2-topk6-guard4 --base base-soft4-guard.json --skip-train --tag g >> $D/stage3.out 2>&1; log "EXIT $? finalize-g7"
# 2. HumanEval rerun for affinity 0.14 + top-k 6 (guard), fresh name
$PY -u $D/quality-ladder.py --config $D/s3a-2k/g6-aff014-topk6-guard4.json --name g6-aff014-topk6-guard4-he2 --track 35 --steps full164 --continue-on-fail >> $D/stage3.out 2>&1; log "EXIT $? ladder-g6-he2"
# 3. mix2 full corpus
$PY -u $D/quality-ladder.py --config $D/s3a-2k/g7-mix2-topk6-guard4.json --name g7-mix2-topk6-guard4-full --track 35 --steps full8192 --continue-on-fail >> $D/stage3.out 2>&1; log "EXIT $? ladder-g7-full8192"
echo STAGE3G_DONE >> $D/stage3.out
# 4. prefill: single 8K pass on the candidate, then the streaming binary after GPU parity tests
$PY -u $D/screen.py --name s3i-8kpass --base $D/base-soft4-guard.json --cases $D/cases/s3i.json --tokens 128 --trials 1 --prompt build-tr16-20261009/prompts8k/holdout-code.ids --prefill-batch 8192 --control-rate 34.0815 --control-stdout $D/s2k8/k0-control/000-baseline.stdout >> $D/stage3.out 2>&1; log "EXIT $? s3i-8kpass"
if grep -q STREAM_BUILD_DONE $D/stream-build.out 2>/dev/null; then
  ok=1
  for t in glm_q23_parity glm_q8_test glm_mailbox_test glm_prefill_parity glm_parity; do
    $D/stream-cuda/$t > $D/stream-$t.log 2>&1 || { ok=0; log "PARITY_FAILED $t"; }
  done
  if [ $ok -eq 1 ]; then
    $PY - <<'PY'
import json,hashlib,time
from pathlib import Path
R=Path('/mnt/nvme02/work/Strata/build-tr16-35-20261010'); b=R/'stream-cuda/strata-glm-decode'
(R/'stream-build-complete.json').write_text(json.dumps(dict(cuda_parities_passed=True,hip_build_passed=False,model_controls_passed=False,quality_qualified=False,time=time.time(),
  sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (b, R/'stream-incremental.patch')}),indent=2)+'\n')
PY
    grep -q "'stream-cuda'" $D/screen.py || sed -i "s#'merged-cuda': R / 'merged-build-complete.json'#'merged-cuda': R / 'merged-build-complete.json', 'stream-cuda': R / 'stream-build-complete.json'#" $D/screen.py
    $PY -u $D/screen.py --name s3j-8kpass-stream --base $D/base-soft4-stream.json --cases $D/cases/s3j.json --tokens 128 --trials 1 --prompt build-tr16-20261009/prompts8k/holdout-code.ids --prefill-batch 8192 --control-rate 34.0815 --control-stdout $D/s2k8/k0-control/000-baseline.stdout >> $D/stage3.out 2>&1; log "EXIT $? s3j-8kpass-stream"
  fi
else log "STREAM_BUILD_NOT_READY"; fi
echo STAGE3I_DONE >> $D/stage3.out
# 5. exploratory loss-reduction screens + pilots, then affinity 0.12 at 8K (only if time remains; each step checks the deadline)
$PY -u $D/screen.py --name s3f-2k --base $D/base-soft4-guard.json --cases $D/cases/s3f.json --tokens 512 --trials 1 --control-rate 32.9658 --control-stdout $D/s2/x0-control/000-baseline.stdout >> $D/stage3.out 2>&1; log "EXIT $? s3f-2k"
for n in h1-aff014-topk6-unbiased-guard4 h2-late-topk6-guard4; do
  [ -f $D/s3f-2k/$n.json ] && { $PY -u $D/quality-ladder.py --config $D/s3f-2k/$n.json --name $n --track 35 --steps pilot2048 >> $D/stage3.out 2>&1; log "EXIT $? pilot-$n"; }
done
echo STAGE3F_DONE >> $D/stage3.out
$PY -u $D/finalize.py --cases g5-aff012-topk6-guard4 --base base-soft4-guard.json --skip-train --skip-2k --tag h >> $D/stage3.out 2>&1; log "EXIT $? finalize-g5-8k"
echo STAGE3H_DONE >> $D/stage3.out; echo CHAIN2_DONE >> $D/stage3.out
