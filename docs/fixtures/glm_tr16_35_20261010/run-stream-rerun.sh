#!/bin/bash
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010; PY=build-exl3-venv/bin/python
until grep -q SOFT4_HE_DONE $D/stage3.out; do sleep 20; done
if grep -q STREAM_BUILD_DONE $D/stream-build.out; then
  ok=1; for t in glm_q23_parity glm_q8_test glm_mailbox_test glm_prefill_parity glm_parity; do $D/stream-cuda/$t > $D/stream-$t.log 2>&1 || { ok=0; echo "PARITY_FAILED $t" >> $D/stage3.out; }; done
  if [ $ok -eq 1 ]; then
    $PY - <<'PY'
import json,hashlib,time
from pathlib import Path
R=Path('/mnt/nvme02/work/Strata/build-tr16-35-20261010'); b=R/'stream-cuda/strata-glm-decode'
(R/'stream-build-complete.json').write_text(json.dumps(dict(cuda_parities_passed=True,hip_build_passed=False,model_controls_passed=False,quality_qualified=False,time=time.time(),sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (b, R/'stream-incremental.patch')}),indent=2)+'\n')
PY
    grep -q "'stream-cuda'" $D/screen.py || sed -i "s#'merged-cuda': R / 'merged-build-complete.json'#'merged-cuda': R / 'merged-build-complete.json', 'stream-cuda': R / 'stream-build-complete.json'#" $D/screen.py
    $PY -u $D/screen.py --name s3j2-8kpass-stream --base $D/base-soft4-stream.json --cases $D/cases/s3j.json --tokens 128 --trials 1 --prompt build-tr16-20261009/prompts8k/holdout-code.ids --prefill-batch 8192 --control-rate 34.0815 --control-stdout $D/s2k8/k0-control/000-baseline.stdout --wait-idle-seconds 3600 >> $D/stage3.out 2>&1; echo "EXIT $? s3j2-8kpass-stream" >> $D/stage3.out
  fi
fi
echo STREAM_RERUN_DONE >> $D/stage3.out
