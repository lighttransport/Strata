#!/bin/bash
cd /mnt/nvme02/work/Strata || exit 1
PY=build-exl3-venv/bin/python; D=build-tr16-35-20261010
until grep -q STAGE2K8_DONE $D/stage2.out; do sleep 10; done
X0=$(python3 -c "import json;print(json.load(open('$D/s2/summary.json'))[0]['decode_tok_s'])")
$PY -u $D/screen.py --name s2g --base $D/base-soft4-guard.json --cases $D/cases/s2g.json --tokens 512 --trials 1 --control-rate $X0 --control-stdout $D/s2/x0-control/000-baseline.stdout >> $D/stage2.out 2>&1
echo "EXIT $? s2g" >> $D/stage2.out; echo STAGE2G_DONE >> $D/stage2.out
