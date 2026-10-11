#!/bin/bash
cd /mnt/nvme02/work/Strata || exit 1
PY=build-exl3-venv/bin/python; D=build-tr16-35-20261010
until grep -q STAGE2X_DONE $D/stage2.out; do sleep 10; done
$PY -u $D/screen.py --name s2k8 --base $D/base-soft4.json --cases $D/cases/s2k8.json --tokens 512 --trials 1 --prompt build-tr16-20261009/prompts8k/holdout-code.ids >> $D/stage2.out 2>&1
echo "EXIT $? s2k8" >> $D/stage2.out; echo STAGE2K8_DONE >> $D/stage2.out
