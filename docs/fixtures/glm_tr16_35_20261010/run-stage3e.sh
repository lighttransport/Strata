#!/bin/bash
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010; PY=build-exl3-venv/bin/python
until grep -q STAGE3D_DONE $D/stage3.out; do sleep 10; done
$PY -u $D/quality-ladder.py --config $D/s3a-2k/g6-aff014-topk6-guard4.json --name g6-aff014-topk6-guard4-full --track 35 --steps full8192 full164 --continue-on-fail >> $D/stage3.out 2>&1
echo "EXIT $? ladder-g6-full" >> $D/stage3.out; echo STAGE3E_DONE >> $D/stage3.out
