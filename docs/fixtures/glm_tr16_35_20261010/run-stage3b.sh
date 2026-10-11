#!/bin/bash
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010
until grep -q STAGE3A_DONE $D/stage3.out; do sleep 10; done
build-exl3-venv/bin/python -u $D/quality-ladder.py --config $D/s2g2-2k/g4-aff016-topk6-guard4.json --name x11-aff016-topk6-guard4 --track 35 --steps task31 smoke20 pilot2048 full8192 full164 >> $D/stage3.out 2>&1
echo "EXIT $? ladder-x11" >> $D/stage3.out; echo STAGE3B_DONE >> $D/stage3.out
