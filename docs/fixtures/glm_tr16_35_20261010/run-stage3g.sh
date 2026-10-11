#!/bin/bash
# mix2 + top-k 6 (guard 4): held-out 2K/8K 512x3, then full 8192 corpus. Runs after the affinity 0.14 full evaluation.
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010; PY=build-exl3-venv/bin/python
until grep -q STAGE3E_DONE $D/stage3.out; do sleep 10; done
$PY -u $D/finalize.py --cases g7-mix2-topk6-guard4 --base base-soft4-guard.json --skip-train --tag g >> $D/stage3.out 2>&1
$PY -u $D/quality-ladder.py --config $D/s3a-2k/g7-mix2-topk6-guard4.json --name g7-mix2-topk6-guard4-full --track 35 --steps full8192 --continue-on-fail >> $D/stage3.out 2>&1
echo "EXIT $? ladder-g7-full8192" >> $D/stage3.out; echo STAGE3G_DONE >> $D/stage3.out
