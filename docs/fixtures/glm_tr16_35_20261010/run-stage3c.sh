#!/bin/bash
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010
until grep -q STAGE3B_DONE $D/stage3.out; do sleep 10; done
build-exl3-venv/bin/python -u $D/finalize.py --cases g6-aff014-topk6-guard4 --base base-soft4-guard.json --skip-train --tag c >> $D/stage3.out 2>&1
echo STAGE3C_DONE >> $D/stage3.out
