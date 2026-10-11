#!/bin/bash
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010
until grep -q STAGE2F_DONE $D/stage2b.out; do sleep 10; done
build-exl3-venv/bin/python -u $D/finalize.py --cases k1-topk6 k4-mix2-topk6 --skip-train --skip-2k --tag b >> $D/stage2b.out 2>&1
echo STAGE2F2_DONE >> $D/stage2b.out
