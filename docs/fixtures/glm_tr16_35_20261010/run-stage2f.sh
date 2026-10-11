#!/bin/bash
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010
until grep -q STAGE2G_DONE $D/stage2.out; do sleep 10; done
build-exl3-venv/bin/python -u $D/finalize.py --cases x11-aff016-topk6 --skip-train >> $D/stage2b.out 2>&1
echo STAGE2F_DONE >> $D/stage2b.out
