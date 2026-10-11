#!/bin/bash
# Waits for Stage 1, reruns the interference-lost case, then runs Stage 2 combination screens at 512x1.
cd /mnt/nvme02/work/Strata || exit 1
PY=build-exl3-venv/bin/python; D=build-tr16-35-20261010
until grep -q STAGE1_DONE $D/stage1.out; do sleep 10; done
run() { $PY -u $D/screen.py "$@" >> $D/stage2.out 2>&1; echo "EXIT $? $*" >> $D/stage2.out; }
run --name s1c-rerun --base $D/base-soft4.json --cases $D/cases/s1c_rerun.json --tokens 128 --trials 1 --control-rate 29.9766
run --name s2 --base $D/base-soft4.json --cases $D/cases/s2.json --tokens 512 --trials 1
echo STAGE2_DONE >> $D/stage2.out
