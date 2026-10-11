#!/bin/bash
# Stage 1 short screens, serial. Priority: A (acceptance), A on draft-route, C (affinity), E (exact), B (tier), D (GPU draft).
cd /mnt/nvme02/work/Strata || exit 1
PY=build-exl3-venv/bin/python; D=build-tr16-35-20261010
run() { $PY -u $D/screen.py "$@" >> $D/stage1.out 2>&1; echo "EXIT $? $*" >> $D/stage1.out; }
run --name s1a --base $D/base-soft4.json --cases $D/cases/s1a.json --tokens 128 --trials 1 --control-rate 29.9766
run --name s1a-dr --base $D/base-soft4-guard.json --cases $D/cases/s1a_dr.json --tokens 128 --trials 1 --control-rate 29.9766 \
    --control-stdout build-tr16-goal-20261010/q22-repeat-screen/00-control/000-baseline.stdout
run --name s1c --base $D/base-soft4.json --cases $D/cases/s1c.json --tokens 128 --trials 1 --control-rate 29.9766
run --name s1e --base $D/base-soft4.json --cases $D/cases/s1e.json --tokens 128 --trials 1 --control-rate 29.9766
run --name s1b --base $D/base-soft4.json --cases $D/cases/s1b.json --tokens 128 --trials 1 --control-rate 29.9766
run --name s1d --base $D/base-soft4.json --cases $D/cases/s1d.json --tokens 128 --trials 1 --control-rate 29.9766
echo STAGE1_DONE >> $D/stage1.out
