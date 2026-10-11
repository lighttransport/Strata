#!/bin/bash
# Exploratory loss-reduction screens + pilots. Runs after the mix2 evaluation.
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010; PY=build-exl3-venv/bin/python
until grep -q STAGE3G_DONE $D/stage3.out; do sleep 10; done
$PY -u $D/screen.py --name s3f-2k --base $D/base-soft4-guard.json --cases $D/cases/s3f.json --tokens 512 --trials 1 --control-rate 32.9658 --control-stdout $D/s2/x0-control/000-baseline.stdout >> $D/stage3.out 2>&1
echo "EXIT $? s3f-2k" >> $D/stage3.out
for n in h1-aff014-topk6-unbiased-guard4 h2-late-topk6-guard4; do
  $PY -u $D/quality-ladder.py --config $D/s3f-2k/$n.json --name $n --track 35 --steps pilot2048 >> $D/stage3.out 2>&1
  echo "EXIT $? pilot-$n" >> $D/stage3.out
done
echo STAGE3F_DONE >> $D/stage3.out
