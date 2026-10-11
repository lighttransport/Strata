#!/bin/bash
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010; PY=build-exl3-venv/bin/python
until grep -q STAGE3C_DONE $D/stage3.out; do sleep 10; done
for n in g6-aff014-topk6-guard4 g7-mix2-topk6-guard4 g5-aff012-topk6-guard4; do
  $PY -u $D/quality-ladder.py --config $D/s3a-2k/$n.json --name $n --track 35 --steps pilot2048 >> $D/stage3.out 2>&1
  echo "EXIT $? pilot-$n" >> $D/stage3.out
done
echo STAGE3D_DONE >> $D/stage3.out
