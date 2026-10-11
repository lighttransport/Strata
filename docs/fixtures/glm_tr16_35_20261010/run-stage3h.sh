#!/bin/bash
# affinity 0.12 + top-k 6 (guard 4): 8K held-out 512x3 only (possible 8K-only 35 profile with lower loss). Last in the chain.
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010; PY=build-exl3-venv/bin/python
until grep -q STAGE3F_DONE $D/stage3.out; do sleep 10; done
$PY -u $D/finalize.py --cases g5-aff012-topk6-guard4 --base base-soft4-guard.json --skip-train --skip-2k --tag h >> $D/stage3.out 2>&1
echo STAGE3H_DONE >> $D/stage3.out
