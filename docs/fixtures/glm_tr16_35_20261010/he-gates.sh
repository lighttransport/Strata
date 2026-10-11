#!/bin/bash
# Release the upload hashing only once the HumanEval decoder is running (its idle wait is over), and mark its completion.
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010; Q=$D/q-soft4-aff08-guard4-he
until [ -f $Q/full164/engine.log ] && pgrep -f 'strata-glm-decode' > /dev/null; do sleep 15; done
sleep 120
grep -q CHAIN2_DONE $D/stage3.out || echo CHAIN2_DONE >> $D/stage3.out
echo "UPLOADS_RELEASED $(date +%T)" >> $D/stage3.out
until [ -f $Q/complete.json ] || ! pgrep -f 'quality-ladder.py --config build-tr16-goal-20261010/q22-repeat-screen/01-repeat4.json' > /dev/null; do sleep 30; done
echo "EXIT ? ladder-soft4-he $(date +%T)" >> $D/stage3.out; echo SOFT4_HE_DONE >> $D/stage3.out
