#!/bin/bash
# Synthetic resumable workload for testing jobctl. Counts up to TARGET (default 200 steps of 10 s),
# saves its position in $CTL_WORK/counter after every step, resumes from it in the next allocation, and
# follows the exit-code rule of agent.sh: 0 = finished, 75 = not finished (continue in the next allocation).
# Setting CTL_TEST_MPI=1 also runs one 12-rank mpiexec hostname at the start (needs pjsub --mpi proc=12).
TARGET=${COUNTER_TARGET:-200}
STEP=${COUNTER_STEP_SEC:-10}
f=$CTL_WORK/counter
n=$(cat "$f" 2>/dev/null || echo 0)
trap 'echo "SIGTERM at $n"; exit 75' TERM
echo "$(date -u +%FT%TZ) seq=$CTL_SEQ resume at $n"
if [ "${CTL_TEST_MPI:-0}" = 1 ] && [ "$n" = 0 ]; then
  mpiexec -np 12 hostname | sort | uniq -c
fi
while [ "$n" -lt "$TARGET" ]; do
  sleep "$STEP" & wait $!
  n=$((n + 1))
  echo "$n" > "$f.tmp" && mv "$f.tmp" "$f"
done
echo "$(date -u +%FT%TZ) finished at $n"
exit 0
