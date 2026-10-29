#!/usr/bin/env bash
# batch.sh "NAME ENV=VAL ..." ...: runs exp.sh for each spec in turn on the persistent worker; stops if either node is short of memory.
cd "$(dirname "$0")"
for spec in "$@"; do
  a=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
  b=$(ssh -o BatchMode=yes b550 "awk '/MemAvailable/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo")
  if awk -v a="$a" -v b="${b:-0}" 'BEGIN{exit !(a<4 || b<2.5)}'; then echo "STOP: memory tr16=$a b550=$b"; exit 1; fi
  [ -z "${USE_WORKER:-}" ] || pgrep -x strata-ds41-cpu >/dev/null || { echo "STOP: no persistent worker"; exit 1; }
  # shellcheck disable=SC2086
  ./exp.sh $spec
done
echo "batch finished"
