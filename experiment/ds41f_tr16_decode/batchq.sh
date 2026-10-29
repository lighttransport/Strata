#!/usr/bin/env bash
# batchq.sh "NAME PROMPT ENV=VAL ..." ...: qual.sh for each spec in turn
cd "$(dirname "$0")"
for spec in "$@"; do
  a=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
  b=$(ssh -o BatchMode=yes b550 "awk '/MemAvailable/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo")
  if awk -v a="$a" -v b="${b:-0}" 'BEGIN{exit !(a<4 || b<2.5)}'; then echo "STOP: memory tr16=$a b550=$b"; exit 1; fi
  # shellcheck disable=SC2086
  ./qual.sh $spec
done
echo "batch finished"
