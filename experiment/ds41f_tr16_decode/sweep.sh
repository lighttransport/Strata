#!/usr/bin/env bash
# sweep.sh TAG LEN STEPS S...: sess.sh for each session count in turn (persistent worker), one summary + worker line each into logs/TAG.out
cd "$(dirname "$0")"
tag=$1 len=$2 steps=$3; shift 3
for S in "$@"; do ./sess.sh ${tag}$S $S $len $steps | head -1; tail -1 logs/${tag}$S.worker; done
