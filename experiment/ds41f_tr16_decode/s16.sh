#!/usr/bin/env bash
cd "$(dirname "$0")"
A="STRATA_DS41_ADAPT_GAP_MS=30 STRATA_DS41_HELPER_CPUS=14,15,30,31 STRATA_DS41_ADAPT_OWNED_WEIGHT=2 STRATA_DS41_DECODE_TIMING=1"
ADAPT=3000 ./sess.sh m16 16 120 24 $A STRATA_DS41_DENSE_MMQ=1 | head -1 | cut -c1-300
grep -hE "decode_gpu_ms" logs/m16.log | cut -c1-400
ADAPT=3000 ./sess.sh b16 16 120 24 $A | head -1 | cut -c1-300
grep -hE "decode_gpu_ms" logs/b16.log | cut -c1-400
