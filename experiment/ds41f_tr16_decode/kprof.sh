#!/usr/bin/env bash
# kprof.sh NAME: single-node c_osd200 / 16-token decode under rocprofv3 with the canonical tier; canon kernel stats
cd "$(dirname "$0")"; rm -rf /tmp/ds41-rocprof
./run.sh $1 eval/c_osd200.txt 16 ${KP_ENV:-} STRATA_DS41_DECODE_BIN=$PWD/prof_wrap.sh STRATA_DS41_CANON_MIB=4096 STRATA_DS41_CANON_PROFILE=$PWD/prof/routes_code4k.i32 \
  STRATA_DS41_VERIFY_FAST_TRUNK=1 STRATA_DS41_GPU_TOPK_MIN=1 STRATA_DS41_ATTN_MULTI=1 > /dev/null 2>&1
rm -rf prof_out/$1; mv /tmp/ds41-rocprof prof_out/$1
python3 -c "
import csv
for a in csv.DictReader(open('prof_out/$1/run_kernel_stats.csv')):
    if 'canon' in a['Name'] or 'quantize_q8_K' in a['Name']: print('$1', a['Calls'], round(float(a['TotalDurationNs'])/1e6,1), 'ms avg', round(float(a['AverageNs'])/1e3,1), 'us', a['Name'][:60])"
grep '^tokens:' logs/$1.log | md5sum
