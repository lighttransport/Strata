#!/usr/bin/env bash
# exp.sh NAME [ENV=VAL ...]: one 128-token run on the persistent worker (rev_node.sh wstart first), one summary line in logs/exp.txt.
# EXACT=0 skips the token comparison against the CUDA reference (lossy levers). Memory guard: scratchpad guard.sh is run by hand.
set -u
E=$(cd "$(dirname "$0")" && pwd); cd "$E"
name=$1; shift
rm -f logs/$name.*
USE_WORKER=${USE_WORKER:-} STEPS=${STEPS:-128} ./rev_node.sh run $name STRATA_DS41_CANON_MIB=${CANON:-5600} STRATA_DS41_DESKTOP_MIB=512 "$@" > logs/$name.run 2>&1
log=logs/$name.log
same=skip
if [ "${EXACT:-1}" = 1 ]; then
  if [ "$(grep '^tokens:' $log | head -1)" = "$(cat logs/ref_cuda_tokens.txt)" ]; then same=SAME; else same=DIFF; fi
fi
f() { grep -hoE "$1" $log | head -1; }
line="$name $(f 'decode_tok_s=[0-9.]+') $(f 'prompt_tok_s=[0-9.]+') ids=$same $(f 'tokens_per_step=[0-9.]+') $(f 'verify_ms_by_width=[0-9.,]+') $(f 'canon_hit=[0-9.]+') $(grep -hoE 'per layer call ms [0-9. /]+' $log | head -1) args: $*"
echo "$line" | tee -a logs/exp.txt
