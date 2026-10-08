#!/bin/sh
# Opt-in Linux launcher for the measured GLM profiles. The allocation policy
# applies only to this process and its threads; the decoder places expert rows.
set -eu
: "${STRATA_GLM_NUMA_EXECUTABLE:?set STRATA_GLM_NUMA_EXECUTABLE to the decoder executable}"
if ! command -v numactl >/dev/null 2>&1; then
    echo 'GLM NUMA launcher requires numactl' >&2
    exit 127
fi
exec numactl --interleave=all "$STRATA_GLM_NUMA_EXECUTABLE" "$@"
