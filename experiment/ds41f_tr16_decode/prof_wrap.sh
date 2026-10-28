#!/usr/bin/env bash
# rocprofv3 kernel trace around the decoder (STRATA_DS41_DECODE_BIN for tools/ds41f_decode_guard.sh); output in $PROF_OUT
out=${PROF_OUT:-/tmp/ds41-rocprof}
exec /opt/rocm/core-10.1/bin/rocprofv3 --kernel-trace --memory-copy-trace --stats -d "$out" -o run --output-format csv -- /mnt/nvme02/work/Strata/build-hip-rocm101-tr16/strata-glm-decode "$@"
