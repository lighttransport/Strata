#!/bin/bash
# CUDA build of draft-route + streaming-prefetch source at low priority (runs beside the HumanEval validation; its wall time is not a claim).
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010; S=$D/stream-source; B=$D/stream-cuda; GG=build-tr16-goal-20261010/ggml-reference
nice -n 19 cmake -S $S -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_GGML_DIR=$GG -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=ON > $D/stream-cuda-configure.log 2>&1 || { echo CONFIGURE_FAILED >> $D/stream-build.out; exit 1; }
nice -n 19 cmake --build $B --target strata-glm-decode glm_q23_parity glm_q8_test glm_mailbox_test glm_prefill_parity glm_parity -j 4 > $D/stream-cuda-build.log 2>&1
code=$?
echo "BUILD_EXIT $code $(date +%T)" >> $D/stream-build.out
[ $code -eq 0 ] && sha256sum $B/strata-glm-decode >> $D/stream-build.out && echo STREAM_BUILD_DONE >> $D/stream-build.out
