#!/bin/bash
# Build the root tree (loader change for assembled files) at low priority, only after the soft4 HumanEval has loaded its model.
cd /mnt/nvme02/work/Strata || exit 1
D=build-tr16-35-20261010
until grep -q 'START full164' <(tail -n 40 $D/stage3.out) && grep -q CHAIN2_DONE $D/stage3.out; do sleep 20; done
sleep 300
echo "BUILD_START $(date +%T)" >> $D/release-build.out
nice -n 19 cmake -S . -B build-glm-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_GGML_DIR=build-tr16-goal-20261010/ggml-reference > $D/release-build-configure.log 2>&1 || { echo CONFIGURE_FAILED >> $D/release-build.out; exit 1; }
nice -n 19 cmake --build build-glm-release --target strata-glm-decode strata-glm-assemble strata-glm-q2-pack glm_q23_parity glm_q8_test glm_mailbox_test glm_prefill_parity -j 4 > $D/release-build.log 2>&1
echo "BUILD_EXIT $? $(date +%T)" >> $D/release-build.out
sha256sum build-glm-release/strata-glm-decode build-glm-release/strata-glm-assemble >> $D/release-build.out 2>/dev/null
echo RELEASE_BUILD_DONE >> $D/release-build.out
