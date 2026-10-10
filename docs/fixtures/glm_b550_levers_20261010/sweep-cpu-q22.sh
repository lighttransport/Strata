#!/bin/bash
set -euo pipefail
cd /home/syoyo/work/Strata
root=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010
export EXPERT_PACK=/mnt/disk01/models/glm53f-reap/REAP50-Q22-down-only-experimental.gguf
common=(STRATA_GLM_HC_FUSED=1 STRATA_GLM_TIER_DIRECT=1 STRATA_GLM_MLA_KERNELS=1 STRATA_GLM_HEADS_BLOCKS=1 STRATA_GLM_FUSE_ROUTE=1 STRATA_GLM_NORM_Q8_FUSED=1 STRATA_GLM_TIER_LOOKUP_ONCE=1 STRATA_GLM_DEFER_EXPERTS=1 STRATA_GLM_DEFER_ROUTES=2 STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit4.json")
THREADS=8 bash "$root/steptrace.sh" q22fit4-t8 "${common[@]}"
THREADS=4 bash "$root/steptrace.sh" q22fit4-t4 "${common[@]}"
bash "$root/steptrace.sh" q22fit4-task2 "${common[@]}" STRATA_NATIVE_TASKS_PER_THREAD=2
bash "$root/steptrace.sh" q22fit4-task1 "${common[@]}" STRATA_NATIVE_TASKS_PER_THREAD=1
