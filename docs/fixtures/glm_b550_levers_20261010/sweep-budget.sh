#!/bin/bash
set -euo pipefail
cd /home/syoyo/work/Strata
root=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010
export EXPERT_PACK=/mnt/disk01/models/glm53f-reap/REAP50-Q22-down-only-experimental.gguf
common=(STRATA_GLM_HC_FUSED=1 STRATA_GLM_TIER_DIRECT=1 STRATA_GLM_MLA_KERNELS=1 STRATA_GLM_HEADS_BLOCKS=1 STRATA_GLM_FUSE_ROUTE=1 STRATA_GLM_NORM_Q8_FUSED=1 STRATA_GLM_DEFER_EXPERTS=1 STRATA_GLM_DEFER_ROUTES=2)
GPU_BUDGET_MIB=16000 bash "$root/steptrace.sh" q22fit4-b16000 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit4.json"
GPU_BUDGET_MIB=16000 bash "$root/steptrace.sh" q22fit4-b16000-r256 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit4.json" STRATA_GLM_TIER_RUNTIME_RESERVE_MIB=256
GPU_BUDGET_MIB=16000 bash "$root/steptrace.sh" q22fit4q3-b16000-r256 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit4.json" STRATA_GLM_TIER_RUNTIME_RESERVE_MIB=256 STRATA_GLM_DENSE_Q4=q3mixers
GPU_BUDGET_MIB=15360 bash "$root/steptrace.sh" q22fit8q3cap16 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit8.json" STRATA_GLM_DENSE_Q4=q3mixers STRATA_GLM_DEFER_MAX_SHARE=0.16
GPU_BUDGET_MIB=16000 bash "$root/steptrace.sh" q22fit8cap16-b16000-r256 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit8.json" STRATA_GLM_TIER_RUNTIME_RESERVE_MIB=256 STRATA_GLM_DEFER_MAX_SHARE=0.16
