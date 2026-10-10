#!/bin/bash
set -euo pipefail
cd /home/syoyo/work/Strata
root=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010
common=(STRATA_GLM_HC_FUSED=1 STRATA_GLM_TIER_DIRECT=1 STRATA_GLM_MLA_KERNELS=1 STRATA_GLM_HEADS_BLOCKS=1 STRATA_GLM_FUSE_ROUTE=1 STRATA_GLM_DEFER_EXPERTS=1)
bash "$root/steptrace.sh" fit1p2 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit1.json" STRATA_GLM_DEFER_ROUTES=2
bash "$root/steptrace.sh" fit2p2 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit2.json" STRATA_GLM_DEFER_ROUTES=2
bash "$root/steptrace.sh" fit3p2 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit3.json" STRATA_GLM_DEFER_ROUTES=2
bash "$root/steptrace.sh" fit4p2 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit4.json" STRATA_GLM_DEFER_ROUTES=2
