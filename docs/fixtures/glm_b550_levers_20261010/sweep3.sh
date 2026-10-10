#!/bin/bash
set -euo pipefail
cd /home/syoyo/work/Strata
root=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010
common=(STRATA_GLM_HC_FUSED=1 STRATA_GLM_TIER_DIRECT=1 STRATA_GLM_MLA_KERNELS=1 STRATA_GLM_HEADS_BLOCKS=1 STRATA_GLM_FUSE_ROUTE=1 STRATA_GLM_DEFER_EXPERTS=1 STRATA_GLM_DEFER_ROUTES=2)
bash "$root/steptrace.sh" fit7p2 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit7.json"
bash "$root/steptrace.sh" fit6nodefer "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit6.json" STRATA_GLM_DEFER_EXPERTS=UNSET STRATA_GLM_DEFER_ROUTES=UNSET
bash "$root/steptrace.sh" fit6full "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit6.json" STRATA_GLM_DEFER_ROUTES=UNSET
bash "$root/steptrace.sh" fit6p2c64 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit6.json" STRATA_GLM_KDA_COLUMNS=64
bash "$root/steptrace.sh" fit6p2q4all "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit6.json" STRATA_GLM_DENSE_Q4=all
