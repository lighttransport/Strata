#!/bin/bash
set -euo pipefail
cd /home/syoyo/work/Strata
root=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010
common=(STRATA_GLM_HC_FUSED=1 STRATA_GLM_TIER_DIRECT=1 STRATA_GLM_MLA_KERNELS=1 STRATA_GLM_HEADS_BLOCKS=1 STRATA_GLM_FUSE_ROUTE=1 STRATA_GLM_DEFER_EXPERTS=1 STRATA_GLM_TIER_LOOKUP_ONCE=1 STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit8.json")
LIMIT=10 bash "$root/humaneval.sh" fit8full-smoke "${common[@]}"
bash "$root/steptrace.sh" fit8fullrows4 "${common[@]}" STRATA_GLM_KDA_ROW_PARTS=4
bash "$root/steptrace.sh" fit8full-repeat1 "${common[@]}"
bash "$root/steptrace.sh" mix2p2-confirm "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json" STRATA_GLM_DEFER_ROUTES=2 STRATA_GLM_TIER_LOOKUP_ONCE=UNSET
bash "$root/steptrace.sh" fit8full-repeat2 "${common[@]}"
