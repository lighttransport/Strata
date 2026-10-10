#!/bin/bash
set -euo pipefail
cd /home/syoyo/work/Strata
root=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010
common=(STRATA_GLM_HC_FUSED=1 STRATA_GLM_TIER_DIRECT=1 STRATA_GLM_MLA_KERNELS=1 STRATA_GLM_HEADS_BLOCKS=1 STRATA_GLM_FUSE_ROUTE=1 STRATA_GLM_DEFER_EXPERTS=1 STRATA_GLM_DEFER_ROUTES=2)
bash "$root/steptrace.sh" fit5p2 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit5.json"
bash "$root/steptrace.sh" fit6p2 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit6.json"
bash "$root/steptrace.sh" mix2p2rows4 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json" STRATA_GLM_KDA_ROW_PARTS=4
bash "$root/steptrace.sh" mix2p2tiles4 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json" STRATA_GLM_KDA_ROW_PARTS=4 STRATA_GLM_KDA_COLUMN_TILES=1
bash "$root/steptrace.sh" mix2p2q5 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json" STRATA_GLM_DENSE_Q4=q5
bash "$root/steptrace.sh" mix2p2q4all "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json" STRATA_GLM_DENSE_Q4=all
