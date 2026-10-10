#!/bin/bash
set -euo pipefail
cd /home/syoyo/work/Strata
root=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010
common=(STRATA_GLM_HC_FUSED=1 STRATA_GLM_TIER_DIRECT=1 STRATA_GLM_MLA_KERNELS=1 STRATA_GLM_HEADS_BLOCKS=1 STRATA_GLM_FUSE_ROUTE=1 STRATA_GLM_DEFER_EXPERTS=1 STRATA_GLM_DEFER_ROUTES=2)
bash "$root/steptrace.sh" lookup-default "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json"
bash "$root/steptrace.sh" lookup-mix2 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json" STRATA_GLM_TIER_LOOKUP_ONCE=1
bash "$root/steptrace.sh" lookup-fit7 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit7.json" STRATA_GLM_TIER_LOOKUP_ONCE=1
bash "$root/steptrace.sh" lookup-fit8full "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit8.json" STRATA_GLM_TIER_LOOKUP_ONCE=1 STRATA_GLM_DEFER_ROUTES=UNSET
bash "$root/steptrace.sh" lookup-fit6rows4 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit6.json" STRATA_GLM_TIER_LOOKUP_ONCE=1 STRATA_GLM_KDA_ROW_PARTS=4
