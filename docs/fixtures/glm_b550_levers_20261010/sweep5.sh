#!/bin/bash
set -euo pipefail
cd /home/syoyo/work/Strata
root=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010
common=(STRATA_GLM_HC_FUSED=1 STRATA_GLM_TIER_DIRECT=1 STRATA_GLM_MLA_KERNELS=1 STRATA_GLM_HEADS_BLOCKS=1 STRATA_GLM_FUSE_ROUTE=1 STRATA_GLM_DEFER_EXPERTS=1 STRATA_GLM_DEFER_ROUTES=2)
bash "$root/steptrace.sh" norm-default "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json"
bash "$root/steptrace.sh" norm-mix2 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json" STRATA_GLM_NORM_Q8_FUSED=1
bash "$root/steptrace.sh" norm-fit4 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit4.json" STRATA_GLM_NORM_Q8_FUSED=1
bash "$root/steptrace.sh" norm-fit9 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit9.json" STRATA_GLM_NORM_Q8_FUSED=1
bash "$root/steptrace.sh" norm-fit10 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit10.json" STRATA_GLM_NORM_Q8_FUSED=1
bash "$root/steptrace.sh" norm-q3mix2 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json" STRATA_GLM_NORM_Q8_FUSED=1 STRATA_GLM_DENSE_Q4=q3mixers
bash "$root/steptrace.sh" norm-q3fit4 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit4.json" STRATA_GLM_NORM_Q8_FUSED=1 STRATA_GLM_DENSE_Q4=q3mixers
bash "$root/steptrace.sh" norm-fit6cap16 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit6.json" STRATA_GLM_NORM_Q8_FUSED=1 STRATA_GLM_DEFER_MAX_SHARE=0.16
