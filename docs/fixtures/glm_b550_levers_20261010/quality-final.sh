#!/bin/bash
set -euo pipefail
cd /home/syoyo/work/Strata
root=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010
common=(STRATA_GLM_HC_FUSED=1 STRATA_GLM_TIER_DIRECT=1 STRATA_GLM_MLA_KERNELS=1 STRATA_GLM_HEADS_BLOCKS=1 STRATA_GLM_FUSE_ROUTE=1 STRATA_GLM_NORM_Q8_FUSED=1 STRATA_GLM_TIER_LOOKUP_ONCE=1 STRATA_GLM_DEFER_EXPERTS=1 STRATA_GLM_DEFER_ROUTES=2)
LIMIT=30 bash "$root/humaneval.sh" q23mix2-reference30 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json"
export EXPERT_PACK=/mnt/disk01/models/glm53f-reap/REAP50-Q22-down-only-experimental.gguf
LIMIT=20 OFFSET=10 bash "$root/humaneval.sh" q22fit5rows4-rest20 "${common[@]}" STRATA_GLM_KDA_ROW_PARTS=4 STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit5.json"
LIMIT=20 OFFSET=10 bash "$root/humaneval.sh" q22fit4-rest20 "${common[@]}" STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit4.json"
bash "$root/evalkl.sh" q22fit5rows4 ref "${common[@]}" STRATA_GLM_EVAL_WIDTH=1 STRATA_GLM_KDA_ROW_PARTS=4 STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit5.json"
bash "$root/evalkl.sh" q22fit4 ref "${common[@]}" STRATA_GLM_EVAL_WIDTH=1 STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-fit4.json"
