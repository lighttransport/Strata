#!/bin/bash
set -euo pipefail
cd /home/syoyo/work/Strata
root=/home/syoyo/work/Strata/docs/fixtures/glm_b550_levers_20261010
common=(STRATA_GLM_HC_FUSED=1 STRATA_GLM_TIER_DIRECT=1 STRATA_GLM_MLA_KERNELS=1 STRATA_GLM_HEADS_BLOCKS=1 STRATA_GLM_FUSE_ROUTE=1 STRATA_GLM_DEFER_EXPERTS=1 STRATA_GLM_DEFER_ROUTES=2 STRATA_GLM_ROUTE_AFFINITY_PROFILE="$root/prof-mix2.json" STRATA_GLM_EVAL_WIDTH=1)
CORPUS="$root/parity4x64.ids" REF=/tmp/strata-b550-old-default.logits EXE=/tmp/strata-b550-before-lookup bash "$root/evalkl.sh" parity-old save "${common[@]}"
CORPUS="$root/parity4x64.ids" REF=/tmp/strata-b550-new-default.logits bash "$root/evalkl.sh" parity-default save "${common[@]}"
CORPUS="$root/parity4x64.ids" REF=/tmp/strata-b550-new-fused.logits bash "$root/evalkl.sh" parity-fused save "${common[@]}" STRATA_GLM_NORM_Q8_FUSED=1 STRATA_GLM_TIER_LOOKUP_ONCE=1
cmp /tmp/strata-b550-old-default.logits /tmp/strata-b550-new-default.logits
cmp /tmp/strata-b550-old-default.logits /tmp/strata-b550-new-fused.logits
sha256sum /tmp/strata-b550-{old-default,new-default,new-fused}.logits > "$root/logits-parity.sha256"
