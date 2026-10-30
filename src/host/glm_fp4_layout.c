#include "strata/host/glm_fp4_layout.h"
int strata_glm_fp4_layout_role(unsigned layer, unsigned expert, unsigned first_moe,
                               unsigned main_layers, unsigned expert_count, unsigned remote_count) {
    if (!expert_count || first_moe >= main_layers || layer < first_moe || layer > main_layers ||
        expert >= expert_count || remote_count > expert_count) return -1;
    return layer < main_layers && expert < remote_count;
}
int strata_glm_fp4_layout_owner(unsigned layer, unsigned expert, unsigned remote_count) {
    return strata_glm_fp4_layout_role(layer, expert, 3, 45, 288, remote_count);
}
int strata_glm_fp4_layout_span(uint64_t offset, uint64_t bytes, uint64_t file_bytes) {
    return offset <= file_bytes && bytes <= file_bytes - offset;
}
int strata_glm_fp4_layout_admit(uint64_t weights, uint64_t reserve, uint64_t allowance) {
    return weights <= allowance && reserve <= allowance - weights;
}
