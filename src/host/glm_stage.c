#include "strata/host/glm_stage.h"
#include <stdlib.h>
#include <string.h>

int strata_stage_valid(const struct strata_stage_buffer *b, uint64_t sequence, uint32_t vocab, uint32_t max_decode_tokens) {
    if (!b || b->magic != STRATA_STAGE_MAGIC || b->version != STRATA_STAGE_VERSION || !sequence || b->sequence != sequence ||
        b->command < STRATA_STAGE_BATCH || b->command > STRATA_STAGE_COMMAND_LAST || b->prefill > 1 ||
        b->reserved0 || b->reserved1)
        return 0;
    if (b->command == STRATA_STAGE_BATCH || b->command == STRATA_STAGE_VERIFY) {
        if (!b->nt || b->nt > STRATA_STAGE_TOKENS || (!b->prefill && b->nt > max_decode_tokens) ||
            (b->command == STRATA_STAGE_VERIFY && b->prefill) || b->position < 0)
            return 0;
        for (uint32_t t = 0; t < b->nt; ++t)
            if (b->tokens[t] < 0 || (uint32_t)b->tokens[t] >= vocab) return 0;
    } else if (b->nt || b->prefill) return 0;
    if ((b->command == STRATA_STAGE_RESTORE_VERIFIED || b->command == STRATA_STAGE_ENABLE_HISTORY) && b->count < 0) return 0;
    if ((b->command == STRATA_STAGE_SAVE || b->command == STRATA_STAGE_RESTORE) && b->id < 0) return 0;
    return 1;
}

int strata_stage_range(const char *text, uint32_t layers, uint32_t *first, uint32_t *end) {
    if (!text || !first || !end || !*text) return 0;
    char *colon = NULL;
    const unsigned long a = strtoul(text, &colon, 10);
    if (colon == text || *colon != ':' || text[0] == '-' || colon[1] == '-' || colon[1] == '+' || text[0] == '+') return 0;
    char *tail = NULL;
    const unsigned long b = strtoul(colon + 1, &tail, 10);
    if (tail == colon + 1 || *tail || a >= b || b > layers) return 0;
    *first = (uint32_t)a; *end = (uint32_t)b;
    return 1;
}
