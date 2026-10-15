#include "strata/host/glm_stage.h"
#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    uint32_t a = 0, b = 0;
    assert(strata_stage_range("0:24", 45, &a, &b) && a == 0 && b == 24);
    assert(strata_stage_range("24:45", 45, &a, &b) && a == 24 && b == 45);
    assert(!strata_stage_range("24:46", 45, &a, &b) && !strata_stage_range("24:24", 45, &a, &b));
    assert(!strata_stage_range("", 45, &a, &b) && !strata_stage_range("24", 45, &a, &b) && !strata_stage_range("-1:5", 45, &a, &b));
    assert(!strata_stage_range("3:x", 45, &a, &b) && !strata_stage_range("3:5 ", 45, &a, &b) && !strata_stage_range("+3:5", 45, &a, &b));
    struct strata_stage_buffer *s = calloc(1, sizeof *s);
    assert(s);
    s->magic = STRATA_STAGE_MAGIC; s->version = STRATA_STAGE_VERSION; s->sequence = 7;
    s->command = STRATA_STAGE_BATCH; s->nt = 4; s->tokens[0] = 5; s->tokens[3] = 154879;
    assert(strata_stage_valid(s, 7, 154880, 8));
    assert(!strata_stage_valid(s, 8, 154880, 8));
    s->tokens[3] = 154880; assert(!strata_stage_valid(s, 7, 154880, 8)); s->tokens[3] = 1;
    s->nt = 9; assert(!strata_stage_valid(s, 7, 154880, 8));
    s->prefill = 1; assert(strata_stage_valid(s, 7, 154880, 8));
    s->nt = 8193; assert(!strata_stage_valid(s, 7, 154880, 8)); s->nt = 4;
    s->command = STRATA_STAGE_VERIFY; assert(!strata_stage_valid(s, 7, 154880, 8)); s->prefill = 0;
    assert(strata_stage_valid(s, 7, 154880, 8));
    s->command = STRATA_STAGE_RESET; assert(!strata_stage_valid(s, 7, 154880, 8)); s->nt = 0;
    assert(strata_stage_valid(s, 7, 154880, 8));
    s->command = STRATA_STAGE_RESTORE; s->id = -1; assert(!strata_stage_valid(s, 7, 154880, 8)); s->id = 0;
    assert(strata_stage_valid(s, 7, 154880, 8));
    s->command = 13; assert(!strata_stage_valid(s, 7, 154880, 8));
    s->command = STRATA_STAGE_STOP; s->reserved0 = 1; assert(!strata_stage_valid(s, 7, 154880, 8));
    free(s);
    return 0;
}
