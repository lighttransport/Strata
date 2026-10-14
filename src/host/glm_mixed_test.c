#include "strata/host/glm_mixed.h"
#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
int main(void) {
    struct strata_mixed_buffer *b = calloc(1, sizeof *b);
    assert(b);
    b->magic=STRATA_MIXED_MAGIC; b->version=STRATA_MIXED_VERSION;
    b->sequence=b->generation=1; b->nt=1; b->routes=1; b->expert[0]=10;
    assert(strata_mixed_valid(b,48,288));
    b->destination[0]=8; assert(!strata_mixed_valid(b,48,288)); b->destination[0]=0;
    b->primary_owned[10]=1; assert(!strata_mixed_valid(b,48,288)); b->primary_owned[10]=0;
    b->routes=2; b->expert[1]=11; assert(!strata_mixed_valid(b,48,288));
    b->destination[1]=1; assert(strata_mixed_valid(b,48,288));
    b->nt=9; assert(!strata_mixed_valid(b,48,288));
    b->prefill=1; assert(strata_mixed_valid(b,48,288));
    b->nt=STRATA_MIXED_TOKENS+1; assert(!strata_mixed_valid(b,48,288)); b->nt=1;
    b->routes=0; b->prefill=2; assert(strata_mixed_valid(b,48,288));
    assert(strata_mixed_prompt_valid(b,288));
    b->chunk_tag=1;assert(strata_mixed_prompt_valid(b,288));
    b->chunk_tag=2;assert(!strata_mixed_prompt_valid(b,288));b->chunk_tag=0;
    b->route_weight[287]=NAN;assert(!strata_mixed_prompt_valid(b,288));
    b->route_weight[287]=-1;assert(!strata_mixed_prompt_valid(b,288));
    b->route_weight[287]=123;assert(strata_mixed_prompt_valid(b,288));
    assert(!strata_mixed_prompt_valid(b,289));b->route_weight[287]=0;
    b->prefill=5; assert(!strata_mixed_valid(b,48,288)); b->prefill=2;
    b->swiglu_limit=NAN; assert(!strata_mixed_valid(b,48,288));
    b->swiglu_limit=0; b->nt=1; b->chunk_rows=1; b->chunk_destination[0]=3; b->chunk_tag=1;
    uint8_t *wanted=calloc(STRATA_MIXED_ROUTES,1), *seen=calloc(STRATA_MIXED_ROUTES,1); assert(wanted && seen);
    wanted[3]=1; assert(strata_mixed_chunk_valid(b,wanted,seen));
    assert(!strata_mixed_chunk_valid(b,wanted,seen));
    b->chunk_rows=0; b->chunk_tag=STRATA_MIXED_FINAL|2; assert(strata_mixed_chunk_valid(b,wanted,seen));
    b->prefill=STRATA_MIXED_PREFILL_PARTIAL;b->nt=2;b->routes=1;b->destination[0]=3;b->route_weight[3]=0.5f;
    assert(strata_mixed_valid(b,48,288));
    b->route_weight[3]=NAN;assert(!strata_mixed_valid(b,48,288));b->route_weight[3]=0.5f;
    b->chunk_rows=1;b->chunk_tag=1;b->chunk_destination[0]=3;assert(!strata_mixed_chunk_valid(b,wanted,seen));
    b->chunk_destination[0]=0;wanted[0]=1;assert(strata_mixed_chunk_valid(b,wanted,seen));
    b->prefill=STRATA_MIXED_DECODE_PARTIAL;b->nt=2;b->routes=1;
    assert(strata_mixed_valid(b,48,288));
    b->route_weight[0]=1;assert(!strata_mixed_valid(b,48,288));b->route_weight[0]=0;
    b->nt=9;assert(!strata_mixed_valid(b,48,288));b->nt=2;
    {
        int order[6]={2,3,0,1,4,5};
        double score[6]={10,9,12,11,8,7};
        float counts[6]={0,0,1,8,0,0};
        uint8_t previous[6]={1,1,0,0,0,0},primary[6]={0},selected[6]={0};
        assert(strata_mixed_select_cache(6,2,order,score,counts,previous,primary,1,6,10,selected));
        assert(selected[0] && selected[3] && !selected[1] && !selected[2]);
        /* A higher threshold retains both old owners; low-count newcomers do not churn. */
        assert(strata_mixed_select_cache(6,2,order,score,counts,previous,primary,4,6,50,selected));
        assert(selected[0] && selected[1] && !selected[2] && !selected[3]);
        /* Primary ownership forces a refill even when every count is below the gate. */
        primary[0]=1;
        assert(strata_mixed_select_cache(6,2,order,score,counts,previous,primary,0,100,50,selected));
        assert(!selected[0] && selected[1] && selected[2]);
        assert(strata_mixed_select_cache(6,0,order,score,counts,previous,primary,4,6,10,selected));
        for (int i=0;i<6;++i) assert(!selected[i]);
        memset(selected,0xa5,sizeof selected);order[5]=2;
        assert(!strata_mixed_select_cache(6,2,order,score,counts,previous,primary,4,6,10,selected));
        for (int i=0;i<6;++i) assert(selected[i]==0xa5);
        order[5]=5;score[2]=NAN;
        assert(!strata_mixed_select_cache(6,2,order,score,counts,previous,primary,4,6,10,selected));
    }
    free(wanted);free(seen);
    free(b); return 0;
}
