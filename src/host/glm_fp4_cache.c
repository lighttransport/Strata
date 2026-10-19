#include "strata/host/glm_fp4_cache.h"
#include <stdlib.h>
static int hottest_first(const void *a, const void *b) {
    const strata_glm_fp4_cache_entry *x=a, *y=b;
    if(x->frequency!=y->frequency)return x->frequency>y->frequency?-1:1;
    if(x->layer!=y->layer)return x->layer<y->layer?-1:1;
    return (x->expert>y->expert)-(x->expert<y->expert);
}
int strata_glm_fp4_cache_select(strata_glm_fp4_cache_entry *entries, size_t count,
                               const size_t slots[2], unsigned devices, int mode) {
    size_t used[2]={0,0};
    if((!entries&&count)||!slots||devices<1||devices>2||(mode!=1&&mode!=2))return 0;
    for(size_t i=0;i<count;++i)
        if(entries[i].layer<0||entries[i].expert<0)return 0;
    if(count>1)qsort(entries,count,sizeof(*entries),hottest_first);
    for(size_t i=0;i<count;++i) {
        strata_glm_fp4_cache_entry *e=entries+i;
        unsigned owner=(unsigned)e->expert%devices;
        e->owner=-1;
        if(used[owner]>=slots[owner]&&mode==2&&devices==2)owner=1-owner;
        if(used[owner]<slots[owner]){e->owner=(int)owner;++used[owner];}
    }
    return 1;
}
int strata_glm_fp4_cache_adapt(const uint64_t frequency[288], uint8_t keep[288],
                              unsigned rank, unsigned devices, unsigned limit,
                              uint64_t hysteresis, int removed[288], int added[288]) {
    if(!frequency||!keep||!removed||!added||devices<1||devices>2||rank>=devices||limit>288)return -1;
    for(unsigned id=0;id<288;++id)
        if(keep[id]>1||(keep[id]&&id%devices!=rank))return -1;
    unsigned changed=0;
    while(changed<limit) {
        int hot=-1,cold=-1;
        for(int id=0;id<288;++id)if((unsigned)id%devices==rank) {
            if(keep[id]) {
                if(cold<0||frequency[id]<frequency[cold]||
                   (frequency[id]==frequency[cold]&&id>cold))cold=id;
            }else if(hot<0||frequency[id]>frequency[hot]||
                     (frequency[id]==frequency[hot]&&id<hot))hot=id;
        }
        if(hot<0||cold<0||frequency[hot]<=frequency[cold]||
           frequency[hot]-frequency[cold]<hysteresis)break;
        keep[cold]=0;keep[hot]=1;removed[changed]=cold;added[changed]=hot;++changed;
    }
    return (int)changed;
}
