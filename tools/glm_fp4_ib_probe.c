/* Bounded activation-size IB probe; never registers a large benchmark arena. */
#define _POSIX_C_SOURCE 200809L
#include "ucomm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static void check(ucomm_status s){if(s!=UCOMM_OK){fprintf(stderr,"ERROR %s\n",ucomm_strerror(s));exit(1);}}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int order(const void* a,const void* b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
int main(int argc,char** argv){
    if(argc!=4){fprintf(stderr,"usage: probe RANK ROOT:PORT IB_PORT\n");return 2;}
    int rank=atoi(argv[1]),port=atoi(argv[3]);if(rank<0||rank>1||port<1||port>2)return 2;
    ucomm_config cfg;ucomm_config_default(&cfg);cfg.rank=rank;cfg.world=2;cfg.root_addr=argv[2];cfg.ib_port=port;cfg.backend=UCOMM_BACKEND_IB;cfg.timeout_ms=15000;
    ucomm_t* c=NULL;check(ucomm_init(&cfg,&c));
    unsigned char* buf=malloc(262144);if(!buf)return 1;memset(buf,42,262144);
    ucomm_mr_t* mr=NULL;check(ucomm_mr_reg(c,buf,262144,&mr));
    printf("TRANSPORT %s rank=%d registered_bytes=262144\n",ucomm_backend_name(c),rank);
    const size_t sizes[]={16384,262144};
    for(int s=0;s<2;++s){
        double times[20];size_t bytes=sizes[s];
        for(int i=-2;i<20;++i){double start=now();size_t got=0;
            if(rank==0){check(ucomm_send(c,1,1,buf,bytes));check(ucomm_recv(c,1,2,buf,bytes,&got));}
            else{check(ucomm_recv(c,0,1,buf,bytes,&got));check(ucomm_send(c,0,2,buf,bytes));}
            if(got!=bytes){fprintf(stderr,"ERROR length mismatch\n");return 1;}
            if(i>=0)times[i]=(now()-start)*1e6;
        }
        if(rank==0){qsort(times,20,sizeof(double),order);printf("ROUNDTRIP bytes_each_way=%zu median_us=%.3f min_us=%.3f max_us=%.3f samples=20\n",bytes,(times[9]+times[10])/2,times[0],times[19]);}
    }
    ucomm_mr_dereg(mr);ucomm_finalize(c);free(buf);puts("PASS bounded IB probe");return 0;
}
