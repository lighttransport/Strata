#define _POSIX_C_SOURCE 200809L
#include "ucomm.h"
#include "engram_wire.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define CHECK(x) do { if(!(x)) {fprintf(stderr,"failed line %d\n",__LINE__);exit(1);} } while(0)
int main(int argc,char **argv) {
 CHECK(argc==2);ucomm_config cfg;ucomm_config_default(&cfg);
 cfg.rank=1;cfg.world=2;cfg.root_addr=argv[1];cfg.backend=UCOMM_BACKEND_UDP;
 ucomm_t*c;CHECK(ucomm_init(&cfg,&c)==UCOMM_OK);
 struct eg_hello h;size_t n;CHECK(ucomm_recv(c,0,EG_HELLO,&h,sizeof h,&n)==UCOMM_OK);
 CHECK(n==sizeof h && h.magic==EG_MAGIC && h.version==1 && h.size==16);
 struct eg_request r[3]={{0,4,0},{4,4,0},{0,4,0}};struct eg_status s;char data[12];
 CHECK(ucomm_send(c,0,EG_REQUEST,r,sizeof r)==UCOMM_OK);
 CHECK(ucomm_recv(c,0,EG_STATUS,&s,sizeof s,&n)==UCOMM_OK && s.code==0 && s.bytes==12);
 CHECK(ucomm_recv(c,0,EG_DATA,data,sizeof data,&n)==UCOMM_OK && n==12);
 CHECK(!memcmp(data,"GGUF0123GGUF",12));
 r[0].offset=15;r[0].length=2;
 CHECK(ucomm_send(c,0,EG_REQUEST,r,sizeof r[0])==UCOMM_OK);
 CHECK(ucomm_recv(c,0,EG_STATUS,&s,sizeof s,&n)==UCOMM_OK && s.code!=0 && !s.bytes);
 const struct eg_request invalid[]={{UINT64_MAX,4,0},{0,EG_MAX_ROW+1,0},{0,4,1},{0,0,0}};
 for(size_t i=0;i<sizeof invalid/sizeof invalid[0];i++) {
  CHECK(ucomm_send(c,0,EG_REQUEST,&invalid[i],sizeof invalid[i])==UCOMM_OK);
  CHECK(ucomm_recv(c,0,EG_STATUS,&s,sizeof s,&n)==UCOMM_OK && s.code!=0 && !s.bytes);
 }
 CHECK(ucomm_send(c,0,EG_REQUEST,r,1)==UCOMM_OK);
 CHECK(ucomm_recv(c,0,EG_STATUS,&s,sizeof s,&n)==UCOMM_OK && s.code!=0 && !s.bytes);
 // Errors must not leave payload bytes queued ahead of a subsequent valid request.
 r[0].offset=0;r[0].length=4;r[0].reserved=0;
 CHECK(ucomm_send(c,0,EG_REQUEST,r,sizeof r[0])==UCOMM_OK);
 CHECK(ucomm_recv(c,0,EG_STATUS,&s,sizeof s,&n)==UCOMM_OK && !s.code && s.bytes==4);
 CHECK(ucomm_recv(c,0,EG_DATA,data,4,&n)==UCOMM_OK && n==4 && !memcmp(data,"GGUF",4));
 CHECK(ucomm_send(c,0,EG_REQUEST,r,0)==UCOMM_OK);ucomm_finalize(c);return 0;
}
