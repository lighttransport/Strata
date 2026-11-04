/* Read-only Engram byte-range service. One context, one owning thread.
 * Protocol v1: size handshake, bounded ordered requests, explicit read status.
 * Native little-endian ABI, as with the existing DS41 experimental protocol. */
#define _POSIX_C_SOURCE 200809L
#include "ucomm.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "engram_wire.h"
#include <openssl/evp.h>
#include <sys/wait.h>
#include <pthread.h>
#include <time.h>
static int check(ucomm_status s) {
  if (s == UCOMM_OK) return 1;
  fprintf(stderr, "engram: %s\n", ucomm_strerror(s)); return 0;
}
struct read_pool {
  pthread_mutex_t mu;
  pthread_cond_t wake, done;
  int fd, stop, failed;
  size_t count, next, pending;
  struct eg_request *req;
  size_t *offsets;
  unsigned char *reply;
};
static void *read_worker(void *arg) {
  struct read_pool *p=arg;
  pthread_mutex_lock(&p->mu);
  for(;;) {
    while(p->next==p->count && !p->stop)pthread_cond_wait(&p->wake,&p->mu);
    if(p->stop)break;
    size_t i=p->next++;pthread_mutex_unlock(&p->mu);
    size_t n=0;int failed=0;
    while(n<p->req[i].length) {
      ssize_t k=pread(p->fd,p->reply+p->offsets[i]+n,p->req[i].length-n,(off_t)(p->req[i].offset+n));
      if(k<0 && errno==EINTR)continue;
      if(k<=0){failed=1;break;}n+=(size_t)k;
    }
    pthread_mutex_lock(&p->mu);
    p->failed|=failed;
    if(!--p->pending)pthread_cond_signal(&p->done);
  }
  pthread_mutex_unlock(&p->mu);return NULL;
}
static int serve_session(int fd, const struct stat *source, const unsigned char digest[32], const char *address) {
  struct stat st=*source;

  ucomm_config cfg; ucomm_config_default(&cfg);
  cfg.rank=0; cfg.world=2; cfg.root_addr=address;
  /* Explicit UDP is available for loopback tests; production requires IB. */
  cfg.backend = getenv("UCOMM_BACKEND") && !strcmp(getenv("UCOMM_BACKEND"),"udp") ? UCOMM_BACKEND_UDP : UCOMM_BACKEND_IB;
  if (cfg.timeout_ms<=0) cfg.timeout_ms=60000;
  ucomm_t *c=NULL;
  if (!check(ucomm_init(&cfg,&c))) return 1;
  struct eg_hello hello={EG_MAGIC,1,(uint64_t)st.st_size,{0}};
  memcpy(hello.sha256,digest,32);
  if (!check(ucomm_send(c,1,EG_HELLO,&hello,sizeof hello))) return 1;
  struct eg_request *req=malloc(sizeof(*req)*EG_MAX_ENTRIES);
  unsigned char *reply=malloc(EG_MAX_BYTES);
  if (!req || !reply) return 1;
  ucomm_mr_t *region=NULL;
  if(ucomm_mr_reg(c,reply,EG_MAX_BYTES,&region)!=UCOMM_OK)region=NULL;
  int readers=getenv("STRATA_ENGRAM_READERS")?atoi(getenv("STRATA_ENGRAM_READERS")):8;
  if(readers<1 || readers>64){fprintf(stderr,"invalid STRATA_ENGRAM_READERS\n");return 2;}
  struct read_pool pool={0};pool.fd=fd;pool.req=req;pool.reply=reply;
  pool.offsets=malloc(EG_MAX_ENTRIES*sizeof(size_t));if(!pool.offsets)return 1;
  pthread_mutex_init(&pool.mu,NULL);pthread_cond_init(&pool.wake,NULL);pthread_cond_init(&pool.done,NULL);
  pthread_t workers[64];
  for(int i=0;i<readers;i++)if(pthread_create(&workers[i],NULL,read_worker,&pool))return 1;
  uint64_t batches=0,rows=0,transferred=0;
  for (;;) {
    size_t got=0;
    if (!check(ucomm_recv(c,1,EG_REQUEST,req,sizeof(*req)*EG_MAX_ENTRIES,&got))) return 1;
    if (!got) break;
    struct eg_status status={0,0};
    if (got%sizeof(*req)) status.code=EINVAL;
    size_t count=got/sizeof(*req), total=0;
    for(size_t i=0;!status.code && i<count;i++) {
      if (req[i].reserved || !req[i].length || req[i].length>EG_MAX_ROW || req[i].offset>(uint64_t)st.st_size ||
          req[i].length>(uint64_t)st.st_size-req[i].offset || req[i].length>EG_MAX_BYTES-total) status.code=EINVAL;
      else total+=req[i].length;
    }
    if(!status.code) {
      size_t at=0;
      for(size_t i=0;i<count;i++){pool.offsets[i]=at;at+=req[i].length;}
      pthread_mutex_lock(&pool.mu);pool.next=0;pool.count=count;pool.pending=count;pool.failed=0;
      pthread_cond_broadcast(&pool.wake);
      while(pool.pending)pthread_cond_wait(&pool.done,&pool.mu);
      if(pool.failed)status.code=EIO;
      pthread_mutex_unlock(&pool.mu);
    }
    status.bytes=status.code?0:(uint32_t)total;
    if(!status.code){++batches;rows+=count;transferred+=total;}
    if(!check(ucomm_send(c,1,EG_STATUS,&status,sizeof status))) return 1;
    if(!status.code && !check(ucomm_send(c,1,EG_DATA,reply,total))) return 1;
  }
  fprintf(stderr,"engram session: batches=%llu rows=%llu bytes=%llu\n",(unsigned long long)batches,(unsigned long long)rows,(unsigned long long)transferred);
  pthread_mutex_lock(&pool.mu);pool.stop=1;pthread_cond_broadcast(&pool.wake);pthread_mutex_unlock(&pool.mu);
  for(int i=0;i<readers;i++)pthread_join(workers[i],NULL);
  pthread_cond_destroy(&pool.wake);pthread_cond_destroy(&pool.done);pthread_mutex_destroy(&pool.mu);
  free(pool.offsets);if(region)ucomm_mr_dereg(region);
  free(req);free(reply);close(fd);ucomm_finalize(c);return 0;
}

static int describe(const char *address) {
  ucomm_config cfg;ucomm_config_default(&cfg);cfg.rank=1;cfg.world=2;cfg.root_addr=address;
  cfg.backend=getenv("UCOMM_BACKEND") && !strcmp(getenv("UCOMM_BACKEND"),"udp")?UCOMM_BACKEND_UDP:UCOMM_BACKEND_IB;
  if(cfg.timeout_ms<=0)cfg.timeout_ms=60000;
  ucomm_t*c=NULL;struct eg_hello h;size_t got=0;
  if(!check(ucomm_init(&cfg,&c)) || !check(ucomm_recv(c,0,EG_HELLO,&h,sizeof h,&got)))return 1;
  if(got!=sizeof h || h.magic!=EG_MAGIC || h.version!=1)return 1;
  struct eg_request r={0,(uint32_t)(h.size<EG_MAX_ROW?h.size:EG_MAX_ROW),0};
  struct eg_status status;unsigned char bytes[EG_MAX_ROW];
  if(!check(ucomm_send(c,0,EG_REQUEST,&r,sizeof r)) ||
     !check(ucomm_recv(c,0,EG_STATUS,&status,sizeof status,&got)) || got!=sizeof status || status.code || status.bytes!=r.length ||
     !check(ucomm_recv(c,0,EG_DATA,bytes,r.length,&got)) || got!=r.length)return 1;
  printf("{\"version\":1,\"size\":%llu,\"sha256\":\"",(unsigned long long)h.size);
  for(int i=0;i<32;i++)printf("%02x",h.sha256[i]);
  printf("\",\"header_hex\":\"");for(uint32_t i=0;i<r.length;i++)printf("%02x",bytes[i]);printf("\"}\n");
  if(!check(ucomm_send(c,0,EG_REQUEST,NULL,0)))return 1;
  ucomm_finalize(c);return 0;
}
int main(int argc, char **argv) {
  if(argc==3 && !strcmp(argv[1],"describe"))return describe(argv[2]);
  if (argc != 4 || strcmp(argv[1], "serve")) {
    fprintf(stderr,"usage: strata-dsv41-engram serve FILE HOST:PORT | describe HOST:PORT\n"); return 2;
  }
  int fd=open(argv[2],O_RDONLY); struct stat st;
  if(fd<0 || fstat(fd,&st) || !S_ISREG(st.st_mode)) {perror("engram file");return 2;}
  unsigned char magic[4];
  if(pread(fd,magic,4,0)!=4 || memcmp(magic,"GGUF",4)) {fprintf(stderr,"not GGUF\n");return 2;}
  EVP_MD_CTX *md=EVP_MD_CTX_new(); unsigned char digest[32],buf[1048576]; unsigned int n=0;
  if(!md || !EVP_DigestInit_ex(md,EVP_sha256(),NULL)) return 1;
  ssize_t k;
  while((k=read(fd,buf,sizeof buf))!=0) {
    if(k<0) {if(errno==EINTR)continue;perror("hash read");return 1;}
    if(!EVP_DigestUpdate(md,buf,(size_t)k))return 1;
  }
  if(!EVP_DigestFinal_ex(md,digest,&n) || n!=32)return 1;
  EVP_MD_CTX_free(md);
  fprintf(stderr,"engram source verified: %llu bytes\n",(unsigned long long)st.st_size);
  /* A fresh child owns each context. A failed peer cannot poison the next session. */
  for(;;) {
    struct stat current;
    if(fstat(fd,&current) || current.st_size!=st.st_size || current.st_mtime!=st.st_mtime || current.st_ctime!=st.st_ctime) {
      fprintf(stderr,"engram source changed; restart required\n");return 1;
    }
    pid_t pid=fork();
    if(pid<0){perror("fork");return 1;}
    if(!pid) _exit(serve_session(fd,&st,digest,argv[3]));
    int status;
    while(waitpid(pid,&status,0)<0)if(errno!=EINTR)return 1;
    if(getenv("STRATA_ENGRAM_ONCE"))return WIFEXITED(status)?WEXITSTATUS(status):1;
    if(!WIFEXITED(status) || WEXITSTATUS(status)) {struct timespec pause={1,0};nanosleep(&pause,NULL);}
    /* Bootstrap retries also cover clients arriving between sessions. */
  }
}
