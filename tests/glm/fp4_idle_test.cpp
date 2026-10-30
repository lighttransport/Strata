#include "strata/net/glm_fp4_worker.hpp"
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
namespace { int posted=0,waited=0;void* target=nullptr;size_t size=0; }
extern "C" {
ucomm_status ucomm_irecv(ucomm_t*,int,uint32_t,void* p,size_t n,ucomm_req_t** out) {
    ++posted;target=p;size=n;*out=reinterpret_cast<ucomm_req_t*>(1);return UCOMM_OK;
}
ucomm_status ucomm_wait(ucomm_req_t* request,size_t* bytes) {
    if(request!=reinterpret_cast<ucomm_req_t*>(1))return UCOMM_ERR_ARG;
    if(++waited<3)return UCOMM_ERR_TIMEOUT;
    std::memset(target,0x5a,size);*bytes=size;return UCOMM_OK;
}
const char* ucomm_strerror(ucomm_status) {return "test failure";}
}
int main() {
    uint32_t header[8]={};
    strata::net::glm_fp4_receive_idle(nullptr,0,22,header,sizeof header);
    if(posted!=1||waited!=3||header[0]!=0x5a5a5a5a)return 1;
    int channel[2];if(pipe(channel))return 2;
    const auto child=fork();
    if(child<0)return 3;
    if(!child) {
        close(channel[0]);waited=0;
        struct Unwind {int fd;~Unwind(){const char marker='x';(void)write(fd,&marker,1);}} guard{channel[1]};
        // The first timeout expires the total idle deadline. No stack buffer
        // may be unwound while that request is still live.
        strata::net::glm_fp4_receive_idle(nullptr,0,22,header,sizeof header,0);
        return 4;
    }
    close(channel[1]);int status=0;if(waitpid(child,&status,0)!=child)return 5;
    char marker;const auto bytes=read(channel[0],&marker,1);close(channel[0]);
    if(!WIFEXITED(status)||WEXITSTATUS(status)!=1||bytes!=0)return 6;
    return 0;
}
