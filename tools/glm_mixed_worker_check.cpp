// Deterministic worker component check using the shared C11 protocol definition.
// Writes compact raw expert rows for comparison of old/new binaries per backend.
#include "strata/core/glm_mixed.hpp"
#include <sys/syscall.h>
#include <sys/wait.h>
#include <linux/memfd.h>
#include <fcntl.h>
#include <signal.h>
#include <fstream>
#include <cmath>

int main(int argc, char **argv) {
    if (argc!=4) {
        std::cerr << "usage: glm_mixed_worker_check WORKER MODEL OUTPUT\n";
        return 2;
    }
    int memory=-1, sockets[2]={-1,-1};
    pid_t child=-1;
    try {
        memory=int(syscall(SYS_memfd_create,"glm-worker-check",MFD_CLOEXEC));
        if (memory<0 || ftruncate(memory,sizeof(strata_mixed_buffer)) ||
            socketpair(AF_UNIX,SOCK_SEQPACKET,0,sockets))
            throw std::runtime_error("worker check transport creation failed");
        child=fork();
        if (child<0) throw std::runtime_error("worker check fork failed");
        if (!child) {
            close(sockets[0]);
            if (fcntl(memory,F_SETFD,0)<0) _exit(126);
            std::string mem=std::to_string(memory),sock=std::to_string(sockets[1]);
            execl(argv[1],argv[1],argv[2],mem.c_str(),sock.c_str(),"512","14336",nullptr);
            _exit(127);
        }
        close(sockets[1]);sockets[1]=-1;
        std::ofstream output(argv[3],std::ios::binary);
        if (!output) throw std::runtime_error("worker check output creation failed");
        {
            strata::core::MixedClient client(memory,sockets[0]);
            auto &b=*client.buffer;
            for (int repeat=0;repeat<2;++repeat) for (uint32_t rows: {1u,2u,4u,8u}) {
                b.layer=3;b.nt=rows;b.routes=rows*2;b.prefill=0;b.canon=1;
                b.swiglu_limit=0;b.gate_skip=0;b.chunk_tag=0;
                std::memset(b.primary_owned,0,sizeof b.primary_owned);
                for (uint32_t row=0;row<rows;++row) {
                    for (uint32_t h=0;h<4096;++h)
                        b.activation[size_t(row)*4096+h]=float(int(h%23)-11)*float(row+1)*.001f;
                    for (uint32_t expert=0;expert<2;++expert) {
                        const uint32_t i=row*2+expert;
                        b.expert[i]=int(expert);b.destination[i]=int(row*8+expert);
                    }
                }
                client.submit();client.wait();
                const size_t count=size_t(b.routes)*4096;
                bool nonzero=false;
                for (size_t i=0;i<count;++i) {
                    if (!std::isfinite(b.output[i])) throw std::runtime_error("nonfinite expert row");
                    nonzero|=b.output[i]!=0.f;
                }
                if (!nonzero) throw std::runtime_error("all expert rows are zero");
                output.write(reinterpret_cast<const char *>(b.output),std::streamsize(count*sizeof(float)));
                if (!output) throw std::runtime_error("worker check output write failed");
            }
        }
        int status=0;
        if (waitpid(child,&status,0)!=child || !WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error("worker check child failed");
        child=-1;close(sockets[0]);close(memory);
        std::cout << "WORKER_CHECK rows=1,2,4,8 experts=2 repeats=2 finite=1\n";
        return 0;
    } catch (const std::exception &error) {
        if (child>0) {kill(child,SIGTERM);waitpid(child,nullptr,0);}
        for (int fd: {memory,sockets[0],sockets[1]}) if (fd>=0) close(fd);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
