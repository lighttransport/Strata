#include "strata/core/glm_mixed.hpp"
#include "strata/prefill/partition.hpp"
#include <sys/wait.h>
#include <sys/syscall.h>
#include <linux/memfd.h>
#undef NDEBUG
#include <cassert>
#include <vector>

void check_fast_transport() {
    setenv("STRATA_GLM_MIXED_PROTOCOL","5",1);
    const auto &layout=strata::core::mixed_layout();
    const int memory=syscall(SYS_memfd_create,"mixed-fast-test",MFD_CLOEXEC);
    assert(memory>=0 && ftruncate(memory,layout.bytes)==0);
    int sockets[2];assert(socketpair(AF_UNIX,SOCK_SEQPACKET,0,sockets)==0);
    const pid_t child=fork();assert(child>=0);
    if (!child) {
        close(sockets[0]);strata::core::MixedMapping mapping(memory,sockets[1]);
        uint64_t seq=STRATA_MIXED_MAGIC;strata::core::mixed_signal(sockets[1],seq,true);
        strata::core::mixed_signal(sockets[1],seq,false);
        assert(seq==1 && mapping.valid(48,224) && mapping.activation_half());
        auto &b=*mapping.buffer;b.chunk_rows=b.nt;b.chunk_tag=1;
        for (uint32_t i=0;i<b.nt;++i) mapping.destinations()[i]=i;
        mapping.output()[size_t(b.nt)*4096-1]=123.f;
        strata_mixed_message packet{seq,1,STRATA_MIXED_CHUNK};
        strata::core::mixed_packet(sockets[1],packet,true);strata::core::mixed_packet(sockets[1],packet,false);
        assert(packet.sequence==seq && packet.tag==1 && packet.kind==STRATA_MIXED_ACK);
        b.chunk_rows=0;b.chunk_tag=2|STRATA_MIXED_FINAL;
        packet={seq,b.chunk_tag,STRATA_MIXED_CHUNK};strata::core::mixed_packet(sockets[1],packet,true);
        strata::core::mixed_signal(sockets[1],seq,false);assert(seq==0);_exit(0);
    }
    close(sockets[1]);
    {
        strata::core::MixedClient client(memory,sockets[0]);auto &b=*client.buffer;
        b.layer=3;b.nt=8192;b.routes=1;b.prefill=STRATA_MIXED_PREFILL_PARTIAL;b.canon=0;
        client.set_activation_half(true);client.submit();
        assert(client.wait_chunk() && b.chunk_rows==8192 && client.destinations()[8191]==8191);
        assert(client.output()[size_t(8192)*4096-1]==123.f);
        client.acknowledge_chunk();assert(!client.wait_chunk());
    }
    int status=0;assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
    close(sockets[0]);close(memory);
    // A v5 selection must never map a legacy-size buffer.
    const int small=syscall(SYS_memfd_create,"mixed-wrong-size",MFD_CLOEXEC);
    assert(small>=0 && ftruncate(small,sizeof(strata_mixed_buffer))==0);
    bool rejected=false;
    try {strata::core::MixedMapping wrong(small,-1);} catch (const std::runtime_error &) {rejected=true;}
    assert(rejected);close(small);unsetenv("STRATA_GLM_MIXED_PROTOCOL");
}

void check_stable_ownership() {
#ifdef __linux__
    setenv("STRATA_GLM_MIXED_STABLE_OWNERSHIP","1",1);
    setenv("STRATA_GLM_MIXED_PROMPT_PRIOR","1",1);
    const int memory=syscall(SYS_memfd_create,"mixed-owner-test",MFD_CLOEXEC);
    assert(memory>=0 && ftruncate(memory,sizeof(strata_mixed_buffer))==0);
    int sockets[2];assert(socketpair(AF_UNIX,SOCK_SEQPACKET,0,sockets)==0);
    const pid_t child=fork();assert(child>=0);
    if (!child) {
        close(sockets[0]);strata::core::MixedMapping mapping(memory,sockets[1]);
        uint64_t value=STRATA_MIXED_MAGIC;strata::core::mixed_signal(sockets[1],value,true);
        auto &b=*mapping.buffer;
        for (uint64_t expected=1;expected<=5;++expected) {
            strata::core::mixed_signal(sockets[1],value,false);assert(value==expected);
            if (expected==3) {
                assert(b.prefill==1);b.chunk_rows=1;b.chunk_tag=1;b.chunk_destination[0]=3;
                strata_mixed_message packet{value,1,STRATA_MIXED_CHUNK};
                strata::core::mixed_packet(sockets[1],packet,true);strata::core::mixed_packet(sockets[1],packet,false);
                assert(packet.sequence==value && packet.kind==STRATA_MIXED_ACK);
                b.chunk_rows=0;b.chunk_tag=2|STRATA_MIXED_FINAL;packet={value,b.chunk_tag,STRATA_MIXED_CHUNK};
                strata::core::mixed_packet(sockets[1],packet,true);
            } else {
                assert(b.prefill==2 && strata_mixed_prompt_valid(&b,224));
                assert(b.route_weight[3]==(expected==5 ? 99.f : 17.f));
                assert(b.chunk_tag==(expected==5 ? 1u : 0u));
                std::memset(b.worker_owned+3*288,0,288);
                b.worker_owned[3*288+(expected==1 ? 5 : 6)]=1;
                strata::core::mixed_signal(sockets[1],value,true);
            }
        }
        strata::core::mixed_signal(sockets[1],value,false);assert(value==0);_exit(0);
    }
    close(sockets[1]);
    {
        strata::core::MixedClient client(memory,sockets[0]);std::array<unsigned long long,288> primary{};
        std::array<uint64_t,288> counts{};counts[3]=17;
        client.configure(3,224,primary.data(),counts.data());
        primary[10]=1;client.configure(3,224,primary.data(),counts.data()); // disjoint: no control request
        primary[5]=1;client.configure(3,224,primary.data(),counts.data()); // overlap: must update
        auto &b=*client.buffer;b.layer=3;b.nt=1;b.routes=1;b.prefill=1;b.canon=0;b.expert[0]=20;b.destination[0]=3;
        client.submit();assert(client.wait_chunk());client.acknowledge_chunk();assert(!client.wait_chunk());
        client.configure(3,224,primary.data(),counts.data()); // unchanged primary, but prefill invalidated residency
        counts[3]=99;client.configure(3,224,primary.data(),counts.data(),true);
    }
    int status=0;assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
    close(sockets[0]);close(memory);unsetenv("STRATA_GLM_MIXED_STABLE_OWNERSHIP");
    unsetenv("STRATA_GLM_MIXED_PROMPT_PRIOR");
#endif
}
int main() {
    check_fast_transport();
    check_stable_ownership();
    for (int groups: {14,18}) for (int primary=1;primary<groups;++primary) {
        int owned=0;
        for (int g=0;g<groups;++g) {
            const bool p=strata::prefill::owns_expert_group(g,0,2,primary,groups);
            assert(p!=strata::prefill::owns_expert_group(g,1,2,primary,groups));
            owned+=p;
        }
        assert(owned==primary);
    }
#ifdef __linux__
    const int memory = syscall(SYS_memfd_create, "glm-mixed-test", MFD_CLOEXEC);
    assert(memory>=0);
    assert(ftruncate(memory,sizeof(strata_mixed_buffer))==0);
    int sockets[2]; assert(socketpair(AF_UNIX,SOCK_SEQPACKET,0,sockets)==0);
    const pid_t child=fork(); assert(child>=0);
    if (!child) {
        close(sockets[0]);
        strata::core::MixedMapping mapping(memory,sockets[1]);
        uint64_t value=STRATA_MIXED_MAGIC;
        strata::core::mixed_signal(sockets[1],value,true);
        strata::core::mixed_signal(sockets[1],value,false);
        assert(value==1 && strata_mixed_valid(mapping.buffer,48,288));
        for (unsigned i=0; i<mapping.buffer->routes; ++i)
            for (unsigned j=0; j<4096; ++j) mapping.buffer->output[size_t(i)*4096+j]=float(i+1);
        strata::core::mixed_signal(sockets[1],value,true);
        strata::core::mixed_signal(sockets[1],value,false);
        assert(value==2 && mapping.buffer->prefill==1);
        auto &b=*mapping.buffer;
        for (unsigned i=0;i<2;++i) {
            b.chunk_rows=1;b.chunk_tag=i+1;b.chunk_destination[0]=i ? 3 : 1;
            b.output[0]=float(10+i);
            strata_mixed_message packet{value,b.chunk_tag,STRATA_MIXED_CHUNK};
            strata::core::mixed_packet(sockets[1],packet,true);
            strata::core::mixed_packet(sockets[1],packet,false);
            assert(packet.sequence==value && packet.tag==i+1 && packet.kind==STRATA_MIXED_ACK);
        }
        b.chunk_rows=0;b.chunk_tag=3|STRATA_MIXED_FINAL;
        strata_mixed_message packet{value,b.chunk_tag,STRATA_MIXED_CHUNK};
        strata::core::mixed_packet(sockets[1],packet,true);
        strata::core::mixed_signal(sockets[1],value,false);
        assert(value==3 && b.prefill==STRATA_MIXED_PREFILL_PARTIAL);
        for (unsigned i=0;i<b.nt;++i) {
            b.chunk_rows=1;b.chunk_tag=i+1;b.chunk_destination[0]=i;b.output[0]=float(20+i);
            strata_mixed_message part{value,b.chunk_tag,STRATA_MIXED_CHUNK};
            strata::core::mixed_packet(sockets[1],part,true);strata::core::mixed_packet(sockets[1],part,false);
            assert(part.sequence==value && part.tag==i+1 && part.kind==STRATA_MIXED_ACK);
        }
        b.chunk_rows=0;b.chunk_tag=(b.nt+1)|STRATA_MIXED_FINAL;
        strata_mixed_message done{value,b.chunk_tag,STRATA_MIXED_CHUNK};strata::core::mixed_packet(sockets[1],done,true);
        strata::core::mixed_signal(sockets[1],value,false);
        assert(value==4 && b.prefill==STRATA_MIXED_DECODE_PARTIAL && strata_mixed_valid(&b,48,288));
        std::fill_n(b.output,4096,0.5f);strata::core::mixed_signal(sockets[1],value,true);
        strata::core::mixed_signal(sockets[1],value,false);
        --value; // A stale completion must not release the current buffer lease.
        strata::core::mixed_signal(sockets[1],value,true);
        _exit(0);
    }
    close(sockets[1]);
    strata::core::MixedClient client(memory,sockets[0]);
    auto &b=*client.buffer; b.layer=3;b.nt=3;b.routes=3;b.canon=1;b.prefill=0;
    b.expert[0]=10;b.expert[1]=20;b.expert[2]=30;
    b.destination[0]=3;b.destination[1]=11;b.destination[2]=17;
    client.submit();
    std::vector<float> rows(24*4096,0.f);rows[4096]=7.f;client.finish(rows.data());
    assert(rows[3*4096]==1.f && rows[11*4096]==2.f && rows[17*4096]==3.f);
    assert(rows[4096]==7.f && rows[0]==0.f);
    b.nt=1;b.routes=2;b.destination[0]=3;b.destination[1]=1;
    b.prefill=1;client.submit();
    assert(client.wait_chunk() && b.chunk_destination[0]==1 && b.output[0]==10.f);
    client.acknowledge_chunk();
    assert(client.wait_chunk() && b.chunk_destination[0]==3 && b.output[0]==11.f);
    client.acknowledge_chunk();assert(!client.wait_chunk());
    b.prefill=STRATA_MIXED_PREFILL_PARTIAL;b.nt=2;b.routes=1;client.submit();
    for (unsigned i=0;i<b.nt;++i) {
        assert(client.wait_chunk() && b.chunk_destination[0]==int(i) && b.output[0]==float(20+i));
        client.acknowledge_chunk();
    }
    assert(!client.wait_chunk());
    b.prefill=STRATA_MIXED_DECODE_PARTIAL;b.nt=1;b.route_weight[3]=0.5f;client.submit();
    std::vector<float> sum(4096,1.f);client.finish_partial(sum.data());assert(sum[0]==1.5f && sum.back()==1.5f);
    b.prefill=0;client.submit();bool rejected=false;
    try {client.finish(rows.data());} catch (const std::runtime_error &) {rejected=true;}
    assert(rejected);
    int status=0;assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
    rejected=false;
    try {client.wait();} catch (const std::runtime_error &) {rejected=true;}
    assert(rejected);close(sockets[0]);close(memory);
#endif
}
