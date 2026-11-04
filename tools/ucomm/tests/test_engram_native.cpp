#include "ds4_engram_remote.hpp"
#include "ds4_engram_io.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
int main(int argc,char**argv) {
    if(argc!=3)return 2;
    strata::ds4::EngramRemote remote;std::string error;
    if(!remote.connect(argv[1],16,argv[2],error)){std::fprintf(stderr,"%s\n",error.c_str());return 3;}
    // Cross the request entry limit while preserving the caller's order and duplicates.
    std::vector<strata::ds4::EgRead> rows(8200);
    std::vector<unsigned char> out(rows.size()*4);
    for(size_t i=0;i<rows.size();i++){rows[i].n=4;rows[i].off=i%2?4:0;rows[i].dst=out.data()+i*4;}
    if(!remote.read(rows.data(),rows.size()))return 4;
    for(size_t i=0;i<rows.size();i++)if(std::memcmp(out.data()+i*4,i%2?"0123":"GGUF",4))return 5;
    // Exercise the same idle worker path used by the GPU engine.
    strata::ds4::EgReader reader;
    reader.remote_keepalive_ms=50;
    reader.remote_read=[&](strata::ds4::EgRead* r,size_t n){return remote.read(r,n);};
    reader.start({},1);
    std::this_thread::sleep_for(std::chrono::milliseconds(750));
    reader.stop();
    return 0;
}
