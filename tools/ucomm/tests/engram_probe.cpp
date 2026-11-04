// Hardware validation helper: fetch an ordered list of exact row ranges through the production reader.
#include "ds4_engram_remote.hpp"
#include "ds4_engram_io.hpp"
#include <fstream>
#include <iostream>
#include <limits>
int main(int argc,char**argv) {
    if(argc!=6){std::cerr<<"usage: engram_probe HOST:PORT SIZE SHA256 RANGES OUT\n";return 2;}
    strata::ds4::EngramRemote remote;std::string error;
    if(!remote.connect(argv[1],std::stoull(argv[2]),argv[3],error)){std::cerr<<error<<'\n';return 1;}
    std::ifstream input(argv[4]);std::vector<strata::ds4::EgRead> rows;
    uint64_t offset;uint32_t length;size_t total=0;
    while(input>>offset>>length){
        if(!length || length>16384 || total>size_t(256)*1024*1024-length)return 2;
        strata::ds4::EgRead r;r.off=offset;r.n=length;rows.push_back(r);total+=length;
    }
    if(!input.eof() || rows.empty())return 2;
    std::vector<uint8_t> bytes(total);size_t at=0;
    for(auto& r:rows){r.dst=bytes.data()+at;at+=r.n;}
    if(!remote.read(rows.data(),rows.size()))return 1;
    std::ofstream output(argv[5],std::ios::binary);output.write(reinterpret_cast<char*>(bytes.data()),bytes.size());
    return output?0:1;
}
