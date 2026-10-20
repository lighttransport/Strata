from pathlib import Path
import subprocess,json
p=Path('build-fp4-local/q3-signed-code');s=Path('src/kernels/cpu/q23_avx2.cpp').read_text()
s=s.replace('template<int TY,int NT,int H> inline void chunk','template<int TY,int NT,int H,bool Signed=false> inline void chunk')
s=s.replace('template<int TY,int NT,int... H> inline void chunks','template<int TY,int NT,bool Signed=false,int... H> inline void chunks')
s=s.replace('chunk<TY,NT,H>','chunk<TY,NT,H,Signed>')
s=s.replace('bool Expanded=false, bool Nta=false> void dot','bool Expanded=false, bool Nta=false, bool Signed=false> void dot')
s=s.replace('chunks<TY,NT>','chunks<TY,NT,Signed>')
old='''        __m256i p=_mm256_maddubs_epi16(q,v);
        if constexpr(TY==11)p=_mm256_sub_epi16(p,p4[(t*kMaxBlocks+b)*8+H]);'''
new='''        __m256i p;
        if constexpr(Signed) {
            const auto signed_q=_mm256_sub_epi8(q,_mm256_set1_epi8(4));
            p=_mm256_maddubs_epi16(_mm256_sign_epi8(signed_q,signed_q),_mm256_sign_epi8(v,signed_q));
        } else {
            p=_mm256_maddubs_epi16(q,v);
            if constexpr(TY==11)p=_mm256_sub_epi16(p,p4[(t*kMaxBlocks+b)*8+H]);
        }'''
assert old in s;s=s.replace(old,new)
needle='    if constexpr(TY==10) { if(q2k_lut())'
s=s.replace(needle,'''    if constexpr(TY==11 && NT==1 && !Expanded) {
        if (!p4) {
            if(pf<0) dot<TY,NT,Expanded,true,true>(w,blocks,act,result,nullptr,-pf);
            else dot<TY,NT,Expanded,false,true>(w,blocks,act,result,nullptr,pf);
            return;
        }
    }
'''+needle)
a=s.index('template<int TY,bool Expanded=false> void dispatch');start=s.index('template<int TY,int NT,bool Expanded=false> void rows');part=s[start:a]
needle='    if constexpr(TY==11) {'
part=part.replace(needle,'''    bool signed_safe=false;
    if constexpr(TY==11 && NT==1 && !Expanded) {
        signed_safe=true;
        for(int b=0;b<n/256;++b) for(int h=0;h<8;++h) {
            const auto v=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const block_q8_K*>(act[0])[b].qs+32*h));
            if(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v,_mm256_set1_epi8(-128)))) signed_safe=false;
        }
    }
    const __m256i* correction=signed_safe?nullptr:p4;
    if constexpr(TY==11) { if(!signed_safe) {''',1)
part=part.replace('    int pf=prefetch_distance();','    }\n    int pf=prefetch_distance();')
part=part.replace('act,a,p4,pf','act,a,correction,pf').replace('act,u,p4,pf','act,u,correction,pf')
s=s[:start]+part+s[a:]
(p/'candidate.cpp').write_text(s)
(p/'probe.cpp').write_text('#include "strata/kernels/cpu/q23_avx2.hpp"\n#include "ggml.h"\n#include "ggml-cpu.h"\n#define GGML_COMMON_DECL_CPP\n#include "ggml-common.h"\n#include <vector>\n#include <random>\n#include <iostream>\n#include <cmath>\n#include <stdexcept>\n#include <cstring>\n#include <chrono>\n#include <ctime>\n#include <cstdlib>\nnamespace strata::kernels::cpu {\nvoid q23_candidate_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int);\n}\nusing namespace strata::kernels::cpu;\n// CPU kernel screen; optional disjoint weight copies exceed cache capacity.\n// Packing, replication, activation quantization and reference checks are untimed.\nvoid bench(int ty,int nt,int repeats,const std::string& layout,int copies=1) {\n    if(copies<1||copies>1024||(ty!=10&&ty!=11)||nt<1||nt>4||repeats<1||repeats>1000000||\n       (layout!="row"&&layout!="direct"&&layout!="lut"))throw std::runtime_error("invalid benchmark arguments");\n    ggml_cpu_init();std::mt19937 rng(731);\n    constexpr int cols=4096,rows=128;\n    const auto type=ggml_type(ty);\n    const size_t stride=ggml_row_size(type,cols),astride=ggml_row_size(GGML_TYPE_Q8_K,cols);\n    std::vector<float>w(rows*cols),decoded(w.size());std::vector<uint8_t>q(rows*stride),packed(q.size());\n    for(auto& v:w)v=float(int(rng()%20001)-10000)*.001f;\n    ggml_quantize_chunk(type,w.data(),q.data(),0,rows,cols,nullptr);\n    ggml_get_type_traits(type)->to_float(q.data(),decoded.data(),decoded.size());\n    q23_pack(ty,q.data(),packed.data(),cols,rows);\n    std::vector<std::vector<uint8_t>> acts(nt,std::vector<uint8_t>(astride));\n    std::vector<std::vector<float>> a(nt,std::vector<float>(cols)),y(nt,std::vector<float>(rows));\n    const void* ap[4];float* op[4];\n    for(int t=0;t<nt;++t) {\n        for(auto& v:a[t])v=float(int(rng()%2001)-1000)*.003f;\n        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(a[t].data(),acts[t].data(),cols);\n        if(ty==11) {\n            auto* blocks=reinterpret_cast<block_q8_K*>(acts[t].data());\n            for(int b=0;b<cols/256;++b) for(auto& q:blocks[b].qs) if(q==-128)q=-127;\n            if(std::getenv("STRATA_PROBE_MIN128"))blocks[0].qs[0]=-128;\n        }\n        const auto* qa=reinterpret_cast<const block_q8_K*>(acts[t].data());\n        for(int j=0;j<cols;++j)a[t][j]=qa[j/256].d*qa[j/256].qs[j%256];\n        ap[t]=acts[t].data();op[t]=y[t].data();\n    }\n    const auto& source=layout=="row"?q:packed;\n    std::vector<uint8_t> ring;\n    if(copies>1) {\n        ring.resize(source.size()*size_t(copies));\n        for(int i=0;i<copies;++i)std::memcpy(ring.data()+size_t(i)*source.size(),source.data(),source.size());\n        for(int i=0;i<copies;++i)if(std::memcmp(ring.data()+size_t(i)*source.size(),source.data(),source.size()))\n            throw std::runtime_error("weight replication mismatch");\n    }\n    bool candidate=false;\n    auto run=[&](int copy) {\n        const auto* weights=copies==1?source.data():ring.data()+size_t(copy)*source.size();\n        if(layout=="row")(candidate?q23_candidate_rows:q23_rows)(ty,weights,stride,cols,ap,nt,op,0,rows);\n        else q23_packed_rows(ty,weights,cols,ap,nt,op,0,rows,layout=="lut");\n    };\n    run(0);\n    for(int t=0;t<nt;++t)for(int r=0;r<rows;++r) {\n        double ref=0,norm=0;\n        for(int j=0;j<cols;++j){const double v=double(decoded[r*cols+j])*a[t][j];ref+=v;norm+=std::abs(v);}\n        if(!std::isfinite(y[t][r])||std::abs(y[t][r]-ref)>1e-5*norm+1e-5)throw std::runtime_error("benchmark FP64 check failed");\n    }\n    const auto expected=y;\n    // Warm every physical copy and verify dispatch into every ring slot.\n    for(int i=0;i<copies;++i) {\n        run(i);\n        for(int t=0;t<nt;++t)if(std::memcmp(y[t].data(),expected[t].data(),rows*sizeof(float)))\n            throw std::runtime_error("replicated weight output mismatch");\n    }\n    for(int round=0;round<12;++round)for(int order=0;order<2;++order) {\n        candidate=bool(order^(round%2));\n        run(0);\n        const auto start=std::chrono::steady_clock::now();const auto cpu=std::clock();\n        for(int i=0;i<repeats;++i)run((round*repeats+i)%copies);\n        const double cpu_ms=1000.*(std::clock()-cpu)/CLOCKS_PER_SEC;\n        const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();\n        for(int t=0;t<nt;++t)if(std::memcmp(y[t].data(),expected[t].data(),rows*sizeof(float)))\n            throw std::runtime_error("paired candidate output differs from baseline");\n        std::cout<<"round="<<round<<" candidate="<<candidate<<" nt="<<nt<<" copies="<<copies\n                 <<" calls="<<repeats<<" bytes="<<source.size()*size_t(copies)\n                 <<" cpu_ms="<<cpu_ms<<" wall_ms="<<wall_ms<<"\\n";\n    }\n}\nint main(int argc,char** argv){try {\n    if(argc!=3 && argc!=4)throw std::runtime_error("usage: paired_probe TOKENS COPIES [TYPE=10|11]");\n    bench(argc==4?std::stoi(argv[3]):10,std::stoi(argv[1]),64,"row",std::stoi(argv[2]));\n}catch(const std::exception& e){std::cerr<<e.what()<<\'\\n\';return 1;}}\n')
flags=['g++','-O3','-DNDEBUG','-std=c++20','-mavx2','-mfma','-mf16c','-DSTRATA_NATIVE_EXPERTS=1','-DSTRATA_Q2_HALF_PAIR_NT1=1','-Iinclude','-Ithird_party/ggml','-Ibuild-tr16-goal-20261010/ggml-reference/ggml/include']
renames=['-Dq23_rows=q23_candidate_rows','-Dq23_gu_rows=q23_candidate_gu_rows','-Dq3k_expanded_rows=candidate_q3k_expanded_rows','-Dq3k_expanded_gu_rows=candidate_q3k_expanded_gu_rows']
libs=['build-hip-rocm101-tr16/'+f for f in ['libstrata_kernels_cpu.a','libstrata_host.a','ggml/src/libggml-cpu.a','ggml/src/libggml-base.a']]
cmds=[flags+renames+['-c',str(p/'candidate.cpp'),'-o',str(p/'candidate.o')],flags+[str(p/'probe.cpp'),str(p/'candidate.o')]+libs+['-lpthread','-ldl','-lm','-o',str(p/'probe')]]
with (p/'build.log').open('w') as f:
 for c in cmds:subprocess.run(c,check=True,stdout=f,stderr=subprocess.STDOUT)
(p/'build.json').write_text(json.dumps(cmds,indent=2)+'\n')
