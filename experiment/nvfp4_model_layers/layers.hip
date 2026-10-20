// Reuse the frozen, validated W4A16 kernels without changing measured sources.
#define main synthetic_w4a16_main
#include "../nvfp4_rdna4/w4a16.hip"
#undef main

struct Weight {
    uint32_t layer,n,k;
    float alpha;
    std::vector<uint8_t> w,s;
};
static void read_exact(std::ifstream& f,void* ptr,size_t n) {
    f.read(static_cast<char*>(ptr),n); require(bool(f),"truncated input");
}
static std::vector<Weight> load(const char* path) {
    std::ifstream f(path,std::ios::binary);require(bool(f),"cannot open input");
    char magic[8];read_exact(f,magic,8);require(std::string(magic,8)=="NVLAYER1","bad magic");
    uint32_t count;read_exact(f,&count,4);require(count==6,"expected six projections");
    std::vector<Weight> weights;
    for(unsigned i=0;i<count;++i) {
        Weight t;read_exact(f,&t.layer,4);read_exact(f,&t.n,4);read_exact(f,&t.k,4);read_exact(f,&t.alpha,4);
        require(t.layer==i/3 && t.n==(i%3==2?4096u:12288u) && t.k==(i%3==2?12288u:4096u),"unexpected layer shape/order");
        require(std::isfinite(t.alpha)&&t.alpha>0,"invalid global scale");
        t.w.resize(size_t(t.n)*t.k/2);t.s.resize(size_t(t.n)*t.k/16);
        read_exact(f,t.w.data(),t.w.size());read_exact(f,t.s.data(),t.s.size());
        for(auto s:t.s) require(s<127,"nonfinite or negative E4M3 scale");
        weights.push_back(std::move(t));
    }
    require(f.peek()==std::char_traits<char>::eof(),"trailing input");return weights;
}
template<bool Half> static void measure(const Weight& t,int m,const std::string& output) {
    Case<Half> c(m,t.n,t.k,"random");c.w=t.w;c.s=t.s;c.alpha=t.alpha;
    // Independent host decode: explicit E2M1 table and arithmetic FP8 formula.
    constexpr float table[]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
    for(size_t i=0;i<c.expanded.size();++i) {
        unsigned code=(c.w[i/2]>>((i%2)*4))&15, scale=c.s[i/16];
        unsigned exp=scale>>3, mant=scale&7;
        float sv=exp?std::ldexp(1.0f+mant/8.0f,int(exp)-7):mant/512.0f;
        float x=table[code]*sv;
        c.expanded[i]=encode16<Half>(x);
        require(decode16<Half>(c.expanded[i])==x,"inexact local weight decode");
    }
    std::ofstream csv(output);require(bool(csv),"cannot create CSV");csv.precision(10);
    csv<<"activation,m,n,k,kernel,round,ms,tops,max_abs_fp64_error,rel_l2_fp64\n";
    run(c,true,&csv);require(bool(csv),"CSV write failure");
}

template<bool Half> static uint16_t rounded16(float x) {
    if constexpr(Half) return encode16<Half>(x);
    else { uint32_t bits;std::memcpy(&bits,&x,4);bits+=0x7fff+((bits>>16)&1);return uint16_t(bits>>16); }
}
template<bool Half> static std::vector<float> chain_projection(const Weight& t,const std::vector<uint16_t>& input,int m) {
    size_t count=size_t(m)*t.n;
    Buffer<uint16_t> a(input.size()),expanded(size_t(t.n)*t.k);
    Buffer<uint8_t> w(t.w.size()),s(t.s.size());
    Buffer<float> packed(count),baseline(count);Buffer<double> ref(count),absolute(count);
    a.put(input);w.put(t.w);s.put(t.s);
    expand<Half><<<(expanded.n+255)/256,256>>>(w.p,s.p,expanded.p,expanded.n);
    simple<Half,false><<<dim3((t.n+15)/16,(m+15)/16),32>>>(a.p,w.p,s.p,expanded.p,baseline.p,m,t.n,t.k,t.alpha);
    tiled<Half,true,64><<<dim3((t.n+63)/64,(m+63)/64),128>>>(a.p,w.p,s.p,expanded.p,packed.p,m,t.n,t.k,t.alpha);
    reference<Half><<<(count+127)/128,128>>>(a.p,w.p,s.p,ref.p,absolute.p,m,t.n,t.k,t.alpha);
    HIP(hipGetLastError());HIP(hipDeviceSynchronize());
    auto got=packed.get(),base=baseline.get();auto expected=ref.get(),absref=absolute.get();
    double u=std::numeric_limits<float>::epsilon()/2,gamma=((t.k+1)*u)/(1-(t.k+1)*u);
    double maxerr=0;
    for(size_t i=0;i<count;++i) {
        double e=std::abs(double(got[i])-expected[i]);
        require(std::isfinite(got[i])&&got[i]==base[i]&&e<=gamma*absref[i],"chained projection mismatch");
        maxerr=std::max(maxerr,e);
    }
    std::printf("CHAIN projection n=%u k=%u max_abs_fp64_error=%.9g\n",t.n,t.k,maxerr);
    return got;
}
template<bool Half> static void chain(const std::vector<Weight>& weights,unsigned layer) {
    constexpr int m=16;
    Case<Half> seed(m,1,4096,"random");
    // Scale synthetic input into a smaller normal range for the MLP smoke test.
    for(auto& x:seed.a) x=rounded16<Half>(decode16<Half>(x)*0.125f);
    auto gate=chain_projection<Half>(weights[layer*3],seed.a,m);
    auto up=chain_projection<Half>(weights[layer*3+1],seed.a,m);
    std::vector<uint16_t> hidden(gate.size());
    for(size_t i=0;i<hidden.size();++i) {
        float g=std::min(decode16<Half>(rounded16<Half>(gate[i])),10.0f);
        float v=std::clamp(decode16<Half>(rounded16<Half>(up[i])),-10.0f,10.0f);
        float silu=decode16<Half>(rounded16<Half>(g/(1.0f+std::exp(-g))));
        hidden[i]=rounded16<Half>(silu*v);
    }
    auto out=chain_projection<Half>(weights[layer*3+2],hidden,m);
    double norm=0;for(float x:out) norm+=double(x)*x;
    require(norm>0&&std::isfinite(norm),"invalid chain output");
    std::printf("CHAIN PASS layer=%u activation=%s m=16 output_l2=%.9g\n",layer,Half?"fp16":"bf16",std::sqrt(norm));
}
int main(int argc,char** argv) {
    try {
        require(argc==3,"usage: layers INPUT.bin OUTPUT_PREFIX");
        std::setvbuf(stdout,nullptr,_IOLBF,0);
        HIP(hipSetDevice(0));hipDeviceProp_t p{};HIP(hipGetDeviceProperties(&p,0));
        require(std::string(p.gcnArchName).find("gfx1201")==0,"requires gfx1201");
        std::printf("GPU %s %s; real NVFP4 model weights, synthetic full-mantissa activations\n",p.name,p.gcnArchName);
        auto weights=load(argv[1]);const char* names[]={"gate","up","down"};
        for(size_t i=0;i<weights.size();++i) for(int m:{1,16,512}) {
            const auto& t=weights[i];
            std::printf("MODEL layer=%u projection=%s n=%u k=%u global_scale=%.9g m=%d\n",t.layer,names[i%3],t.n,t.k,t.alpha,m);
            std::string prefix=std::string(argv[2])+"-l"+std::to_string(t.layer)+"-"+names[i%3]+"-m"+std::to_string(m);
            measure<false>(t,m,prefix+"-bf16.csv");
            measure<true>(t,m,prefix+"-fp16.csv");
        }
        std::puts("PASS all 36 real-weight cases across six modes");
        for(unsigned layer=0;layer<2;++layer) {chain<false>(weights,layer);chain<true>(weights,layer);}
        std::puts("PASS all four MLP chain smoke cases");return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
}
