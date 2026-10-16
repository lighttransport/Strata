#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
namespace {
__device__ float e5(unsigned v){return v?ldexpf(1.f+float(v&7)/8,int(v>>3)-15):0.f;}
__device__ unsigned state(const uint32_t* p,int r,int c){unsigned t=(r&1)+((r&6)<<2)+((r&8)>>2)+((c&7)<<5)+((c&8)>>1);
    unsigned end=(t+257)*2,start=end-16,shift=31-((end-1)&31);return unsigned(((uint64_t(p[(start/32)%16])<<32)|p[((end-1)/32)%16])>>shift)&65535;}
__device__ float decode(unsigned s,int kind){
    if(kind==0){unsigned v=(s*0xcbac1fedu&0x8fff8fffu)^0x3b603b60u;return __half2float(__hadd(__ushort_as_half(v&65535),__ushort_as_half(v>>16)));}
    if(kind==1){unsigned v=s*0x83dcd12du;unsigned sum=(v&255)+((v>>8)&255)+((v>>16)&255)+(v>>24);
        return __half2float(__float2half_rn(fmaf(__half2float(__ushort_as_half(0x6400+sum)),__half2float(__ushort_as_half(0x1eee)),__half2float(__ushort_as_half(0xc931)))));}
    unsigned h=s>>2,p=(__popc(h&0x1089)&1)|((__popc(h&0x2112)&1)<<1),q=(__popc(h&0x0a24)&1)|((__popc(h&0x2441)&1)<<1);
    const float palettes[16]={-6,-1.5,0,2,-4,-1,.5,3,-3,-.5,1,4,-2,0,1.5,6};return palettes[p*4+((s&3)^q)];
}
__device__ float xor_byte_decode(unsigned s){
    unsigned h=s>>2,p=(__popc(h&0x1089)&1)|((__popc(h&0x2112)&1)<<1);
    unsigned q=(__popc(h&0x0a24)&1)|((__popc(h&0x2441)&1)<<1),index=p*4+((s&3)^q);
    unsigned low=__byte_perm(0x0400fdf4u,0x0601fef8u,index&7);
    unsigned high=__byte_perm(0x0802fffau,0x0c0300fcu,index&7);
    int value=int((index<8?low:high)<<24)>>24;return .5f*value;
}
__global__ void xor_decode_probe(float* output){unsigned s=blockIdx.x*256+threadIdx.x;if(s<65536)output[s]=xor_byte_decode(s);}
__global__ void quant_act(const float* x,int8_t* q,float* scale,int n){
    __shared__ float maxima[256];float peak=0;for(int i=threadIdx.x;i<n;i+=256)peak=fmaxf(peak,fabsf(x[i]));maxima[threadIdx.x]=peak;__syncthreads();
    for(int d=128;d;d/=2){if(threadIdx.x<d)maxima[threadIdx.x]=fmaxf(maxima[threadIdx.x],maxima[threadIdx.x+d]);__syncthreads();}
    float s=maxima[0]>0?maxima[0]/127:1;if(!threadIdx.x)*scale=s;for(int i=threadIdx.x;i<n;i+=256)q[i]=(int8_t)__float2int_rn(x[i]/s);
}
template<bool Q8,bool FAST=false> __global__ void mv(const uint8_t* packed,const float* scales,const uint8_t* s8,const float* grid,
                                    const float* x,const int8_t* aq,const float* as,float* y,int in,int out,int group,int kind){
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(row>=out)return;float sum=0;
    for(int c=lane;c<in;c+=32){float ws=kind==2?e5(s8[size_t(row)*(in/group)+c/group]):scales[size_t(row)*(in/group)+c/group];
        if(kind<=2){auto p=(const uint32_t*)(packed+(size_t(row/16)*(in/16)+c/16)*64);unsigned s=state(p,c%16,row%16);
            if constexpr(Q8){unsigned v=s*0x83dcd12du;int a=int(aq[c]);unsigned ap=unsigned(a+128)*0x01010101u;
                int bs=int((v&255)+((v>>8)&255)+((v>>16)&255)+(v>>24));
                int dot=int(__dp4a(v,ap,0u))-128*bs;
                float unit=__half2float(__ushort_as_half(0x1eee)),bias=1024*unit+__half2float(__ushort_as_half(0xc931));
                sum+=(float(dot)*unit+float(a)*bias)*(*as)*ws;
            }else sum+=(FAST&&kind==2?xor_byte_decode(s):decode(s,kind))*ws*x[c];
        }else {size_t i=size_t(row)*in+c;unsigned bits=kind==3?4:2,v=(packed[i*bits/8]>>(i*bits%8))&((1<<bits)-1);sum+=grid[v]*ws*x[c];}
    }
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);if(!lane)y[row]=sum;
}
// One packed byte holds four exact Qift levels. Read it once and process four
// consecutive activations, keeping source weights packed throughout the GEMV.
__global__ void qift_packed(const uint8_t* packed,const float* scales,const float* x,float* y,int in,int out){
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(row>=out)return;float sum=0;
    for(int c=lane*4;c<in;c+=128){unsigned byte=__ldg(packed+size_t(row)*(in/4)+c/4);
        const float4 input=__ldg(reinterpret_cast<const float4*>(x+c));
        const float values[4]={input.x,input.y,input.z,input.w};
#pragma unroll
        for(int j=0;j<4;++j){int code=(byte>>(j*2))&3;float w=float(2*code-3)*.5f;sum=fmaf(w,values[j],sum);}
    }
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);if(!lane)y[row]=sum*scales[row];
}
}
void research_xor_decode_probe(float* output){xor_decode_probe<<<256,256>>>(output);}
void research_gemv(const uint8_t* p,const float* scales,const uint8_t* s8,const float* grid,const float* x,
                   int8_t* q,float* as,float* y,int in,int out,int group,int kind,bool q8,bool fast){
    if(q8){quant_act<<<1,256>>>(x,q,as,in);mv<true><<<(out+7)/8,256>>>(p,scales,s8,grid,x,q,as,y,in,out,group,kind);}
    else if(fast&&kind==4)qift_packed<<<(out+7)/8,256>>>(p,scales,x,y,in,out);
    else if(fast)mv<false,true><<<(out+7)/8,256>>>(p,scales,s8,grid,x,q,as,y,in,out,group,kind);
    else mv<false><<<(out+7)/8,256>>>(p,scales,s8,grid,x,q,as,y,in,out,group,kind);
}
