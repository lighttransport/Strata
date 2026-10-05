#include "strata/kernels/exl3.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include "strata/kernels/cpu/exl3/dense_q8.hpp"
#include "strata/artifact/glm_exl3.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <bit>
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void real_projection(const char* path) {
 strata::artifact::GlmExl3 model(path);
 for(const char* projection:{"gate_proj","up_proj","down_proj"}) {
  auto m=model.load(3,17,projection);int in=m.in,out=m.out;
  std::vector<uint8_t> packed(m.trellis.size());
  for(int ot=0;ot<out/16;++ot)for(int it=0;it<in/16;++it)
   std::copy_n(m.trellis.data()+(it*(out/16)+ot)*64,64,packed.data()+(ot*(in/16)+it)*64);
  std::vector<float> x(in),y(out),ref(out);for(int i=0;i<in;++i)x[i]=std::sin(float(i)*.017f);
  strata::cpu::exl3::linear(m.trellis.data(),m.suh.data(),m.svh.data(),in,out,x.data(),ref.data());
  uint8_t* dp;uint16_t *si,*so;float *dw,*dx,*dh,*dy;
  check(cudaMalloc(&dp,packed.size()));check(cudaMalloc(&si,in*2));check(cudaMalloc(&so,out*2));
  check(cudaMalloc(&dw,size_t(in)*out*4));check(cudaMalloc(&dx,in*4));check(cudaMalloc(&dh,in*4));check(cudaMalloc(&dy,out*4));
  check(cudaMemcpy(dp,packed.data(),packed.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(si,m.suh.data(),in*2,cudaMemcpyHostToDevice));check(cudaMemcpy(so,m.svh.data(),out*2,cudaMemcpyHostToDevice));check(cudaMemcpy(dx,x.data(),in*4,cudaMemcpyHostToDevice));
  strata::kernels::exl3_reconstruct(dp,dw,in,out,nullptr);strata::kernels::exl3_hadamard(dx,dh,si,in,1,true,nullptr);
  cublasHandle_t b;if(cublasCreate(&b)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("real cuBLAS create");const float one=1,zero=0;
  if(cublasSgemm(b,CUBLAS_OP_T,CUBLAS_OP_N,out,1,in,&one,dw,in,dh,in,&zero,dy,out)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("real cuBLAS projection");
  strata::kernels::exl3_hadamard(dy,dy,so,out,1,false,nullptr);check(cudaMemcpy(y.data(),dy,out*4,cudaMemcpyDeviceToHost));
  double err=0,energy=0;for(int i=0;i<out;++i){err+=double(y[i]-ref[i])*(y[i]-ref[i]);energy+=double(ref[i])*ref[i];}
  double rel=std::sqrt(err/std::max(energy,1e-30));std::cout<<projection<<" relative_L2="<<rel<<'\n';
  strata::kernels::exl3_packed_gemv(dp,dh,dy,in,out,nullptr);
  strata::kernels::exl3_hadamard(dy,dy,so,out,1,false,nullptr);check(cudaMemcpy(y.data(),dy,out*4,cudaMemcpyDeviceToHost));
  err=0;for(int i=0;i<out;++i)err+=double(y[i]-ref[i])*(y[i]-ref[i]);
  double packed_rel=std::sqrt(err/std::max(energy,1e-30));std::cout<<projection<<" packed_gemv_relative_L2="<<packed_rel<<'\n';
  if(!std::isfinite(packed_rel)||packed_rel>1e-5)throw std::runtime_error("real packed GEMV parity");
  cublasDestroy(b);cudaFree(dp);cudaFree(si);cudaFree(so);cudaFree(dw);cudaFree(dx);cudaFree(dh);cudaFree(dy);
  if(!std::isfinite(rel)||energy<=0||rel>.005)throw std::runtime_error("real CUDA projection parity");
 }
}
int main(int argc,char** argv) {
 int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count){std::cerr<<"CUDA driver/device unavailable\n";return 77;}
 try {
    constexpr int n=128;std::vector<uint8_t> packed(n*n/4);for(size_t i=0;i<packed.size();++i)packed[i]=(i*53+17)&255;
    std::vector<float> expected(n*n),actual(n*n),tile(256),x(n),y(n),reference(n);std::vector<uint16_t>scales(n);
    for(int ot=0;ot<n/16;++ot)for(int it=0;it<n/16;++it) {
        strata::cpu::exl3::tile(packed.data()+(ot*(n/16)+it)*64,tile.data());
        for(int r=0;r<16;++r)for(int c=0;c<16;++c)expected[(ot*16+c)*n+it*16+r]=tile[r*16+c];
    }
    uint8_t* dp;float* dw;float* dx;float* dy;uint16_t* ds;
    check(cudaMalloc(&dp,packed.size()));check(cudaMalloc(&dw,actual.size()*4));check(cudaMalloc(&dx,n*4));check(cudaMalloc(&dy,n*4));check(cudaMalloc(&ds,n*2));
    check(cudaMemcpy(dp,packed.data(),packed.size(),cudaMemcpyHostToDevice));strata::kernels::exl3_reconstruct(dp,dw,n,n,nullptr);
    check(cudaMemcpy(actual.data(),dw,actual.size()*4,cudaMemcpyDeviceToHost));if(actual!=expected)throw std::runtime_error("CUDA trellis mismatch");
    for(int i=0;i<n;++i){x[i]=std::sin(float(i));scales[i]=strata::cpu::exl3::half_round(.25f+float(i)/512);reference[i]=x[i]*strata::cpu::exl3::half_value(scales[i]);}
    strata::cpu::exl3::hadamard(reference.data(),n);check(cudaMemcpy(dx,x.data(),n*4,cudaMemcpyHostToDevice));check(cudaMemcpy(ds,scales.data(),n*2,cudaMemcpyHostToDevice));
    strata::kernels::exl3_hadamard(dx,dy,ds,n,1,true,nullptr);check(cudaMemcpy(y.data(),dy,n*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i)if(std::abs(y[i]-reference[i])>1e-5)throw std::runtime_error("CUDA Hadamard mismatch");
    // Full projection: Hadamard, FP32 GEMV, Hadamard and output scales.
    cublasHandle_t blas;if(cublasCreate(&blas)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS create");
    cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH);const float one=1,zero=0;
    if(cublasSgemv(blas,CUBLAS_OP_T,n,n,&one,dw,n,dy,1,&zero,dx,1)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS projection");
    strata::kernels::exl3_hadamard(dx,dx,ds,n,1,false,nullptr);check(cudaMemcpy(y.data(),dx,n*4,cudaMemcpyDeviceToHost));
    std::vector<uint8_t> source(packed.size());for(int it=0;it<n/16;++it)for(int ot=0;ot<n/16;++ot)std::copy_n(packed.data()+(ot*(n/16)+it)*64,64,source.data()+(it*(n/16)+ot)*64);
    strata::cpu::exl3::linear(source.data(),scales.data(),scales.data(),n,n,x.data(),reference.data());
    double error=0,energy=0;float max_error=0,max_ref=1;
    for(int i=0;i<n;++i){double diff=y[i]-reference[i];error+=diff*diff;energy+=double(reference[i])*reference[i];max_error=std::max(max_error,float(std::abs(diff)));max_ref=std::max(max_ref,std::abs(reference[i]));}
    if(std::sqrt(error/std::max(energy,1e-20))>.005||max_error>.01*max_ref)throw std::runtime_error("CUDA full projection parity");
    std::vector<uint16_t> bf16(n*n);std::vector<float> bf_reference(n),bf_values(n*n);
    for(int i=0;i<n*n;++i){bf16[i]=std::bit_cast<uint32_t>(std::sin(float(i)*.1f))>>16;bf_values[i]=std::bit_cast<float>(uint32_t(bf16[i])<<16);}
    for(int row=0;row<n;++row){float sum=0;for(int col=0;col<n;++col)sum+=bf_values[row*n+col]*x[col];bf_reference[row]=sum;}
    uint16_t* db;check(cudaMalloc(&db,bf16.size()*2));check(cudaMemcpy(db,bf16.data(),bf16.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(dx,x.data(),n*4,cudaMemcpyHostToDevice));
    strata::kernels::exl3_bf16_gemv(db,dx,dy,n,n,nullptr);check(cudaMemcpy(y.data(),dy,n*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i)if(std::abs(y[i]-bf_reference[i])>1e-4*std::max(1.f,std::abs(bf_reference[i])))throw std::runtime_error("BF16 GEMV parity");
    strata::kernels::exl3_bf16_to_float(db,dw,n*n,nullptr);check(cudaMemcpy(actual.data(),dw,actual.size()*4,cudaMemcpyDeviceToHost));
    if(actual!=bf_values)throw std::runtime_error("BF16 panel conversion parity");
    std::vector<strata::cpu::exl3::DenseQ8Block> q8(n*n/32);
    std::vector<float> q8_values(n*n),q8_reference(n);
    for(int row=0;row<n;++row)strata::cpu::exl3::dense_q8_row(bf16.data()+row*n,q8.data()+row*n/32,n);
    for(int i=0;i<n*n;++i)q8_values[i]=strata::cpu::exl3::half_value(q8[i/32].scale)*q8[i/32].values[i%32];
    for(int row=0;row<n;++row)for(int col=0;col<n;++col)q8_reference[row]+=q8_values[row*n+col]*x[col];
    void* dq8;check(cudaMalloc(&dq8,q8.size()*sizeof(q8[0])));
    check(cudaMemcpy(dq8,q8.data(),q8.size()*sizeof(q8[0]),cudaMemcpyHostToDevice));
    strata::kernels::exl3_dense_q8_to_float(dq8,dw,n*n,nullptr);check(cudaMemcpy(actual.data(),dw,actual.size()*4,cudaMemcpyDeviceToHost));
    if(actual!=q8_values)throw std::runtime_error("dense Q8 panel conversion parity");
    strata::kernels::exl3_dense_q8_gemv(dq8,dx,dy,n,n,nullptr);check(cudaMemcpy(y.data(),dy,n*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i)if(std::abs(y[i]-q8_reference[i])>1e-4*std::max(1.f,std::abs(q8_reference[i])))throw std::runtime_error("dense Q8 GEMV parity");
    cudaFree(dq8);
    std::vector<strata::cpu::exl3::DenseQ6Block> q6(n*n/32);
    std::vector<float> q6_values(n*n),q6_reference(n);
    for(int row=0;row<n;++row)strata::cpu::exl3::dense_q6_row(bf16.data()+row*n,q6.data()+row*n/32,n);
    for(int i=0;i<n*n;++i)q6_values[i]=strata::cpu::exl3::dense_q6_value(q6[i/32],i%32);
    for(int row=0;row<n;++row)for(int col=0;col<n;++col)q6_reference[row]+=q6_values[row*n+col]*x[col];
    void* dq6;check(cudaMalloc(&dq6,q6.size()*sizeof(q6[0])));
    check(cudaMemcpy(dq6,q6.data(),q6.size()*sizeof(q6[0]),cudaMemcpyHostToDevice));
    strata::kernels::exl3_dense_q6_to_float(dq6,dw,n*n,nullptr);check(cudaMemcpy(actual.data(),dw,actual.size()*4,cudaMemcpyDeviceToHost));
    if(actual!=q6_values)throw std::runtime_error("dense Q6 panel conversion parity");
    strata::kernels::exl3_dense_q6_gemv(dq6,dx,dy,n,n,nullptr);check(cudaMemcpy(y.data(),dy,n*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i)if(std::abs(y[i]-q6_reference[i])>1e-4*std::max(1.f,std::abs(q6_reference[i])))throw std::runtime_error("dense Q6 GEMV parity");
    cudaFree(dq6);
    cudaFree(db);
    cublasDestroy(blas);
    cudaFree(dp);cudaFree(dw);cudaFree(dx);cudaFree(dy);cudaFree(ds);if(argc>1)real_projection(argv[1]);std::cout<<"EXL3 CUDA PASS\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
