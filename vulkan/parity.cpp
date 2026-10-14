#include "projections.hpp"
#include "runtime.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/canon_expert.hpp"
#include "strata/kernels/cpu/q23_avx2.hpp"
#include "strata/kernels/glm_reference.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
namespace cpu = strata::kernels::cpu;
namespace ref = strata::kernels::glm_reference;
void close(const std::vector<float> &a, const std::vector<float> &b,
           float tolerance, const char *name) {
  if (a.size() != b.size())
    throw std::runtime_error("parity geometry differs");
  double error = 0, energy = 0;
  float maximum = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      throw std::runtime_error("parity nonfinite");
    error += (a[i] - b[i]) * (a[i] - b[i]);
    energy += b[i] * b[i];
    maximum = std::max(maximum, std::abs(a[i] - b[i]));
  }
  double relative = std::sqrt(error / std::max(1e-20, energy));
  std::cerr << "PARITY name=" << name << " relative_l2=" << relative
            << " max_abs=" << maximum << '\n';
  if (relative > tolerance)
    throw std::runtime_error(std::string(name) + " parity failed");
}
int main(int argc, char **argv) {
  try {
    strata::vulkan::Projections projections(0, 512ULL * 1024 * 1024);
    strata::vulkan::Compute gpu(projections.device_name());
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> random(-.2f, .2f);
    auto values = [&](size_t n) {
      std::vector<float> v(n);
      for (float &x : v)
        x = random(rng);
      return v;
    };
    for (int nt : {1, 2, 8, 64}) {
      int in = 256, out = 128;
      auto w = values(in * out), x = values(in * nt);
      std::span<const uint8_t> bytes(
          reinterpret_cast<const uint8_t *>(w.data()), w.size() * 4);
      auto actual = projections.synthetic("f32-" + std::to_string(nt),
                                          GGML_TYPE_F32, bytes, in, out, x, nt);
      std::vector<float> expected(out * nt);
      for (int t = 0; t < nt; t++)
        for (int r = 0; r < out; r++)
          for (int i = 0; i < in; i++)
            expected[t * out + r] += w[r * in + i] * x[t * in + i];
      close(actual, expected, 2e-5, "projection-f32");
    }
    for (auto type :
         {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
          GGML_TYPE_Q6_K, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0}) {
      const int in = 4096, out = 128;
      auto w = values(in * out), x = values(in);
      std::vector<uint8_t> packed(ggml_row_size(type, in) * out);
      const size_t written = ggml_quantize_chunk(type, w.data(), packed.data(),
                                                 0, out, in, nullptr);
      if (written != packed.size())
        throw std::runtime_error("quantized fixture size differs");
      std::vector<float> decoded(w.size()), expected(out);
      auto quantized_x = strata::vulkan::decode_activations(x);
      ggml_get_type_traits(type)->to_float(packed.data(), decoded.data(),
                                           decoded.size());
      for (int r = 0; r < out; ++r)
        for (int i = 0; i < in; ++i)
          expected[r] += decoded[r * in + i] * quantized_x[i];
      if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0) {
        const int blocks = in / 32;
        for (int r = 0; r < out; ++r)
          for (int b = 0; b < blocks; ++b) {
            float reconstructed = 0;
            for (int i = 0; i < 32; ++i)
              reconstructed += quantized_x[b * 32 + i];
            auto sum = [&](auto &&self, int first, int stride) -> float {
              if (stride == 32)
                return x[b * 32 + first];
              return self(self, first, stride * 2) +
                     self(self, first + stride, stride * 2);
            };
            float original = sum(sum, 0, 1);
            original = ggml_fp16_to_fp32(ggml_fp32_to_fp16(original));
            ggml_fp16_t d;
            std::memcpy(&d,
                        packed.data() +
                            (size_t(r) * blocks + b) * ggml_type_size(type),
                        sizeof(d));
            expected[r] += (type == GGML_TYPE_Q4_0 ? -8.f : -16.f) *
                           ggml_fp16_to_fp32(d) * (original - reconstructed);
          }
      }
      auto actual = projections.synthetic("quant-" + std::to_string(type), type,
                                          packed, in, out, x);
      close(actual, expected, 2e-5f, "projection-quantized");
      auto repeat = projections.synthetic("quant-" + std::to_string(type), type,
                                          packed, in, out, x);
      if (actual != repeat)
        throw std::runtime_error("Cached projection repeat differs");
    }
    if (const char *check = std::getenv("STRATA_VULKAN_AVX2_ACTIVATIONS");
        check && std::string(check) == "1") {
      auto input = values(32 * 128);
      for (int block = 0; block < 64; ++block) {
        input[block * 32 + 31] = 127.f;
        input[block * 32] = .5f;
        input[block * 32 + 1] = -.5f;
        input[block * 32 + 2] = std::nextafter(.5f, 0.f);
        input[block * 32 + 3] = std::nextafter(.5f, 1.f);
        input[block * 32 + 4] = -0.f;
      }
      auto vectorized = strata::vulkan::decode_activations(input);
#ifdef _WIN32
      _putenv_s("STRATA_VULKAN_AVX2_ACTIVATIONS", "0");
#else
      setenv("STRATA_VULKAN_AVX2_ACTIVATIONS", "0", 1);
#endif
      auto scalar = strata::vulkan::decode_activations(input);
#ifdef _WIN32
      _putenv_s("STRATA_VULKAN_AVX2_ACTIVATIONS", "1");
#else
      setenv("STRATA_VULKAN_AVX2_ACTIVATIONS", "1", 1);
#endif
      if (std::memcmp(scalar.data(), vectorized.data(),
                      scalar.size() * sizeof(float)))
        throw std::runtime_error("AVX2 activation bytes differ from scalar");
      std::cerr << "PARITY name=activation-avx2 bitwise_equal=1 values="
                << input.size() << '\n';
    }
    const int heads = 64, dim = 128, n = heads * dim;
    std::vector<float> state(size_t(n) * dim), actual(n), expected(n);
    for (int step = 0; step < 5; step++) {
      auto q = values(n), k = values(n), v = values(n), g = values(n),
           beta = values(heads);
      for (float &x : g)
        x = -std::abs(x);
      gpu.kda(0, q, k, v, g, beta, actual, heads, dim);
      ref::kda(state.data(), q.data(), k.data(), v.data(), g.data(),
               beta.data(), expected.data(), heads, dim);
      close(actual, expected, 2e-5, "kda-output");
      close(gpu.kda_state(0), state, 2e-5, "kda-state");
    }
    // Fused recurrent path: convolution history, gate equations and output RMS.
    for (int convolution : {1, 4}) {
      const int count = heads * dim, layer = 20 + convolution;
      auto weights = values(3 * count * convolution), bias = values(count),
           a = values(heads), norm = values(dim);
      for (float &value : a)
        value = -std::abs(value) - 0.1f;
      std::vector<float> history(3 * count * (convolution - 1)),
          expected_state(size_t(count) * dim), output(count);
      for (int step = 0; step < 5; ++step) {
        auto raw = values(3 * count), decay = values(count),
             beta = values(heads), gate = values(count);
        std::array<std::vector<float>, 3> convolved;
        for (int part = 0; part < 3; ++part) {
          convolved[part].resize(count);
          for (int channel = 0; channel < count; ++channel) {
            const int wi = (part * count + channel) * convolution,
                      hi = (part * count + channel) * (convolution - 1);
            float sum =
                raw[part * count + channel] * weights[wi + convolution - 1];
            for (int k = 0; k < convolution - 1; ++k)
              sum += history[hi + k] * weights[wi + k];
            for (int k = 0; k < convolution - 2; ++k)
              history[hi + k] = history[hi + k + 1];
            if (convolution > 1)
              history[hi + convolution - 2] = raw[part * count + channel];
            convolved[part][channel] = sum * ref::sigmoid(sum);
          }
        }
        auto g = decay, b = beta;
        std::vector<float> expected(count);
        for (int channel = 0; channel < count; ++channel)
          g[channel] = -5.f * ref::sigmoid(-a[channel / dim] *
                                           (decay[channel] + bias[channel]));
        ref::kda(expected_state.data(), convolved[0].data(),
                 convolved[1].data(), convolved[2].data(), g.data(), b.data(),
                 expected.data(), heads, dim);
        for (int h = 0; h < heads; ++h) {
          float energy = 0;
          for (int j = 0; j < dim; ++j)
            energy += expected[h * dim + j] * expected[h * dim + j];
          const float inv = 1 / std::sqrt(energy / dim + 1e-6f);
          for (int j = 0; j < dim; ++j)
            expected[h * dim + j] = expected[h * dim + j] * inv * norm[j] *
                                    ref::sigmoid(gate[h * dim + j]);
        }
        gpu.kda_fused(layer, raw, decay, beta, gate, weights, bias, a, norm,
                      output, heads, convolution, -5.f, 1e-6f);
        close(output, expected, 2e-5, "kda-fused-output");
        close(gpu.kda_state(layer), expected_state, 2e-5, "kda-fused-state");
        if (convolution > 1)
          close(gpu.kda_history(layer), history, 0, "kda-fused-history");
      }
    }
    auto r = values(4 * 256), p = values(24), base = values(24),
         scale = values(3), c = values(24), x = values(256), ec = values(24),
         ex = values(256);
    ref::mhc(p.data(), base.data(), scale.data(), ec.data(), 20, 1e-6f);
    for (int d = 0; d < 256; d++) {
      ex[d] = 0;
      for (int i = 0; i < 4; i++)
        ex[d] += ec[i] * r[i * 256 + d];
    }
    gpu.mhc_read(r, p, base, scale, c, x, 20, 1e-6f);
    close(c, ec, 2e-5, "mhc-coefficients");
    close(x, ex, 2e-5, "mhc-read");
    auto y = values(256);
    std::vector<float> out(1024), eo(1024);
    for (int j = 0; j < 4; j++)
      for (int d = 0; d < 256; d++) {
        eo[j * 256 + d] = c[4 + j] * y[d];
        for (int i = 0; i < 4; i++)
          eo[j * 256 + d] += c[8 + i * 4 + j] * r[i * 256 + d];
      }
    gpu.mhc_write(r, c, y, out);
    close(out, eo, 2e-5, "mhc-write");
    int latent = 512;
    std::vector<float> cache;
    for (int pos = 0; pos < 17; pos++) {
      auto row = values(latent), q = values(heads * latent);
      cache.insert(cache.end(), row.begin(), row.end());
      std::vector<int> ids;
      for (int i = 0; i <= pos; i++)
        if (i % 3 != 1)
          ids.push_back(i);
      std::vector<float> a(heads * latent), e(heads * latent);
      gpu.mla(1, row, pos, 32, q, ids, a, heads, .125f);
      for (int h = 0; h < heads; h++) {
        std::vector<float> scores;
        for (int id : ids) {
          float s = 0;
          for (int d = 0; d < latent; d++)
            s += q[h * latent + d] * cache[id * latent + d];
          scores.push_back(s * .125f);
        }
        float mx = *std::max_element(scores.begin(), scores.end()), total = 0;
        for (float &v : scores) {
          v = std::exp(v - mx);
          total += v;
        }
        for (size_t s = 0; s < scores.size(); s++)
          for (int d = 0; d < latent; d++)
            e[h * latent + d] += scores[s] / total * cache[ids[s] * latent + d];
      }
      close(a, e, 2e-5, "mla");
    }
    bool rejected = false;
    try {
      std::vector<float> q(heads * latent), y(heads * latent), row(latent);
      std::vector<int> ids{999};
      gpu.mla(1, row, 17, 32, q, ids, y, heads, 1);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("MLA invalid ID not rejected");
    // Chunked MLA must preserve causal selections and retained cache contents.
    gpu.reset();
    std::vector<std::vector<float>> batch_rows,batch_queries;
    std::vector<std::vector<int>> batch_ids;
    std::vector<float> sequential;
    for(int pos=0;pos<8;++pos) {
      batch_rows.push_back(values(latent));batch_queries.push_back(values(heads*latent));
      std::vector<int> ids;for(int i=0;i<=pos;++i)if(i%3!=1)ids.push_back(i);batch_ids.push_back(ids);
      std::vector<float> output(heads*latent);
      gpu.mla(1,batch_rows.back(),pos,32,batch_queries.back(),ids,output,heads,.125f);
      sequential.insert(sequential.end(),output.begin(),output.end());
    }
    gpu.reset();std::vector<float> batched;
    for(int base:{0,4}) {
      std::vector<float> rows,queries;std::vector<std::vector<int>> ids;
      for(int t=base;t<base+4;++t) {
        rows.insert(rows.end(),batch_rows[t].begin(),batch_rows[t].end());
        queries.insert(queries.end(),batch_queries[t].begin(),batch_queries[t].end());ids.push_back(batch_ids[t]);
      }
      auto output=gpu.mla_batch(1,rows,base,32,queries,ids,heads,.125f);
      batched.insert(batched.end(),output.begin(),output.end());
    }
    close(batched,sequential,0,"mla-batched-causal-chunks");
    if(std::memcmp(batched.data(),sequential.data(),batched.size()*4))throw std::runtime_error("Batched MLA bits differ");
    {
      // Exercise the real head/latent geometry and the sparse-selection
      // boundary, rather than relying only on tiny attention caches.
      strata::vulkan::Compute attention(projections.device_name());
      constexpr int seed=2048,width=512,nh=64,nt=3,capacity=4096;
      auto rows=values(seed*width);std::vector<float> zero_queries(seed*width);
      std::vector<std::vector<int>> seed_ids(seed);
      for(int t=0;t<seed;++t)seed_ids[t]={t};
      attention.mla_batch(90,rows,0,capacity,zero_queries,seed_ids,1,.0625f);
      auto snapshot=attention.checkpoint();
      auto next_rows=values(nt*width),queries=values(nt*nh*width);
      std::vector<std::vector<int>> selections(nt);std::vector<float> singles;
      for(int t=0;t<nt;++t) {
        selections[t].resize(seed+t+1);std::iota(selections[t].begin(),selections[t].end(),0);
        std::vector<float> output(nh*width);
        attention.mla(90,{next_rows.data()+t*width,width},seed+t,capacity,
                      {queries.data()+t*nh*width,nh*width},selections[t],output,nh,.0625f);
        singles.insert(singles.end(),output.begin(),output.end());
      }
      attention.restore(snapshot);
      auto actual=attention.mla_batch(90,next_rows,seed,capacity,queries,selections,nh,.0625f);
      close(actual,singles,0,"mla-real-geometry-selection-boundary");
      if(std::memcmp(actual.data(),singles.data(),actual.size()*sizeof(float)))
        throw std::runtime_error("Real-geometry batched MLA bits differ");
    }
    const auto state_before=gpu.kda_state(0),conv_before=gpu.kda_history(24);
    auto saved=gpu.checkpoint();
    auto newq=values(n),newk=values(n),newv=values(n),newg=values(n),newbeta=values(heads);
    std::vector<float> newoutput(n);
    gpu.kda(0,newq,newk,newv,newg,newbeta,newoutput,heads,dim);
    gpu.restore(saved);
    close(gpu.kda_state(0),state_before,0,"snapshot-kda-state");
    close(gpu.kda_history(24),conv_before,0,"snapshot-convolution-history");
    std::vector<float> row=values(latent),query=values(heads*latent),once(heads*latent),again(heads*latent);
    std::vector<int> selected{0,2,7,8};
    gpu.mla(1,row,8,32,query,selected,once,heads,.125f);
    gpu.restore(saved);
    gpu.mla(1,row,8,32,query,selected,again,heads,.125f);
    close(once,again,0,"snapshot-mla-append-replay");
    try {
      gpu.mla_batch(1,row,9,32,query,{{10}},heads,.125f);
      throw std::runtime_error("Batched MLA accepted a future row");
    } catch(const std::invalid_argument &) {}
    for(auto [ih,idim]:{std::pair{4,16},std::pair{32,128}}) {
      const int pools=257;
      auto pooled=values(pools*idim),queries=values(3*ih*idim),weights=values(3*ih);
      std::vector<int> counts{0,3,pools};
      std::vector<float> expected(3*pools);
      for(int t=0;t<3;++t)for(int row=0;row<counts[t];++row) {
        float score=0;
        for(int head=0;head<ih;++head) {
          float dot=0;
          for(int d=0;d<idim;++d)dot+=queries[(t*ih+head)*idim+d]*pooled[row*idim+d];
          score+=weights[t*ih+head]*std::max(dot/std::sqrt(float(idim)),0.f);
        }
        expected[t*pools+row]=score/std::sqrt(float(ih));
      }
      auto actual=gpu.index_scores(pooled,queries,weights,counts,ih,idim);
      close(actual,expected,0,"indexer-causal-scores");
      if(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)))
        throw std::runtime_error("Indexer score bits differ from canonical CPU selection");
      counts[1]=pools+1;
      try {
        gpu.index_scores(pooled,queries,weights,counts,ih,idim);
        throw std::runtime_error("Indexer accepted a future pool");
      } catch(const std::invalid_argument &) {}
    }
    {
      // 1024 tokens * 64 heads crosses Polaris's 65535-group X limit.
      const int nt=1024,nh=64,width=8;
      auto rows=values(nt*width),queries=values(nt*nh*width);
      std::vector<std::vector<int>> selected(nt,std::vector<int>{0});
      auto output=gpu.mla_batch(91,rows,0,nt,queries,selected,nh,.125f);
      std::vector<float> expected(output.size());
      for(size_t i=0;i<expected.size();++i)expected[i]=rows[i%width];
      close(output,expected,0,"mla-large-grid-causal");
      if(std::memcmp(output.data(),expected.data(),output.size()*4))
        throw std::runtime_error("Large MLA grid bits differ");
      const int pools=8192;
      std::vector<float> keys(pools),q(nt,1.f),weights(nt,1.f);
      std::vector<int> counts(nt);
      for(int row=0;row<pools;++row)keys[row]=1.f+float(row%7)/8;
      for(int token=0;token<nt;++token)counts[token]=(token%3)*pools/2;
      auto scores=gpu.index_scores(keys,q,weights,counts,1,1);
      for(int token=0;token<nt;++token)for(int row=0;row<pools;++row)
        if(scores[size_t(token)*pools+row]!=(row<counts[token]?keys[row]:0.f))
          throw std::runtime_error("Large causal index grid differs");
      std::cerr << "PARITY name=indexer-large-grid passed=1\n";
      gpu.release_working_buffers();
    }
    {
      // Match the live state/cache footprint near a 32K context. A scratch
      // resize must release its old allocation before charging the new one.
      strata::vulkan::Compute limits(projections.device_name());
      constexpr int nt=1024,heads=64,latent=512,capacity=32768;
      std::vector<float> state(64*128),beta(64),state_output(state.size());
      std::vector<float> raw(3*state.size()),conv(12*state.size()),negative_a(64,-1.f),norm(128,1.f);
      for(int layer=200;layer<236;++layer)
        limits.kda_fused(layer,raw,state,beta,state,conv,state,negative_a,norm,
                         state_output,64,4,-8.f,1e-6f);
      std::vector<float> row(latent,1.f),query(heads*latent),single(query.size());
      for(int layer=100;layer<109;++layer)
        limits.mla(layer,row,0,capacity,query,std::vector<int>{0},single,heads,.125f);
      std::vector<float> pooled(8192*128),iq(nt*32*128),iw(nt*32);
      std::vector<int> counts(nt);
      limits.index_scores(pooled,iq,iw,counts,32,128);
      std::vector<float> rows(nt*latent,1.f),queries(size_t(nt)*heads*latent);
      std::vector<std::vector<int>> ids(nt,std::vector<int>{0});
      auto output=limits.mla_batch(100,rows,1,capacity,queries,ids,heads,.125f);
      if(!std::all_of(output.begin(),output.end(),[](float v){return v==1.f;}))
        throw std::runtime_error("32K working-buffer result differs");
      std::vector<uint8_t> gate(8*1024*1024),down(4*1024*1024);
      limits.upload_expert(0,gate,gate,down);
      std::cerr << "PARITY name=mla-32k-working-budget peak_MiB=" << limits.peak_bytes()/1048576.0 << '\n';
    }
    if (argc == 2) {
      strata::core::ModelArtifact artifact(argv[1]);
      for (int layer : {3, 13}) {
        const auto prefix = "blk."+std::to_string(layer)+".";
        const auto &G=artifact.at(prefix+"ffn_gate_exps.weight"),
                   &U=artifact.at(prefix+"ffn_up_exps.weight"),
                   &D=artifact.at(prefix+"ffn_down_exps.weight");
        const auto &m=artifact.descriptor();
        strata::kernels::cpu::NativeFmt format;std::string error;
        if(!strata::kernels::cpu::native_fmt(G.tensor->type,D.tensor->type,m.hidden,
                          m.layers[layer].intermediate,format,error)) throw std::runtime_error(error);
        format.canon=true;format.swiglu_limit=m.layers[layer].swiglu_limit;
        for(int id : {0,33,143}) {
          const int key=layer*144+id;
          const auto gb=G.bytes/m.experts, db=D.bytes/m.experts;
          gpu.upload_expert(key,{G.data()+id*gb,gb},{U.data()+id*gb,gb},{D.data()+id*db,db});
          for(int nt:{1,2,8}) {
            std::vector<uint8_t> act(nt*format.act_bytes),hq(nt*format.h_bytes);
            std::vector<const void*> acts(nt),hqs(nt);
            std::vector<float> hidden(nt*format.n_ff),expected_hidden(hidden.size()),expected(nt*m.hidden);
            std::vector<float*> hs(nt),outs(nt);
            for(int t=0;t<nt;++t) {
              auto input=values(m.hidden);
              cpu::native_quant_act(format,input.data(),act.data()+t*format.act_bytes);
              acts[t]=act.data()+t*format.act_bytes;hs[t]=expected_hidden.data()+t*format.n_ff;
              hqs[t]=hq.data()+t*format.h_bytes;outs[t]=expected.data()+t*m.hidden;
            }
            auto gu=gpu.expert_dot(key,act,m.hidden,2*format.n_ff,G.tensor->type,0,nt);
            for(int t=0;t<nt;++t) {
              for(int i=0;i<format.n_ff;++i)hidden[t*format.n_ff+i]=cpu::canon_swiglu(
                gu[t*2*format.n_ff+i],gu[t*2*format.n_ff+format.n_ff+i],format.swiglu_limit);
              cpu::native_quant_h(format,hidden.data()+t*format.n_ff,hq.data()+t*format.h_bytes);
            }
            cpu::native_gu_rows(format,G.data()+id*gb,acts.data(),nt,hs.data(),0,format.n_ff,U.data()+id*gb);
            close(hidden,expected_hidden,0,"expert-canonical-gu");
            if(std::memcmp(hidden.data(),expected_hidden.data(),hidden.size()*4))throw std::runtime_error("Canonical Vulkan GU bits differ");
            auto actual=gpu.expert_dot(key,hq,format.n_ff,m.hidden,D.tensor->type,gb*2,nt);
            cpu::native_down_rows(format,G.data()+id*gb,hqs.data(),nt,outs.data(),0,m.hidden,D.data()+id*db);
            if(actual!=expected) {
              std::cerr << "EXPERT_CASE layer=" << layer << " id=" << id << " tokens=" << nt << '\n';
              for(int t=0;t<nt;++t) {
                auto single=gpu.expert_dot(key,{hq.data()+t*format.h_bytes,format.h_bytes},format.n_ff,m.hidden,D.tensor->type,gb*2);
                std::vector<float> oracle(m.hidden);const void *qa[]={hq.data()+t*format.h_bytes};float *qo[]={oracle.data()};
                cpu::native_down_rows(format,G.data()+id*gb,qa,1,qo,0,m.hidden,D.data()+id*db);
                std::vector<float> chunk(actual.begin()+t*m.hidden,actual.begin()+(t+1)*m.hidden);
                close(single,oracle,2e-5,"debug-single-down");
                close(chunk,single,2e-5,"debug-batch-versus-single");
              }
            }
            close(actual,expected,0,"expert-canonical-down");
            if(std::memcmp(actual.data(),expected.data(),actual.size()*4))throw std::runtime_error("Canonical Vulkan down bits differ");
            auto repeated=gpu.expert_dot(key,hq,format.n_ff,m.hidden,D.tensor->type,gb*2,nt);
            if(repeated!=actual)throw std::runtime_error("Canonical Vulkan expert repeat differs");
          }
        }
      }
      gpu.clear_experts();

      {
        std::set<std::tuple<int,uint64_t,uint64_t>> geometries;
        for(const auto &[name,t]:artifact.tensors()) {
          if(!name.starts_with("blk.") || t.tensor->shape.size()!=2 ||
             !geometries.emplace(t.tensor->type,t.tensor->shape[0],t.tensor->shape[1]).second)continue;
          // Include narrow KDA products and small router/mHC outputs. One
          // geometry per backend lifetime keeps the fixture's VRAM bounded.
          strata::vulkan::Projections mmv(0,512ULL*1024*1024);
          const int in=t.tensor->shape[0],out=t.tensor->shape[1];
          const auto type=ggml_type(t.tensor->type);const size_t bytes=ggml_row_size(type,in)*out;
          const std::string key="mmv-type-"+std::to_string(t.tensor->type);
          for(int nt:{2,3,4,5,6,7,8}) {
            auto x=values(size_t(in)*nt);
            auto batch=mmv.synthetic(key,type,{t.data(),bytes},in,out,x,nt);
            std::vector<float> sequential;
            for(int token=0;token<nt;++token) {
              auto y=mmv.synthetic(key,type,{t.data(),bytes},in,out,{x.data()+size_t(token)*in,size_t(in)},1);
              sequential.insert(sequential.end(),y.begin(),y.end());
            }
            std::cerr << "MMV_CASE type=" << t.tensor->type << " in=" << in << " out=" << out
                      << " tokens=" << nt << " tensor=" << name << '\n';
            close(batch,sequential,0,"projection-mmv-canonical");
            if(std::memcmp(batch.data(),sequential.data(),batch.size()*4))
              throw std::runtime_error("Vulkan multi-vector projection bits differ");
          }
        }
      }

      std::vector<std::pair<std::string, const strata::core::ArtifactTensor *>>
          items;
      for (const auto *part :
           {"attn_q", "attn_k", "attn_v", "ssm_f_a", "ssm_g_a", "ssm_beta"}) {
        std::string name = std::string("blk.1.") + part + ".weight";
        items.emplace_back(name, &artifact.at(name));
      }
      auto input = values(artifact.descriptor().hidden);
      auto grouped = projections.joined(items, input);
      auto repeated = projections.joined(items, input);
      for (size_t i = 0; i < items.size(); ++i) {
        auto individual =
            projections.mat(items[i].first, *items[i].second, input);
        close(grouped[i], individual, 0, "projection-group");
        if (grouped[i] != repeated[i])
          throw std::runtime_error("Grouped projection repeat differs");
      }
      auto &fa = artifact.at("blk.1.ssm_f_b.weight");
      auto &ga = artifact.at("blk.1.ssm_g_b.weight");
      auto f = values(fa.tensor->shape[0]), g = values(ga.tensor->shape[0]);
      auto pair = projections.multiple(
          {{"blk.1.ssm_f_b.weight", &fa}, {"blk.1.ssm_g_b.weight", &ga}},
          {f, g});
      close(pair[0], projections.mat("blk.1.ssm_f_b.weight", fa, f), 0,
            "projection-distinct-input-f");
      close(pair[1], projections.mat("blk.1.ssm_g_b.weight", ga, g), 0,
            "projection-distinct-input-g");
    } else if (argc != 1) {
      throw std::invalid_argument("Usage: glm_vulkan_parity [model.gguf]");
    }
    gpu.reset();
    for (int layer : {21, 24}) {
      auto state = gpu.kda_state(layer), history = gpu.kda_history(layer);
      if (std::any_of(state.begin(), state.end(),
                      [](float value) { return value != 0; }) ||
          std::any_of(history.begin(), history.end(),
                      [](float value) { return value != 0; }))
        throw std::runtime_error("Fused KDA reset left recurrent history");
    }
    std::cout << "{\"backend\":\"vulkan\",\"passed\":true}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
