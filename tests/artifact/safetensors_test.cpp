#include "strata/artifact/safetensors.hpp"
#include <iostream>
#include <chrono>
#ifdef __linux__
#include <linux/memfd.h>
#include <sys/syscall.h>
#endif
struct Fixture {
 std::filesystem::path root;
 Fixture(){root=std::filesystem::temp_directory_path()/("strata-sft-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));std::filesystem::create_directory(root);}
 ~Fixture(){std::error_code e;std::filesystem::remove_all(root,e);}
 void write(const std::string& header,const std::string& index,int payload=4){
  std::ofstream f(root/"a.safetensors",std::ios::binary);uint64_t n=header.size();for(int i=0;i<8;++i)f.put(char(n>>(8*i)));f<<header;for(int i=0;i<payload;++i)f.put(char(i));f.close();std::ofstream(root/"model.safetensors.index.json")<<index;
 }
};
int main(){try{
 const std::string good=R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",index=R"({"metadata":{"total_size":4},"weight_map":{"x":"a.safetensors"}})";
 Fixture f;f.write(good,index);strata::artifact::Safetensors model(f.root);uint8_t value[4];model.read("x",0,value,4);if(value[3]!=3)throw std::runtime_error("read mismatch");
 for(const auto& bad:std::vector<std::string>{R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})",R"({"x":{"dtype":"F32","shape":[18446744073709551615],"data_offsets":[0,4]}})",R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}})",R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,3]}})",R"({"x":{"dtype":"F64","shape":[1],"data_offsets":[0,4]}})"}){
  f.write(bad,index);bool rejected=false;try{strata::artifact::Safetensors invalid(f.root);}catch(...){rejected=true;}if(!rejected)throw std::runtime_error("invalid tensor accepted");
 }
 for(const auto& bad:std::vector<std::string>{R"({"weight_map":{"x":"../a.safetensors"}})",R"({"weight_map":{"y":"a.safetensors"}})",R"({"weight_map":{"x":"missing.safetensors"}})"}){
  f.write(good,bad);bool rejected=false;try{strata::artifact::Safetensors invalid(f.root);}catch(...){rejected=true;}if(!rejected)throw std::runtime_error("invalid index accepted");
 }
 f.write(good,index,3);bool rejected=false;try{strata::artifact::Safetensors invalid(f.root);}catch(...){rejected=true;}if(!rejected)throw std::runtime_error("truncation accepted");
 {
  Fixture nested;nested.write(good,index);
  std::filesystem::create_directory(nested.root/"tensors");
  std::filesystem::rename(nested.root/"a.safetensors",nested.root/"tensors"/"native.safetensors");
  std::ofstream(nested.root/"model.safetensors.index.json")<<R"({"metadata":{"total_size":4},"weight_map":{"x":"tensors/native.safetensors"}})";
  bool refused=false;try{strata::artifact::Safetensors strict(nested.root);}catch(...){refused=true;}
  if(!refused)throw std::runtime_error("strict index accepted nested shard");
  strata::artifact::Safetensors selected(nested.root,true);
  selected.read("x",0,value,4);if(value[3]!=3)throw std::runtime_error("nested shard read mismatch");
#ifndef _WIN32
  const auto* p=selected.data("x");
  if(p!=selected.data(selected.at("x")) || p[3]!=3)throw std::runtime_error("shared mapping mismatch");
  Fixture escaped;
  std::filesystem::create_symlink(nested.root/"tensors"/"native.safetensors",escaped.root/"a.safetensors");
  std::ofstream(escaped.root/"model.safetensors.index.json")<<index;
  refused=false;try{strata::artifact::Safetensors invalid(escaped.root,true);}catch(...){refused=true;}
  if(!refused)throw std::runtime_error("nested index accepted escaping symlink");
#endif
 }
 {
  Fixture subset;subset.write(R"({"x":{"dtype":"U16","shape":[1],"data_offsets":[0,2]},"unused":{"dtype":"U16","shape":[1],"data_offsets":[2,4]}})",R"({"metadata":{"total_size":2},"weight_map":{"x":"a.safetensors"}})");
  strata::artifact::Safetensors selected(subset.root,true);
  if(selected.tensors().size()!=1 || selected.at("x").bytes!=2)throw std::runtime_error("indexed subset mismatch");
 }
#ifndef _WIN32
 {
  Fixture ram;ram.write(good,index);
  strata::artifact::Safetensors staged(ram.root,true,true),direct(ram.root);
  staged.read("x",0,value,4);
  const auto* p=staged.data("x");
  auto maps=staged.ram_mappings();
  if(maps.size()!=1||maps[0].second!=staged.file_bytes()||
     static_cast<const uint8_t*>(maps[0].first)+8+good.size()!=p)
      throw std::runtime_error("RAM DMA mapping differs from tensor backing");
  bool refused=false;try{direct.ram_mappings();}catch(...){refused=true;}
  if(!refused)throw std::runtime_error("file-backed weights accepted as RAM DMA mappings");
  if(value[0]!=0 || value[3]!=3 || p[3]!=3)throw std::runtime_error("shared RAM initial bytes mismatch");
  {
   std::fstream file(ram.root/"a.safetensors",std::ios::in|std::ios::out|std::ios::binary);
   file.seekp(8+good.size());file.put(char(77));
  }
  direct.read("x",0,value,4);
  if(value[0]!=77)throw std::runtime_error("source fixture update failed");
  staged.read("x",0,value,4);
  if(value[0]!=0 || p!=staged.data("x") || p[0]!=0)throw std::runtime_error("RAM snapshot reads source after staging");
  staged.warm(false);staged.drop(staged.at("x"));
  if(p[3]!=3)throw std::runtime_error("RAM snapshot lifetime mismatch");
 }
 {
  constexpr int boundary=4*1024*1024,bytes=boundary+19;
  const auto header=std::string("{\"x\":{\"dtype\":\"U8\",\"shape\":[")+std::to_string(bytes)+"],\"data_offsets\":[0,"+std::to_string(bytes)+"]}}";
  Fixture ram;ram.write(header,R"({"weight_map":{"x":"a.safetensors"}})",bytes);
  strata::artifact::Safetensors staged(ram.root,true,true);
  uint8_t slice[32];staged.read("x",boundary-13,slice,sizeof(slice));
  for(int i=0;i<32;++i)if(slice[i]!=uint8_t(boundary-13+i))throw std::runtime_error("RAM staging chunk boundary mismatch");
 }
#endif
#ifdef __linux__
 {
  Fixture ram;ram.write(good,index);
  std::ifstream input(ram.root/"a.safetensors",std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
  const int fd=int(::syscall(SYS_memfd_create,"strata-test",MFD_CLOEXEC|MFD_ALLOW_SEALING));
  if(fd<0 || ::write(fd,bytes.data(),bytes.size())!=ssize_t(bytes.size()))throw std::runtime_error("cannot create RAM fixture");
  std::filesystem::remove(ram.root/"a.safetensors");
  std::filesystem::create_symlink("/proc/self/fd/"+std::to_string(fd),ram.root/"a.safetensors");
  bool refused=false;try{strata::artifact::Safetensors invalid(ram.root,true,false,true);}catch(...){refused=true;}
  if(!refused)throw std::runtime_error("unsealed RAM file accepted");
  if(::fcntl(fd,F_ADD_SEALS,F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL))throw std::runtime_error("cannot seal RAM fixture");
  {
   strata::artifact::Safetensors cached(ram.root,true,false,true);
   cached.read("x",0,value,4);
   if(value[3]!=3 || cached.data("x")[3]!=3)throw std::runtime_error("sealed RAM reads differ");
   cached.warm(true);
   auto maps=cached.ram_mappings();
   if(maps.size()!=1||static_cast<const uint8_t*>(maps[0].first)+8+good.size()!=cached.data("x"))
       throw std::runtime_error("sealed RAM DMA view lost source identity");
   ::close(fd);
   if(cached.data("x")[3]!=3)throw std::runtime_error("RAM reader lost its own descriptor");
  }
  Fixture regular;regular.write(good,index);
  refused=false;try{strata::artifact::Safetensors invalid(regular.root,true,false,true);}catch(...){refused=true;}
  if(!refused)throw std::runtime_error("ordinary file accepted as sealed RAM");
 }
#endif
 std::cout<<"safetensors PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
