#include "strata/artifact/safetensors.hpp"
#include <iostream>
#include <chrono>
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
 std::cout<<"safetensors PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
