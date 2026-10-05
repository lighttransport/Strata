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
 std::cout<<"safetensors PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
