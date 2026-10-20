"""Fresh, disjoint coding fixtures for the second native-FP4 optimization campaign.

Freeze this file's hash before tuning. Holdouts are used once after selection;
none are the tasks used in the original dual-V100 qualification.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from glm_coding_quality import code_block, INCLUDES

PREFIX = ('Return exactly one complete C++17 code block implementing the function below. '
          'Include needed headers, but no main function. Follow the edge cases exactly.\n')
FIXTURES = {}


def add(name, split, contract, reference, checks):
    FIXTURES[name] = dict(split=split, prompt=PREFIX+contract,
                          reference=reference, driver='int main(){'+checks+'}')


add('sum_small', 'training', 'long long sum_small(const std::vector<int>& a); sum the elements. The sum fits long long.',
    'long long sum_small(const std::vector<int>& a){long long s=0;for(int x:a)s+=x;return s;}',
    'assert(sum_small({})==0);assert(sum_small({INT_MAX,INT_MAX,-1})==4294967293LL);')
add('count_byte', 'training', 'size_t count_byte(std::string_view s, unsigned char b); count byte b, including NUL and bytes above 127.',
    'size_t count_byte(std::string_view s,unsigned char b){size_t n=0;for(unsigned char c:s)n+=c==b;return n;}',
    'std::string s;for(int i=0;i<1024;++i)s+=char(i);for(int b=0;b<256;++b)assert(count_byte(s,b)==4);assert(count_byte({},0)==0);')
add('ascii_upper', 'training', 'std::string ascii_upper(std::string_view s); uppercase ASCII a-z only, preserving every other byte.',
    "std::string ascii_upper(std::string_view s){std::string r(s);for(char&c:r)if(c>='a'&&c<='z')c-=32;return r;}",
    "std::string s;for(int i=0;i<256;++i)s+=char(i);auto r=ascii_upper(s);assert(r.size()==256);for(int i=0;i<256;++i)assert((unsigned char)r[i]==(i>='a'&&i<='z'?i-32:i));")
add('suffix_match', 'training', 'bool suffix_match(std::string_view s, std::string_view suffix) noexcept; byte-exact suffix test, empty suffix matches.',
    'bool suffix_match(std::string_view s,std::string_view t)noexcept{return s.size()>=t.size()&&s.substr(s.size()-t.size())==t;}',
    'assert(suffix_match("abc","bc"));assert(suffix_match("", ""));assert(!suffix_match("a","ba"));assert(!suffix_match("abc","ab"));')
add('argmin_first', 'training', 'size_t argmin_first(const std::vector<int>& a) noexcept; first minimum index, or size for empty input.',
    'size_t argmin_first(const std::vector<int>&a)noexcept{return std::min_element(a.begin(),a.end())-a.begin();}',
    'assert(argmin_first({})==0);assert(argmin_first({3,-1,-1,2})==1);assert(argmin_first({INT_MAX})==0);')
add('clamp_int', 'training', 'int clamp_int(int x,int lo,int hi) noexcept; lo<=hi; clamp x into the inclusive interval without overflow.',
    'int clamp_int(int x,int lo,int hi)noexcept{return x<lo?lo:x>hi?hi:x;}',
    'assert(clamp_int(INT_MIN,-1,1)==-1);assert(clamp_int(INT_MAX,-1,1)==1);assert(clamp_int(3,3,3)==3);')
add('adjacent_changes', 'training', 'size_t adjacent_changes(const std::vector<int>& a) noexcept; count adjacent pairs with different values.',
    'size_t adjacent_changes(const std::vector<int>&a)noexcept{size_t n=0;for(size_t i=1;i<a.size();++i)n+=a[i]!=a[i-1];return n;}',
    'assert(adjacent_changes({})==0);assert(adjacent_changes({1})==0);assert(adjacent_changes({1,1,2,3,3,1})==3);')
add('zip_sum', 'training', 'std::vector<long long> zip_sum(const std::vector<int>& a,const std::vector<int>& b); element sums; throw std::invalid_argument if sizes differ.',
    'std::vector<long long> zip_sum(const std::vector<int>&a,const std::vector<int>&b){if(a.size()!=b.size())throw std::invalid_argument("size");std::vector<long long>r;for(size_t i=0;i<a.size();++i)r.push_back((long long)a[i]+b[i]);return r;}',
    'assert(zip_sum({},{}).empty());assert(zip_sum({INT_MAX,INT_MIN},{INT_MAX,INT_MIN})==std::vector<long long>({4294967294LL,-4294967296LL}));bool e=false;try{zip_sum({1},{});}catch(const std::invalid_argument&){e=true;}assert(e);')

add('exclusive_prefix', 'holdout', 'std::vector<uint64_t> exclusive_prefix(const std::vector<uint64_t>& a); output[i] is the sum before i modulo 2^64; output length equals input length.',
    'std::vector<uint64_t> exclusive_prefix(const std::vector<uint64_t>&a){std::vector<uint64_t>r;uint64_t s=0;for(auto x:a){r.push_back(s);s+=x;}return r;}',
    'assert(exclusive_prefix({}).empty());assert(exclusive_prefix({UINT64_MAX,2,5})==std::vector<uint64_t>({0,UINT64_MAX,1}));std::mt19937_64 rng(507);for(int n=0;n<100;++n){std::vector<uint64_t>a(n);for(auto&x:a)x=rng();auto r=exclusive_prefix(a);assert(r.size()==a.size());for(int i=0;i<n;++i){uint64_t s=0;for(int j=0;j<i;++j)s+=a[j];assert(r[i]==s);}}')
add('run_lengths', 'holdout', 'std::vector<std::pair<int,size_t>> run_lengths(const std::vector<int>& a); encode each maximal consecutive run as value,count, preserving run order.',
    'std::vector<std::pair<int,size_t>> run_lengths(const std::vector<int>&a){std::vector<std::pair<int,size_t>>r;for(int x:a){if(r.empty()||r.back().first!=x)r.emplace_back(x,1);else ++r.back().second;}return r;}',
    'assert(run_lengths({}).empty());std::mt19937 rng(508);for(int n=0;n<200;++n){std::vector<int>a(n);for(auto&x:a)x=rng()%4;auto r=run_lengths(a);std::vector<int>b;for(size_t i=0;i<r.size();++i){assert(r[i].second>0);if(i)assert(r[i].first!=r[i-1].first);b.insert(b.end(),r[i].second,r[i].first);}assert(a==b);}')
add('popcount64', 'holdout', 'unsigned popcount64(uint64_t x) noexcept; count set bits.',
    'unsigned popcount64(uint64_t x)noexcept{unsigned n=0;while(x){x&=x-1;++n;}return n;}',
    'assert(popcount64(0)==0);assert(popcount64(UINT64_MAX)==64);std::mt19937_64 r(509);for(int i=0;i<4000;++i){auto x=r();unsigned n=0;for(int b=0;b<64;++b)n+=(x>>b)&1;assert(popcount64(x)==n);}')
add('reverse_bits64', 'holdout', 'uint64_t reverse_bits64(uint64_t x) noexcept; reverse all 64 bits.',
    'uint64_t reverse_bits64(uint64_t x)noexcept{uint64_t y=0;for(int i=0;i<64;++i){y=(y<<1)|(x&1);x>>=1;}return y;}',
    'for(int b=0;b<64;++b)assert(reverse_bits64(uint64_t(1)<<b)==(uint64_t(1)<<(63-b)));std::mt19937_64 r(510);for(int i=0;i<2000;++i){auto x=r(),y=reverse_bits64(x);for(int b=0;b<64;++b)assert(((x>>b)&1)==((y>>(63-b))&1));}')
add('gcd_u64', 'holdout', 'uint64_t gcd_u64(uint64_t a,uint64_t b) noexcept; greatest common divisor; gcd(0,0)=0.',
    'uint64_t gcd_u64(uint64_t a,uint64_t b)noexcept{while(b){auto t=a%b;a=b;b=t;}return a;}',
    'assert(gcd_u64(0,0)==0);assert(gcd_u64(UINT64_MAX,0)==UINT64_MAX);for(uint64_t a=0;a<80;++a)for(uint64_t b=0;b<80;++b){uint64_t g=0;for(uint64_t d=1;d<=std::max(a,b);++d)if(a%d==0&&b%d==0)g=d;assert(gcd_u64(a,b)==g);}')
add('median_three', 'holdout', 'int median_three(int a,int b,int c) noexcept; return the median including duplicates, without overflow.',
    'int median_three(int a,int b,int c)noexcept{int v[]={a,b,c};std::sort(v,v+3);return v[1];}',
    'for(int a:{INT_MIN,-1,0,1,INT_MAX})for(int b:{INT_MIN,-1,0,1,INT_MAX})for(int c:{INT_MIN,-1,0,1,INT_MAX}){std::vector<int>v{a,b,c};std::sort(v.begin(),v.end());assert(median_three(a,b,c)==v[1]);}')
add('ceil_div_u64', 'holdout', 'bool ceil_div_u64(uint64_t a,uint64_t b,uint64_t& out) noexcept; mathematical ceiling a/b; b=0 fails and preserves out; avoid overflow.',
    'bool ceil_div_u64(uint64_t a,uint64_t b,uint64_t&out)noexcept{if(!b)return false;out=a/b+(a%b!=0);return true;}',
    'for(uint64_t a:{uint64_t(0),uint64_t(1),uint64_t(7),UINT64_MAX})for(uint64_t b:{uint64_t(0),uint64_t(1),uint64_t(2),UINT64_MAX}){uint64_t out=77;bool ok=ceil_div_u64(a,b,out);assert(ok==(b!=0));if(b)assert(out==uint64_t(((unsigned __int128)a+b-1)/b));else assert(out==77);}')
add('ceil_power_two', 'holdout', 'bool ceil_power_two(uint64_t x,uint64_t& out) noexcept; smallest power of two >=x, treating 0 as 1. Fail and preserve out if result exceeds uint64_t.',
    'bool ceil_power_two(uint64_t x,uint64_t&out)noexcept{if(x>(uint64_t(1)<<63))return false;uint64_t p=1;while(p<x)p<<=1;out=p;return true;}',
    'for(int b=0;b<64;++b){uint64_t p=uint64_t(1)<<b,out=0;assert(ceil_power_two(p,out)&&out==p);if(b>1)assert(ceil_power_two(p-1,out)&&out==p);}uint64_t out=77;assert(!ceil_power_two(UINT64_MAX,out)&&out==77);assert(ceil_power_two(0,out)&&out==1);')
add('byte_swap64', 'holdout', 'uint64_t byte_swap64(uint64_t x) noexcept; reverse the order of the eight bytes.',
    'uint64_t byte_swap64(uint64_t x)noexcept{uint64_t y=0;for(int i=0;i<8;++i){y=(y<<8)|(x&255);x>>=8;}return y;}',
    'assert(byte_swap64(0x0123456789abcdefULL)==0xefcdab8967452301ULL);std::mt19937_64 r(511);for(int i=0;i<2000;++i){auto x=r(),y=byte_swap64(x);for(int b=0;b<8;++b)assert(((x>>(8*b))&255)==((y>>(8*(7-b)))&255));}')
add('hamming_bytes', 'holdout', 'size_t hamming_bytes(std::string_view a,std::string_view b); count differing byte positions; throw std::invalid_argument if lengths differ.',
    'size_t hamming_bytes(std::string_view a,std::string_view b){if(a.size()!=b.size())throw std::invalid_argument("length");size_t n=0;for(size_t i=0;i<a.size();++i)n+=a[i]!=b[i];return n;}',
    'assert(hamming_bytes({}, {})==0);assert(hamming_bytes("abc","adc")==1);std::string a(256,0),b;for(int i=0;i<256;++i)b+=char(i);assert(hamming_bytes(a,b)==255);bool e=false;try{hamming_bytes("a","");}catch(const std::invalid_argument&){e=true;}assert(e);')
add('stable_negative_first', 'holdout', 'std::vector<int> stable_negative_first(const std::vector<int>& a); negative values first, then nonnegative values; preserve order within both groups and leave input unchanged.',
    'std::vector<int> stable_negative_first(const std::vector<int>&a){auto r=a;std::stable_partition(r.begin(),r.end(),[](int x){return x<0;});return r;}',
    'std::mt19937 r(512);for(int n=0;n<200;++n){std::vector<int>a(n),ref;for(auto&x:a)x=int(r()%11)-5;auto copy=a;for(int x:a)if(x<0)ref.push_back(x);for(int x:a)if(x>=0)ref.push_back(x);assert(stable_negative_first(a)==ref&&a==copy);}')
add('longest_equal_run', 'holdout', 'size_t longest_equal_run(const std::vector<int>& a) noexcept; length of the longest consecutive run of equal elements; zero for empty.',
    'size_t longest_equal_run(const std::vector<int>&a)noexcept{size_t best=0,n=0;for(size_t i=0;i<a.size();++i){n=i&&a[i]==a[i-1]?n+1:1;best=std::max(best,n);}return best;}',
    'std::mt19937 r(513);for(int n=0;n<200;++n){std::vector<int>a(n);for(auto&x:a)x=r()%4;size_t best=0;for(int i=0;i<n;++i){size_t k=i;while(k<a.size()&&a[k]==a[i])++k;best=std::max(best,k-i);}assert(longest_equal_run(a)==best);}')
add('interleave_vectors', 'holdout', 'std::vector<int> interleave_vectors(const std::vector<int>& a,const std::vector<int>& b); a[0],b[0],a[1],b[1],...; append the remainder of the longer input.',
    'std::vector<int> interleave_vectors(const std::vector<int>&a,const std::vector<int>&b){std::vector<int>r;for(size_t i=0;i<std::max(a.size(),b.size());++i){if(i<a.size())r.push_back(a[i]);if(i<b.size())r.push_back(b[i]);}return r;}',
    'assert(interleave_vectors({},{}).empty());assert(interleave_vectors({1,2,3},{4})==std::vector<int>({1,4,2,3}));assert(interleave_vectors({1},{2,3,4})==std::vector<int>({1,2,3,4}));assert(interleave_vectors({},{2,3})==std::vector<int>({2,3}));')
add('transpose_rect', 'holdout', 'std::vector<int> transpose_rect(const std::vector<int>& a,size_t rows,size_t cols); transpose row-major data. Reject dimension overflow or a size mismatch with std::invalid_argument. Zero dimensions are valid when input is empty.',
    'std::vector<int> transpose_rect(const std::vector<int>&a,size_t rows,size_t cols){if((cols&&rows>SIZE_MAX/cols)||rows*cols!=a.size())throw std::invalid_argument("shape");std::vector<int>b(a.size());if(a.empty())return b;for(size_t r=0;r<rows;++r)for(size_t c=0;c<cols;++c)b[c*rows+r]=a[r*cols+c];return b;}',
    'for(size_t r=0;r<12;++r)for(size_t c=0;c<12;++c){std::vector<int>a(r*c);for(size_t i=0;i<a.size();++i)a[i]=i;auto b=transpose_rect(a,r,c);assert(b.size()==a.size());for(size_t i=0;i<r;++i)for(size_t j=0;j<c;++j)assert(b[j*r+i]==a[i*c+j]);}bool e=false;try{transpose_rect({},SIZE_MAX,2);}catch(const std::invalid_argument&){e=true;}assert(e);assert(transpose_rect({},SIZE_MAX,0).empty());')
add('delta_decode', 'holdout', 'std::vector<uint64_t> delta_decode(const std::vector<uint64_t>& deltas,uint64_t initial); inclusive cumulative addition modulo 2^64 starting from initial.',
    'std::vector<uint64_t> delta_decode(const std::vector<uint64_t>&a,uint64_t v){std::vector<uint64_t>r;for(auto x:a){v+=x;r.push_back(v);}return r;}',
    'assert(delta_decode({},9).empty());assert(delta_decode({1,2,UINT64_MAX},UINT64_MAX)==std::vector<uint64_t>({0,2,1}));')
add('rotate_bits', 'holdout', 'uint64_t rotate_bits(uint64_t x,unsigned k) noexcept; rotate 64-bit x left by k modulo 64; no undefined shifts.',
    'uint64_t rotate_bits(uint64_t x,unsigned k)noexcept{k%=64;return k?(x<<k)|(x>>(64-k)):x;}',
    'for(unsigned k:{0u,1u,63u,64u,65u,UINT_MAX})for(int b=0;b<64;++b)assert(rotate_bits(uint64_t(1)<<b,k)==(uint64_t(1)<<((b+k%64)%64)));assert(rotate_bits(UINT64_MAX,31)==UINT64_MAX);')
add('is_subsequence', 'holdout', 'bool is_subsequence(std::string_view needle,std::string_view text) noexcept; whether needle can be obtained by deleting zero or more bytes from text, preserving order.',
    'bool is_subsequence(std::string_view a,std::string_view b)noexcept{size_t i=0;for(char c:b)if(i<a.size()&&a[i]==c)++i;return i==a.size();}',
    'assert(is_subsequence("",""));assert(is_subsequence("abc","axbxc"));assert(!is_subsequence("abc","acb"));assert(!is_subsequence("aa","a"));assert(is_subsequence(std::string("a\\0",2),std::string("aX\\0",3)));')
add('common_prefix_bytes', 'holdout', 'size_t common_prefix_bytes(std::string_view a,std::string_view b) noexcept; length of their identical initial byte sequence.',
    'size_t common_prefix_bytes(std::string_view a,std::string_view b)noexcept{size_t i=0;while(i<a.size()&&i<b.size()&&a[i]==b[i])++i;return i;}',
    'assert(common_prefix_bytes("abc","abd")==2);assert(common_prefix_bytes("abc","abc")==3);assert(common_prefix_bytes("","x")==0);assert(common_prefix_bytes(std::string("a\\0b",3),std::string("a\\0c",3))==2);')
add('flatten_rows', 'holdout', 'std::vector<int> flatten_rows(const std::vector<std::vector<int>>& rows); concatenate rows in order, including empty rows.',
    'std::vector<int> flatten_rows(const std::vector<std::vector<int>>&a){std::vector<int>r;for(const auto&v:a)r.insert(r.end(),v.begin(),v.end());return r;}',
    'assert(flatten_rows({}).empty());assert(flatten_rows({{}, {1,2}, {}, {3}, {}})==std::vector<int>({1,2,3}));')
add('count_inversions_small', 'holdout', 'uint64_t count_inversions_small(const std::vector<int>& a); count pairs i<j with a[i]>a[j]; size<=1000; input unchanged.',
    'uint64_t count_inversions_small(const std::vector<int>&a){uint64_t n=0;for(size_t i=0;i<a.size();++i)for(size_t j=i+1;j<a.size();++j)n+=a[i]>a[j];return n;}',
    'assert(count_inversions_small({})==0);assert(count_inversions_small({2,2,1})==2);std::mt19937 r(514);for(int n=0;n<100;++n){std::vector<int>a(n);for(auto&x:a)x=int(r()%21)-10;auto copy=a;uint64_t swaps=0;for(int i=0;i<n;++i)for(int j=1;j<n-i;++j)if(copy[j-1]>copy[j]){std::swap(copy[j-1],copy[j]);++swaps;}assert(count_inversions_small(a)==swaps);}')
assert sum(x['split']=='training' for x in FIXTURES.values()) == 8
assert sum(x['split']=='holdout' for x in FIXTURES.values()) == 20


def check_output(text, name):
    fixture=FIXTURES[name]
    with tempfile.TemporaryDirectory(prefix='glm-fp4-quality-') as directory:
        root=Path(directory);source=root/'case.cpp';binary=root/'case'
        source.write_text(INCLUDES+code_block(text)+'\n'+fixture['driver']+'\n')
        built=subprocess.run(['g++','-O2','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True,timeout=60)
        if built.returncode: raise ValueError('compile: '+built.stderr[-2000:])
        tested=subprocess.run([str(binary)],capture_output=True,text=True,timeout=20)
        if tested.returncode: raise ValueError('oracle: '+tested.stderr[-1000:])
    return 1


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--check-oracles',action='store_true');a=p.parse_args()
    if a.check_oracles:
        for name, fixture in FIXTURES.items():
            check_output('```cpp\n'+fixture['reference']+'\n```',name)
            print('PASS',name,flush=True)
    print(json.dumps({'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'splits':{k:sum(v['split']==k for v in FIXTURES.values()) for k in ('training','holdout')}},sort_keys=True))
