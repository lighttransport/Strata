"""Frozen C++ task specifications and independent compiled correctness oracles.

Training tasks are distinct from the twenty held-out specifications. Holdouts
must not be used to choose tuning settings. A passing compile alone is not pass.
"""
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'Implement the following C++17 function. Include required headers. Return exactly one complete cpp code block, no main or explanation. Keep reasoning brief. '
TRAINING = {
    'parse_u64': PREFIX+'bool parse_u64(std::string_view s, std::uint64_t& out) noexcept; accept nonempty ASCII decimal digits, allow leading zeroes, reject signs, whitespace, NUL, non-ASCII and overflow. Leave out unchanged on failure. No allocations.',
    'checked_slice': PREFIX+'bool checked_slice(const std::vector<int>& in, std::size_t start, std::size_t count, std::vector<int>& out); copy the requested range to out, including when in and out alias. Reject any invalid range without changing out. Empty ranges at the end are valid. Avoid start+count overflow.',
    'reduce_routed': PREFIX+'std::vector<float> reduce_routed(const std::vector<std::vector<float>>& experts, const std::vector<float>& weights); sum each coordinate in expert order using float arithmetic. Empty input returns empty. Throw std::invalid_argument on a weight-count or row-width mismatch. Do not modify inputs.',
}
TRAINING['parse_u64'] += '\nCurrent parser to repair: bool parse_u64(std::string_view s, std::uint64_t& out) noexcept { out=0; for(char c:s) { if(c<\"0\"[0] || c>\"9\"[0]) return false; out=out*10+(c-\"0\"[0]); } return true; }'
TRAINING['checked_slice'] += '\nCurrent bounds/copy code to repair: if(start+count>in.size()) return false; out.clear(); for(size_t i=start;i<start+count;++i) out.push_back(in[i]); return true;'
TRAINING['reduce_routed'] += '\nCurrent routed reduction to repair: std::vector<float> out(experts[0].size()); for(size_t i=0;i<weights.size();++i) for(size_t j=0;j<out.size();++j) out[j]+=experts[i][j]*weights[i]; return out;'
INCLUDES = '#include <algorithm>\n#include <cassert>\n#include <cstdint>\n#include <climits>\n#include <limits>\n#include <string>\n#include <string_view>\n#include <vector>\n#include <stdexcept>\n#include <random>\n#include <set>\n#include <utility>\n'
TRAIN_DRIVERS = {
    'parse_u64': '''int main(){uint64_t x=77; for(auto s:{"","-1","+1"," 1","1 ","18446744073709551616"}){x=77;assert(!parse_u64(s,x)&&x==77);} assert(!parse_u64(std::string_view("1\\0",2),x)&&x==77);assert(parse_u64("18446744073709551615",x)&&x==UINT64_MAX);std::mt19937_64 r(901);for(int i=0;i<4000;++i){auto n=r();auto s=std::to_string(n);assert(parse_u64(s,x)&&x==n);assert(parse_u64("000"+s,x)&&x==n);}}''',
    'checked_slice': '''int main(){std::vector<int> a{1,2,3,4},out{77};assert(!checked_slice(a,SIZE_MAX,1,out)&&out==std::vector<int>{77});assert(!checked_slice(a,1,SIZE_MAX,out)&&out==std::vector<int>{77});assert(checked_slice(a,4,0,out)&&out.empty());for(size_t i=0;i<=a.size();++i)for(size_t n=0;n<=a.size()-i;++n){auto b=a;std::vector<int> ref(a.begin()+i,a.begin()+i+n);assert(checked_slice(b,i,n,b)&&b==ref);}out={77};assert(!checked_slice(a,4,1,out)&&out==std::vector<int>{77});}''',
    'reduce_routed': '''int main(){assert(reduce_routed({},{}).empty());bool threw=false;try{reduce_routed({{1}},{});}catch(const std::invalid_argument&){threw=true;}assert(threw);threw=false;try{reduce_routed({{1},{1,2}},{1,2});}catch(const std::invalid_argument&){threw=true;}assert(threw);std::mt19937 r(17);for(int t=0;t<300;++t){size_t n=r()%8+1,m=r()%65;std::vector<std::vector<float>> x(n,std::vector<float>(m));std::vector<float>w(n),expected(m);for(size_t j=0;j<n;++j){w[j]=int(r()%9)-4;for(size_t i=0;i<m;++i){x[j][i]=int(r()%17)-8;expected[i]+=w[j]*x[j][i];}}auto original=x;assert(reduce_routed(x,w)==expected&&x==original);}}''',
}
HOLDOUTS = {}
DRIVERS = {}

def add(name, contract, driver):
    HOLDOUTS[name] = PREFIX+contract
    DRIVERS[name] = driver

for op in ['add','mul']:
    for unsigned in [False,True]:
        name=f'{op}_'+('u64' if unsigned else 'i64')
        typ='std::uint64_t' if unsigned else 'std::int64_t'
        wide='unsigned __int128' if unsigned else '__int128'
        expr='a+b' if op=='add' else 'a*b'
        minimum='0' if unsigned else 'std::numeric_limits<T>::min()'
        add(name,f'bool {name}({typ} a, {typ} b, {typ}& out) noexcept; compute checked {op}, rejecting overflow without modifying out. No undefined behavior.',
            f'''int main(){{using T={typ};auto check=[](T a,T b){{T out=17;{wide} aa=a,bb=b;auto v={expr.replace('a','aa').replace('b','bb')};bool good=v>={minimum}&&v<=std::numeric_limits<T>::max();bool ok={name}(a,b,out);assert(ok==good);assert(out==(good?T(v):T(17)));}};for(T a:{{T(0),T(1),T(-1),std::numeric_limits<T>::min(),std::numeric_limits<T>::max()}})for(T b:{{T(0),T(1),T(-1),T(2),std::numeric_limits<T>::min(),std::numeric_limits<T>::max()}})check(a,b);std::mt19937_64 r(1337);for(int i=0;i<2000;++i)check(T(r()),T(r()));}}''')
add('parse_i64','bool parse_i64(std::string_view s, std::int64_t& out) noexcept; accept decimal digits with optional single leading minus; reject plus, whitespace, embedded NUL, empty input and overflow; allow leading zeroes, preserve out on failure.',
    '''int main(){int64_t x=17;for(auto s:{"","-","+1"," 1","1 ","9223372036854775808","-9223372036854775809"}){x=17;assert(!parse_i64(s,x)&&x==17);}assert(parse_i64("-9223372036854775808",x)&&x==INT64_MIN);assert(!parse_i64(std::string_view("1\\0",2),x)&&x==INT64_MIN);std::mt19937_64 r(1337);for(int i=0;i<2000;++i){int64_t v=r();assert(parse_i64(std::to_string(v),x)&&x==v);}}''')
add('parse_hex','bool parse_hex(std::string_view s, std::uint64_t& out) noexcept; accept nonempty ASCII hexadecimal digits in either case, no prefix or signs; allow leading zeroes, reject overflow and all other bytes, preserve out on failure.',
    '''int main(){uint64_t x=17;for(auto s:{"","0x1","+1","-1","g","1 ","10000000000000000"}){x=17;assert(!parse_hex(s,x)&&x==17);}assert(parse_hex("fFfFfFfFfFfFfFfF",x)&&x==UINT64_MAX);assert(parse_hex("0000",x)&&x==0);std::mt19937_64 r(1337);const char*d="0123456789abcdef";for(int i=0;i<2000;++i){uint64_t v=r(),a=v;std::string s;do{s+=d[a&15];a>>=4;}while(a);std::reverse(s.begin(),s.end());assert(parse_hex(s,x)&&x==v);}}''')
for upper in [False,True]:
    name='upper_index' if upper else 'lower_index'
    operation='upper_bound' if upper else 'lower_bound'
    add(name,f'std::size_t {name}(const std::vector<int>& sorted, int key) noexcept; return the first index whose value is '+('greater than' if upper else 'at least')+' key, or size; input is sorted ascending.',
        f'''int main(){{std::mt19937 r(1337);for(int t=0;t<500;++t){{std::vector<int>a(r()%200);for(auto&x:a)x=int(r()%31)-15;std::sort(a.begin(),a.end());for(int k=-17;k<=17;++k)assert({name}(a,k)==size_t(std::{operation}(a.begin(),a.end(),k)-a.begin()));}}}}''')
for sorted_ in [False,True]:
    name='sorted_unique' if sorted_ else 'stable_unique'
    contract='sort values ascending and remove duplicates' if sorted_ else 'remove duplicate values preserving first occurrence order'
    oracle='std::set<int>s(a.begin(),a.end());std::vector<int> ref(s.begin(),s.end());' if sorted_ else 'std::set<int>s;std::vector<int>ref;for(auto x:a)if(s.insert(x).second)ref.push_back(x);'
    add(name,f'std::vector<int> {name}(const std::vector<int>& a); {contract}; leave input unchanged.',f'''int main(){{std::mt19937 r(1337);for(int t=0;t<500;++t){{std::vector<int>a(r()%200);for(auto&x:a)x=int(r()%31)-15;auto copy=a;{oracle}assert({name}(a)==ref&&a==copy);}}}}''')
for right in [False,True]:
    name='rotate_right' if right else 'rotate_left'
    pivot='a.size()-n' if right else 'n'
    add(name,f'void {name}(std::vector<int>& a, std::size_t k); rotate '+('right' if right else 'left')+' by k modulo size, preserve order; empty is a no-op; handle SIZE_MAX.',
        f'''int main(){{for(size_t m=0;m<100;++m)for(size_t k:{{size_t(0),size_t(1),size_t(77),SIZE_MAX}}){{std::vector<int>a(m);for(size_t i=0;i<m;++i)a[i]=int(i);auto ref=a;if(m){{size_t n=k%m;std::rotate(ref.begin(),ref.begin()+({pivot}),ref.end());}}{name}(a,k);assert(a==ref);}}}}''')
for remove in [False,True]:
    name='remove_all' if remove else 'replace_all'
    signature=f'std::string {name}(std::string_view input, std::string_view needle'+(');' if remove else ', std::string_view replacement);')
    call=f'{name}(input,needle)' if remove else f'{name}(input,needle,replacement)'
    replacement='std::string replacement;' if remove else 'std::string replacement=t%2?"aba":"xyz";'
    add(name,signature+' process non-overlapping matches left to right in the original input, without reprocessing inserted text. Empty needle returns input unchanged. Support embedded NUL.',
        f'''int main(){{std::mt19937 r(1337);for(int t=0;t<500;++t){{std::string input;for(unsigned i=0,n=r()%100;i<n;++i)input+=char(r()%4);std::string needle;for(unsigned i=0,n=r()%6;i<n;++i)needle+=char(r()%4);{replacement}std::string ref;if(needle.empty())ref=input;else{{size_t p=0;while(p<input.size()){{if(input.compare(p,needle.size(),needle)==0){{ref+=replacement;p+=needle.size();}}else ref+=input[p++];}}}}assert({call}==ref);}}}}''')
for touching in [False,True]:
    name='merge_touching' if touching else 'merge_overlap'
    cmp='<=' if touching else '<'
    add(name,f'std::vector<std::pair<int,int>> {name}(std::vector<std::pair<int,int>> intervals); valid start<end intervals in arbitrary order. Return sorted merged intervals; merge '+('overlapping or touching' if touching else 'strictly overlapping, but not merely touching')+' intervals. Empty input returns empty.',
        f'''int main(){{std::mt19937 r(1337);for(int t=0;t<500;++t){{std::vector<std::pair<int,int>>a;for(unsigned i=0,n=r()%100;i<n;++i){{int start=int(r()%100)-50;a.push_back({{start,start+int(r()%10)+1}});}}auto sorted=a;std::sort(sorted.begin(),sorted.end());std::vector<std::pair<int,int>>ref;for(auto x:sorted){{if(!ref.empty()&&x.first{cmp}ref.back().second)ref.back().second=std::max(ref.back().second,x.second);else ref.push_back(x);}}assert({name}(a)==ref);}}}}''')
for upper in [False,True]:
    name='hex_upper' if upper else 'hex_lower';digits='0123456789ABCDEF' if upper else '0123456789abcdef'
    add(name,f'std::string {name}(std::string_view bytes); encode every byte as exactly two '+('uppercase' if upper else 'lowercase')+' hexadecimal digits, including NUL and bytes>=128.',
        f'''int main(){{std::string a;for(int i=0;i<256;++i)a+=char(i);const char*d="{digits}";for(size_t n=0;n<=256;++n){{auto s=a.substr(0,n);std::string ref;for(unsigned char c:s){{ref+=d[c>>4];ref+=d[c&15];}}assert({name}(s)==ref);}}}}''')
add('fnv1a32','std::uint32_t fnv1a32(std::string_view bytes) noexcept; implement 32-bit FNV-1a over unsigned bytes: initial value 2166136261, XOR each byte then multiply by 16777619 modulo 2^32.',
    "int main(){assert(fnv1a32(\"\")==2166136261u);assert(fnv1a32(\"hello\")==0x4f9f2cabu);std::mt19937 r(1337);for(int t=0;t<500;++t){std::string s;for(unsigned i=0,n=r()%2000;i<n;++i)s+=char(r());uint64_t h=2166136261u;for(unsigned char c:s)h=((h^c)*16777619u)&0xffffffffu;assert(fnv1a32(s)==uint32_t(h));}}")
add('adler32','std::uint32_t adler32(std::string_view bytes) noexcept; initial A=1,B=0; for every unsigned byte set A=(A+byte)%65521 and B=(B+A)%65521; return (B<<16)|A. Support arbitrary bytes and empty input.',
    "int main(){assert(adler32(\"\")==1);assert(adler32(\"Wikipedia\")==0x11e60398u);std::mt19937 r(1337);for(int t=0;t<500;++t){std::string s;for(unsigned i=0,n=r()%5000;i<n;++i)s+=char(r());uint64_t a=1,b=0;for(unsigned char c:s){a+=c;a%=65521;b+=a;b%=65521;}assert(adler32(s)==uint32_t((b<<16)|a));}}")
assert len(HOLDOUTS)==20


def code_block(text):
    blocks = re.findall(r'```(?:cpp|c\+\+)\s*\n(.*?)```', text, re.S)
    if len(blocks)!=1:
        raise ValueError('expected exactly one complete C++ code block')
    return blocks[0]


def check_output(text, fixture):
    if fixture in ['prime','json_escape','csv']:
        from glm_q2_coding_check import check
        with tempfile.TemporaryDirectory(prefix='glm-quality-') as directory:
            root=Path(directory);(root/(fixture+'.output.md')).write_text(text)
            return check(root,fixture)
    code=code_block(text)
    with tempfile.TemporaryDirectory(prefix='glm-quality-') as directory:
        root=Path(directory);source=root/'test.cpp';binary=root/'test'
        if fixture=='long_cpp':
            if len(re.findall(r'\bTEST(?:_F)?\s*\(',code))<12:
                raise ValueError('long C++ answer needs at least 12 tests')
            if len(re.findall(r'\b(?:EXPECT|ASSERT)_\w+\s*\(',code))<12 or not all(s in code for s in ['NumaTensor','reduce_routed','View']):
                raise ValueError('long C++ answer lacks required API/assertion coverage')
            source.write_text(code)
            command=['g++','-O2','-std=c++17','-mavx2','-pthread','-I'+str(ROOT/'docs/fixtures/glm_q2_long_cpp'),
                     '-I'+str(ROOT/'include'),str(source),'-o',str(binary),'-lgtest_main','-lgtest']
        else:
            driver=TRAIN_DRIVERS.get(fixture,DRIVERS.get(fixture))
            if driver is None: raise ValueError('unknown coding fixture: '+fixture)
            source.write_text(INCLUDES+code+'\n'+driver+'\n')
            command=['g++','-O2','-std=c++17',str(source),'-o',str(binary)]
        compiled=subprocess.run(command,capture_output=True,text=True,timeout=60)
        if compiled.returncode:
            raise ValueError('C++ compilation failed: '+compiled.stderr[-3000:])
        checked=subprocess.run([str(binary)],capture_output=True,text=True,timeout=20)
        if checked.returncode:
            raise ValueError('independent C++ oracle failed: '+checked.stderr[-1000:])
        return 12 if fixture=='long_cpp' else 1
