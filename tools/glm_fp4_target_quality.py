"""Next-round FP4 quality fixtures: reused public training, twenty fresh holdouts.

Freeze this file before scoring candidates. Consume holdouts only after freezing
one selected policy; the previous campaign's holdouts are not rerun here.
"""
import hashlib,json,subprocess,tempfile
from pathlib import Path
from glm_coding_quality import code_block,INCLUDES
from glm_fp4_optimization_quality import FIXTURES as PREVIOUS
PREFIX='Return exactly one complete C++17 code block implementing the function below. Include needed headers, but no main function. Follow the edge cases exactly.\n'
FIXTURES={k:dict(v) for k,v in PREVIOUS.items() if v['split']=='training'}
def add(name,contract,reference,checks):
    FIXTURES[name]=dict(split='holdout',prompt=PREFIX+contract,reference=reference,driver='int main(){'+checks+'}')
add('bytes_to_hex','std::string bytes_to_hex(std::string_view s); lowercase hexadecimal, exactly two digits per byte, including NUL and bytes above127.',
    'std::string bytes_to_hex(std::string_view s){const char*h="0123456789abcdef";std::string r;for(unsigned char b:s){r+=h[b>>4];r+=h[b&15];}return r;}',
    'std::string s;for(int i=0;i<256;++i)s+=char(i);auto r=bytes_to_hex(s);assert(r.size()==512);const std::string h="0123456789abcdef";for(int i=0;i<256;++i)assert(r.substr(i*2,2)==std::string()+h[i>>4]+h[i&15]);assert(bytes_to_hex({}).empty());')
add('hex_to_bytes','bool hex_to_bytes(std::string_view s,std::string& out); accept even-length ASCII hexadecimal, either case, empty accepted. Reject other bytes or odd length; preserve out on failure. Input may view out.',
    'bool hex_to_bytes(std::string_view s,std::string&out){if(s.size()%2)return false;auto digit=[](unsigned char c){return c>=48&&c<=57?c-48:c>=65&&c<=70?c-55:c>=97&&c<=102?c-87:-1;};std::string r;for(size_t i=0;i<s.size();i+=2){int a=digit(s[i]),b=digit(s[i+1]);if(a<0||b<0)return false;r+=char(a*16+b);}out=r;return true;}',
    'std::string out="keep";for(auto s:{"0","gg","0 ","-1"})assert(!hex_to_bytes(s,out)&&out=="keep");assert(hex_to_bytes("00fF7a",out)&&out==std::string("\\0\\xffz",3));out="4142";assert(hex_to_bytes(out,out)&&out=="AB");assert(hex_to_bytes("",out)&&out.empty());')
add('escape_csv_field','std::string escape_csv_field(std::string_view s); CSV-quote iff s contains comma, quote, CR or LF. Quoted fields start/end with quote and double each internal quote. Preserve all bytes.',
    'std::string escape_csv_field(std::string_view s){if(s.find_first_of(",\\\"\\r\\n")==s.npos)return std::string(s);std::string r="\\\"";for(char c:s){r+=c;if(c==34)r+=c;}return r+"\\\"";}',
    'assert(escape_csv_field("")=="");assert(escape_csv_field("abc")=="abc");assert(escape_csv_field("a,b")=="\\\"a,b\\\"");assert(escape_csv_field("a\\\"b")=="\\\"a\\\"\\\"b\\\"");assert(escape_csv_field("\\n")=="\\\"\\n\\\"");')
add('join_nonempty','std::string join_nonempty(const std::vector<std::string>& a,std::string_view separator); concatenate only nonempty elements, separators only between included elements. Preserve NUL bytes.',
    'std::string join_nonempty(const std::vector<std::string>&a,std::string_view s){std::string r;bool first=true;for(auto&x:a)if(!x.empty()){if(!first)r+=s;r+=x;first=false;}return r;}',
    'assert(join_nonempty({"","a","","b",""},"::")=="a::b");assert(join_nonempty({"",""},"x").empty());assert(join_nonempty({"a","b"},"")=="ab");assert(join_nonempty({std::string("a\\0",2),"b"},",")==std::string("a\\0,b",4));')
add('scan_decimal_u32','bool scan_decimal_u32(std::string_view s,uint32_t& out) noexcept; nonempty ASCII decimal, leading zeroes accepted. Reject signs, whitespace, NUL, non-ASCII and overflow. Preserve out on failure.',
    'bool scan_decimal_u32(std::string_view s,uint32_t&out)noexcept{if(s.empty())return false;uint32_t v=0;for(unsigned char c:s){if(c<48||c>57||v>(UINT32_MAX-(c-48))/10)return false;v=v*10+c-48;}out=v;return true;}',
    'uint32_t out=77;for(auto s:{"","-1","+1"," 1","4294967296","999999999999"})assert(!scan_decimal_u32(s,out)&&out==77);assert(!scan_decimal_u32(std::string("1\\0",2),out)&&out==77);std::mt19937 r(771);for(int i=0;i<2000;++i){uint32_t x=r();assert(scan_decimal_u32(std::to_string(x),out)&&out==x);}assert(scan_decimal_u32("000",out)&&out==0);')
add('saturating_add_i32','int32_t saturating_add_i32(int32_t a,int32_t b) noexcept; clamp mathematical sum to INT32_MIN..INT32_MAX without signed overflow.',
    'int32_t saturating_add_i32(int32_t a,int32_t b)noexcept{int64_t s=int64_t(a)+b;return s<INT32_MIN?INT32_MIN:s>INT32_MAX?INT32_MAX:int32_t(s);}',
    'for(int32_t a:{INT32_MIN,-1,0,1,INT32_MAX})for(int32_t b:{INT32_MIN,-1,0,1,INT32_MAX}){int64_t s=int64_t(a)+b;assert(saturating_add_i32(a,b)==std::max<int64_t>(INT32_MIN,std::min<int64_t>(INT32_MAX,s)));}')
add('ordered_unique','std::vector<int> ordered_unique(const std::vector<int>& a); keep each distinct value once in first-occurrence order; input unchanged.',
    'std::vector<int> ordered_unique(const std::vector<int>&a){std::set<int>s;std::vector<int>r;for(int x:a)if(s.insert(x).second)r.push_back(x);return r;}',
    'assert(ordered_unique({2,1,2,3,1})==std::vector<int>({2,1,3}));std::mt19937 r(772);for(int n=0;n<100;++n){std::vector<int>a(n),ref;for(int&x:a)x=int(r()%17)-8;for(int x:a)if(std::find(ref.begin(),ref.end(),x)==ref.end())ref.push_back(x);auto copy=a;assert(ordered_unique(a)==ref&&a==copy);}')
add('multiset_intersection','std::vector<int> multiset_intersection(const std::vector<int>& a,const std::vector<int>& b); inputs sorted ascending. Sorted intersection with minimum occurrence count for each value. Inputs unchanged.',
    'std::vector<int> multiset_intersection(const std::vector<int>&a,const std::vector<int>&b){std::vector<int>r;size_t i=0,j=0;while(i<a.size()&&j<b.size()){if(a[i]<b[j])++i;else if(b[j]<a[i])++j;else{r.push_back(a[i]);++i;++j;}}return r;}',
    'std::mt19937 g(773);for(int n=0;n<80;++n){std::vector<int>a(n),b(n/2),ref;for(int&x:a)x=g()%11;for(int&x:b)x=g()%11;std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());for(int x=0;x<11;++x)for(int c=std::min(std::count(a.begin(),a.end(),x),std::count(b.begin(),b.end(),x));c;--c)ref.push_back(x);assert(multiset_intersection(a,b)==ref);}')
add('remove_prefix_bytes','bool remove_prefix_bytes(std::string& s,std::string_view prefix); remove matching prefix and return true, else preserve s and return false. Empty prefix matches. Prefix may view s.',
    'bool remove_prefix_bytes(std::string&s,std::string_view p){if(s.size()<p.size()||std::string_view(s).substr(0,p.size())!=p)return false;auto n=p.size();s.erase(0,n);return true;}',
    'std::string s="abcdef";auto p=std::string_view(s).substr(0,3);assert(remove_prefix_bytes(s,p)&&s=="def");assert(!remove_prefix_bytes(s,"dx")&&s=="def");assert(remove_prefix_bytes(s,"")&&s=="def");s=std::string("a\\0b",3);assert(remove_prefix_bytes(s,std::string_view(s.data(),2))&&s=="b");')
add('index_of_nth','size_t index_of_nth(std::string_view s,unsigned char byte,size_t n) noexcept; position of zero-based nth occurrence, or s.size() if absent. Handle NUL and SIZE_MAX n.',
    'size_t index_of_nth(std::string_view s,unsigned char b,size_t n)noexcept{for(size_t i=0;i<s.size();++i)if((unsigned char)s[i]==b){if(n==0)return i;--n;}return s.size();}',
    'assert(index_of_nth("ababa",97,0)==0);assert(index_of_nth("ababa",97,2)==4);assert(index_of_nth("ababa",97,3)==5);assert(index_of_nth("a",97,SIZE_MAX)==1);assert(index_of_nth(std::string("x\\0x",3),0,0)==1);')
add('is_sorted_rotated','bool is_sorted_rotated(const std::vector<int>& a) noexcept; whether some cyclic rotation is nondecreasing. Duplicates allowed; empty and single-element inputs true.',
    'bool is_sorted_rotated(const std::vector<int>&a)noexcept{if(a.size()<2)return true;int drops=0;for(size_t i=0;i<a.size();++i)if(a[i]>a[(i+1)%a.size()]&&++drops>1)return false;return true;}',
    'for(int n=0;n<7;++n){int combinations=1;for(int i=0;i<n;++i)combinations*=3;for(int c=0;c<combinations;++c){std::vector<int>a(n);int t=c;for(int&x:a){x=t%3;t/=3;}bool ref=a.empty();for(int k=0;k<n;++k){auto b=a;std::rotate(b.begin(),b.begin()+k,b.end());ref|=std::is_sorted(b.begin(),b.end());}assert(is_sorted_rotated(a)==ref);}}')
add('parse_hhmm','bool parse_hhmm(std::string_view s,unsigned& minutes) noexcept; exact five-byte 24-hour HH:MM, hours00..23 and minutes00..59. Reject any other form and preserve output.',
    'bool parse_hhmm(std::string_view s,unsigned&out)noexcept{if(s.size()!=5||s[2]!=58)return false;for(int i:{0,1,3,4})if(s[i]<48||s[i]>57)return false;unsigned h=(s[0]-48)*10+s[1]-48,m=(s[3]-48)*10+s[4]-48;if(h>23||m>59)return false;out=h*60+m;return true;}',
    'unsigned out=77;for(auto s:{"","1:00","24:00","23:60","-1:00","12:3x"})assert(!parse_hhmm(s,out)&&out==77);for(unsigned h=0;h<24;++h)for(unsigned m=0;m<60;++m){std::string s="00:00";s[0]+=h/10;s[1]+=h%10;s[3]+=m/10;s[4]+=m%10;assert(parse_hhmm(s,out)&&out==h*60+m);}')
add('to_big_endian','std::vector<unsigned char> to_big_endian(uint64_t x); exactly eight bytes, most significant first.',
    'std::vector<unsigned char> to_big_endian(uint64_t x){std::vector<unsigned char>r(8);for(int i=7;i>=0;--i){r[i]=x&255;x>>=8;}return r;}',
    'std::mt19937_64 r(774);for(int i=0;i<1000;++i){auto x=r();auto b=to_big_endian(x);assert(b.size()==8);uint64_t y=0;for(auto v:b)y=(y<<8)|v;assert(y==x);}')
add('from_big_endian','uint64_t from_big_endian(const std::vector<unsigned char>& b); exactly eight bytes, most significant first. Throw std::invalid_argument for any other length.',
    'uint64_t from_big_endian(const std::vector<unsigned char>&b){if(b.size()!=8)throw std::invalid_argument("size");uint64_t x=0;for(auto v:b)x=(x<<8)|v;return x;}',
    'for(size_t n:{0,7,9}){bool e=false;try{from_big_endian(std::vector<unsigned char>(n));}catch(const std::invalid_argument&){e=true;}assert(e);}assert(from_big_endian({1,35,69,103,137,171,205,239})==0x0123456789abcdefULL);assert(from_big_endian(std::vector<unsigned char>(8,255))==UINT64_MAX);')
add('is_ascii_identifier','bool is_ascii_identifier(std::string_view s) noexcept; nonempty, first byte A-Z/a-z/underscore, subsequent bytes may also be0-9. Reject all other bytes including NUL and above127.',
    'bool is_ascii_identifier(std::string_view s)noexcept{auto letter=[](unsigned char c){return(c>=65&&c<=90)||(c>=97&&c<=122)||c==95;};if(s.empty()||!letter(s[0]))return false;for(unsigned char c:s)if(!letter(c)&&!(c>=48&&c<=57))return false;return true;}',
    'assert(!is_ascii_identifier(""));assert(is_ascii_identifier("_a9"));assert(!is_ascii_identifier("9a"));for(int b=0;b<256;++b){std::string s="a";s+=char(b);bool ok=(b>=65&&b<=90)||(b>=97&&b<=122)||(b>=48&&b<=57)||b==95;assert(is_ascii_identifier(s)==ok);}')
add('split_once_byte','std::pair<std::string,std::string> split_once_byte(std::string_view s,unsigned char separator); split at first matching byte, omit separator. If absent return entire input and empty string. Preserve NUL bytes.',
    'std::pair<std::string,std::string> split_once_byte(std::string_view s,unsigned char b){auto n=s.find(char(b));if(n==s.npos)return{std::string(s),{}};return{std::string(s.substr(0,n)),std::string(s.substr(n+1))};}',
    'assert(split_once_byte("a:b:c",58)==std::make_pair(std::string("a"),std::string("b:c")));assert(split_once_byte("abc",58)==std::make_pair(std::string("abc"),std::string()));assert(split_once_byte(std::string("a\\0b",3),0)==std::make_pair(std::string("a"),std::string("b")));')
add('interval_distance_i64','uint64_t interval_distance_i64(int64_t a,int64_t b) noexcept; absolute mathematical distance between a and b, fits uint64_t even for INT64_MIN and INT64_MAX. Avoid signed overflow.',
    'uint64_t interval_distance_i64(int64_t a,int64_t b)noexcept{return a>=b?uint64_t(a)-uint64_t(b):uint64_t(b)-uint64_t(a);}',
    'for(int64_t a:{INT64_MIN,int64_t(-1),int64_t(0),int64_t(1),INT64_MAX})for(int64_t b:{INT64_MIN,int64_t(-1),int64_t(0),int64_t(1),INT64_MAX}){__int128 d=(__int128)a-b;if(d<0)d=-d;assert(interval_distance_i64(a,b)==(uint64_t)d);}')
add('sum_mod_u64','uint64_t sum_mod_u64(uint64_t a,uint64_t b,uint64_t m) noexcept; m>0, a<m, b<m. Mathematical(a+b)%m without uint64 overflow.',
    'uint64_t sum_mod_u64(uint64_t a,uint64_t b,uint64_t m)noexcept{return a>=m-b?a-(m-b):a+b;}',
    'std::mt19937_64 r(775);for(int i=0;i<20000;++i){uint64_t m=r();if(!m)m=1;auto a=r()%m,b=r()%m;assert(sum_mod_u64(a,b,m)==uint64_t(((unsigned __int128)a+b)%m));}assert(sum_mod_u64(UINT64_MAX-1,UINT64_MAX-1,UINT64_MAX)==UINT64_MAX-2);')
add('normalize_ascii_space','std::string normalize_ascii_space(std::string_view s); trim and coalesce ASCII space/tab/CR/LF/form-feed/vertical-tab to a single space. Preserve every other byte including NUL.',
    'std::string normalize_ascii_space(std::string_view s){std::string r;bool pending=false;for(unsigned char c:s){bool ws=c==32||(c>=9&&c<=13);if(ws){pending=!r.empty();continue;}if(pending)r+=32;r+=char(c);pending=false;}return r;}',
    'assert(normalize_ascii_space(" \\t a\\r\\n b \\f")=="a b");assert(normalize_ascii_space("\\t\\n").empty());assert(normalize_ascii_space(std::string(" a\\0 b ",6))==std::string("a\\0 b",4));for(int c=0;c<256;++c)if(c!=32&&!(c>=9&&c<=13)){std::string s(1,char(c));assert(normalize_ascii_space(s)==s);}')
add('rle_expand_checked','bool rle_expand_checked(const std::vector<std::pair<unsigned char,size_t>>& runs,size_t limit,std::string& out); concatenate count copies of byte per run. Zero counts allowed. Fail and preserve out if total length exceeds limit; avoid length-sum overflow.',
    'bool rle_expand_checked(const std::vector<std::pair<unsigned char,size_t>>&a,size_t limit,std::string&out){size_t n=0;for(auto p:a){if(p.second>limit-n)return false;n+=p.second;}std::string r;for(auto p:a)r.append(p.second,char(p.first));out=std::move(r);return true;}',
    'std::string out="keep";assert(!rle_expand_checked({{97,SIZE_MAX},{98,1}},SIZE_MAX,out)&&out=="keep");assert(!rle_expand_checked({{97,3}},2,out)&&out=="keep");assert(rle_expand_checked({{97,2},{0,1},{98,0}},3,out)&&out==std::string("aa\\0",3));assert(rle_expand_checked({},0,out)&&out.empty());')
assert sum(v['split']=='holdout' for v in FIXTURES.values())==20
assert not ({k for k,v in FIXTURES.items() if v['split']=='holdout'} & PREVIOUS.keys())
def check_output(text,name):
    f=FIXTURES[name]
    with tempfile.TemporaryDirectory(prefix='glm-target-quality-') as d:
        root=Path(d);source=root/'case.cpp';binary=root/'case'
        source.write_text(INCLUDES+code_block(text)+'\n'+f['driver']+'\n')
        p=subprocess.run(['g++','-O2','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True,timeout=60)
        if p.returncode:raise ValueError('compile: '+p.stderr[-2000:])
        p=subprocess.run([str(binary)],capture_output=True,text=True,timeout=20)
        if p.returncode:raise ValueError('oracle: '+p.stderr[-1000:])
    return 1
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--check-oracles',action='store_true');a=p.parse_args()
    if a.check_oracles:
        for name,f in FIXTURES.items():check_output('```cpp\n'+f['reference']+'\n```',name);print('PASS',name,flush=True)
    print(json.dumps({'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training':8,'holdout':20}))
