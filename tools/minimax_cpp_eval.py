#!/usr/bin/env python3
"""A small functional C++ benchmark for the MiniMax engine's lossy decode settings.

26 single-function tasks with unit tests.  The model answers each prompt (thinking off) with greedy decoding; the first code
block of the answer is compiled together with the task's test main and run.  A task passes when the program compiles and exits 0.

    python3 tools/minimax_cpp_eval.py selftest                      # the reference solutions pass every test (harness check)
    python3 tools/minimax_cpp_eval.py prepare SHARD1.gguf OUTDIR    # prompts as token ids + manifest.json
    python3 tools/minimax_cpp_eval.py score SHARD1.gguf RESULTS.jsonl [--report out.md]

RESULTS.jsonl has one object per line: {"cfg": name, "task": name, "tokens": [ids], "tok_s": float}.
"""
import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

PRE = "#include <bits/stdc++.h>\nusing namespace std;\n"

# (name, signature, description, reference solution, test main body)
TASKS = [
    ("reverse_words", "std::string reverse_words(const std::string& s);",
     "Return the words of s in reverse order, separated by single spaces, with no leading or trailing space. Words are separated by one or more spaces.",
     'string reverse_words(const string& s){istringstream in(s);vector<string> w;string x;while(in>>x)w.push_back(x);string r;for(int i=(int)w.size()-1;i>=0;--i){r+=w[i];if(i)r+=" ";}return r;}',
     'assert(reverse_words("hello world")=="world hello");assert(reverse_words("  a   b c ")=="c b a");assert(reverse_words("")=="");assert(reverse_words("one")=="one");'),
    ("is_prime", "bool is_prime(long long n);",
     "Return true when n is a prime number (n may be up to 1e12; numbers below 2 are not prime).",
     'bool is_prime(long long n){if(n<2)return false;for(long long i=2;i*i<=n;++i)if(n%i==0)return false;return true;}',
     'assert(!is_prime(0));assert(!is_prime(1));assert(is_prime(2));assert(is_prime(3));assert(!is_prime(4));assert(is_prime(97));assert(!is_prime(100));assert(is_prime(7919));assert(is_prime(1000000007LL));assert(!is_prime(1000000000000LL));assert(is_prime(999999999989LL));'),
    ("lcm", "long long lcm_ll(long long a, long long b);",
     "Return the least common multiple of two non-negative integers; if either is 0 the result is 0. Values fit in 32 bits, the result fits in 64 bits.",
     'long long lcm_ll(long long a,long long b){if(!a||!b)return 0;return a/std::__gcd(a,b)*b;}',
     'assert(lcm_ll(4,6)==12);assert(lcm_ll(21,6)==42);assert(lcm_ll(1,1)==1);assert(lcm_ll(0,5)==0);assert(lcm_ll(1000000007LL,998244353LL)==1000000007LL*998244353LL);'),
    ("fib_mod", "unsigned long long fib_mod(unsigned long long n, unsigned long long m);",
     "Return F(n) mod m where F(0)=0, F(1)=1. n can be as large as 1e18 and m is at most 1e9+7, so it must be fast (O(log n)).",
     'typedef unsigned long long u;pair<u,u> fd(u n,u m){if(!n)return{0,1%m};auto p=fd(n/2,m);u a=p.first,b=p.second;u c=a*((2*b%m+m-a)%m)%m;u d=(a*a+b*b)%m;if(n%2==0)return{c,d};return{d,(c+d)%m};}u fib_mod(u n,u m){return fd(n,m).first;}',
     'assert(fib_mod(10,1000)==55);assert(fib_mod(0,7)==0);assert(fib_mod(1,7)==1);assert(fib_mod(100,1000000007ULL)==687995182ULL);assert(fib_mod(1000000000000000000ULL,1000000007ULL)==fib_mod(1000000000000000000ULL,1000000007ULL));assert(fib_mod(90,1000000000000ULL)==2880067194370816120ULL%1000000000000ULL);'),
    ("lower_bound_idx", "int lower_bound_idx(const std::vector<int>& a, int x);",
     "a is sorted ascending. Return the smallest index i with a[i] >= x, or a.size() if there is none. Use binary search.",
     'int lower_bound_idx(const vector<int>& a,int x){int lo=0,hi=a.size();while(lo<hi){int m=(lo+hi)/2;if(a[m]>=x)hi=m;else lo=m+1;}return lo;}',
     'assert(lower_bound_idx({1,3,5,7},4)==2);assert(lower_bound_idx({1,3,5,7},1)==0);assert(lower_bound_idx({1,3,5,7},8)==4);assert(lower_bound_idx({},3)==0);assert(lower_bound_idx({2,2,2},2)==0);'),
    ("merge_sorted", "std::vector<int> merge_sorted(const std::vector<int>& a, const std::vector<int>& b);",
     "Merge two ascending sorted vectors into one ascending sorted vector in linear time.",
     'vector<int> merge_sorted(const vector<int>&a,const vector<int>&b){vector<int> r;size_t i=0,j=0;while(i<a.size()||j<b.size()){if(j==b.size()||(i<a.size()&&a[i]<=b[j]))r.push_back(a[i++]);else r.push_back(b[j++]);}return r;}',
     'assert((merge_sorted({1,3,5},{2,4,6})==vector<int>{1,2,3,4,5,6}));assert((merge_sorted({},{1})==vector<int>{1}));assert((merge_sorted({1,1},{1})==vector<int>{1,1,1}));assert((merge_sorted({},{})==vector<int>{}));'),
    ("max_subarray", "long long max_subarray(const std::vector<int>& a);",
     "Return the maximum sum of a non-empty contiguous subarray of a (a is non-empty).",
     'long long max_subarray(const vector<int>&a){long long best=a[0],cur=a[0];for(size_t i=1;i<a.size();++i){cur=max<long long>(a[i],cur+a[i]);best=max(best,cur);}return best;}',
     'assert(max_subarray({-2,1,-3,4,-1,2,1,-5,4})==6);assert(max_subarray({-3,-1,-2})==-1);assert(max_subarray({5})==5);assert(max_subarray({1,2,3})==6);'),
    ("roman_to_int", "int roman_to_int(const std::string& s);",
     "Convert a valid Roman numeral (I V X L C D M, with subtractive forms like IV and CM) to an integer.",
     'int roman_to_int(const string&s){map<char,int> v{{\'I\',1},{\'V\',5},{\'X\',10},{\'L\',50},{\'C\',100},{\'D\',500},{\'M\',1000}};int r=0;for(size_t i=0;i<s.size();++i){int a=v[s[i]];if(i+1<s.size()&&a<v[s[i+1]])r-=a;else r+=a;}return r;}',
     'assert(roman_to_int("III")==3);assert(roman_to_int("IV")==4);assert(roman_to_int("IX")==9);assert(roman_to_int("LVIII")==58);assert(roman_to_int("MCMXCIV")==1994);'),
    ("balanced", "bool balanced(const std::string& s);",
     "Return true when the brackets ()[]{} in s are balanced and properly nested; other characters are ignored.",
     'bool balanced(const string&s){string st;for(char c:s){if(c==\'(\'||c==\'[\'||c==\'{\')st+=c;else if(c==\')\'||c==\']\'||c==\'}\'){if(st.empty())return false;char o=st.back();st.pop_back();if((c==\')\'&&o!=\'(\')||(c==\']\'&&o!=\'[\')||(c==\'}\'&&o!=\'{\'))return false;}}return st.empty();}',
     'assert(balanced("()[]{}"));assert(balanced("{[()]}"));assert(!balanced("(]"));assert(!balanced("(("));assert(!balanced(")("));assert(balanced("a(b)c[d]"));assert(balanced(""));'),
    ("lcp", "std::string lcp(const std::vector<std::string>& v);",
     "Return the longest common prefix of all strings in v (empty string when v is empty or there is none).",
     'string lcp(const vector<string>&v){if(v.empty())return "";string p=v[0];for(auto&s:v){size_t i=0;while(i<p.size()&&i<s.size()&&p[i]==s[i])++i;p.resize(i);}return p;}',
     'assert(lcp({"flower","flow","flight"})=="fl");assert(lcp({"dog","racecar","car"})=="");assert(lcp({})=="");assert(lcp({"abc"})=="abc");assert(lcp({"ab","ab"})=="ab");'),
    ("two_sum", "std::vector<int> two_sum(const std::vector<int>& a, int target);",
     "Return {i, j} with i < j and a[i] + a[j] == target. Exactly one such pair exists.",
     'vector<int> two_sum(const vector<int>&a,int t){unordered_map<int,int> m;for(int j=0;j<(int)a.size();++j){auto it=m.find(t-a[j]);if(it!=m.end())return{it->second,j};m[a[j]]=j;}return{};}',
     'assert((two_sum({2,7,11,15},9)==vector<int>{0,1}));assert((two_sum({3,2,4},6)==vector<int>{1,2}));assert((two_sum({3,3},6)==vector<int>{0,1}));'),
    ("rle", "std::string rle(const std::string& s);",
     "Run-length encode s: each maximal run of the same character becomes the character followed by the run length in decimal, e.g. \"aaabcc\" -> \"a3b1c2\". The empty string maps to the empty string.",
     'string rle(const string&s){string r;for(size_t i=0;i<s.size();){size_t j=i;while(j<s.size()&&s[j]==s[i])++j;r+=s[i];r+=to_string(j-i);i=j;}return r;}',
     'assert(rle("aaabcc")=="a3b1c2");assert(rle("")=="");assert(rle("z")=="z1");assert(rle("aaaaaaaaaaaa")=="a12");'),
    ("rotate_left", "std::vector<int> rotate_left(std::vector<int> a, int k);",
     "Rotate the vector left by k positions (k >= 0, possibly larger than the size) and return it. rotate_left({1,2,3,4,5}, 2) is {3,4,5,1,2}.",
     'vector<int> rotate_left(vector<int> a,int k){if(a.empty())return a;k%=a.size();rotate(a.begin(),a.begin()+k,a.end());return a;}',
     'assert((rotate_left({1,2,3,4,5},2)==vector<int>{3,4,5,1,2}));assert((rotate_left({1,2,3},0)==vector<int>{1,2,3}));assert((rotate_left({1,2,3},7)==vector<int>{2,3,1}));assert((rotate_left({},3)==vector<int>{}));'),
    ("is_anagram", "bool is_anagram(const std::string& a, const std::string& b);",
     "Return true when a and b contain the same letters with the same counts, ignoring case and ignoring all non-letter characters.",
     'bool is_anagram(const string&a,const string&b){auto f=[](const string&s){array<int,26> c{};for(char ch:s)if(isalpha((unsigned char)ch))c[tolower(ch)-\'a\']++;return c;};return f(a)==f(b);}',
     'assert(is_anagram("listen","silent"));assert(is_anagram("Dormitory","dirty room"));assert(!is_anagram("abc","abd"));assert(is_anagram("",""));assert(!is_anagram("aab","ab"));'),
    ("popcount64", "int popcount64(unsigned long long x);",
     "Return the number of set bits in x.",
     'int popcount64(unsigned long long x){int c=0;while(x){x&=x-1;++c;}return c;}',
     'assert(popcount64(0)==0);assert(popcount64(1)==1);assert(popcount64(255)==8);assert(popcount64(0xFFFFFFFFFFFFFFFFULL)==64);assert(popcount64(0x8000000000000001ULL)==2);'),
    ("camel_to_snake", "std::string camel_to_snake(const std::string& s);",
     "Convert camelCase to snake_case: put an underscore before every uppercase letter (except at position 0) and lowercase it. \"helloWorldFoo\" -> \"hello_world_foo\".",
     'string camel_to_snake(const string&s){string r;for(size_t i=0;i<s.size();++i){if(isupper((unsigned char)s[i])){if(i)r+=\'_\';r+=tolower(s[i]);}else r+=s[i];}return r;}',
     'assert(camel_to_snake("helloWorldFoo")=="hello_world_foo");assert(camel_to_snake("abc")=="abc");assert(camel_to_snake("")=="");assert(camel_to_snake("aB")=="a_b");'),
    ("transpose", "std::vector<std::vector<int>> transpose(const std::vector<std::vector<int>>& m);",
     "Return the transpose of a rectangular matrix (all rows have equal length; the matrix may be empty).",
     'vector<vector<int>> transpose(const vector<vector<int>>&m){if(m.empty())return {};vector<vector<int>> r(m[0].size(),vector<int>(m.size()));for(size_t i=0;i<m.size();++i)for(size_t j=0;j<m[0].size();++j)r[j][i]=m[i][j];return r;}',
     'assert((transpose({{1,2,3},{4,5,6}})==vector<vector<int>>{{1,4},{2,5},{3,6}}));assert((transpose({{1}})==vector<vector<int>>{{1}}));assert((transpose({})==vector<vector<int>>{}));'),
    ("pow_mod", "long long pow_mod(long long b, long long e, long long m);",
     "Return b^e mod m for 0 <= b, 0 <= e <= 1e18 and 1 <= m <= 1e9+7 (pow_mod(x, 0, m) is 1 % m).",
     'long long pow_mod(long long b,long long e,long long m){long long r=1%m;b%=m;while(e){if(e&1)r=r*b%m;b=b*b%m;e>>=1;}return r;}',
     'assert(pow_mod(2,10,1000)==24);assert(pow_mod(3,0,7)==1);assert(pow_mod(5,3,1)==0);assert(pow_mod(2,1000000000000000000LL,1000000007LL)==pow_mod(2,1000000000000000000LL,1000000007LL));assert(pow_mod(7,13,1000000007LL)==96889010407LL%1000000007LL);'),
    ("count_islands", "int count_islands(const std::vector<std::string>& grid);",
     "Count the 4-connected groups of '1' cells in a grid of '0'/'1' characters (rows may be empty).",
     'int count_islands(const vector<string>&g){int n=g.size(),c=0;vector<vector<char>> seen(n);for(int i=0;i<n;++i)seen[i].assign(g[i].size(),0);function<void(int,int)> d=[&](int i,int j){if(i<0||i>=n||j<0||j>=(int)g[i].size()||g[i][j]!=\'1\'||seen[i][j])return;seen[i][j]=1;d(i+1,j);d(i-1,j);d(i,j+1);d(i,j-1);};for(int i=0;i<n;++i)for(int j=0;j<(int)g[i].size();++j)if(g[i][j]==\'1\'&&!seen[i][j]){++c;d(i,j);}return c;}',
     'assert(count_islands({"11000","11000","00100","00011"})==3);assert(count_islands({})==0);assert(count_islands({"000"})==0);assert(count_islands({"101","010","101"})==5);'),
    ("top_k_frequent", "std::vector<int> top_k_frequent(const std::vector<int>& a, int k);",
     "Return the k most frequent values of a, ordered by decreasing frequency; ties are broken by the smaller value first.",
     'vector<int> top_k_frequent(const vector<int>&a,int k){map<int,int> c;for(int x:a)c[x]++;vector<pair<int,int>> v(c.begin(),c.end());sort(v.begin(),v.end(),[](auto&x,auto&y){return x.second!=y.second?x.second>y.second:x.first<y.first;});vector<int> r;for(int i=0;i<k&&i<(int)v.size();++i)r.push_back(v[i].first);return r;}',
     'assert((top_k_frequent({1,1,1,2,2,3},2)==vector<int>{1,2}));assert((top_k_frequent({4,4,5,5,6},2)==vector<int>{4,5}));assert((top_k_frequent({7},1)==vector<int>{7}));assert((top_k_frequent({3,1,2},3)==vector<int>{1,2,3}));'),
    ("lis_length", "int lis_length(const std::vector<int>& a);",
     "Return the length of the longest strictly increasing subsequence of a (O(n log n) expected for n up to 1e5).",
     'int lis_length(const vector<int>&a){vector<int> t;for(int x:a){auto it=lower_bound(t.begin(),t.end(),x);if(it==t.end())t.push_back(x);else *it=x;}return t.size();}',
     'assert(lis_length({10,9,2,5,3,7,101,18})==4);assert(lis_length({})==0);assert(lis_length({5,5,5})==1);assert(lis_length({1,2,3,4})==4);assert(lis_length({4,3,2,1})==1);'),
    ("edit_distance", "int edit_distance(const std::string& a, const std::string& b);",
     "Return the Levenshtein distance between a and b (insert, delete, substitute each cost 1).",
     'int edit_distance(const string&a,const string&b){int n=a.size(),m=b.size();vector<vector<int>> d(n+1,vector<int>(m+1));for(int i=0;i<=n;++i)d[i][0]=i;for(int j=0;j<=m;++j)d[0][j]=j;for(int i=1;i<=n;++i)for(int j=1;j<=m;++j)d[i][j]=min({d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]!=b[j-1])});return d[n][m];}',
     'assert(edit_distance("kitten","sitting")==3);assert(edit_distance("","abc")==3);assert(edit_distance("same","same")==0);assert(edit_distance("flaw","lawn")==2);'),
    ("hex_encode", "std::string hex_encode(const std::string& s);",
     "Return the lowercase hexadecimal encoding of the bytes of s, two digits per byte, e.g. \"Hi\" -> \"4869\".",
     'string hex_encode(const string&s){static const char*h="0123456789abcdef";string r;for(unsigned char c:s){r+=h[c>>4];r+=h[c&15];}return r;}',
     'assert(hex_encode("Hi")=="4869");assert(hex_encode("")=="");assert(hex_encode(string("\\x00\\xff",2))=="00ff");assert(hex_encode("abc")=="616263");'),
    ("add_big", "std::string add_big(const std::string& a, const std::string& b);",
     "Add two non-negative integers given as decimal strings of arbitrary length (no leading zeros except \"0\") and return the sum as a decimal string.",
     'string add_big(const string&a,const string&b){string r;int i=a.size()-1,j=b.size()-1,c=0;while(i>=0||j>=0||c){int s=c+(i>=0?a[i--]-\'0\':0)+(j>=0?b[j--]-\'0\':0);r+=char(\'0\'+s%10);c=s/10;}reverse(r.begin(),r.end());return r.empty()?"0":r;}',
     'assert(add_big("123","456")=="579");assert(add_big("999","1")=="1000");assert(add_big("0","0")=="0");assert(add_big("99999999999999999999","1")=="100000000000000000000");'),
    ("next_perm", "bool next_perm(std::vector<int>& a);",
     "Rearrange a into the lexicographically next greater permutation and return true; if a is already the greatest permutation, rearrange it into the smallest (ascending) and return false. Do not call std::next_permutation.",
     'bool next_perm(vector<int>&a){int i=(int)a.size()-2;while(i>=0&&a[i]>=a[i+1])--i;if(i<0){reverse(a.begin(),a.end());return false;}int j=a.size()-1;while(a[j]<=a[i])--j;swap(a[i],a[j]);reverse(a.begin()+i+1,a.end());return true;}',
     'vector<int> a{1,2,3};assert((next_perm(a)&&a==vector<int>{1,3,2}));a={3,2,1};assert((!next_perm(a)&&a==vector<int>{1,2,3}));a={1,1,5};assert((next_perm(a)&&a==vector<int>{1,5,1}));a={};assert(!next_perm(a));'),
    ("merge_intervals", "std::vector<std::pair<int,int>> merge_intervals(std::vector<std::pair<int,int>> v);",
     "Merge all overlapping closed intervals (touching intervals such as [1,2] and [2,3] also merge) and return them sorted by start.",
     'vector<pair<int,int>> merge_intervals(vector<pair<int,int>> v){sort(v.begin(),v.end());vector<pair<int,int>> r;for(auto&p:v){if(!r.empty()&&p.first<=r.back().second)r.back().second=max(r.back().second,p.second);else r.push_back(p);}return r;}',
     'typedef vector<pair<int,int>> V;assert((merge_intervals({{1,3},{2,6},{8,10},{15,18}})==V{{1,6},{8,10},{15,18}}));assert((merge_intervals({{1,4},{4,5}})==V{{1,5}}));assert((merge_intervals({})==V{}));assert((merge_intervals({{5,6},{1,2}})==V{{1,2},{5,6}}));'),

    # ---- harder tasks (classes, graphs, dynamic programming, parsing, string algorithms)
    ("lru_cache", "class LRUCache {\npublic:\n    explicit LRUCache(int capacity);\n    int get(int key);\n    void put(int key, int value);\n};",
     "Implement a least-recently-used cache with the given capacity. get returns the value or -1 when the key is absent and marks the key as most recently used; put inserts or updates a key (also making it most recently used) and, when the capacity is exceeded, evicts the least recently used key. Both operations must be O(1). Write the whole class.",
     'class LRUCache{int cap;list<pair<int,int>> l;unordered_map<int,list<pair<int,int>>::iterator> m;public:explicit LRUCache(int c):cap(c){}int get(int k){auto it=m.find(k);if(it==m.end())return -1;l.splice(l.begin(),l,it->second);return it->second->second;}void put(int k,int v){auto it=m.find(k);if(it!=m.end()){it->second->second=v;l.splice(l.begin(),l,it->second);return;}l.emplace_front(k,v);m[k]=l.begin();if((int)l.size()>cap){m.erase(l.back().first);l.pop_back();}}};',
     'LRUCache c(2);c.put(1,1);c.put(2,2);assert(c.get(1)==1);c.put(3,3);assert(c.get(2)==-1);c.put(4,4);assert(c.get(1)==-1);assert(c.get(3)==3);assert(c.get(4)==4);c.put(3,9);assert(c.get(3)==9);'),
    ("dijkstra", "std::vector<long long> dijkstra(int n, const std::vector<std::tuple<int,int,int>>& edges, int src);",
     "Single-source shortest paths on a directed graph with n vertices (0..n-1) and non-negative weighted edges (u, v, w). Return the distance to every vertex, or -1 for unreachable vertices.",
     'vector<long long> dijkstra(int n,const vector<tuple<int,int,int>>&e,int src){vector<vector<pair<int,int>>> g(n);for(auto&[u,v,w]:e)g[u].push_back({v,w});vector<long long> d(n,-1);priority_queue<pair<long long,int>,vector<pair<long long,int>>,greater<>> q;d[src]=0;q.push({0,src});while(!q.empty()){auto[dd,u]=q.top();q.pop();if(dd>d[u])continue;for(auto[v,w]:g[u])if(d[v]<0||dd+w<d[v]){d[v]=dd+w;q.push({d[v],v});}}return d;}',
     'typedef vector<tuple<int,int,int>> E;assert((dijkstra(5,E{{0,1,4},{0,2,1},{2,1,2},{1,3,1},{2,3,5}},0)==vector<long long>{0,3,1,4,-1}));assert((dijkstra(1,E{},0)==vector<long long>{0}));assert((dijkstra(3,E{{1,2,1}},0)==vector<long long>{0,-1,-1}));'),
    ("topo_smallest", "std::vector<int> topo_smallest(int n, const std::vector<std::pair<int,int>>& edges);",
     "Topological order of a directed graph with n vertices (0..n-1) and edges u -> v. Among the vertices that are available at each step always take the smallest index. Return an empty vector when the graph has a cycle.",
     'vector<int> topo_smallest(int n,const vector<pair<int,int>>&e){vector<vector<int>> g(n);vector<int> in(n);for(auto&[u,v]:e){g[u].push_back(v);in[v]++;}priority_queue<int,vector<int>,greater<>> q;for(int i=0;i<n;++i)if(!in[i])q.push(i);vector<int> r;while(!q.empty()){int u=q.top();q.pop();r.push_back(u);for(int v:g[u])if(--in[v]==0)q.push(v);}if((int)r.size()!=n)r.clear();return r;}',
     'typedef vector<pair<int,int>> E;assert((topo_smallest(4,E{{3,0},{3,1},{0,2},{1,2}})==vector<int>{3,0,1,2}));assert((topo_smallest(2,E{{0,1},{1,0}})==vector<int>{}));assert((topo_smallest(3,E{})==vector<int>{0,1,2}));'),
    ("eval_expr", "long long eval_expr(const std::string& s);",
     "Evaluate an arithmetic expression with non-negative integers, the binary operators + - * / (usual precedence, left associative, division truncates toward zero), parentheses and arbitrary spaces. The expression is valid.",
     'struct P{const string&s;size_t i=0;P(const string&x):s(x){}void ws(){while(i<s.size()&&s[i]==\' \')++i;}long long num(){ws();if(s[i]==\'(\'){++i;long long v=expr();ws();++i;return v;}long long v=0;while(i<s.size()&&isdigit((unsigned char)s[i]))v=v*10+s[i++]-\'0\';return v;}long long term(){long long v=num();for(;;){ws();if(i<s.size()&&s[i]==\'*\'){++i;v*=num();}else if(i<s.size()&&s[i]==\'/\'){++i;v/=num();}else return v;}}long long expr(){long long v=term();for(;;){ws();if(i<s.size()&&s[i]==\'+\'){++i;v+=term();}else if(i<s.size()&&s[i]==\'-\'){++i;v-=term();}else return v;}}};long long eval_expr(const string&s){P p(s);return p.expr();}',
     'assert(eval_expr("1+2*3")==7);assert(eval_expr("(1+2)*3")==9);assert(eval_expr(" 10 / 3 - 1")==2);assert(eval_expr("2*(3+4)*5")==70);assert(eval_expr("100")==100);assert(eval_expr("7-2-1")==4);assert(eval_expr("((2))*(3 + 4) / 7")==2);'),
    ("knapsack", "int knapsack(const std::vector<int>& w, const std::vector<int>& v, int cap);",
     "0/1 knapsack: item i has weight w[i] and value v[i]; return the maximum total value of a subset whose total weight is at most cap.",
     'int knapsack(const vector<int>&w,const vector<int>&v,int cap){vector<int> d(cap+1);for(size_t i=0;i<w.size();++i)for(int c=cap;c>=w[i];--c)d[c]=max(d[c],d[c-w[i]]+v[i]);return d[cap];}',
     'assert(knapsack({1,3,4,5},{1,4,5,7},7)==9);assert(knapsack({1,3,4,5},{1,4,5,7},0)==0);assert(knapsack({5},{10},4)==0);assert(knapsack({2,2,2},{3,3,3},4)==6);'),
    ("coin_change", "int coin_change(const std::vector<int>& coins, int amount);",
     "Return the minimum number of coins (unlimited supply of each denomination) that sum to amount, or -1 if impossible. amount 0 needs 0 coins.",
     'int coin_change(const vector<int>&c,int a){vector<int> d(a+1,INT_MAX);d[0]=0;for(int x=1;x<=a;++x)for(int k:c)if(k<=x&&d[x-k]!=INT_MAX)d[x]=min(d[x],d[x-k]+1);return d[a]==INT_MAX?-1:d[a];}',
     'assert(coin_change({1,2,5},11)==3);assert(coin_change({2},3)==-1);assert(coin_change({1},0)==0);assert(coin_change({186,419,83,408},6249)==20);'),
    ("lcs_len", "int lcs_len(const std::string& a, const std::string& b);",
     "Return the length of the longest common subsequence of a and b.",
     'int lcs_len(const string&a,const string&b){vector<vector<int>> d(a.size()+1,vector<int>(b.size()+1));for(size_t i=1;i<=a.size();++i)for(size_t j=1;j<=b.size();++j)d[i][j]=a[i-1]==b[j-1]?d[i-1][j-1]+1:max(d[i-1][j],d[i][j-1]);return d[a.size()][b.size()];}',
     'assert(lcs_len("abcde","ace")==3);assert(lcs_len("abc","def")==0);assert(lcs_len("","a")==0);assert(lcs_len("AGGTAB","GXTXAYB")==4);'),
    ("n_queens", "int n_queens(int n);",
     "Return the number of ways to place n non-attacking queens on an n x n chessboard (1 <= n <= 10).",
     'int n_queens(int n){int c=0;vector<int> col(n),d1(2*n),d2(2*n);function<void(int)> go=[&](int r){if(r==n){++c;return;}for(int x=0;x<n;++x){if(col[x]||d1[r+x]||d2[r-x+n])continue;col[x]=d1[r+x]=d2[r-x+n]=1;go(r+1);col[x]=d1[r+x]=d2[r-x+n]=0;}};go(0);return c;}',
     'assert(n_queens(1)==1);assert(n_queens(4)==2);assert(n_queens(6)==4);assert(n_queens(8)==92);'),
    ("primes_upto", "std::vector<int> primes_upto(int n);",
     "Return all primes <= n in ascending order using a sieve; n can be up to 5,000,000.",
     'vector<int> primes_upto(int n){vector<char> c(n+1);vector<int> r;for(int i=2;i<=n;++i){if(!c[i]){r.push_back(i);for(long long j=(long long)i*i;j<=n;j+=i)c[j]=1;}}return r;}',
     'assert((primes_upto(10)==vector<int>{2,3,5,7}));assert((primes_upto(1)==vector<int>{}));assert((primes_upto(2)==vector<int>{2}));assert(primes_upto(100000).size()==9592);assert(primes_upto(5000000).size()==348513);'),
    ("my_atoi", "int my_atoi(const std::string& s);",
     "Parse an integer like C atoi: skip leading spaces, accept an optional + or - sign, read decimal digits until the first non-digit, return 0 when there are no digits, and clamp the result to the int range (INT_MIN..INT_MAX) instead of overflowing.",
     'int my_atoi(const string&s){size_t i=0;while(i<s.size()&&s[i]==\' \')++i;int sg=1;if(i<s.size()&&(s[i]==\'+\'||s[i]==\'-\')){if(s[i]==\'-\')sg=-1;++i;}long long v=0;while(i<s.size()&&isdigit((unsigned char)s[i])){v=v*10+s[i++]-\'0\';if(v>(1LL<<40))break;}v*=sg;if(v>INT_MAX)return INT_MAX;if(v<INT_MIN)return INT_MIN;return (int)v;}',
     'assert(my_atoi("42")==42);assert(my_atoi("   -42")==-42);assert(my_atoi("4193 with words")==4193);assert(my_atoi("words 987")==0);assert(my_atoi("-91283472332")==INT_MIN);assert(my_atoi("91283472332")==INT_MAX);assert(my_atoi("+1")==1);assert(my_atoi("")==0);'),
    ("kth_largest", "int kth_largest(std::vector<int> a, int k);",
     "Return the k-th largest element of a (1-indexed, duplicates counted: in {3,2,3,1} the 2nd largest is 3). 1 <= k <= a.size().",
     'int kth_largest(vector<int> a,int k){nth_element(a.begin(),a.begin()+(k-1),a.end(),greater<int>());return a[k-1];}',
     'assert(kth_largest({3,2,1,5,6,4},2)==5);assert(kth_largest({3,2,3,1,2,4,5,5,6},4)==4);assert(kth_largest({1},1)==1);assert(kth_largest({3,2,3,1},2)==3);'),
    ("count_components", "int count_components(int n, const std::vector<std::pair<int,int>>& edges);",
     "Return the number of connected components of an undirected graph with n vertices (0..n-1) and the given edges (use union-find or a DFS).",
     'int count_components(int n,const vector<pair<int,int>>&e){vector<int> p(n);iota(p.begin(),p.end(),0);function<int(int)> f=[&](int x){return p[x]==x?x:p[x]=f(p[x]);};int c=n;for(auto [a,b]:e){a=f(a);b=f(b);if(a!=b){p[a]=b;--c;}}return c;}',
     'typedef vector<pair<int,int>> E;assert(count_components(5,E{{0,1},{1,2},{3,4}})==2);assert(count_components(4,E{})==4);assert(count_components(0,E{})==0);assert(count_components(3,E{{0,1},{1,2},{0,2}})==1);'),
    ("longest_palindrome", "std::string longest_palindrome(const std::string& s);",
     "Return the longest palindromic substring of s; if several have the same length return the one that starts earliest. The empty string maps to the empty string.",
     'string longest_palindrome(const string&s){int n=s.size(),bs=0,bl=n?1:0;for(int c=0;c<n;++c)for(int t=0;t<2;++t){int l=c,r=c+t;while(l>=0&&r<n&&s[l]==s[r]){--l;++r;}int len=r-l-1;if(len>bl||(len==bl&&l+1<bs)){bl=len;bs=l+1;}}return s.substr(bs,bl);}',
     'assert(longest_palindrome("babad")=="bab");assert(longest_palindrome("cbbd")=="bb");assert(longest_palindrome("a")=="a");assert(longest_palindrome("forgeeksskeegfor")=="geeksskeeg");assert(longest_palindrome("")=="");'),
    ("word_break", "bool word_break(const std::string& s, const std::vector<std::string>& dict);",
     "Return true when s can be segmented into a sequence of one or more words from dict (words may be reused).",
     'bool word_break(const string&s,const vector<string>&d){vector<char> ok(s.size()+1);ok[0]=1;for(size_t i=1;i<=s.size();++i)for(auto&w:d)if(w.size()<=i&&ok[i-w.size()]&&s.compare(i-w.size(),w.size(),w)==0){ok[i]=1;break;}return ok[s.size()];}',
     'assert(word_break("leetcode",{"leet","code"}));assert(word_break("applepenapple",{"apple","pen"}));assert(!word_break("catsandog",{"cats","dog","sand","and","cat"}));assert(!word_break("a",{}));'),
    ("unique_perms", "std::vector<std::string> unique_perms(std::string s);",
     "Return all distinct permutations of the characters of s in lexicographic order (the empty string has one permutation, the empty string).",
     'vector<string> unique_perms(string s){sort(s.begin(),s.end());vector<string> r;do r.push_back(s);while(next_permutation(s.begin(),s.end()));return r;}',
     'assert((unique_perms("aab")==vector<string>{"aab","aba","baa"}));assert(unique_perms("abc").size()==6);assert(unique_perms("abc").front()=="abc"&&unique_perms("abc").back()=="cba");assert((unique_perms("")==vector<string>{""}));'),
    ("mul_big", "std::string mul_big(const std::string& a, const std::string& b);",
     "Multiply two non-negative integers given as decimal strings of arbitrary length and return the product as a decimal string without leading zeros (\"0\" for zero).",
     'string mul_big(const string&a,const string&b){vector<int> r(a.size()+b.size());for(int i=a.size()-1;i>=0;--i)for(int j=b.size()-1;j>=0;--j){int s=(a[i]-\'0\')*(b[j]-\'0\')+r[i+j+1];r[i+j+1]=s%10;r[i+j]+=s/10;}string o;for(int x:r)if(!o.empty()||x)o+=char(\'0\'+x);return o.empty()?"0":o;}',
     'assert(mul_big("123","456")=="56088");assert(mul_big("0","999")=="0");assert(mul_big("99999999999","99999999999")=="9999999999800000000001");assert(mul_big("1","1")=="1");'),
    ("spiral", "std::vector<int> spiral(const std::vector<std::vector<int>>& m);",
     "Return the elements of a (possibly non-square, possibly empty) matrix in clockwise spiral order starting at the top-left corner.",
     'vector<int> spiral(const vector<vector<int>>&m){vector<int> r;if(m.empty())return r;int t=0,b=m.size()-1,l=0,rr=m[0].size()-1;while(t<=b&&l<=rr){for(int j=l;j<=rr;++j)r.push_back(m[t][j]);++t;for(int i=t;i<=b;++i)r.push_back(m[i][rr]);--rr;if(t<=b){for(int j=rr;j>=l;--j)r.push_back(m[b][j]);--b;}if(l<=rr){for(int i=b;i>=t;--i)r.push_back(m[i][l]);++l;}}return r;}',
     'assert((spiral({{1,2,3},{4,5,6},{7,8,9}})==vector<int>{1,2,3,6,9,8,7,4,5}));assert((spiral({{1,2,3,4},{5,6,7,8},{9,10,11,12}})==vector<int>{1,2,3,4,8,12,11,10,9,5,6,7}));assert((spiral({})==vector<int>{}));assert((spiral({{1},{2}})==vector<int>{1,2}));'),
    ("median_sorted", "double median_sorted(const std::vector<int>& a, const std::vector<int>& b);",
     "Return the median of the combined multiset of two sorted vectors (at least one element in total). For an even count it is the mean of the two middle values.",
     'double median_sorted(const vector<int>&a,const vector<int>&b){vector<int> v(a);v.insert(v.end(),b.begin(),b.end());sort(v.begin(),v.end());size_t n=v.size();return n%2?v[n/2]:(v[n/2-1]+v[n/2])/2.0;}',
     'assert(fabs(median_sorted({1,3},{2})-2.0)<1e-9);assert(fabs(median_sorted({1,2},{3,4})-2.5)<1e-9);assert(fabs(median_sorted({},{5})-5.0)<1e-9);assert(fabs(median_sorted({1,1},{1,1})-1.0)<1e-9);'),
    ("min_window", "std::string min_window(const std::string& s, const std::string& t);",
     "Return the shortest substring of s that contains every character of t with at least its multiplicity in t; return \"\" when none exists; if several shortest windows exist return the leftmost.",
     'string min_window(const string&s,const string&t){if(t.empty())return "";array<int,256> need{};for(unsigned char c:t)need[c]++;int missing=t.size(),bs=0,bl=INT_MAX,l=0;for(int r=0;r<(int)s.size();++r){if(need[(unsigned char)s[r]]-->0)--missing;while(!missing){if(r-l+1<bl){bl=r-l+1;bs=l;}if(++need[(unsigned char)s[l]]>0)++missing;++l;}}return bl==INT_MAX?"":s.substr(bs,bl);}',
     'assert(min_window("ADOBECODEBANC","ABC")=="BANC");assert(min_window("a","a")=="a");assert(min_window("a","aa")=="");assert(min_window("ab","b")=="b");'),
    ("gen_parens", "std::vector<std::string> gen_parens(int n);",
     "Return all balanced strings of n pairs of parentheses in lexicographic order ('(' sorts before ')'). n = 0 yields a single empty string.",
     'void gp(int o,int c,int n,string&cur,vector<string>&r){if((int)cur.size()==2*n){r.push_back(cur);return;}if(o<n){cur+=\'(\';gp(o+1,c,n,cur,r);cur.pop_back();}if(c<o){cur+=\')\';gp(o,c+1,n,cur,r);cur.pop_back();}}vector<string> gen_parens(int n){vector<string> r;string cur;gp(0,0,n,cur,r);return r;}',
     'assert((gen_parens(1)==vector<string>{"()"}));assert((gen_parens(3)==vector<string>{"((()))","(()())","(())()","()(())","()()()"}));assert((gen_parens(0)==vector<string>{""}));assert(gen_parens(8).size()==1430);'),
]

INSTR = ("Write a complete C++17 implementation of the following function. Reply with exactly one ```cpp code block that contains the function "
         "(plus any helper functions it needs) and nothing else: no main(), no explanation. The headers <bits/stdc++.h> are already included "
         "and `using namespace std;` is in effect.\n\nSignature:\n{sig}\n\nBehaviour: {desc}")


def build_program(code, test):
    return f"{PRE}\n{code}\n\nint main() {{\n{test}\nreturn 0;\n}}\n"


def run_program(src, timeout=10):
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d)
        (p / "t.cpp").write_text(src)
        c = subprocess.run(["g++", "-std=c++17", "-O1", "-w", "-o", str(p / "t"), str(p / "t.cpp")], capture_output=True, text=True, timeout=120)
        if c.returncode:
            return "compile", c.stderr[-300:]
        try:
            r = subprocess.run([str(p / "t")], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout", ""
        return ("pass" if r.returncode == 0 else "fail"), r.stderr[-200:]


def extract_code(text):
    m = re.search(r"```(?:cpp|c\+\+|cc|C\+\+)?\s*\n(.*?)```", text, re.S)
    if not m:
        m = re.search(r"```(?:cpp|c\+\+|cc|C\+\+)?\s*\n(.*)$", text, re.S)   # a block cut off by the token limit
    code = m.group(1) if m else text
    code = re.sub(r"^\s*#include\s*<bits/stdc\+\+\.h>\s*$", "", code, flags=re.M)
    code = re.sub(r"^\s*using\s+namespace\s+std\s*;\s*$", "", code, flags=re.M)
    code = re.sub(r"int\s+main\s*\([^)]*\)\s*\{.*?\n\}\s*$", "", code, flags=re.S) if re.search(r"int\s+main\s*\(", code) else code
    return code


def cmd_selftest(_):
    bad = 0
    for name, sig, desc, ref, test in TASKS:
        status, err = run_program(build_program(ref, test))
        print(f"{name:18s} {status}")
        if status != "pass":
            bad += 1
            print(err)
    print("selftest:", "all reference solutions pass" if not bad else f"{bad} failing")
    return 1 if bad else 0


def load_tokenizer(shard):
    from strata_tokenizer import Tokenizer
    return Tokenizer.from_gguf(pathlib.Path(shard))


def cmd_prepare(a):
    from gguf_reader import GGUFFile
    from jinja2.sandbox import SandboxedEnvironment
    meta = GGUFFile(pathlib.Path(a.shard)).metadata
    tk = load_tokenizer(a.shard)
    env = SandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda s: (_ for _ in ()).throw(ValueError(s))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, sig, desc, ref, test in TASKS:
        chat = env.from_string(meta["tokenizer.chat_template"]).render(
            messages=[{"role": "user", "content": INSTR.format(sig=sig, desc=desc)}], tools=None, add_generation_prompt=True)
        chat += "</mm:think>\n"   # thinking off: the reply starts after the think-end prefix
        ids = tk.encode(chat, parse_special=True)
        (out / f"{name}.ids").write_text(" ".join(map(str, ids)) + "\n")
        manifest.append({"task": name, "ids": f"{name}.ids", "n_prompt": len(ids)})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(manifest)} prompts, {min(m['n_prompt'] for m in manifest)}-{max(m['n_prompt'] for m in manifest)} tokens, in {out}")
    return 0


def cmd_score(a):
    meta_tk = load_tokenizer(a.shard)
    from gguf_reader import GGUFFile
    meta = GGUFFile(pathlib.Path(a.shard)).metadata
    stops = {int(meta[k]) for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id") if k in meta}
    tasks = {t[0]: t for t in TASKS}
    rows = collections.defaultdict(list)
    for line in pathlib.Path(a.results).read_text().splitlines():
        if line.strip():
            rows[json.loads(line)["cfg"]].append(json.loads(line))
    report = ["| config | pass | compile errors | wrong | mean tok/s | mean tokens |", "|---|---|---|---|---|---|"]
    detail = []
    for cfg, rs in rows.items():
        cnt = collections.Counter()
        speeds, lens = [], []
        for r in rs:
            text = meta_tk.decode([t for t in r["tokens"] if t not in stops])
            name = r["task"]
            _, sig, desc, ref, test = tasks[name]
            status, err = run_program(build_program(extract_code(text), test))
            cnt[status] += 1
            detail.append(f"{cfg}\t{name}\t{status}")
            if r.get("tok_s"):
                speeds.append(r["tok_s"])
            lens.append(len(r["tokens"]))
        n = len(rs)
        report.append(f"| {cfg} | {cnt['pass']}/{n} | {cnt['compile']} | {cnt['fail'] + cnt['timeout']} | "
                      f"{sum(speeds) / max(1, len(speeds)):.1f} | {sum(lens) / max(1, len(lens)):.0f} |")
    text = "\n".join(report)
    print(text)
    if a.report:
        pathlib.Path(a.report).write_text(text + "\n\nper task:\n" + "\n".join(detail) + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    p = sub.add_parser("prepare")
    p.add_argument("shard")
    p.add_argument("out")
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("score")
    p.add_argument("shard")
    p.add_argument("results")
    p.add_argument("--report")
    p.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
