#pragma once
// Small strict metadata parser. Integers retain their spelling: offsets never pass through double.
#include <charconv>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strata::artifact {
struct Json {
    enum Kind { Null, Boolean, Number, String, Array, Object } kind = Null;
    bool boolean = false;
    std::string text;
    std::vector<Json> array;
    std::map<std::string, Json> object;
    const Json& at(const std::string& name) const {
        if (kind != Object || !object.contains(name)) throw std::runtime_error("JSON: missing key " + name);
        return object.at(name);
    }
    const Json* find(const std::string& name) const {
        auto i = object.find(name); return kind == Object && i != object.end() ? &i->second : nullptr;
    }
    const std::string& string() const {
        if (kind != String) throw std::runtime_error("JSON: expected string");
        return text;
    }
    bool flag() const {
        if (kind != Boolean) throw std::runtime_error("JSON: expected boolean");
        return boolean;
    }
    uint64_t integer() const {
        uint64_t value = 0;
        auto [p, e] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (kind != Number || text.empty() || e != std::errc() || p != text.data() + text.size())
            throw std::runtime_error("JSON: expected nonnegative integer");
        return value;
    }
    float real() const {
        if (kind != Number) throw std::runtime_error("JSON: expected number");
        size_t end; float v = std::stof(text, &end);
        if (end != text.size() || !std::isfinite(v)) throw std::runtime_error("JSON: nonfinite number");
        return v;
    }
};
class JsonParser {
    std::string_view input;
    size_t pos = 0, values = 0;
    [[noreturn]] void fail(const char* what) const { throw std::runtime_error("JSON at byte " + std::to_string(pos) + ": " + what); }
    void whitespace() { while (pos < input.size() && (input[pos]==' ' || input[pos]=='\n' || input[pos]=='\r' || input[pos]=='\t')) ++pos; }
    bool take(char c) { whitespace(); if (pos < input.size() && input[pos] == c) { ++pos; return true; } return false; }
    void need(char c) { if (!take(c)) fail("unexpected character"); }
    uint32_t hex4() {
        uint32_t v=0;
        for (int i=0;i<4;++i) {
            if (pos == input.size()) fail("truncated unicode escape");
            char c=input[pos++]; unsigned d;
            if (c>='0'&&c<='9') d=c-'0'; else if(c>='a'&&c<='f') d=c-'a'+10; else if(c>='A'&&c<='F') d=c-'A'+10; else fail("invalid unicode escape");
            v=v*16+d;
        }
        return v;
    }
    static void utf8(std::string& s, uint32_t c) {
        if(c<128) s+=char(c);
        else if(c<2048) { s+=char(0xc0|(c>>6)); s+=char(0x80|(c&63)); }
        else if(c<65536) { s+=char(0xe0|(c>>12)); s+=char(0x80|((c>>6)&63)); s+=char(0x80|(c&63)); }
        else { s+=char(0xf0|(c>>18)); s+=char(0x80|((c>>12)&63)); s+=char(0x80|((c>>6)&63)); s+=char(0x80|(c&63)); }
    }
    std::string string() {
        need('"'); std::string out;
        while(pos<input.size()) {
            unsigned char c=input[pos++];
            if(c=='"') return out;
            if(c<32) fail("control byte in string");
            if(c!='\\') { out+=char(c); continue; }
            if(pos==input.size()) fail("truncated escape");
            c=input[pos++];
            switch(c) {
            case '"': case '\\': case '/': out+=char(c); break;
            case 'b': out+='\b'; break; case 'f': out+='\f'; break; case 'n': out+='\n'; break; case 'r': out+='\r'; break; case 't': out+='\t'; break;
            case 'u': {
                uint32_t u=hex4();
                if(u>=0xd800 && u<=0xdbff) {
                    if(pos+2>input.size() || input.substr(pos,2)!="\\u") fail("missing low surrogate");
                    pos+=2; uint32_t lo=hex4(); if(lo<0xdc00||lo>0xdfff) fail("invalid low surrogate");
                    u=0x10000+((u-0xd800)<<10)+(lo-0xdc00);
                } else if(u>=0xdc00 && u<=0xdfff) fail("unpaired low surrogate");
                utf8(out,u); break;
            }
            default: fail("invalid escape");
            }
        }
        fail("unterminated string");
    }
    Json value(unsigned depth) {
        if(depth>128 || ++values>2000000) fail("metadata complexity limit");
        whitespace(); if(pos==input.size()) fail("missing value"); Json v;
        if(input[pos]=='{') {
            ++pos; v.kind=Json::Object; if(take('}')) return v;
            do { std::string key=string(); need(':'); if(!v.object.emplace(key,value(depth+1)).second) fail("duplicate object key"); } while(take(','));
            need('}'); return v;
        }
        if(input[pos]=='[') {
            ++pos; v.kind=Json::Array; if(take(']')) return v;
            do { v.array.push_back(value(depth+1)); } while(take(',')); need(']'); return v;
        }
        if(input[pos]=='"') { v.kind=Json::String; v.text=string(); return v; }
        for(auto word : {std::string_view("true"),std::string_view("false"),std::string_view("null")})
            if(input.substr(pos,word.size())==word) { pos+=word.size(); v.kind=word=="null"?Json::Null:Json::Boolean; v.boolean=word=="true"; return v; }
        size_t begin=pos;
        if(input[pos]=='-') ++pos;
        if(pos==input.size()) fail("truncated number");
        if(input[pos]=='0') ++pos;
        else { if(input[pos]<'1'||input[pos]>'9') fail("invalid value"); while(pos<input.size()&&input[pos]>='0'&&input[pos]<='9') ++pos; }
        if(pos<input.size()&&input[pos]=='.') { ++pos; size_t b=pos; while(pos<input.size()&&input[pos]>='0'&&input[pos]<='9') ++pos; if(b==pos) fail("invalid fraction"); }
        if(pos<input.size()&&(input[pos]=='e'||input[pos]=='E')) { ++pos; if(pos<input.size()&&(input[pos]=='+'||input[pos]=='-')) ++pos; size_t b=pos; while(pos<input.size()&&input[pos]>='0'&&input[pos]<='9') ++pos; if(b==pos) fail("invalid exponent"); }
        v.kind=Json::Number; v.text=input.substr(begin,pos-begin); return v;
    }
public:
    explicit JsonParser(std::string_view s) : input(s) {}
    Json parse() { Json v=value(0); whitespace(); if(pos!=input.size()) fail("trailing input"); return v; }
};
inline Json parse_json(std::string_view text) { return JsonParser(text).parse(); }
inline Json read_json(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f) throw std::runtime_error("JSON: cannot open " + path);
    auto size=f.tellg(); if(size<0 || size>64*1024*1024) throw std::runtime_error("JSON: oversized metadata " + path);
    std::string text(static_cast<size_t>(size),'\0'); f.seekg(0);
    if(!f.read(text.data(),size)) throw std::runtime_error("JSON: truncated " + path);
    return parse_json(text);
}
} // namespace strata::artifact
