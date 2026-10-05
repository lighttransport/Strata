"""Compile generated coding fixtures and check them against independent oracles."""
import argparse
import csv
import io
import json
import pathlib
import random
import re
import subprocess
import tempfile

DRIVER = r'''
#include <iostream>
#include <string>
#include <vector>
static std::string unhex(const std::string& s) {
    std::string r;
    for(size_t i=0;i<s.size();i+=2) r+=char(std::stoi(s.substr(i,2),nullptr,16));
    return r;
}
static std::string hex(const std::string& s) {
    std::string r; const char* digits="0123456789abcdef";
    for(unsigned char c:s) {r+=digits[c>>4]; r+=digits[c&15];} return r;
}
'''
PRIME = r'''
#include <vector>
#include <iostream>
#include <climits>
int main() {
    const unsigned limit=400532;std::vector<bool> sieve(limit+1,true);sieve[0]=sieve[1]=false;
    for(unsigned i=2;i<=limit/i;++i)if(sieve[i])for(unsigned j=i*i;j<=limit;j+=i)sieve[j]=false;
    for(unsigned i=0;i<=limit;++i)if(is_prime(i)!=sieve[i])return 1;
    for(auto pair:{std::pair<unsigned,bool>{4294967291U,true},{UINT_MAX,false},{UINT_MAX-1,false},{2147483647U,true},{2147483646U,false}})
        if(is_prime(pair.first)!=pair.second)return 2;
    std::cout<<"400538";
}
'''


def check(root, fixture):
    text = (root / (fixture + ".output.md")).read_text()
    blocks = re.findall(r"```(?:cpp|c\+\+)\s*\n(.*?)```", text, re.S)
    if len(blocks) != 1:
        raise ValueError(f"{fixture}: expected one complete C++ code block")
    driver = PRIME if fixture == "prime" else DRIVER + (r'''
int main(){std::string s;while(std::getline(std::cin,s))std::cout<<hex(json_escape(unhex(s)))<<'\n';}
''' if fixture == "json_escape" else r'''
int main(){std::string s;while(std::getline(std::cin,s)) {
    std::vector<std::string> out{"sentinel"};bool ok=parse_csv(unhex(s),out);
    std::cout<<ok<<'|'<<out.size();for(const auto& field:out)std::cout<<'|'<<hex(field);std::cout<<'\n';
}}
''')
    with tempfile.TemporaryDirectory(prefix="glm-coding-check-") as temp:
        cpp, exe = pathlib.Path(temp) / "check.cpp", pathlib.Path(temp) / "check"
        cpp.write_text(blocks[0] + driver)
        subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True, capture_output=True, text=True)
        if fixture == "prime":
            result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=10)
            return int(result.stdout)
        rng = random.Random(1950)
        if fixture == "json_escape":
            cases = [b"", bytes(range(256)), bytes(c for c in range(32, 256) if c not in (34, 92)), b'"\\\x00\n\r\t']
            cases += [bytes(rng.randrange(256) for _ in range(rng.randrange(257))) for _ in range(2048)]
        else:
            cases = [("", [""]), (",", ["", ""]), ("a,", ["a", ""]), ('"a,b","c""d",', ["a,b", 'c"d', ""])]
            for _ in range(2048):
                fields = ["".join(rng.choice('abc,\" \x00') for _ in range(rng.randrange(12))) for _ in range(rng.randrange(1, 8))]
                buffer = io.StringIO();csv.writer(buffer, lineterminator="\n").writerow(fields)
                record = buffer.getvalue()[:-1]
                # Python's CSV parser is the independent valid-record oracle.
                assert next(csv.reader([record], strict=True)) == fields
                cases.append((record, fields))
            cases += [(s, None) for s in ('"', 'a"b', '"a"x', '"a",b"', '"a" ', '"""')]
        inputs = cases if fixture == "json_escape" else [s.encode() for s, _ in cases]
        result = subprocess.run([str(exe)], input="\n".join(s.hex() for s in inputs) + "\n", check=True, capture_output=True, text=True, timeout=10)
        lines = result.stdout.splitlines()
        if len(lines) != len(cases):
            raise ValueError("missing oracle results")
        for case, line in zip(cases, lines):
            if fixture == "json_escape":
                raw = bytes.fromhex(line)
                if all(c >= 32 and c not in (34, 92) for c in case) and raw != case:
                    raise ValueError("JSON escaping changed a byte that needed no escape")
                escaped = raw.decode("latin1")
                if json.loads('"' + escaped + '"') != case.decode("latin1"):
                    raise ValueError("JSON round trip failed")
            else:
                _, expected = case
                ok, count, *fields = line.split("|")
                fields = [bytes.fromhex(f).decode() for f in fields]
                if int(count) != len(fields) or bool(int(ok)) != (expected is not None) or fields != (expected if expected is not None else ["sentinel"]):
                    raise ValueError(f"CSV oracle/failure preservation mismatch: {case!r}")
        return len(cases)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=pathlib.Path)
    ap.add_argument("--fixtures", nargs="+", choices=("prime", "json_escape", "csv"), default=["prime", "json_escape", "csv"])
    args = ap.parse_args()
    results = {fixture: dict(passed_cases=check(args.directory, fixture)) for fixture in args.fixtures}
    (args.directory / "quality.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
