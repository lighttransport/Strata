from pathlib import Path
import subprocess,json
p=Path('build-fp4-local/q3-direct-scale-loads');old=Path('build-fp4-local/q3-signed-code')
s=Path('src/kernels/cpu/q23_avx2.cpp').read_text()
needle='uint32_t a[4];std::memcpy(a,x.scales,12);const uint32_t t=a[2],m=0x0f0f0f0f,u=0x03030303;'
assert s.count(needle)==1
s=s.replace(needle,'''uint32_t a[4];
            if constexpr(NT==1) {
                // Read exactly the 12 scale bytes; early-clobber outputs cannot alias the pointer.
                // The explicit memory input describes the complete read to the optimizer.
                asm("movl 0(%3), %0\\n\\tmovl 4(%3), %1\\n\\tmovl 8(%3), %2"
                    : "=&r"(a[0]), "=&r"(a[1]), "=&r"(a[2])
                    : "r"(x.scales), "m"(*reinterpret_cast<const uint8_t (*)[12]>(x.scales)));
            } else std::memcpy(a,x.scales,12);
            const uint32_t t=a[2],m=0x0f0f0f0f,u=0x03030303;''')
(p/'candidate.cpp').write_text(s);(p/'probe.cpp').write_bytes((old/'probe.cpp').read_bytes())
cmds=[[a.replace(str(old),str(p)) for a in c] for c in json.loads((old/'build.json').read_text())]
with (p/'build.log').open('w') as f:
 for c in cmds:subprocess.run(c,check=True,stdout=f,stderr=subprocess.STDOUT)
(p/'build.json').write_text(json.dumps(cmds,indent=2)+'\n')
for f in ['run.py','analyze.py']:(p/f).write_text((old/f).read_text().replace(old.name,p.name))
