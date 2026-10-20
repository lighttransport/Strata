from pathlib import Path
import json,subprocess
p=Path('build-fp4-local/fp4-compact-scale-products');s=Path('src/kernels/cpu/nvfp4_avx2.cpp').read_text()
s=s.replace('    __m256i correction[kMaxBlocks * 2];','    __m256i correction[kMaxBlocks * 2];\n    alignas(32) float products[kMaxBlocks * 2][129];')
a=s.index('struct OffsetActs : Acts');b=s.index('template <int NT, bool PackedK32',a);part=s[a:b]
needle='''        }
    }
};''';assert part.count(needle)==1
part=part.replace(needle,'''        }
        if (safe) for (int ib = 0; ib < 2 * nb; ++ib)
            for (int scale = 0; scale < 129; ++scale)
                products[ib][scale] = kUe4m3.v[scale == 128 ? 127 : (scale | 128)] * dy[ib];
    }
};''');s=s[:a]+part+s[b:]
a=s.index('inline void row_dot(const uint8_t* row, const OffsetActs&');b=s.index('// Distinct types',a);part=s[a:b]
part=part.replace('kUe4m3.v[x[ib].d[1]]','a.products[2*ib][x[ib].d[1]]').replace('kUe4m3.v[x[ib].d[0]]','a.products[2*ib][x[ib].d[0]]').replace('kUe4m3.v[x[ib].d[3]]','a.products[2*ib+1][x[ib].d[3]]').replace('kUe4m3.v[x[ib].d[2]]','a.products[2*ib+1][x[ib].d[2]]')
part=part.replace('_mm256_mul_ps(u01, _mm256_broadcast_ss(a.dy + 2*ib))','u01').replace('_mm256_mul_ps(u23, _mm256_broadcast_ss(a.dy + 2*ib + 1))','u23');s=s[:a]+part+s[b:];s=s.replace('a.products[2*ib][x[ib].d[', 'a.products[2*ib][compact_scale(x[ib].d[').replace('a.products[2*ib+1][x[ib].d[', 'a.products[2*ib+1][compact_scale(x[ib].d[');
for i in range(4): s=s.replace('compact_scale(x[ib].d['+str(i)+']]', 'compact_scale(x[ib].d['+str(i)+'])]')
s=s.replace('struct OffsetActs : Acts', 'inline unsigned compact_scale(unsigned code) { return code == 127 ? 128 : (code & 127); }\nstruct OffsetActs : Acts');(p/'candidate.cpp').write_text(s)
(p/'probe.cpp').write_text(Path('build-fp4-local/fp4-offset-aligned-acts/probe.cpp').read_text().replace('b.d=0x1c00','b.d=uint16_t(0x1000+rng()%0x3000)'))
cmds=[[a.replace('fp4-offset-aligned-acts','fp4-compact-scale-products') for a in c] for c in json.loads(Path('build-fp4-local/fp4-offset-aligned-acts/build.json').read_text())]
with (p/'build.log').open('w') as f:
 for c in cmds:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
(p/'build.json').write_text(json.dumps(cmds,indent=2)+'\n')
for name in ('run.py','summarize.py'):(p/name).write_text(Path('build-fp4-local/fp4-offset-aligned-acts',name).read_text().replace('fp4-offset-aligned-acts','fp4-compact-scale-products'))
