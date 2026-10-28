import sys
sys.path.insert(0, sys.argv[1]); from ds4tok import load
t = load(); src, skip_lines, n, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
txt = ''.join(open(src, encoding='utf-8', errors='replace').readlines()[skip_lines:])
ids = t.encode(txt).ids[:n]
assert len(ids) == n, (src, len(ids))
open(out, 'w').write(' '.join(map(str, ids)) + '\n')
print(out, len(ids), repr(t.decode(ids[:40])))
