"""Copy completed evidence (configs, summaries, results, logs, patches, drivers) into docs/fixtures/glm_tr16_35_20261010 with a SHA-256 manifest.
Never promotes a profile. Skips source trees, build directories, binaries and large logs."""
import hashlib, json, shutil, time
from pathlib import Path
R = Path(__file__).resolve().parent; OUT = R.parent / 'docs/fixtures/glm_tr16_35_20261010'; OUT.mkdir(parents=True, exist_ok=True)
SKIP_DIRS = {'merged-source', 'merged-cuda', 'merged-hip', 'stream-source', 'stream-cuda', 'stream-hip', '__pycache__'}
KEEP = {'.json', '.py', '.sh', '.log', '.stdout', '.txt', '.patch', '.md', '.ids'}
manifest = {}
for p in sorted(R.rglob('*')):
    if not p.is_file() or p.suffix not in KEEP: continue
    rel = p.relative_to(R)
    if set(rel.parts) & SKIP_DIRS or rel.name.startswith('merged-test_'): continue
    if rel.name.startswith('HumanEval_') or rel.name.endswith('.hf.log'): continue   # per-task generations stay in the research dir
    if p.stat().st_size > 64 * 2**20: continue
    dest = OUT / rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    manifest[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
(OUT / 'manifest.json').write_text(json.dumps(dict(exported=time.strftime('%Y-%m-%dT%H:%M:%S%z'), files=manifest, quality_qualified=False,
    note='Research evidence from build-tr16-35-20261010; no profile promoted; merged source/binaries reproducible from merged.patch on e5232ca3'), indent=2) + '\n')
(OUT / 'README.txt').write_text('TR16 35/45 tokens/s follow-up (2026-10-10). See goal.json for the window and gates, s*/summary.json for screens, all-screens.json for the table, s0-step-trace/decomposition.json for the host-timeline decomposition. Nothing here qualifies a profile; quality ladder scripts were prepared but not run in this window.\n')
print(len(manifest), 'files exported to', OUT)
