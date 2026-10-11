"""Run task quality only after successful full-corpus completion within PPL10%."""
import json,subprocess,time
from pathlib import Path
r=Path('/mnt/nvme02/work/Strata/build-tr16-goal-20261010')
end=json.loads((r/'goal.json').read_text())['deadline']
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
pid=3298523
assert b'build-tr16-goal-20261010/quality-q22-full.py' in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
stamp=identity(pid);assert stamp
print('WAIT_FULL_CORPUS_BEFORE_HE',pid,stamp,flush=True)
while identity(pid)==stamp:
 if time.time()+4200>=end:raise SystemExit('Insufficient full-task validation reserve')
 time.sleep(5)
assert json.loads((r/'q22-full-corpus/complete.json').read_text())['full_corpus_complete']
rows=json.loads((r/'q22-full-corpus/summary.json').read_text())
assert len(rows)==1 and rows[0]['complete']
if rows[0]['ppl_increase_percent']>10:raise SystemExit('Full-corpus PPL exceeds exploratory moderate budget; no task run')
print('START_FULL164_HE',flush=True)
code=subprocess.call([str(r.parent/'build-exl3-venv/bin/python'),'-u',str(r/'quality-q22-humaneval.py')])
print('FULL164_HE_FINISHED',code,flush=True)
raise SystemExit(code)
