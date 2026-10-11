"""Serialize a diagnostic behind the already-running combination; never restart it."""
import json, subprocess, time
from pathlib import Path
r=Path('/mnt/nvme02/work/Strata/build-tr16-goal-20261010')
end=json.loads((r/'goal.json').read_text())['deadline']
def identity(pid):
 try:
  stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return stat[19] if stat[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
pid=3273509
args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert b'build-tr16-goal-20261010/screen-q22-cold-combination.py' in args
stamp=identity(pid);assert stamp
print('WAIT_FOR_COMBINATION',pid,stamp,flush=True)
while identity(pid)==stamp:
 if time.time()+5400>=end:raise SystemExit('Insufficient profiling and validation reserve')
 time.sleep(5)
marker=r/'q22-cold-combination/complete.json'
assert marker.exists(), 'Combination ended without completion; inspect before profiling'
assert json.loads(marker.read_text())['screen_cases']==1
rows=json.loads((r/'q22-cold-combination/summary.json').read_text())
assert len(rows)==1 and rows[0]['clean'] and rows[0]['exit_code']==0
print('COMBINATION_COMPLETE_START_PROFILE',flush=True)
code=subprocess.call([str(r.parent/'build-exl3-venv/bin/python'),'-u',str(r/'profile-q22-followup.py')])
print('PROFILE_FINISHED',code,flush=True)
raise SystemExit(code)
