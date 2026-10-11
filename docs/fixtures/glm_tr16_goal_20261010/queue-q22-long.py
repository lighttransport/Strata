"""Long-generation diagnostic for the completed30.44 screen, after profiling.

The low track selects the30tok/s speed threshold only. Its precision pilot
already failed provisional low-loss gates; this never qualifies that track.
"""
import json,subprocess,time
from pathlib import Path
r=Path('/mnt/nvme02/work/Strata/build-tr16-goal-20261010')
end=json.loads((r/'goal.json').read_text())['deadline']
def identity(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return f[19] if f[0]!='Z' else None
 except (FileNotFoundError,ProcessLookupError):return None
pid=3278141
assert b'build-tr16-goal-20261010/queue-q22-profile.py' in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
stamp=identity(pid);assert stamp
print('WAIT_FOR_PROFILE_BEFORE_LONG',pid,stamp,flush=True)
while identity(pid)==stamp:
 if time.time()+5100>=end:raise SystemExit('Insufficient long-validation reserve')
 time.sleep(5)
status=json.loads((r/'nsys-q22-q4/status.json').read_text())
assert status['capture_complete'], 'Incomplete profile: inspect before starting another run'
result=r/'q22-cache-screen/depth3-soft4-cache6656/000-baseline.result.json'
print('START_LONG_DIAGNOSTIC_KNOWN_LOW_LOSS_PILOT_FAILURE',flush=True)
code=subprocess.call([str(r.parent/'build-exl3-venv/bin/python'),'-u',str(r/'screen-finalist-long.py'),'--screen-result',str(result),'--track','low','--name','q22-soft4-diagnostic'])
print('LONG_DIAGNOSTIC_FINISHED',code,flush=True)
raise SystemExit(code)
