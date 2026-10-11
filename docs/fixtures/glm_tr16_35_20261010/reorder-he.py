"""After the prefill measurements (STAGE3I_DONE), stop the rest of chain 2 (exploratory screens), run the full HumanEval for the
soft-acceptance control with the guard on (affinity 0.08, Q22/Q4, depth 3, margin 4), and release the upload watchers."""
import os, signal, subprocess, sys, time
from pathlib import Path
ROOT = Path('/mnt/nvme02/work/Strata'); os.chdir(ROOT); D = ROOT / 'build-tr16-35-20261010'; sys.path.insert(0, str(ROOT / 'tools'))
from glm_q2_run_guard import descendants
log = D / 'stage3.out'
while 'STAGE3I_DONE' not in log.read_text(): time.sleep(5)
pids = [int(p) for p in subprocess.run(['pgrep', '-f', 'build-tr16-35-20261010/run-chain2.sh'], capture_output=True, text=True).stdout.split()]
tree = []
for p in pids: tree += [p, *descendants(p)]
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
    for p in tree:
        try: os.kill(p, sig)
        except ProcessLookupError: pass
    time.sleep(10 if sig == signal.SIGINT else 3)
with log.open('a') as f: f.write(f'CHAIN2_STOPPED_AFTER_PREFILL {time.strftime("%H:%M:%S")} killed={tree}\n')
cfg = ROOT / 'build-tr16-goal-20261010/q22-repeat-screen/01-repeat4.json'
p = subprocess.Popen([str(ROOT / 'build-exl3-venv/bin/python'), '-u', str(D / 'quality-ladder.py'), '--config', str(cfg), '--name', 'soft4-aff08-guard4-he',
                      '--track', '35', '--steps', 'full164', '--continue-on-fail', '--deadline', str(int(time.time()) + 3 * 3600)],
                     stdout=log.open('a'), stderr=subprocess.STDOUT, start_new_session=True)
time.sleep(240)   # let the HumanEval pass its idle wait and start the model before the upload hashing begins
with log.open('a') as f: f.write('CHAIN2_DONE\n')
p.wait()
with log.open('a') as f: f.write(f'EXIT {p.returncode} ladder-soft4-he {time.strftime("%H:%M:%S")}\nSOFT4_HE_DONE\n')
