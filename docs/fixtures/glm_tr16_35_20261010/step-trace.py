"""Diagnostic only: STEP_TRACE=2 decomposition of the soft4 control. Never cite its tok/s."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path('/mnt/nvme02/work/Strata'); os.chdir(ROOT); R = ROOT / 'build-tr16-35-20261010'
sys.path.insert(0, str(ROOT / 'tools'))
from glm_tr16_calibrate import command, wait_idle, digest
goal = json.loads((R / 'goal.json').read_text())
assert time.time() + 600 < goal['deadline']
c = json.loads((R / 'base-soft4.json').read_text())
out = R / 's0-step-trace'; out.mkdir(exist_ok=False)
prompt = ROOT / 'build-tr16-20261009/prompts2k/train-code.ids'
cmd = command(c, prompt, 128, 1)
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
env.update(c['env']); env['STRATA_GLM_STEP_TRACE'] = '2'
(out / 'manifest.json').write_text(json.dumps(dict(command=cmd, env=c['env'], step_trace='2', decoder_sha256=digest(c['exe']),
    prompt_sha256=digest(prompt), scope='diagnostic decomposition; instrumentation perturbs timing; not a throughput record'), indent=2) + '\n')
wait_idle(900)
guard = [sys.executable, 'tools/glm_q2_run_guard.py', '--output', str(out / 'run'), '--min-gpu-free-mib', '512',
         '--require-gpu-idle', '--require-idle', '--', *cmd]
print('START step-trace', time.strftime('%H:%M:%S'), flush=True)
code = subprocess.call(guard, env=env)
log = (out / 'run.log').read_text()
lines = [l for l in log.splitlines() if l.startswith(('STEP_TRACE', 'SPECULATIVE', 'MTP_TIMING', 'CPU_EXPERT', 'DECODE_CACHE'))]
(out / 'decomposition.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines)); print('EXIT', code)
