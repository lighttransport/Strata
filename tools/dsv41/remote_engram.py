#!/usr/bin/env python3
"""Manage an opt-in two-host DeepSeek deployment from ignored JSON configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, IndexError):
        return None


def remote_reader_count(cfg):
    readers = cfg.get('remote_readers', 8)
    if type(readers) is not int or not 1 <= readers <= 64:
        raise ValueError('remote_readers must be an integer from 1 to 64')
    return readers


def remote(cfg, action):
    if action == 'start':
        remote_reader_count(cfg)
    # The source string is fixed; configuration travels as one shell-quoted JSON argument.
    script = '''import json,os,pathlib,signal,subprocess,sys
c=json.loads(sys.argv[1]); action=sys.argv[2]; readers=c.get('remote_readers')
d=pathlib.Path(c.get('remote_state_dir','~/.cache/strata-engram-run')).expanduser();d.mkdir(parents=True,exist_ok=True)
p=d/'process.json'
def identity(pid):
 try:
  fields=pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
  return fields[19] if fields[0]!='Z' else None
 except (OSError,IndexError):return None
s=json.loads(p.read_text()) if p.exists() else {}
alive=bool(s.get('pid') and s.get('start') and identity(s['pid'])==s.get('start'))
if action=='stop' and alive:
 os.killpg(s['pid'],signal.SIGTERM);alive=False;p.unlink()
if action=='start' and alive and readers is not None and s.get('readers')!=readers:
 raise SystemExit('owned Engram service has a different or unknown reader count; stop it before changing remote_readers')
if action=='start' and not alive:
 log=open(d/'service.log','ab',buffering=0)
 env=dict(os.environ,UCOMM_BACKEND='ib',UCOMM_TIMEOUT_MS='60000')
 actual_readers=readers if readers is not None else (None if 'STRATA_ENGRAM_READERS' in env else 8)
 if readers is not None:env['STRATA_ENGRAM_READERS']=str(readers)
 proc=subprocess.Popen([str(pathlib.Path(c['remote_exe']).expanduser()),'serve',c['remote_file'],c['engram_endpoint']],stdout=log,stderr=log,start_new_session=True,env=env)
 s={'pid':proc.pid,'start':identity(proc.pid),'readers':actual_readers};p.write_text(json.dumps(s));alive=True
print(json.dumps({'running':alive,'pid':s.get('pid'),'readers':s.get('readers'),'log':str(d/'service.log')}))
'''
    args = ['ssh']
    if cfg.get('ssh_config'):
        args += ['-F', str(Path(cfg['ssh_config']).expanduser())]
    args += ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', cfg['ssh_host'],
             'python3 -c ' + shlex.quote(script) + ' ' + shlex.quote(json.dumps(cfg)) + ' ' + shlex.quote(action)]
    return json.loads(subprocess.check_output(args, text=True, timeout=30))


def verify_local(cfg, state):
    catalog = json.loads((ROOT/'data/dsv41/coder-set.json').read_text())
    cache_path = state/'verified-local.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for name in catalog['shards']:
        path = Path(cfg['model_dir'])/name
        st = path.stat()
        expected_size, expected_hash = catalog['files'][name]
        if st.st_size != expected_size:
            raise ValueError(f'{name}: wrong size')
        signature = [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, expected_hash]
        if cache.get(name) != signature:
            print(f'Verifying {name}', flush=True)
            with path.open('rb') as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(block)
                actual = digest.hexdigest()
            if actual != expected_hash:
                raise ValueError(f'{name}: SHA-256 mismatch')
            cache[name] = signature
            cache_path.write_text(json.dumps(cache))


def describe(cfg, state):
    result = subprocess.check_output([cfg['helper'], 'describe', cfg['engram_endpoint']],
                                     env=dict(os.environ, UCOMM_BACKEND='ib', UCOMM_TIMEOUT_MS='60000'), timeout=90)
    descriptor = json.loads(result)
    catalog = json.loads((ROOT/'data/dsv41/coder-set.json').read_text())
    size, digest = catalog['files'][catalog['engram_shard']]
    if (descriptor['size'], descriptor['sha256']) != (size, digest):
        raise ValueError('remote Engram does not match the pinned Coder source')
    path = state/'engram.json'
    path.write_text(json.dumps(descriptor))
    return path


def api_command(runfile, port):
    return [sys.executable, str(ROOT/'serve/server.py'), '--engine', 'strata', '--config', str(runfile),
            '--host', '127.0.0.1', '--port', str(port)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['preflight', 'start', 'status', 'stop'])
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    state = Path(cfg['state_dir']).resolve()
    state.mkdir(parents=True, exist_ok=True)
    process_file = state/'server.json'
    saved = json.loads(process_file.read_text()) if process_file.exists() else {}
    alive = bool(saved.get('pid') and saved.get('start') and identity(saved['pid']) == saved.get('start'))
    if args.action == 'status':
        print(json.dumps({'api_process_running': alive, 'remote': remote(cfg, 'status'),
                          'api_log': str(state/'api.log')}))
        return
    if args.action == 'stop':
        if alive:
            os.killpg(saved['pid'], signal.SIGTERM)
            process_file.unlink()
        print(json.dumps(remote(cfg, 'stop')))
        return
    if alive:
        print('API process already running; use status or stop.')
        return
    verify_local(cfg, state)
    if not Path('/dev/kfd').exists():
        raise RuntimeError('HIP GPU device /dev/kfd is unavailable')
    if not Path(cfg['engine_exe']).is_file() or not Path(cfg['helper']).is_file():
        raise RuntimeError('build the HIP engine and helper first')
    if args.action == 'preflight':
        print(json.dumps({'local_files_verified': True, 'remote': remote(cfg, 'status')}))
        return
    print(json.dumps(remote(cfg, 'start')), flush=True)
    # Source hashing runs before bootstrap. Wait for this owned service's TCP listener.
    # Do not probe with TCP connect: that would consume ucomm's bootstrap session.
    descriptor = None
    for attempt in range(10):
        try:
            descriptor = describe(cfg, state)
            break
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            if not remote(cfg, 'status')['running']:
                raise RuntimeError('remote service exited during source verification')
    if descriptor is None:
        raise RuntimeError('remote source verification did not finish')
    pack = Path(cfg['pack_dir']).resolve()
    subprocess.run([sys.executable, str(ROOT/'tools/dsv41/pack_gguf.py'), '--model', cfg['model_dir'],
                    '--out', str(pack), '--engram-remote-manifest', str(descriptor)], check=True)
    env = dict(cfg.get('env', {}))
    env['DS4_ENGRAM_CACHE_MB'] = '256'
    run = {'exe': cfg['engine_exe'], 'arch': 'deepseek41', 'model_name': 'deepseek-v4.1-flash-coder',
           'tokenizer': str(pack/'tokenizer'), 'context': 32768, 'parallel': 1, 'drafting': 'off',
           'engine_ready_s': 3600, 'log': str(state/'engine.log'), 'env': env,
           'args': ['--pack', str(pack), '--gpus', '0', '--max-context', '32768',
                    '--prefill-chunk', '8192', '--cpu-threads', '15', '--reserve-mib', '3072', '--experts', '6', '--parallel', '1',
                    '--dspark', 'none', '--vision', 'off', '--mmap-experts', 'on', '--ssd-cache', '0',
                    '--tp', '1', '--threads', 'off', '--expert-profile', str(ROOT/'data/dsv41/profile-coder.bin'),
                    '--profile-readonly', '--engram-remote', cfg['engram_endpoint']]}
    runfile = state/'run.json'
    runfile.write_text(json.dumps(run, indent=2))
    with (state/'api.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(api_command(runfile, cfg.get('port', 8080)),
                                cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
    process_file.write_text(json.dumps({'pid': proc.pid, 'start': identity(proc.pid)}))
    print(f'API process started; readiness and errors: {state / "api.log"}')


if __name__ == '__main__':
    main()
