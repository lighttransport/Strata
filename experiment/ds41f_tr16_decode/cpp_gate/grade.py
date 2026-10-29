"""Grade one frozen DS41 decoder log with the parser oracle in a networkless sandbox."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('log', type=pathlib.Path)
p.add_argument('output', type=pathlib.Path)
a = p.parse_args()
if a.output.exists():
    p.error('output already exists')
sys.path.insert(0, str(ROOT.parent / 'tok'))
from ds4tok import load
log = a.log.read_text()
match = re.search(r'^tokens: (.+)$', log, re.M)
if not match:
    p.error('no generated tokens in log')
text = load().decode([int(x) for x in match[1].split()])
blocks = re.findall(r'```(?:cpp|c\+\+)\s*\n(.*?)```', text, re.S)
record = {'log_sha256': hashlib.sha256(a.log.read_bytes()).hexdigest(),
          'passed': False, 'task': 'parse_u64', 'scope': 'one C++ coding task'}
if len(blocks) != 1:
    record['error'] = 'expected exactly one fenced C++ block'
else:
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    oracle = REPO / manifest['oracle']
    if hashlib.sha256(oracle.read_bytes()).hexdigest() != manifest['oracle_sha256']:
        raise ValueError('oracle hash differs')
    with tempfile.TemporaryDirectory(prefix='ds41-cpp-') as tmp:
        work = pathlib.Path(tmp)
        (work / 'generated.cpp').write_text(blocks[0])
        (work / 'test.cpp').write_bytes(oracle.read_bytes())
        sandbox = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
                   '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib',
                   '--ro-bind', '/lib64', '/lib64', '--proc', '/proc', '--dev', '/dev',
                   '--tmpfs', '/tmp', '--bind', tmp, '/work', '--chdir', '/work',
                   '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin']
        for stage, command, timeout in [
            ('compile', ['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                         '-fsanitize=undefined', '-fno-sanitize-recover=all', 'test.cpp', '-o', 'test'], 60),
            ('run', ['/work/test'], 15)]:
            try:
                result = subprocess.run(sandbox + ['prlimit', '--as=1073741824',
                    '--cpu=30', '--fsize=16777216', '--nofile=64', '--'] + command,
                    capture_output=True, text=True, timeout=timeout)
                record[stage] = {'returncode': result.returncode,
                                 'stdout': result.stdout[-4000:], 'stderr': result.stderr[-4000:]}
            except subprocess.TimeoutExpired:
                record[stage] = {'error': 'timeout'}
                break
            if result.returncode:
                break
        record['passed'] = (record.get('run', {}).get('returncode') == 0
                            and record['run']['stdout'].strip() == 'PASS 400532 cases')
with a.output.open('x') as output:
    json.dump(record, output, indent=2)
    output.write('\n')
print(json.dumps(record))
