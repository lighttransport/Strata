#!/usr/bin/env python3
"""Compile the lane/CPU candidates and link an alternate without changing baseline objects.

Use an already-configured Ninja backend build. Other objects come from that build;
this is not a clean rebuild or a substitute for its backend's GPU validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build-dir', type=Path, required=True)
p.add_argument('--out-dir', type=Path, required=True)
p.add_argument('--name', default='strata-glm-decode-candidate')
a = p.parse_args()
build = a.build_dir.resolve()
out = a.out_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
exe = out / a.name
manifest = out / (a.name + '.build.json')
if exe.exists() or manifest.exists():
    p.error('refusing to overwrite an existing executable or manifest')
commands = subprocess.check_output(
    ['ninja', '-t', 'commands', 'strata-glm-decode'], cwd=build, text=True
).splitlines()
objects = {}
sources = {}
repo = Path(__file__).resolve().parents[2]
for relative in ('src/kernels/cpu/iq_avx2_rows.inl', 'include/strata/kernels/cpu/iq_avx2.hpp'):
    sources[relative] = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
for source in ('expert_lane.cpp', 'iq_avx2.cpp'):
    matches = [c for c in commands if ' -c ' in c and c.endswith('/' + source)]
    if len(matches) != 1:
        raise RuntimeError(f'expected one compile command for {source}')
    cmd = shlex.split(matches[0])
    if Path(cmd[0]).name == 'ccache':
        cmd.pop(0)
    obj = out / (a.name + '-' + source + '.o')
    if obj.exists():
        raise RuntimeError(f'refusing to overwrite {obj.name}')
    original = cmd[cmd.index('-o') + 1]
    sources[source] = hashlib.sha256(Path(cmd[-1]).read_bytes()).hexdigest()
    cmd[cmd.index('-o') + 1] = str(obj)
    if '-MF' in cmd:
        cmd[cmd.index('-MF') + 1] = str(obj) + '.d'
    subprocess.run(cmd, cwd=build, check=True)
    objects[source] = (original, str(obj))
cmd = shlex.split(commands[-1].removeprefix(': && ').removesuffix(' && :'))
if any(c in ('&&', ';', '|') for c in cmd):
    raise RuntimeError('unsupported compound link command')
cmd = [c for c in cmd if not c.startswith('-Wl,--dependency-file=')]
cmd[cmd.index('-o') + 1] = str(exe)
old_lane, new_lane = objects['expert_lane.cpp']
if old_lane not in cmd:
    raise RuntimeError('lane object is not in the expected link command')
cmd[cmd.index(old_lane)] = new_lane
cmd.insert(cmd.index('libstrata_kernels_cpu.a'), objects['iq_avx2.cpp'][1])
subprocess.run(cmd, cwd=build, check=True)
manifest.write_text(json.dumps({
    'source_sha256': sources,
    'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
    'scope': 'new lane/CPU objects plus the configured backend build objects',
}, indent=2) + '\n')
print(exe)
