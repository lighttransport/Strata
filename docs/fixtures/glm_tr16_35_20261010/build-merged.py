"""Build the merged isolated decoder (cold-prune features + draft-route repeat guard). PREPARED; not run in the 4-hour window.

Compilation is CPU-heavy: never run while a timing screen is active. Steps: patch reproduction check,
CUDA configure/build with tests, HIP configure/build (compile only), CUDA parity suites, routing oracles,
MMVQ IL parity, policy unit tests against the merged header, marker JSON with SHA-256 values.
"""
import difflib, hashlib, json, os, signal, subprocess, sys, tempfile, time
from pathlib import Path
ROOT = Path('/mnt/nvme02/work/Strata'); os.chdir(ROOT)
R = ROOT / 'build-tr16-35-20261010'; OLD = ROOT / 'build-tr16-goal-20261010'; sys.path.insert(0, str(ROOT / 'tools'))
from glm_tr16_calibrate import wait_idle
from glm_q2_run_guard import descendants
goal = json.loads((R / 'goal.json').read_text())
end = int(sys.argv[1]) if len(sys.argv) > 1 else goal['deadline']   # pass a fresh deadline (epoch seconds) in the next window
if time.time() + 3600 >= end: raise SystemExit('Need at least one hour before the deadline for the merged build')
s = R / 'merged-source'; cu = R / 'merged-cuda'; hip = R / 'merged-hip'; gg = OLD / 'ggml-reference'
assert s.exists() and gg.exists()
# Reproduce the patch against e5232ca3 from the changed files of both parents.
files = list(dict.fromkeys(json.loads((OLD / 'cold-prune-patch-check.json').read_text())['files'] +
                           json.loads((OLD / 'draft-route-patch-check.json').read_text())['files']))
patch = ''; original = {}
for name in files:
    old = subprocess.run(['git', 'show', 'e5232ca3:' + name], capture_output=True, text=True)
    original[name] = old.stdout if old.returncode == 0 else None
    patch += ''.join(difflib.unified_diff(old.stdout.splitlines(True) if old.returncode == 0 else [], (s / name).read_text().splitlines(True),
                                          fromfile='a/' + name if old.returncode == 0 else '/dev/null', tofile='b/' + name))
(R / 'merged.patch').write_text(patch)
with tempfile.TemporaryDirectory(prefix='strata-merged-patch-') as tmp:
    for name, value in original.items():
        p = Path(tmp) / name; p.parent.mkdir(parents=True, exist_ok=True)
        if value is not None: p.write_text(value)
    subprocess.run(['patch', '-p1', '--batch', '-i', str(R / 'merged.patch')], cwd=tmp, check=True, stdout=subprocess.DEVNULL)
    for name in files: assert (Path(tmp) / name).read_bytes() == (s / name).read_bytes(), name
(R / 'merged-patch-check.json').write_text(json.dumps(dict(applies_to='e5232ca3', reproduces_all_changed_files=True,
    patch_sha256=hashlib.sha256(patch.encode()).hexdigest(), files=files,
    parents=['cold-prune.patch', 'draft-route-incremental.patch'], manual_merge='src/kernels/glm_route_prune_test.cpp oracle line'), indent=2) + '\n')


def identity(pid):
    try:
        f = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split(); return f[19] if f[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError): return None


def run(name, command, seconds=1800, cwd=None):
    if time.time() + 120 >= end: raise SystemExit('Deadline before ' + name)
    print('START', name, time.strftime('%H:%M:%S'), flush=True)
    with (R / (name + '.log')).open('x') as log:
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=cwd)
        try: code = p.wait(timeout=min(seconds, end - time.time() - 30))
        except subprocess.TimeoutExpired:
            owned = {c: identity(c) for c in descendants(p.pid)}
            for sig in (signal.SIGINT, signal.SIGKILL):
                for c, stamp in owned.items():
                    if stamp and identity(c) == stamp:
                        try: os.kill(c, sig)
                        except ProcessLookupError: pass
                if sig == signal.SIGINT: time.sleep(10)
            p.wait(); raise
    if code: raise RuntimeError(f'{name} failed {code}')
    print('PASS', name, flush=True)


wait_idle(900)
for name, b, args in [('merged-cuda', cu, ['-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_ENABLE_HIP=OFF', '-DCMAKE_CUDA_ARCHITECTURES=120', '-DSTRATA_BUILD_TESTS=ON']),
                      ('merged-hip', hip, ['-DSTRATA_ENABLE_CUDA=OFF', '-DSTRATA_ENABLE_HIP=ON', '-DCMAKE_HIP_ARCHITECTURES=gfx1201', '-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0', '-DSTRATA_BUILD_TESTS=OFF'])]:
    run(name + '-configure', ['cmake', '-S', str(s), '-B', str(b), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DSTRATA_GGML_DIR=' + str(gg), *args])
    targets = ['strata-glm-decode']
    if b == cu: targets += ['mmvq_il_parity', 'glm_route_prune_test', 'glm_parity', 'glm_q23_parity', 'glm_q8_test', 'glm_mailbox_test', 'glm_prefill_parity']
    run(name + '-build', ['cmake', '--build', str(b), '--target', *targets, '-j', '12'], 3600)
# Policy unit tests against the merged header (same sources as draft-policy-unit-tests.json).
for test in ('test_repeat_guard', 'test_bounded_accept'):
    run('merged-' + test + '-compile', ['c++', '-std=c++20', '-O2', '-I' + str(s / 'include'), str(OLD / (test + '.cpp')), '-o', str(R / ('merged-' + test))], 600)
    run('merged-' + test, [str(R / ('merged-' + test))], 600)
wait_idle(900)
for name in ('glm_parity', 'glm_q23_parity', 'glm_q8_test', 'glm_mailbox_test', 'glm_prefill_parity'):
    run('merged-' + name, [str(cu / name)], 300)
run('merged-routing-oracles', ['ctest', '--test-dir', str(cu), '-R', '^glm_route_prune_', '--output-on-failure'], 600)
run('merged-mmvq-il-parity', [str(cu / 'mmvq_il_parity')], 600)
files_out = [cu / 'strata-glm-decode', hip / 'strata-glm-decode', R / 'merged.patch', R / 'merged-routing-oracles.log', R / 'merged-mmvq-il-parity.log']
(R / 'merged-build-complete.json').write_text(json.dumps(dict(cuda_parities_passed=True, hip_build_passed=True, route_oracles_passed=True,
    mmvq_il_parity_passed=True, policy_unit_tests_passed=True, model_controls_passed=False, quality_qualified=False, time=time.time(),
    sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files_out}), indent=2) + '\n')
print('MERGED_BUILD_COMPLETE: next run screen.py --name s3-merged-control with base-soft4-merged.json (guard 0 then guard 4) for model parity', flush=True)
