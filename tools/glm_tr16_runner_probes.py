#!/usr/bin/env python3
"""Component probes for the GLM runner simulator on TR16 (tools/sim/glm_runner.py inputs).

Runs, in order and one at a time: NUMA-local CPU expert lanes (flow bench), host bounce latency and
shared-mapping registration on both cards (tools/glm_transfer_probe.cu), worker loopback round trips
for the CUDA and HIP workers, the draft-step cost with CPU and GPU draft experts, and a per-slot GPU
decode decomposition (nsys on the CUDA primary, rocprofv3 on the HIP primary). Each step writes a
receipt into the output directory; `summary.json` records commands, exit codes and elapsed time.

Every step waits for an idle machine first. None of these numbers is a throughput qualification.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROCM = '/opt/rocm/core-10.1'
REPO = pathlib.Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def wait_idle(max_load=2.0, timeout=1800):
    start = time.monotonic()
    while True:
        load = os.getloadavg()[0]
        # Anchored at the start of the command line so that shells whose -c text mentions these names do not match.
        busy = subprocess.run(['pgrep', '-f', r'^(numactl .* )?[^ ]*(strata-glm-decode|strata-glm-gpu-worker)( |$)|^[^ ]*python[^ ]* [^ ]*glm_mixed_bench\.py'],
                              capture_output=True, text=True).stdout.strip()
        if load < max_load and not busy:
            return load
        if time.monotonic() - start > timeout:
            raise SystemExit(f'machine not idle after {timeout}s (load {load:.1f}, busy {busy!r})')
        time.sleep(15)


class Driver:
    def __init__(self, out, env):
        self.out = out; self.env = env; self.steps = []
        out.mkdir(parents=True, exist_ok=True)

    def run(self, name, cmd, env=None, stdout=None, timeout=3600, cwd=None, numa=None):
        wait_idle()
        full = (['numactl', *numa] if numa else []) + [str(x) for x in cmd]
        start = time.monotonic()
        with open(self.out / f'{name}.stdout', 'w') as so, open(self.out / f'{name}.stderr', 'w') as se:
            try:
                rc = subprocess.run(full, env=env or self.env, stdout=so, stderr=se, timeout=timeout, cwd=cwd).returncode
            except subprocess.TimeoutExpired:
                rc = -1
        rec = dict(name=name, command=full, returncode=rc, elapsed_seconds=time.monotonic() - start)
        self.steps.append(rec)
        print(f'{name}: rc={rc} {rec["elapsed_seconds"]:.0f}s', flush=True)
        (self.out / 'summary.json').write_text(json.dumps(dict(steps=self.steps), indent=1) + '\n')
        return rc

    def stdout(self, name):
        return (self.out / f'{name}.stdout').read_text()


def flow_rate(text):
    m = re.findall(r'GB_s=([\d.]+)', text)
    return float(m[-1]) if m else None


def probe_cpu_lanes(d, flow_bench):
    env = dict(d.env, STRATA_GLM_CANON='1')
    results = {}
    for width in (1, 2):
        for node in (0, 1):
            name = f'flow-node{node}-w{width}'
            d.run(name, [flow_bench, width, 96, 300, 'flow', 8], env=env, numa=[f'--cpunodebind={node}', f'--membind={node}'])
            results[name] = dict(gb_s=flow_rate(d.stdout(name)), text=d.stdout(name)[-400:])
        # both nodes at once: two lanes of 8 threads
        wait_idle()
        procs = []
        for node in (0, 1):
            name = f'flow-both-node{node}-w{width}'
            so = open(d.out / f'{name}.stdout', 'w')
            procs.append((name, so, subprocess.Popen(['numactl', f'--cpunodebind={node}', f'--membind={node}', str(flow_bench), str(width), '96', '300', 'flow', '8'], env=env, stdout=so, stderr=subprocess.STDOUT)))
        for name, so, p in procs:
            p.wait(); so.close()
            results[name] = dict(gb_s=flow_rate(d.stdout(name)), returncode=p.returncode)
        name = f'flow-interleave-w{width}'
        d.run(name, [flow_bench, width, 96, 300, 'flow', 15], env=env, numa=['--interleave=all'])
        results[name] = dict(gb_s=flow_rate(d.stdout(name)))
    (d.out / 'cpu-lanes.json').write_text(json.dumps(dict(schema='strata.glm.cpu_lanes.v1', flow_bench=str(flow_bench), sha256=sha(flow_bench),
        scope='Synthetic Q2_K/Q3_K expert flow bench (96 experts, 300 layers, 8 routes per token); node-local 8-thread lanes alone and concurrently, and the 15-thread interleaved reference',
        results=results), indent=1) + '\n')


def probe_transfers(d):
    probes = {}
    for vendor, compiler, extra in (('cuda', 'nvcc', []), ('hip', f'{ROCM}/bin/hipcc', ['-DSTRATA_PROBE_HIP'])):
        binary = d.out / f'probe-{vendor}'
        rc = d.run(f'build-probe-{vendor}', [compiler, '-O3', *extra, REPO / 'tools/glm_transfer_probe.cu', '-o', binary])
        if rc == 0:
            probes[vendor] = binary
    bounce = {}
    for vendor, binary in probes.items():
        for size in (131072, 1048576, 16777216, 67108864, 134217728):
            name = f'bounce-{vendor}-{size}'
            d.run(name, [binary, 'bounce', size, 400 if size >= 16777216 else 2000])
            try:
                bounce[name] = json.loads(d.stdout(name).strip().splitlines()[-1])
            except (ValueError, IndexError):
                bounce[name] = dict(error=d.stdout(name)[-300:])
    register = {}
    if probes:
        size = 64 << 20
        fd = os.memfd_create('strata-register-probe'); os.ftruncate(fd, size)
        for vendor, binary in probes.items():
            name = f'register-{vendor}'
            wait_idle()
            with open(d.out / f'{name}.stdout', 'w') as so:
                rc = subprocess.run([str(binary), 'register', str(fd), str(size)], env=d.env, stdout=so, stderr=subprocess.STDOUT, pass_fds=(fd,)).returncode
            register[name] = dict(returncode=rc, text=(d.out / f'{name}.stdout').read_text()[-400:])
        if len(probes) == 2:
            wait_idle()
            procs = []
            for vendor, binary in probes.items():
                name = f'register-both-{vendor}'
                so = open(d.out / f'{name}.stdout', 'w')
                procs.append((name, so, subprocess.Popen([str(binary), 'register', str(fd), str(size)], env=d.env, stdout=so, stderr=subprocess.STDOUT, pass_fds=(fd,))))
            for name, so, p in procs:
                p.wait(); so.close()
                register[name] = dict(returncode=p.returncode, text=(d.out / f'{name}.stdout').read_text()[-400:])
        os.close(fd)
    (d.out / 'bounce.json').write_text(json.dumps(dict(schema='strata.glm.bounce.v1', bounce=bounce, register=register,
        scope='One D2H + one H2D of SIZE bytes through pinned host memory (wall clock), and HostRegister of a shared memfd mapping per vendor, alone and both at once'), indent=1) + '\n')


def probe_loopback(d, model, python):
    for vendor, build, node in (('cuda', 'build-glm-release', 1), ('hip', 'build-hip-rocm101-tr16', 0)):
        d.run(f'loopback-{vendor}', [python, REPO / 'tools/glm_mixed_worker_loopback.py', '--worker', REPO / build / 'strata-glm-gpu-worker',
                                     '--model', model, '--output', d.out / f'worker-rtt-{vendor}.json', '--numa-node', node,
                                     '--cache-mib', 12288, '--gpu-budget-mib', 15360 if vendor == 'hip' else 14336], timeout=1800)


def decoder_command(build, model, prompt, tokens, gpu_budget, extra=()):
    return ['numactl', '--interleave=all', str(REPO / build / 'strata-glm-decode'), str(model), '@' + str(prompt), str(tokens), '4096', '15',
            '--context=12288', '--prefill-batch=8192', f'--gpu-budget-mib={gpu_budget}', '--cpu-affinity=numa', '--decode-experts=cpu',
            '--decode-cache-mib=12288', '--prefill-experts=f16-batched', '--decode-graphs', '--speculative=mtp', '--draft-depth=3',
            '--bench=1', '--decode-bench=2', *(extra or ['--mtp-experts=cpu'])]


def probe_draft(d, model, prompt):
    results = {}
    # --mtp-experts=gpu is refused together with --decode-cache-mib (the GPU draft bank needs an engine change
    # to coexist with the primary tier); only the CPU draft cost is measured here.
    for mode in ('cpu',):
        name = f'draft-{mode}'
        env = dict(d.env, STRATA_GLM_PREFETCH_GROUPS='14')
        d.run(name, decoder_command('build-hip-rocm101-tr16', model, prompt, 128, 15360, [f'--mtp-experts={mode}']), env=env, timeout=1800)
        text = (d.out / f'{name}.stderr').read_text()
        rows = [dict(re.findall(r'(\w+)=([\d.eE+-]+)', line)) for line in text.splitlines() if line.startswith('MTP_TIMING ') or line.startswith('SPECULATIVE ') or line.startswith('STEP_TRACE ') or line.startswith('MTP ')]
        results[mode] = rows
    (d.out / 'draft-step.json').write_text(json.dumps(dict(schema='strata.glm.draft_step.v1', scope='HIP alone, 128 tokens, 8K holdout code; MTP_TIMING per run with CPU vs GPU draft experts', results=results), indent=1) + '\n')


def probe_decode_profiles(d, model, prompt):
    env = dict(d.env, STRATA_GLM_PREFETCH_GROUPS='14')
    # The desktop (about 2.4 GB) and the profiler's buffers sit on the 5060 Ti: a smaller budget and tier for this kernel decomposition.
    cmd = decoder_command('build-glm-release', model, prompt, 48, 12288)
    # 2048-token prefill chunks keep the prefill scratch inside the smaller budget; the 8K context is unchanged for decode.
    cmd = [x.replace('--decode-cache-mib=12288', '--decode-cache-mib=6144').replace('--prefill-batch=8192', '--prefill-batch=2048') for x in cmd]
    rep = d.out / 'decode-nsys-cuda'
    rc = d.run('nsys-cuda', ['nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none', '-f', 'true', '-o', rep, '--', *cmd], env=dict(env, STRATA_GLM_PREFETCH_GROUPS='8'), timeout=2400)
    if rc == 0 and (rep.with_suffix('.nsys-rep')).exists():
        d.run('nsys-stats-cuda', ['nsys', 'stats', '--report', 'cuda_gpu_kern_sum,cuda_gpu_trace', '--format', 'csv', '-o', str(d.out / 'decode-nsys-cuda'), str(rep.with_suffix('.nsys-rep'))], timeout=2400)
    if os.environ.get('STRATA_PROBE_SKIP_ROCPROF'):
        return
    cmd = decoder_command('build-hip-rocm101-tr16', model, prompt, 48, 15360)
    d.run('rocprof-hip', [f'{ROCM}/bin/rocprofv3', '--kernel-trace', '--memory-copy-trace', '--stats', '--output-format', 'csv',
                          '--output-directory', d.out / 'decode-rocprof-hip', '--output-file', 'decode', '--', *cmd], env=env, timeout=2400)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--model', default='/mnt/nvme01/models/glm53f/prune/GLM-5.3-Flash-Q23-REAP224-assembled.gguf')
    ap.add_argument('--prompt', default=str(REPO / 'docs/fixtures/glm_tr16_20261009/prompts/8k/holdout-code.ids'))
    ap.add_argument('--profile', default=str(REPO / 'docs/benchmarks/tr16_mixed_20261014/numerical_profile.json'))
    ap.add_argument('--prior', default='/mnt/nvme01/models/glm53f/prune/coverage-keep224.json')
    ap.add_argument('--python', default=str(REPO / 'build-exl3-venv/bin/python'))
    ap.add_argument('--steps', default='lanes,transfers,loopback,draft,profiles')
    a = ap.parse_args()
    profile = json.loads(pathlib.Path(a.profile).read_text())
    env = os.environ.copy()
    env.update(profile.get('env', {}))
    env['STRATA_GLM_EXPERT_PRIOR'] = a.prior
    env['LD_LIBRARY_PATH'] = f'{ROCM}/lib' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env.update(STRATA_GLM_CANON='1', STRATA_GLM_CANON_COMBINE='device', STRATA_GLM_STEP_PIPELINE='1', STRATA_GLM_GPU_RESERVE_MIB='1024',
               STRATA_GLM_BUDGET_RESERVE_MIB='1024', STRATA_GLM_STEP_TRACE='2')
    d = Driver(a.output, env)
    (a.output / 'identity.json').write_text(json.dumps(dict(model=a.model, prompt=a.prompt, profile=a.profile, prior=a.prior,
        environment={k: v for k, v in env.items() if k.startswith('STRATA_') or k == 'LD_LIBRARY_PATH'},
        executables={p: sha(REPO / p) for p in ('build-glm-release/strata-glm-decode', 'build-glm-release/strata-glm-gpu-worker',
                                                 'build-hip-rocm101-tr16/strata-glm-decode', 'build-hip-rocm101-tr16/strata-glm-gpu-worker',
                                                 'build-glm-release/strata-glm-q2-flow-bench') if (REPO / p).exists()}), indent=1) + '\n')
    steps = a.steps.split(',')
    if 'lanes' in steps:
        probe_cpu_lanes(d, REPO / 'build-glm-release/strata-glm-q2-flow-bench')
    if 'transfers' in steps:
        probe_transfers(d)
    if 'loopback' in steps:
        probe_loopback(d, a.model, a.python)
    if 'draft' in steps:
        probe_draft(d, a.model, a.prompt)
    if 'profiles' in steps:
        probe_decode_profiles(d, a.model, a.prompt)
    print('done:', a.output / 'summary.json')


if __name__ == '__main__':
    sys.exit(main())
