#!/usr/bin/env python3
"""Loopback round-trip benchmark for the mixed GPU worker (queried protocol layout, no decoder).

Launches `strata-glm-gpu-worker` exactly as tools/glm_mixed_run.py does (memfd + SOCK_SEQPACKET),
takes ownership of one layer's experts, then times the request paths the runner's service thread
takes per slot:

  ownership_noop   an ownership request that changes nothing (the protocol floor: socket + validation)
  decode_rows      prefill=0 requests returning expert rows (routes x 16 KB copied by the worker)
  decode_partial   prefill=4 requests returning one weighted vector per token (STRATA_GLM_MIXED_DECODE_PARTIAL)

for widths 1 and 2 and 1/4/8/16 routes. Wall-clock microseconds per request (p50/p90/p99) are
written as JSON; they are the simulator's `worker.rtt_*` and `worker_request_fixed_ms` inputs.
Needs the model file (the worker loads it) and an idle GPU; run under the quiet-run rules.
"""
import argparse
import hashlib
import json
import os
import random
import socket
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

MAGIC = 0x474d5831


class Worker:
    def __init__(self, binary, model, cache_mib, budget_mib, numa_node, env):
        self.layout = json.loads(subprocess.check_output([str(binary), '--protocol-layout'], env=env, text=True))
        self.offsets = self.layout['offsets']
        size = int(subprocess.check_output([str(binary), '--protocol-size'], env=env, text=True).strip())
        if size != self.layout['bytes']:
            raise SystemExit('worker protocol size differs from its layout')
        self.memory = os.memfd_create('strata-mixed-loopback')
        os.ftruncate(self.memory, size)
        self.view = memoryview(__import__('mmap').mmap(self.memory, size))
        self.client, worker_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        cmd = [str(binary), str(model), str(self.memory), str(worker_end.fileno()), str(cache_mib), str(budget_mib)]
        if numa_node is not None:
            cmd = ['numactl', f'--cpunodebind={numa_node}', f'--membind={numa_node}', *cmd]
        self.process = subprocess.Popen(cmd, env=env, pass_fds=(self.memory, worker_end.fileno()), stderr=subprocess.PIPE, text=True)
        worker_end.close()
        self.sequence = 0
        ready = self.recv()
        if ready != MAGIC:
            raise SystemExit('worker handshake failed')

    def send(self, value):
        self.client.send(struct.pack('<QII', value, 0, 0))

    def recv(self, timeout=600.0):
        self.client.settimeout(timeout)
        packet = self.client.recv(16)
        if len(packet) != 16:
            raise SystemExit('worker disconnected')
        value, tag, kind = struct.unpack('<QII', packet)
        if tag or kind:
            raise SystemExit('unexpected packet kind')
        return value

    def header(self, layer, nt, routes, canon, prefill, chunk_tag=0):
        self.sequence += 1
        struct.pack_into('<8I', self.view, self.offsets['header'], MAGIC, self.layout['version'], layer, nt, routes, canon, prefill, 0)
        struct.pack_into('<2Q', self.view, self.offsets['sequence'], self.sequence, self.sequence)
        struct.pack_into('<2f2I', self.view, self.offsets['limits'], 0.0, 0.0, 0, chunk_tag)

    def request(self):
        start = time.perf_counter()
        self.send(self.sequence)
        reply = self.recv()
        elapsed = time.perf_counter() - start
        seq, gen = struct.unpack_from('<2Q', self.view, self.offsets['sequence'])
        if reply != self.sequence or seq != self.sequence or gen != self.sequence:
            raise SystemExit('stale reply')
        return elapsed

    def ownership(self, layer, counts):
        self.header(layer, 1, 0, 0, 2)
        self.view[self.offsets['primary_owned']:self.offsets['primary_owned'] + 288] = bytes(288)
        struct.pack_into(f'<{288}f', self.view, self.offsets['route_weight'], *counts)
        return self.request()

    def owned(self, layer):
        base = self.offsets['worker_owned'] + layer * 288
        return [e for e in range(288) if self.view[base + e]]

    def decode(self, layer, nt, experts, partial, canon=1):
        routes = len(experts)
        self.header(layer, nt, routes, canon, 4 if partial else 0)
        # one route per (token, slot) pair; destinations unique within nt*8
        dests = [(i % nt) * 8 + (i // nt) for i in range(routes)]
        struct.pack_into(f'<{routes}i', self.view, self.offsets['expert'], *experts)
        struct.pack_into(f'<{routes}i', self.view, self.offsets['destination'], *dests)
        if partial:
            weights = [0.0] * (nt * 8)
            for d in dests:
                weights[d] = 1.0 / 8
            struct.pack_into(f'<{nt * 8}f', self.view, self.offsets['route_weight'], *weights)
        return self.request()

    def close(self):
        struct.pack_into('<I', self.view, 28, 1)  # reserved=1: stop
        try:
            self.send(0)
        except OSError:
            pass
        _, err = self.process.communicate(timeout=120)
        return err


def percentiles(samples):
    s = sorted(samples)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return dict(count=len(s), p50_us=1e6 * pick(0.5), p90_us=1e6 * pick(0.9), p99_us=1e6 * pick(0.99),
                mean_us=1e6 * statistics.fmean(s), min_us=1e6 * s[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--worker', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--layer', type=int, default=20)
    ap.add_argument('--cache-mib', type=int, default=12288)
    ap.add_argument('--gpu-budget-mib', type=int, default=14336)
    ap.add_argument('--numa-node', type=int)
    ap.add_argument('--requests', type=int, default=2000)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--rocm-lib', default='/opt/rocm/core-10.1/lib')
    a = ap.parse_args()
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = a.rocm_lib + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env.setdefault('STRATA_GLM_MIXED_DECODE_GRAPHS', '1')
    env.setdefault('STRATA_GLM_MIXED_DECODE_PARTIAL', '1')
    env['STRATA_GLM_CANON'] = '1'
    random.seed(1)
    worker = Worker(a.worker, a.model, a.cache_mib, a.gpu_budget_mib, a.numa_node, env)
    results = dict(schema='strata.glm.worker_loopback.v1', worker=str(a.worker), worker_sha256=hashlib.sha256(a.worker.read_bytes()).hexdigest(),
                   model=str(a.model), layer=a.layer, cache_mib=a.cache_mib, numa_node=a.numa_node,
                   environment={k: v for k, v in env.items() if k.startswith('STRATA_')},
                   scope='Wall-clock request round trips from a Python client through the real worker protocol; includes the socket, validation, GPU execution and the worker-side copies. Not decoder throughput.',
                   measurements={})
    try:
        counts = [float(288 - e) for e in range(288)]  # prefer low expert ids
        first = worker.ownership(a.layer, counts)
        owned = worker.owned(a.layer)
        results['owned_experts'] = len(owned)
        results['first_ownership_ms'] = 1e3 * first
        if len(owned) < 16:
            raise SystemExit(f'worker owns only {len(owned)} experts of layer {a.layer}')
        # Warm every owned expert once so later requests hit the cache.
        for i in range(0, len(owned), 8):
            worker.decode(a.layer, 1, owned[i:i + 8], partial=True)
        noop = [worker.ownership(a.layer, counts) for _ in range(a.warmup + a.requests)][a.warmup:]
        results['measurements']['ownership_noop'] = percentiles(noop)
        for partial in (True, False):
            for nt in (1, 2):
                for routes in (1, 4, 8, 16):
                    if routes > nt * 8:
                        continue
                    samples = []
                    for i in range(a.warmup + a.requests):
                        chosen = random.sample(owned, routes)
                        samples.append(worker.decode(a.layer, nt, chosen, partial))
                    results['measurements'][f"decode_{'partial' if partial else 'rows'}_w{nt}_r{routes}"] = percentiles(samples[a.warmup:])
                    print(f"{'partial' if partial else 'rows'} width {nt} routes {routes}: p50 {results['measurements'][f'decode_{'partial' if partial else 'rows'}_w{nt}_r{routes}']['p50_us']:.0f} us", flush=True)
    finally:
        results['worker_stderr_tail'] = worker.close()[-2000:]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=1) + '\n')
    print(json.dumps({k: round(v['p50_us']) for k, v in results['measurements'].items()}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
