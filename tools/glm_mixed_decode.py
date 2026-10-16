#!/usr/bin/env python3
"""Decoder-compatible adapter for experimental mixed-GPU evaluation tools.

STRATA_GLM_MIXED_PRIMARY and STRATA_GLM_MIXED_WORKER select backend binaries.
STRATA_GLM_EVAL_WIDTH is still interpreted by the primary decoder.
"""
import os
from pathlib import Path
import sys
from glm_mixed_run import main

if __name__ == '__main__':
    args = sys.argv[1:]
    worker_mode = bool(args and args[0] in ('--worker','--serve'))
    if worker_mode:
        args = args[1:]
    if not args:
        raise SystemExit('usage: glm_mixed_decode.py [--serve|--worker] MODEL RUNNER_ARGS...')
    model, *runner_args = args
    root = Path(__file__).resolve().parents[1]
    primary = os.environ.get('STRATA_GLM_MIXED_PRIMARY', str(root/'build-hip-rocm101-tr16/strata-glm-decode'))
    worker = os.environ.get('STRATA_GLM_MIXED_WORKER', str(root/'build-glm-release/strata-glm-gpu-worker'))
    sys.argv = [sys.argv[0], '--combine', os.environ.get('STRATA_GLM_CANON_COMBINE','strict'), '--primary', primary, '--worker', worker, '--model', model]
    for variable,option in [('STRATA_GLM_MIXED_WORKER_CACHE_MIB','--worker-cache-mib'),
                            ('STRATA_GLM_MIXED_WORKER_GPU_BUDGET_MIB','--worker-gpu-budget-mib'),
                            ('STRATA_GLM_MIXED_WORKER_NUMA_NODE','--worker-numa-node'),
                            ('STRATA_GLM_MIXED_PRIMARY_DEVICE','--primary-device'),
                            ('STRATA_GLM_MIXED_WORKER_DEVICE','--worker-device'),
                            ('STRATA_GLM_MIXED_PRIMARY_CPUS','--primary-cpus'),
                            ('STRATA_GLM_MIXED_WORKER_CPUS','--worker-cpus')]:
        if variable in os.environ: sys.argv += [option,os.environ[variable]]
    if worker_mode:
        sys.argv += ['--primary-worker-mode']
    else:
        runner_args += ['--prefill-experts=f16-batched']
    sys.argv += ['--', *runner_args]
    raise SystemExit(main())
