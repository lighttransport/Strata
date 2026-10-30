"""Use the same C11 ownership/admission rules as the native loader and worker."""
import ctypes
import functools
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


@functools.lru_cache(maxsize=1)
def rules():
    # A developer tool: use a supplied library or build only the tiny C11 rules.
    # No GPU, model access, network, or repository build outputs are involved.
    owner = None
    path = os.environ.get('STRATA_FP4_LAYOUT_RULES')
    if not path:
        owner = tempfile.TemporaryDirectory(prefix='strata-fp4-rules-')
        root = Path(__file__).resolve().parents[1]
        path = str(Path(owner.name) / 'rules.so')
        subprocess.run([*shlex.split(os.environ.get('CC', 'cc')), '-shared', '-fPIC', '-std=c11',
                        '-I' + str(root / 'include'), str(root / 'src/host/glm_fp4_layout.c'), '-o', path], check=True)
    lib = ctypes.CDLL(path)
    lib._temporary_owner = owner
    lib.strata_glm_fp4_layout_admit.argtypes = [ctypes.c_uint64] * 3
    lib.strata_glm_fp4_layout_admit.restype = ctypes.c_int
    lib.strata_glm_fp4_layout_span.argtypes = [ctypes.c_uint64] * 3
    lib.strata_glm_fp4_layout_span.restype = ctypes.c_int
    lib.strata_glm_fp4_layout_role.argtypes = [ctypes.c_uint] * 6
    lib.strata_glm_fp4_layout_role.restype = ctypes.c_int
    return lib
