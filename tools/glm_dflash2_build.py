"""Build the CUDA-only DFlash2 FP32-output GEMM helper for a pinned PyTorch runtime."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', required=True, type=Path)
    p.add_argument('--skinny', action='store_true', help='build optional SM70 half-QPN backend')
    a = p.parse_args()
    a.build_dir.mkdir(parents=True, exist_ok=True)
    from torch.utils.cpp_extension import load
    root=Path(__file__).resolve().parents[1]
    sources=[str(Path(__file__).with_name('glm_dflash2_skinny.cpp' if a.skinny else 'glm_dflash2_ext.cpp').resolve())]
    if a.skinny:sources.append(str(root/'src/kernels/cuda/glm_fp4.cu'))
    library = load(name='strata_dflash2_skinny' if a.skinny else 'strata_dflash2_f32', sources=sources,
                   build_directory=str(a.build_dir.resolve()), with_cuda=True, is_python_module=False,
                   extra_cflags=['-O2'], extra_include_paths=[str(root/'include')],
                   extra_cuda_cflags=['-O3','-lineinfo'], extra_ldflags=['-lcublas'], verbose=True)
    print(library)


if __name__ == '__main__':main()
