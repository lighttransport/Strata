#!/usr/bin/env bash
# A single-request GLM profile for two 32 GiB V100s. Arguments override defaults.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${GLM_Q2_MODEL:-$HOME/models/glm53f-gguf-all/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf}"
python_bin="${STRATA_PYTHON:-$repo_root/.venv-glm/bin/python}"
decoder="${STRATA_GLM_DECODER:-$repo_root/build-glm-v100/strata-glm-decode}"
if [[ ! -x "$python_bin" || ! -x "$decoder" ]]; then
    echo "Build build-glm-v100 and create .venv-glm; see docs/GLM53_V100.md." >&2
    exit 2
fi
export STRATA_NATIVE_NUMA_LOCAL=0
# The host participates in expert work. Count allowed physical cores, not the
# machine-wide CPU count or the number of allowed logical CPUs.
workers="$($python_bin - <<'PY'
import os, pathlib
cores = set()
for cpu in os.sched_getaffinity(0):
    root = pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    cores.add((root.joinpath('physical_package_id').read_text(), root.joinpath('core_id').read_text()))
print(max(1, len(cores) - 1))
PY
)"
command=("$python_bin" "$repo_root/tools/glm_generate.py" "$model"
    --decoder "$decoder" --context 131072 --tokens 512
    --gpu-devices 0,1 --gpu-budget-mib 32768 --prefill-batch 4096 --decode-prefill-cache
    --decode-cache-mib extend --decode-cache-window 256 --decode-cache-adapt --decode-graphs
    --prefill-expert-cache-mib auto --lock-weights --threads "$workers" --cpu-affinity auto
    --decode-experts cpu --cpu-prepack-mib 0 --speculative none "$@")
if command -v numactl >/dev/null 2>&1; then
    exec numactl --interleave=all "${command[@]}"
fi
exec "${command[@]}"
