---
license: mit
base_model:
  - unsloth/GLM-5.3-Flash-GGUF
tags:
  - gguf
  - strata
  - quantization
---

# GLM-5.3-Flash-Strata-Q2Q3-Experts

Experimental GGUF expert sidecars derived from Unsloth's **GLM-5.3-Flash UD-Q2_K_XL** for the [LightTransport Strata fork](https://github.com/lighttransport/Strata).
Eligible routed expert gate/up weights are requantized from IQ2_XS to Q2_K, and down weights from IQ3_XXS to Q3_K. This conversion changes weight values; it is not a lossless repack or a fine-tune.

**These files are not a complete model.** Keep all four original UD-Q2_K_XL GGUF shards from [Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF): dense, attention, router, MTP, and unconverted expert weights are loaded from those files. The sidecar requires Strata's `--expert-pack` loader. Compatibility with other inference engines has not been verified.

## Reproduce the conversion

Linux/CUDA example, following the fork's [conversion procedure](https://github.com/lighttransport/Strata/blob/fb80c8c773810d696736716c4692a6b397f9a3c1/docs/GLM_Q2_REQUANTIZATION.md). Install CUDA, CMake, Ninja, and uv first. Set `CUDA_ARCH` for your GPU (for example, `120` for RTX 5060 Ti).

```bash
git clone https://github.com/lighttransport/Strata.git
cd Strata
git checkout fb80c8c773810d696736716c4692a6b397f9a3c1
git submodule update --init --recursive

uv venv .venv
source .venv/bin/activate
uv pip install huggingface_hub regex jinja2
hf download unsloth/GLM-5.3-Flash-GGUF \
  --include 'UD-Q2_K_XL/*' --local-dir models/glm53f

CUDA_ARCH=120 # Change this for your GPU.
cmake -S . -B build-glm -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build build-glm -j 4

MODEL="$PWD/models/glm53f/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf"
WORK="$PWD/build-q2-redesign"
PACK="$PWD/experts-q23.gguf"
python tools/glm_q2_corpus.py "$MODEL" "$WORK/corpus"
STRATA_GLM_CALIBRATION_DIR="$WORK/calibration" STRATA_GLM_LOCK_RUNTIME=1 \
  python tools/glm_q2_run_guard.py --output "$WORK/calibration-run" -- \
  numactl --interleave=all build-glm/strata-glm-decode "$MODEL" 0 1 4096 15 \
  --context=8192 --prefill-batch=256 --gpu-budget-mib=12288 \
  --cpu-affinity=numa --decode-graphs --eval-corpus="$WORK/corpus/calibration.ids"
build-glm/strata-glm-q2-pack "$MODEL" "$PACK" \
  --calibration="$WORK/calibration" --threads=16 --down=q3
build-glm/strata-glm-q2-pack --verify "$MODEL" "$PACK"
```

The calibration example requires `numactl` and a machine with two NUMA nodes; adapt GPU and memory settings to your hardware. Allow at least 110 GiB free for the sidecar in addition to the original model. The converter refuses to overwrite an existing pack or `.partial` file.

This reproduces the conversion workflow, not a guarantee of identical published bytes. Exact reproduction requires the same source revision, calibration corpus, converter settings, and toolchain. The local calibration corpus used for the Q2/Q3 experiment records Strata revision `f858ea3262292f7fa7534d5cff9e0da518259552`.

## Load and verify

Pass the first original shard as the model and add `--expert-pack=/path/to/experts-q23.gguf` to your `strata-glm-decode` command. All original shards must remain available. Run the `--verify` command above before inference.

**Portability limitation:** this loader fingerprints original shard sizes, headers, and file modification times. A fresh download can fail with `source fingerprint mismatch` even when the original weight bytes match. Regenerate the sidecar against your local shards if verification fails. The published sidecar should not be assumed to load on a fresh installation until this check has been addressed.

This is an experimental conversion. No general quality equivalence or speed improvement is claimed; reconstruction error reports are not a model quality benchmark.

## License and attribution

The original [GLM-5.3-Flash model](https://huggingface.co/zai-org/GLM-5.3-Flash) is by **Z.ai**, under the [MIT license](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE), copyright (c) 2026 Z.AI Co., Ltd. The source GGUF quantization is by **Unsloth** and is [marked MIT](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF).

The original **Strata** engine is by **Niko1221 and the Strata contributors**, under the [MIT license](https://github.com/Niko1221/Strata/blob/main/LICENSE). This conversion uses the [LightTransport fork](https://github.com/lighttransport/Strata), which builds on their work. Quantization kernels come from [GGML](https://github.com/ggml-org/ggml), also under MIT.

These derived weights are distributed under MIT. Retain the applicable upstream copyright and license notices when redistributing them. This is a community conversion, not an official Z.ai, Unsloth, or upstream Strata release.
