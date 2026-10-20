# GLM-5.3-Flash NVFP4: two-layer weight test on RDNA4

This opt-in standalone test reads the gate, up, and down projection weights
from dense MLP layers 0 and 1 of
[nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4).
It tests six actual checkpoint matrices, not complete transformer blocks or
text generation. Attention, residuals, mHC, and routing are not executed.
Inputs are deterministic synthetic BF16/FP16
activations with full mantissa precision, not captured model activations.
No production engine code, format, API, or default path is changed.

## Download only selected tensors

`fetch.py` pins revision `da920bb0b9f4a06727223a349e55468e38352348`.
It fetches metadata and safetensors headers, then uses exact HTTP byte ranges
for 24 tensors: six packed weights, six local scales, six global scales, and
six activation scales. It rejects HTTP 200 or mismatched Content-Range responses
before reading shard bodies. Tensor payload is 169,869,360 bytes (162 MiB),
with a 256 MiB safety cap. No complete shard or full checkpoint is downloaded.

Downloaded tensors, metadata, and a compact runner input remain in ignored
`build/model/`. The compact input duplicates weight/scale payload, so total
local storage is approximately 340 MiB including metadata, plus build outputs.
Tensor SHA-256 hashes and source byte ranges are recorded in `selection.json`.
Model weights must not be committed with this experiment.

```sh
python3 experiment/nvfp4_model_layers/test_fetch.py
make -C experiment/nvfp4_model_layers fetch
make -C experiment/nvfp4_model_layers -j2 all
make -C experiment/nvfp4_model_layers isa bench
```

Only Python's standard library and gfx1201-capable HIP are required. No model
code is downloaded or executed. The HIP runner bounds and validates all input
shapes, file lengths, local scale codes, and global scales.

## Arithmetic and validation

The checkpoint uses packed E2M1, linear `[N,K/16]` E4M3 local scales, and scalar
FP32 `weight_scale_2`. The decoding convention follows
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py):
low nibble holds the even K position; reconstructed weight is E2M1 times local
scale times global scale. Local-scale products are exactly represented in
BF16/FP16. The global scale stays in the FP32 output epilogue. This differs
in rounding from folding the global scale into rounded 16-bit weights first.

`input_scale` is saved for provenance but unused: this is W4A16, while the
published checkpoint also supports NVFP4 activation quantization. Therefore
these tests do not claim numerical parity with its native W4A4 execution.

`layers.hip` reuses the unchanged W4A16 kernels from the adjacent experiment.
Each matrix runs at M=1,16,512 with both activation types and six variants:
packed/predecoded pairs at 16x16, 16x64, and 64x64 tiles. Predecoded inputs
omit the global scale, which both paths apply identically in the epilogue.
An independent host table/arithmetic decode checks all local weight products;
GPU expansion must match it. Every output must agree numerically with the
predecoded baseline and satisfy an FP32 error bound against GPU FP64.
CPU FP64 independently checks 64 sampled outputs per case. Output canaries
and all outputs are checked after every timed round.

Four additional MLP-chain smoke cases (two layers × two activation types, M=16)
feed the actual gate/up outputs through clamped SiLU into the down projection.
They follow the [Transformers MLP formula](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py):
gate is capped at 10, up is clamped to [-10,10]. The host computes SiLU in FP32;
projection outputs, SiLU outputs, and products are rounded to the activation type
with round-to-nearest-even. Each projection compares every output with the
predecoded path and FP64 bound. These smoke cases are untimed, do not include
attention/residuals, and do not claim PyTorch or native NVFP4 activation parity.

Five GEMM warmups, five rounds of 20 launches, rotating mode order, HIP events.
Host preparation, allocation, transfers, and validation are outside timing.
Expansion is separately measured. Resident buffers are reused without cache
flushing, so this does not measure disk or expert streaming. The M=1 kernels
are not specialized GEMV implementations. The miner remains running.

See [RESULTS.md](RESULTS.md) for measurements and source hashes. No full-model
quality, perplexity, generation, tuned-library comparison, idle-GPU baseline,
GPU sanitizer, or NVIDIA parity is claimed. Only standalone HIP code is built;
production CUDA/HIP/SYCL sources are unchanged.
