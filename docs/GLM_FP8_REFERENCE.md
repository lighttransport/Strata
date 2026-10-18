# GLM-5.3-Flash: an FP8 reference and every 128 GB pack measured against it (TR16, 2026-10-17)

Until now the corpus gate measured KL and top-1 against the UD-Q2_K_XL model: a 2-bit model, not the original. A pack
quantized from the original weights then showed KL against UD-Q2_K_XL's own rounding, even when it was closer to the
original. This note adds an exact reference computed from the official FP8 release (zai-org/GLM-5.3-Flash) on one
TR16 GPU, and re-measures every pack that fits 128 GB of RAM against it.

**Result: Q23 quantized from FP8 (Q2_K gate/up, Q3_K down, 110.2 GB) is the recommended 128 GB pack.**
- **Fidelity:** it halves the release Q23's distance from FP8 (KL 0.117 vs 0.212, perplexity +3.5% vs +9.0%).
- **Speed:** it decodes as fast (19.6 tok/s).
- **Coding:** it scores 155/164 HumanEval with exact routing (the release Q23: 154).
- **Canonical mode:** it keeps it, since the types are unchanged.

## The FP8 reference

The FP8 release's routed experts and some fixed matrices are E4M3 with a 128x128 `weight_scale_inv` per block; the
rest is BF16 or F32. Every weight is used as released, with no activation quantization: E4M3 times its scale is
computed in f32 (the product is exact) and every product runs in f32 (`cublasSgemm`). This is the best-defined
"original model" those weights give. Official FP8 serving also quantizes activations to FP8 per 128 values, and does
not.

How it runs (opt-in; GGUF and EXL3 behave as before):
- **Loading:** the native safetensors loader (`include/strata/artifact/glm_exl3.hpp`, `glm_native.hpp`) recognizes
  the release by `quantization_config` (`fp8`, `e4m3`, `[128, 128]`). Its configuration matches the EXL3 model's on
  all 35 geometry fields, and the tokenizers are identical.
- **Fixed matrices:** the 186 FP8 fixed matrices are dequantized to f32 at load. They are the shared experts, the
  three dense layers' MLPs, and the MLA q_a, q_b, kv_a and o projections. An independent Python decoder gives the same
  values.
- **Routed experts:** they run in the GPU prefill (`moe_exl3_gpu`). Each projection is read from the safetensors,
  dequantized on the GPU by `fp8_block_to_float`, and multiplied with `cublasSgemm`.
  - `fp8_block_gpu_test` (ctest) checks the GPU kernel bit for bit against the host on all 256 codes.
  - The reads run 6 experts ahead on threads, with the pages dropped after use (`POSIX_FADV_DONTNEED`).
  - The 304 GB of experts are read once per sequence at the drive's limit: nvme01 is PCIe Gen3 x1, about 0.9 GB/s,
    which gives about 5.5 minutes per 512-token sequence.
- **Scoring:** the evaluator scores prefill rows for this model, one window per sequence (`STRATA_GLM_EVAL_WIDTH=512`,
  `--prefill-batch=512`). Decode refuses the FP8 model.

```sh
STRATA_GLM_EVAL_WIDTH=512 build-hip-rocm101-tr16/strata-glm-decode models/glm53f/fp8 0 1 4096 15 \
  --context=4096 --prefill-batch=512 --gpu-budget-mib=12288 --cpu-affinity=numa \
  --eval-corpus=build-q2-v2/quality/evaluation16.ids --eval-save-logits=eval16-fp8.logits
```

The 16 held-out sequences (8,192 scored tokens) took 83 minutes. FP8 perplexity: **4.449**.

### Determinism

On a quiet GPU the reference is bit-reproducible: across processes, corpus files, position in the corpus and decoder
resets. When another process uses the GPU at the same time, logits from token 64 on change slightly: KL 1.3e-5
against the quiet run, top-1 unchanged. The saved reference was made while another job used the GPU.
- **Ruled out:** rocBLAS. Plain and strided-batched SGEMM at every shape this path uses are bit-identical under load,
  with and without rocBLAS atomics. Float atomics are ruled out too: the only ones are in the GGUF expert scatter,
  which this path does not use.
- **Suspected:** a missing synchronization in a kernel that only runs from token 64 on (the KDA cross-chunk state or
  the MLA tiles). It is not located yet.
- **Effect:** at 1e-5 the effect is four orders of magnitude below the differences measured here. Make references on
  a quiet GPU.

## Every 128 GB pack against FP8

Corpus gate (`tools/glm_tr16_quality.py --eval-prefill`), RX 9070 XT, the same non-canonical settings for every model,
16 held-out sequences against `eval16-fp8.logits`. A sidecar replaces routed expert projections of the four
UD-Q2_K_XL shards (108.7 GB); the release Q23 leaves layer 11 in its original format, which is why it is smaller.

| Pack (routed experts) | Sidecar size | Perplexity | vs FP8 | KL | Top-1 |
|---|---:|---:|---:|---:|---:|
| FP8 (the reference) | 304 GB of experts | 4.449 | | | |
| **Q23 from FP8** (Q2_K gate/up, Q3_K down) | 110.2 GB | **4.606** | **+3.5%** | 0.117 | **88.3%** |
| IQ2_S gate/up + Q3_K down, from FP8 | 108.6 GB | 4.644 | +4.4% | **0.116** | 88.2% |
| IQ2_S, from FP8 | 97.5 GB | 4.673 | +5.0% | 0.132 | 87.8% |
| UD-Q2_K_XL as shipped | none (108.7 GB whole model) | 4.688 | +5.4% | 0.144 | 87.5% |
| Q23 release (requantized from UD-Q2_K_XL's experts) | 105.5 GB | 4.850 | +9.0% | 0.212 | 84.7% |

The release Q23 carries two rounds of quantization (FP8 to UD-Q2_K_XL's 2-bit experts, then to Q2_K/Q3_K); the same
types quantized once from FP8 halve its KL. Against the old UD-Q2_K_XL reference the ranking was different: IQ2_S
then showed a higher KL than Q23.

Weight-space error of the FP8 packs (importance-weighted normalized MSE, mean over the 42 layers, min-max):

| Pack | gate / up | down |
|---|---|---|
| Q23 from FP8 | 0.071 (0.068-0.072) | 0.016 (0.006-0.020) |
| IQ2_S + Q3_K | 0.061 (0.059-0.062) | 0.016 (0.006-0.020) |
| IQ2_S | 0.061 | 0.047 (0.014-0.058) |

```sh
build-glm-release/strata-glm-q2-pack GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf experts-q23-fp8.gguf \
  --calibration=prune/calibration --fp8=fp8 --gate-type=q2_k --down-type=q3_k --threads=14     # 1.9 h on TR16
```

## REAP-50 (64 GB) from FP8

REAP-50 keeps 144 of 288 experts per layer. Its router rows equal the original rows exactly, which gives the original
index of every kept expert (`tools/glm_reap_expert_map.py`); the same 144 experts were then quantized from FP8
(`strata-glm-q2-pack --expert-map`) and assembled into the published REAP-50 file (`strata-glm-assemble --reassemble`).
Same gate, HumanEval with unchanged routing and MTP depth 1, speed in a 60 GiB / no-swap cgroup with the B550 profile:

| REAP-50 file | Perplexity | vs FP8 | KL | Top-1 | HumanEval | Decode 2K / 8K | Prefill 2K / 8K |
|---|---:|---:|---:|---:|---:|---:|---:|
| **experts from FP8** | **5.184** | **+16.5%** | **0.393** | **80.4%** | **154/164** | 16.1 / 16.0 | 219 / 217 |
| first release (requantized from the REAP-50 Q4_K_M experts) | 5.632 | +26.6% | 0.479 | 77.6% | 149/164 | 15.9 / 16.2 | 215 / 192 |

Where REAP-50's error comes from: the same file's experts on the exact-FP8 path (`STRATA_GLM_EXPERT_GGUF` with
`STRATA_GLM_EXPERT_MAP=reap50-expert-map.txt`, which restricts the FP8 router to the 144 kept experts; REAP-50's router
rows and biases are the original ones) give perplexity 5.212 (+17.1%), KL 0.385, top-1 80.7%. So its Q4_K_M dense
weights add almost nothing (0.393 against 0.385). The same Q2_K/Q3_K experts unpruned give KL 0.108 on that path, so
pruning to 144 experts accounts for most of REAP-50's KL. A better 64 GB model has to change which experts are kept
(or keep more of them at fewer bits), not the expert quantizer.

## Expert quantizers alone: llvq Tetra against Q23 (2026-10-18)

The tables above use each pack's own dense weights (UD-Q2_K_XL's). To compare expert quantizers alone, two opt-in
switches run the FP8 reference path with only the routed experts replaced; everything else stays exact FP8:
- `STRATA_GLM_EXPERT_LLVQ=DIR` reads llvq Tetra units (`strata-glm-llvq-encode`, [GLM_LLVQ.md](GLM_LLVQ.md)). Each
  matrix is decoded to f32 on the GPU (`llvq_tetra_dequant_f32`, within 1.8e-7 of the CPU reference) and multiplied
  against the rotated input (`y = W' (Q x)`, rotation in double on the host).
- `STRATA_GLM_EXPERT_GGUF=PACK` reads a GGUF expert pack and dequantizes it on the GPU (`iq_dequant_f32`).

Same 16 sequences and reference as above:

| Routed experts (dense: exact FP8) | Bits per weight | Size | Perplexity | vs FP8 | KL | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Q23 from FP8 (Q2_K gate/up, Q3_K down) | 2.90 | 110.2 GB | **4.583** | **+3.0%** | **0.108** | **88.7%** |
| llvq Tetra, identity Hessian, rotation, 2 gain centroids | 2.07 | 74 GB | 4.732 | +6.4% | 0.172 | 85.9% |

At 29% fewer bits llvq's KL is 1.6 times Q23's. The llvq encode used the cheap setting (identity Hessian; weight
error 0.085 against FP8). The Hessian-weighted variant of the llvq study would cost about 1.5 PFLOP for the full model
and was not run. Q23's dense-weight share is small: with UD-Q2_K_XL's dense weights instead of FP8's its KL is 0.117
(table above), against 0.108 here.

Run times on TR16: 29 minutes (llvq, 74 GB of units read per pass) and 33 minutes (Q23, 110 GB).

```sh
STRATA_GLM_EVAL_WIDTH=512 STRATA_GLM_EXPERT_LLVQ=llvq-tetra \
  build-hip-rocm101-tr16/strata-glm-decode fp8 0 1 4096 15 --context=4096 --prefill-batch=512 --gpu-budget-mib=12288 \
  --cpu-affinity=numa --eval-corpus=build-q2-v2/quality/evaluation16.ids --eval-reference=eval16-fp8.logits
# Q23: STRATA_GLM_EXPERT_GGUF=experts-q23-fp8.gguf instead of STRATA_GLM_EXPERT_LLVQ
```

## Speed and HumanEval

Decode and prefill (`tools/glm_tr16_calibrate.py`): held-out code prompts, 512 generated tokens, median of 3 trials.
The profile is the full-model TR16 one without canonical mode (IQ2_S has none): MTP depth 1, affinity 0.08, RX 9070 XT
alone.

| Pack | Decode 2K | Decode 8K | Prefill 2K | Prefill 8K |
|---|---:|---:|---:|---:|
| Q23 from FP8 | 19.6 | 19.6 | 178 | 497 |
| Q23 release | 19.6 | 20.3 | 176 | 493 |
| IQ2_S + Q3_K | 17.4 | 17.9 | 179 | 550 |
| IQ2_S | 17.3 | 17.7 | 200 | 580 |

The IQ2_S packs decode about 12% slower: the IQ2_S gate/up CPU kernel on Zen 1 costs more per expert than Q2_K, and
fewer bytes do not make up for it. Q3_K down did not change that.

HumanEval (164 tasks, one greedy chat sample, same profile):

| Pack | affinity 0.08 | affinity 0 (exact routing) |
|---|---:|---:|
| Q23 from FP8 | 151 | **155** |
| Q23 release | 156 | 154 |
| IQ2_S + Q3_K | 153 | |
| IQ2_S | 151 | |

With exact routing the FP8 pack scores one task above the release. The lossy routing setting moves each pack by 1-5
tasks in either direction; tune it per pack.

## Fixes found on the way

- **IQ2_S experts in the GPU tier:** the grouped GPU expert kernel (`native_expert_grouped`) had no IQ2_S down
  projection, so decoding an IQ2_S pack with the GPU tier stopped at once. IQ2_S is now in its down formats. The fix is
  checked two ways:
  - `native_grouped_parity` is bitwise equal to the older launch path.
  - Decode-path scoring of sequence 1 at affinity 0 gives KL 0.161 against FP8, against 0.158 from prefill rows.
- **EXL3 scoring and loading:**
  - EXL3 can score prefill rows with `STRATA_EXL3_EVAL_PREFILL=1` (windows of 256).
  - The native prefill arena gained 128 MiB for the per-expert scratch it had no room for.
  - Corpus evaluation scores EXL3's raw padded head instead of masking the 24 padding IDs to -inf.
  - EXL3 on sequence 1 against FP8: perplexity 7.835, KL 0.239, top-1 78.9% (FP8: 7.410). Its dense weights are the
    BF16 originals, not the FP8 release's, which adds to its KL.
- **Runners on a machine without NVIDIA:**
  - `tools/glm_q2_run_guard.py` finds GPU clients and free VRAM through amdgpu sysfs and `/dev/kfd`.
  - `tools/glm_tr16_calibrate.py` records amdgpu facts.
  - `tools/glm_tr16_quality.py` has opt-in `--max-swap-mib` and `--allow-shared-gpu` for quality-only runs, recorded
    in the manifest.
