# Opt-in ordered KDA and fused deferral save

These changes preserve the original FP32 recurrence sum order and avoid a
separate device copy of mHC coefficients during one-token expert deferral.
They do not change model files, routing, quantization, precision or defaults.
They are experiments, not a newly selected inference preset.

Enable the coefficient save with `STRATA_GLM_MHC_SAVE_FUSED=1`. It applies
only while expert deferral is active. The existing mHC write kernel also
stores its 24 coefficients; the following device-to-device copy is skipped.
Decode and rollback paths without deferral retain their existing calls.

For ordered row-partition KDA during long prefill, add these settings to an
existing engine config's `env` object:

```json
{
  "STRATA_GLM_KDA_PREPARE": "1",
  "STRATA_GLM_KDA_ORDERED_ROWS": "1",
  "STRATA_GLM_PREFILL_KDA_ROW_PARTS": "4",
  "STRATA_GLM_PREFILL_KDA_COLUMNS": "128",
  "STRATA_GLM_PREFILL_KDA_MIN_TOKENS": "4096"
}
```

Each thread holds 32 state values. Four row partitions carry a single
accumulator in original row order through shared memory, instead of summing
independently rounded partial results. Output and final-state bits matched
the original column kernel with prepared and unprepared inputs and chunks
of 64, 256 and 2048 tokens. This option works with the existing expert modes;
it does not require BF16. Decode keeps its existing single-part recurrence
when only the prefill row-parts setting is changed. Snapshot/history calls
still require the original single-part path.

## Measurement on 2026-10-11

B550, Ryzen 9 3950X, RX 9070 XT gfx1201 with 16 GiB VRAM, 64 GiB system RAM,
HIP 7.14.60850; RAM at the user's reported 2866 MT/s. The later 2666 MT/s
setting has not been measured. REAP50 Q23 assembly with the experimental
Q22 down-only sidecar and fit4 routing; 12 CPU threads, context/batch 8192,
frozen 7936-token code prefix, 128 generated tokens, custom BF16 WMMA experts,
no speculation, 60 GiB cgroup, zero cgroup swap, 15360 MiB GPU budget and
512 MiB physical reserve. One first-prefill run per arm, no concurrent build
or inference. These are single runs, not medians or a repeatable 36+ claim.

| Arm | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| Prior build, prepared serial KDA columns 32 | 410.785 | 35.8858 |
| Ordered KDA plus fused coefficient save | 466.130 | 36.0570 |

The candidate improved prefill by 13.47% and decode by 0.48% in this pair.
All 128 output IDs, the saved final logits and the cache fingerprint matched
exactly. The two changes were tested together; this does not isolate either
one's contribution. Build hashes and compact result evidence are in
[the measurement record](benchmarks/glm_b550_ordered_kda_20261011.json).

The earlier prepared KDA microbenchmark had different timing scopes: the
serial 0.759 ms included preparation while the ordered 0.167 ms did not.
Do not use their ratio as a matched speedup. The benchmark now includes
preparation in both prepared arms; its revised timing has not been measured.

## Checks and remaining limits

HIP and CUDA builds of the decoder and affected tests passed. HIP tests
passed for mHC output/coefficient bit parity, prepared/unprepared recurrence
bit parity, KDA column tiles, and production-shape BF16 GEMMs with output
guards and sampled CPU references. CUDA is compile-only: no NVIDIA GPU is
available. These GLM GPU sources and APIs are not part of the separate SYCL
implementation.

The BF16 full-model mode remains unselected because its earlier long coding
gate failed. This change does not qualify that mode or relax its quality
gate. The 700 tok/s prefill target is unmet. A 16K timing and broader coding
qualification of the new switches remain pending.

Two full-model screens using library BF16 GEMM lost the GPU from the PCIe
bus and are rejected, not throughput results. The profiled second attempt
finished layers 0-2, then failed before the first MLA layer completed. Small
library and custom WMMA tests passed, including full 16-expert panels and
tail buckets. No preceding GPU VM fault or illegal-instruction report was
found; this does not rule out an application error. The cause is unresolved.
The custom WMMA candidate above completed cleanly. Further library-path
inference needs an isolated reproducer; no power settings were changed here.

Commands for focused checks, without a model:

```bash
ctest --test-dir build-hip -R '^(glm_q8_test|glm_prefill_parity|glm_kda_column_tiles_parity|glm_bf16_prefill_test)$' --output-on-failure -j1
build-hip/glm_bf16_prefill_test --production-shapes
build-hip/glm_bf16_prefill_test --wmma --production-shapes
```

The tensor-prefill caller also checks that each gate/up panel fits its
dequantization buffer before launching a dequantizer or GEMM.
