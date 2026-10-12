### CPU (Threadripper 1950X), dummy weights, 8 experts per call

GB/s for gate+up of 8 experts (71 MB); first row is the read-bandwidth reference.

| kernel | t1 | t2 | t4 | t8 | t16 | t32 | relL2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| read bandwidth | 26.2 | 52.1 | 80.3 | 88.2 | 83.5 | 80.1 | |
| c1_f32 | 3.5 | 6.9 | 13.4 | 25.5 | 45.8 | 50.5 | 2.0e-07 |
| c2_q8_sign | 6.3 | 12.6 | 24.5 | 45.4 | 74.9 | 75.6 | 1.0e-02 |
| c3_q8_off_r1 | 6.9 | 13.7 | 25.8 | 48.2 | 76.8 | 76.0 | 1.0e-02 |
| c3_q8_off_r2 | 4.0 | 8.2 | 15.8 | 29.6 | 49.8 | 55.8 | 1.0e-02 |
| c3_q8_off_r4 | 4.6 | 9.1 | 17.5 | 32.2 | 54.1 | 59.2 | 1.0e-02 |
| c3h_q8_off | 6.0 | 11.7 | 22.7 | 42.4 | 70.9 | 74.6 | 1.0e-02 |

MTP/batch verify on CPU (gate+up, N tokens share the 8 experts), time relative to N=1:

| threads | N=1 | N=2 | N=4 | N=8 |
|---|---:|---:|---:|---:|
| t1 | 10.16 ms (1.00x) | 15.98 ms (1.57x) | 27.59 ms (2.72x) | 53.98 ms (5.31x) |
| t2 | 5.15 ms (1.00x) | 8.05 ms (1.56x) | 14.03 ms (2.73x) | 27.70 ms (5.38x) |
| t4 | 2.62 ms (1.00x) | 4.20 ms (1.60x) | 7.20 ms (2.75x) | 14.01 ms (5.36x) |
| t8 | 1.44 ms (1.00x) | 2.19 ms (1.52x) | 3.75 ms (2.61x) | 7.28 ms (5.06x) |
| t16 | 0.95 ms (1.00x) | 1.30 ms (1.37x) | 2.03 ms (2.14x) | 4.71 ms (4.97x) |
| t32 | 0.94 ms (1.00x) | 1.20 ms (1.27x) | 2.10 ms (2.22x) | 3.95 ms (4.19x) |

Full MoE layer (q8(x), 8x gate/up, SwiGLU, q8(h), 8x down, weighted sum):

| kernel | weights | threads | ms per layer | GB/s | expert-only tok/s (47 layers) |
|---|---|---|---:|---:|---:|
| c2_q8_sign | dummy | t1 | 17.09 | 6.3 | 1.2 |
| c3_q8_off_r2 | dummy | t1 | 26.46 | 4.0 | 0.8 |
| c3h_q8_off | dummy | t1 | 18.01 | 5.9 | 1.2 |
| c2_q8_sign | dummy | t2 | 8.71 | 12.3 | 2.4 |
| c3_q8_off_r2 | dummy | t2 | 13.07 | 8.2 | 1.6 |
| c3h_q8_off | dummy | t2 | 9.42 | 11.4 | 2.3 |
| c2_q8_sign | dummy | t4 | 4.44 | 24.1 | 4.8 |
| c3_q8_off_r2 | dummy | t4 | 6.74 | 15.9 | 3.2 |
| c3h_q8_off | dummy | t4 | 4.80 | 22.3 | 4.4 |
| c2_q8_sign | dummy | t8 | 2.39 | 44.7 | 8.9 |
| c3_q8_off_r2 | dummy | t8 | 3.66 | 29.2 | 5.8 |
| c3h_q8_off | dummy | t8 | 2.61 | 41.0 | 8.1 |
| c2_q8_sign | dummy | t16 | 1.50 | 71.5 | 14.2 |
| c3_q8_off_r2 | dummy | t16 | 2.36 | 45.3 | 9.0 |
| c3h_q8_off | dummy | t16 | 1.60 | 66.8 | 13.3 |
| c2_q8_sign | dummy | t32 | 1.51 | 70.9 | 14.1 |
| c3_q8_off_r2 | dummy | t32 | 3.74 | 28.6 | 5.7 |
| c3h_q8_off | dummy | t32 | 3.06 | 34.9 | 6.9 |
| c2_q8_sign | real L1 | t16 | 1.53 | 69.9 | 13.9 |
| c3_q8_off_r2 | real L1 | t16 | 2.27 | 47.2 | 9.4 |
| c3h_q8_off | real L1 | t16 | 1.55 | 68.9 | 13.7 |
| c2_q8_sign | real L1 | t32 | 1.50 | 71.3 | 14.2 |
| c3_q8_off_r2 | real L1 | t32 | 2.10 | 51.0 | 10.1 |
| c3h_q8_off | real L1 | t32 | 1.51 | 70.9 | 14.1 |

### GPU tensor-core peak per instruction (register-only loop)

| instruction | dense TFLOPS / TOPS |
|---|---:|
| `s8 m16n8k32 (int8 -> s32)` | 151.7 |
| `e4m3 m16n8k32 (fp8 -> f32)` | 76.2 |
| `f16 m16n8k16 (-> f32 acc)` | 38.1 |
| `f16 m16n8k16 (-> f16 acc)` | 75.9 |
| `bf16 m16n8k16 (-> f32 acc)` | 38.1 |
| `f8f6f4 e2m1 m16n8k32 (no scale)` | 76.1 |
| `mxf8f6f4 e2m1xe4m3 m16n8k32 1X` | 151.7 |
| `mxf4 e2m1 m16n8k64 2X ue8m0` | 303.4 |

### GPU decode, dummy weights: 8 experts, best launch config per kernel

N=1: time per call, bandwidth, error vs fp64 (error includes the activation format).

| kernel | gate+up us | GB/s | down us | GB/s | relL2 |
|---|---:|---:|---:|---:|---:|
| g1_dp4a | 170.2 | 419.1 | 86.1 | 414.1 | 1.0e-02 |
| g2_hfma2 | 171.3 | 416.2 | 86.8 | 410.5 | 6.5e-04 |
| g2_hmul_f32 | 173.2 | 411.7 | 88.0 | 405.0 | 2.3e-04 |
| k0_i8_mma | 184.6 | 386.2 | 94.2 | 378.4 | 1.0e-02 |
| k1_f8fold_mma | 188.2 | 379.0 | 97.6 | 365.3 | 2.0e-02 |
| k2_f16fold_mma | 187.8 | 379.6 | 95.8 | 372.2 | 2.0e-04 |
| k3_mxf8f6f4 | 178.8 | 398.8 | 90.8 | 392.6 | 3.0e-02 |
| k4_mxf4 | 177.0 | 402.9 | 89.3 | 399.3 | 1.6e-01 |
| k5_msd_mxf4x2 | 177.9 | 400.7 | 90.4 | 394.4 | 2.4e-02 |

Small batch (MTP verify), gate+up of 8 experts, us per call:

| kernel | N=1 | N=2 | N=4 | N=8 | N=16 |
|---|---:|---:|---:|---:|---:|
| g1_dp4a | 170.2 | 171.2 | 175.7 | 281.8 | 376.6 |
| g2_hfma2 | 171.3 | 179.0 | 266.1 | 513.5 | 1193.3 |
| g2_hmul_f32 | 173.2 | 190.5 | 321.6 | 703.3 | 1329.7 |
| k0_i8_mma | 184.6 | 184.5 | 185.1 | 185.8 | 191.9 |
| k1_f8fold_mma | 188.2 | 189.2 | 189.2 | 211.5 | 192.6 |
| k2_f16fold_mma | 187.8 | 188.1 | 188.4 | 189.9 | 220.4 |
| k3_mxf8f6f4 | 178.8 | 178.2 | 179.4 | 180.7 | 184.3 |
| k4_mxf4 | 177.0 | 177.4 | 178.7 | 179.6 | 181.5 |
| k5_msd_mxf4x2 | 177.9 | 179.3 | 179.0 | 179.9 | 183.8 |

Fused MoE layer (g3, 2 launches):

| measurement | us per layer | GB/s | relL2 |
|---|---:|---:|---:|
| moe_layer | 280.9 | 380.8 | 1.6e-02 |
| moe47_graph | 283.3 | 377.5 | 0.0e+00 |

### GPU decode, real L1+L47 weights: 8 experts, best launch config per kernel

N=1: time per call, bandwidth, error vs fp64 (error includes the activation format).

| kernel | gate+up us | GB/s | down us | GB/s | relL2 |
|---|---:|---:|---:|---:|---:|
| g1_dp4a | 170.5 | 418.2 | 85.9 | 414.9 | 6.7e-03 |
| g2_hfma2 | 171.8 | 414.9 | 87.0 | 409.8 | 5.2e-04 |
| g2_hmul_f32 | 175.0 | 407.5 | 88.1 | 404.6 | 1.9e-04 |
| k0_i8_mma | 184.3 | 387.0 | 94.0 | 379.3 | 6.7e-03 |
| k1_f8fold_mma | 188.1 | 379.0 | 97.5 | 365.6 | 1.4e-02 |
| k2_f16fold_mma | 187.6 | 380.2 | 95.5 | 373.5 | 1.8e-04 |
| k3_mxf8f6f4 | 178.3 | 400.0 | 90.7 | 393.3 | 2.3e-02 |
| k4_mxf4 | 177.1 | 402.6 | 89.2 | 399.9 | 1.6e-01 |
| k5_msd_mxf4x2 | 178.1 | 400.3 | 90.6 | 393.7 | 1.7e-02 |

Small batch (MTP verify), gate+up of 8 experts, us per call:

| kernel | N=1 | N=4 | N=16 |
|---|---:|---:|---:|
| g1_dp4a | 170.5 | 175.7 | 376.6 |
| g2_hfma2 | 171.8 | 265.7 | 1225.6 |
| g2_hmul_f32 | 175.0 | 321.7 | 1233.9 |
| k0_i8_mma | 184.3 | 185.4 | 192.9 |
| k1_f8fold_mma | 188.1 | 189.4 | 192.3 |
| k2_f16fold_mma | 187.6 | 188.7 | 220.3 |
| k3_mxf8f6f4 | 178.3 | 179.8 | 184.2 |
| k4_mxf4 | 177.1 | 178.6 | 181.6 |
| k5_msd_mxf4x2 | 178.1 | 179.0 | 184.1 |

Fused MoE layer (g3, 2 launches):

| measurement | us per layer | GB/s | relL2 |
|---|---:|---:|---:|
| moe_layer | 281.7 | 379.7 | 1.3e-02 |
| moe47_graph | 281.3 | 380.2 | 0.0e+00 |
| moe_layer | 280.8 | 380.9 | 1.5e-02 |
| moe47_graph | 281.3 | 380.2 | 0.0e+00 |

### GPU prefill grouped GEMM, dummy weights, all 256 experts (best BN per kind)

TFLOPS for gate+up / down; relL2 vs fp64 (includes the activation format).

| kernel | T=512 | T=2048 | T=8192 | relL2 |
|---|---:|---:|---:|---:|
| k0_i8_mma | 14.6 / 16.5 | 29.3 / 31.4 | 39.4 / 40.3 | 9.0e-03 |
| k1_f8fold_mma | 15.0 / 17.7 | 33.4 / 34.3 | 40.1 / 40.2 | 2.4e-02 |
| k2_f16fold_mma | 13.2 / 15.0 | 21.6 / 23.3 | 24.4 / 25.4 | 2.0e-04 |
| k3_mxf8f6f4 | 15.4 / 17.5 | 45.4 / 49.2 | 64.5 / 65.3 | 3.2e-02 |
| k4_mxf4 | 23.7 / 23.6 | 74.1 / 71.4 | 107.6 / 101.0 | 1.5e-01 |
| k5_msd_mxf4x2 | 15.3 / 18.0 | 44.5 / 49.8 | 63.0 / 64.1 | 2.1e-02 |

Time per MoE layer (gate+up + down GEMM only), ms:

| kernel | T=512 | T=2048 | T=8192 |
|---|---:|---:|---:|
| k0_i8_mma | 13.6 | 27.6 | 83.1 |
| k1_f8fold_mma | 13.1 | 24.5 | 82.3 |
| k2_f16fold_mma | 15.0 | 37.2 | 133.6 |
| k3_mxf8f6f4 | 12.8 | 17.7 | 50.9 |
| k4_mxf4 | 8.7 | 11.3 | 31.3 |
| k5_msd_mxf4x2 | 12.8 | 17.9 | 52.1 |

### GPU prefill grouped GEMM, real L1 weights, all 256 experts (best BN per kind)

TFLOPS for gate+up / down; relL2 vs fp64 (includes the activation format).

| kernel | T=2048 | relL2 |
|---|---:|---:|
| k0_i8_mma | 30.6 / 31.3 | 9.7e-03 |
| k1_f8fold_mma | 32.1 / 34.3 | 2.5e-02 |
| k2_f16fold_mma | 21.8 / 23.3 | 2.3e-04 |
| k3_mxf8f6f4 | 45.4 / 49.2 | 3.5e-02 |
| k4_mxf4 | 74.2 / 71.5 | 1.5e-01 |
| k5_msd_mxf4x2 | 44.6 / 43.7 | 2.5e-02 |

Time per MoE layer (gate+up + down GEMM only), ms:

| kernel | T=2048 |
|---|---:|
| k0_i8_mma | 26.8 |
| k1_f8fold_mma | 25.1 |
| k2_f16fold_mma | 37.1 |
| k3_mxf8f6f4 | 17.7 |
| k4_mxf4 | 11.3 |
| k5_msd_mxf4x2 | 18.6 |
