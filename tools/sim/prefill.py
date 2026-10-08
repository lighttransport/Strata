"""Prefill model: per MoE layer the GPU runs the mixer, the host stages and uploads all 16-expert groups the
chunk touches (18 for 288 experts), and the GPU runs the routed-expert GEMMs as groups arrive.

Timeline per MoE layer (docs/GLM_Q2_DECODE_REDESIGN.md "Prefill: chunk size", docs/GLM_SINGLE_OPTIMIZATION.md
P07-P11): with staged prefetch, P groups upload during the mixer; the remaining groups go through the 2-slot ring,
whose uploads overlap the GEMMs only partly (prefill_overlap). A deep stream (prefill_stream_depth >= groups)
decouples uploads from the GPU entirely: the layer costs max(mixer + GEMM, uploads).
"""
import dataclasses

import kernels
import model
import routing
from model import GEOMETRY


@dataclasses.dataclass
class PrefillResult:
    tok_s: float
    total_s: float
    chunks: int
    chunk_tokens: int
    pcie_gb_per_chunk: float
    pcie_s_per_chunk: float
    gpu_s_per_chunk: float
    mixer_s_per_chunk: float
    gemm_s_per_chunk: float
    cpu_s_per_chunk: float
    disk_s_per_chunk: float
    bottleneck: str
    experts: str

    def to_dict(self):
        return dataclasses.asdict(self)


def groups_touched(chunk, experts, group=16, top_k=8):
    distinct = experts * (1 - (1 - top_k / experts) ** chunk)
    groups = experts / group
    return 1 - (1 - 1 / groups) ** distinct


REFERENCE_GPU = dict(tflops=47.0, bandwidth_gbps=448.0)   # the terms below were fitted on the RTX 5060 Ti


def _mixer_ms(hw, cfg, params, layer, tokens):
    # the mixers are bandwidth- and latency-bound kernels: scale with the card's memory bandwidth
    scale = REFERENCE_GPU["bandwidth_gbps"] / max(1.0, hw.gpu.bandwidth_gbps)
    if layer in GEOMETRY.mla_layers:
        us = params.prefill_mla_us * (params.prefill_mla_f16_factor if cfg.prefill_mla_f16 else 1.0)
    else:
        us = params.prefill_kda_us * (params.prefill_kda_parts_factor if cfg.prefill_kda_parts else 1.0)
    return tokens * us * 1e-3 * scale + params.prefill_layer_fixed_ms


def _gemm_ms(hw, cfg, params, tokens, groups):
    # the MoE GEMMs are compute-bound: scale with tensor throughput
    scale = REFERENCE_GPU["tflops"] / max(1.0, hw.gpu.tflops)
    us = params.prefill_gemm_us * (params.prefill_legacy_gemm if cfg.prefill_legacy else 1.0) / max(0.05, cfg.prefill_gemm_scale)
    return tokens * us * 1e-3 * scale + groups * params.prefill_group_fixed_ms


def _cpu_expert_ms(hw, cfg, params, pack, layer, tokens, ram_fraction=1.0):
    """CPU computes the chunk's experts of one layer: the union of touched experts streams once, every route is a
    MAC; experts missing from a RAM tier are read from disk first (all routes are known after the router)."""
    g = GEOMETRY
    distinct = pack.experts * (1 - (1 - g.top_k / pack.experts) ** tokens)
    bytes_ = distinct * pack.expert_bytes(layer)
    macs = tokens * g.top_k * g.expert_macs()
    ms, _ = kernels.cpu_roofline(hw, params, bytes_, macs, 1, hw.cpu.workers(cfg.threads), pack.kernel_efficiency(layer))
    if ram_fraction < 1.0:
        ms += kernels.disk_ms(hw, bytes_ * (1 - ram_fraction), parallel=True)
    return ms


def chunk_seconds(hw, cfg, params, pack, chunk, ram_fraction=1.0):
    g = GEOMETRY
    groups_per_layer = pack.experts / 16
    touched = groups_touched(chunk, pack.experts)
    groups = groups_per_layer * touched
    group_bytes = pack.expert_bytes(3) * 16 + 16384
    gpus = max(1, cfg.gpus)
    experts = cfg.prefill_experts
    if experts == "auto":
        gpu_est = groups * g.moe_layers * kernels.pcie_ms(hw, group_bytes)
        cpu_est = sum(_cpu_expert_ms(hw, cfg, params, pack, l, chunk, ram_fraction) for l in g.moe_layer_ids())
        experts = "cpu" if cpu_est < gpu_est else "gpu"

    pcie_bytes = 0.0
    pcie_ms = disk_ms = gemm_ms = mixer_ms = cpu_ms = 0.0
    total_ms = 0.0
    cpu_tokens = 0
    if experts == "gpu" and cfg.prefill_cpu_assist:
        # the CPU takes tokens off the GPU GEMMs while the GPU is upload-bound: as many as its dot rate allows
        rate = hw.cpu.peak_gflops() / 2 * pack.kernel_efficiency(3) * params.cpu_quant_scale * 1e9 / g.routed_macs_per_token()
        upload_s = groups * g.moe_layers * kernels.pcie_ms(hw, group_bytes) / 1e3 / gpus
        cpu_tokens = int(min(chunk // 2, upload_s * rate * 0.8))
    for layer in g.moe_layer_ids():
        mixer = _mixer_ms(hw, cfg, params, layer, chunk)
        mixer_ms += mixer
        if experts == "cpu":
            cpu = _cpu_expert_ms(hw, cfg, params, pack, layer, chunk, ram_fraction)
            cpu_ms += cpu
            total_ms += mixer + cpu + kernels.handoff_ms(hw)
            continue
        per_group = kernels.pcie_ms(hw, group_bytes) / gpus
        if ram_fraction < 1.0:
            per_group += kernels.disk_ms(hw, group_bytes * (1 - ram_fraction)) * params.prefill_lazy_gain / gpus
        uploads = groups * per_group
        pcie_bytes += groups * group_bytes
        pcie_ms += uploads
        gemm = _gemm_ms(hw, cfg, params, chunk - cpu_tokens, groups) / gpus
        gemm_ms += gemm
        if cfg.prefill_stream_depth >= groups_per_layer:
            layer_ms = max(mixer + gemm, uploads)
        else:
            prefetch = min(cfg.prefetch_groups, groups) * per_group
            hidden = min(mixer, prefetch)          # uploads that finish under the mixer
            remaining = uploads - hidden           # the rest overlaps the GEMMs through the ring
            layer_ms = mixer + remaining + gemm - params.prefill_overlap * min(gemm, remaining)
        if cpu_tokens:
            cpu = _cpu_expert_ms(hw, cfg, params, pack, layer, cpu_tokens, ram_fraction)
            cpu_ms += cpu
            layer_ms = max(layer_ms, cpu)
        total_ms += layer_ms
    dense = chunk * params.prefill_dense_us * 1e-3 * REFERENCE_GPU["bandwidth_gbps"] / max(1.0, hw.gpu.bandwidth_gbps)
    total_ms += dense
    gpu_ms = mixer_ms + gemm_ms + dense
    if experts == "cpu":
        bottleneck = "cpu" if cpu_ms > mixer_ms else "gpu"
    elif cfg.prefill_stream_depth >= groups_per_layer:
        bottleneck = "pcie" if pcie_ms >= gpu_ms else "gpu"
    else:
        bottleneck = "pcie" if pcie_ms >= gpu_ms * 0.6 else "gpu"
        if ram_fraction < 1.0 and kernels.disk_ms(hw, pcie_bytes * (1 - ram_fraction)) > pcie_ms * 0.5:
            bottleneck = "disk"
    return dict(seconds=total_ms / 1e3, pcie_bytes=pcie_bytes, pcie_s=pcie_ms / 1e3, gpu_s=gpu_ms / 1e3,
                mixer_s=mixer_ms / 1e3, gemm_s=gemm_ms / 1e3, cpu_s=cpu_ms / 1e3,
                disk_s=(kernels.disk_ms(hw, pcie_bytes * (1 - ram_fraction)) / 1e3 if ram_fraction < 1.0 else 0.0),
                bottleneck=bottleneck, experts=experts)


def simulate(hw, cfg, params=None):
    params = params or kernels.Params()
    pack = model.pack(cfg.pack)
    ram_fraction = 1.0
    if cfg.placement == "ram_tier":
        expert_gib = cfg.ram_expert_gib or max(0.0, hw.memory.gib - 6.0)
        ram_fraction = min(1.0, expert_gib / (pack.routed_total_bytes() / 2 ** 30))
    chunk = max(1, min(cfg.prefill_chunk, cfg.prompt))
    chunks = (cfg.prompt + chunk - 1) // chunk
    total = 0.0
    last = None
    for i in range(chunks):
        tokens = min(chunk, cfg.prompt - i * chunk)
        last = chunk_seconds(hw, cfg, params, pack, tokens, ram_fraction)
        total += last["seconds"]
    return PrefillResult(tok_s=cfg.prompt / total, total_s=total, chunks=chunks, chunk_tokens=chunk,
                         pcie_gb_per_chunk=last["pcie_bytes"] / 1e9, pcie_s_per_chunk=last["pcie_s"],
                         gpu_s_per_chunk=last["gpu_s"], mixer_s_per_chunk=last["mixer_s"], gemm_s_per_chunk=last["gemm_s"],
                         cpu_s_per_chunk=last["cpu_s"], disk_s_per_chunk=last["disk_s"], bottleneck=last["bottleneck"],
                         experts=last["experts"])
