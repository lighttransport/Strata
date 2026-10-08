"""Prefill model: every chunk streams the routed experts of every touched 16-expert group over PCIe while the
GPU runs MMQ GEMMs and attention (docs/GLM_Q2_DECODE_REDESIGN.md "Prefill: chunk size")."""
import dataclasses

import kernels
import model
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
    disk_s_per_chunk: float
    sync_s_per_chunk: float
    bottleneck: str

    def to_dict(self):
        return dataclasses.asdict(self)


def groups_touched(chunk, experts, group=16, top_k=8):
    distinct = experts * (1 - (1 - top_k / experts) ** chunk)
    groups = experts / group
    return 1 - (1 - 1 / groups) ** distinct


def chunk_seconds(hw, cfg, params, pack, chunk, ram_fraction=1.0):
    g = GEOMETRY
    touched = groups_touched(chunk, pack.experts)
    bytes_ = pack.routed_total_bytes() * touched + g.moe_layers * (pack.experts / 16) * 16384
    cached = min(bytes_, cfg.prefill_expert_cache_mib * 2 ** 20)
    bytes_ -= cached
    pcie_s = kernels.pcie_ms(hw, bytes_) / 1e3 / cfg.gpus
    disk_s = kernels.disk_ms(hw, bytes_ * (1 - ram_fraction)) / 1e3 * params.prefill_lazy_gain
    hidden = 0.3 if cfg.prefetch_groups > 0 else 1.0
    sync_s = g.moe_layers * params.prefill_sync_ms_per_layer / 1e3 * hidden
    # the GPU terms were fitted on the 5060 Ti (47 TFLOPS); other cards scale by their tensor throughput
    gpu_s = (params.prefill_fixed_s + params.prefill_ms_per_token / 1e3 * chunk) / cfg.gpus
    if cfg.prefill_legacy:
        gpu_s *= params.prefill_legacy_factor
    gpu_s *= 47.0 / max(1.0, hw.gpu.tflops)
    total = max(pcie_s + disk_s + sync_s, gpu_s)
    bottleneck = "pcie" if pcie_s + disk_s + sync_s >= gpu_s else "gpu"
    if disk_s > pcie_s:
        bottleneck = "disk"
    return total, bytes_, pcie_s, gpu_s, disk_s, sync_s, bottleneck


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
        total += last[0]
    seconds, bytes_, pcie_s, gpu_s, disk_s, sync_s, bottleneck = last
    return PrefillResult(tok_s=cfg.prompt / total, total_s=total, chunks=chunks, chunk_tokens=chunk,
                         pcie_gb_per_chunk=bytes_ / 1e9, pcie_s_per_chunk=pcie_s, gpu_s_per_chunk=gpu_s,
                         disk_s_per_chunk=disk_s, sync_s_per_chunk=sync_s, bottleneck=bottleneck)
