"""Cost primitives shared by the decode and prefill models. All times in milliseconds, bytes in bytes."""
import dataclasses


@dataclasses.dataclass
class Params:
    """Calibrated free parameters. Defaults reproduce the quiet tr16 measurements (see calibration.py)."""
    cpu_bw_scale: float = 1.0          # in-model expert bandwidth / plain stream bandwidth of the same kernel
    cpu_width_penalty: float = 0.04    # bandwidth lost per extra token of width (multi-token kernel overhead)
    cpu_layer_overhead_ms: float = 0.12  # pool phase barriers, quantization, job build per MoE layer
    cpu_quant_scale: float = 1.0       # scales the per-core quantized-dot throughput
    gpu_width_factor: float = 0.185    # dense GEMV cost grows by this per extra column up to 4
    gpu_width_factor_tail: float = 0.30
    gpu_kernels_per_layer: int = 44    # about 1987 launches per token over 45 layers
    gpu_attn_us_per_layer: float = 40.0  # KDA / MLA kernel time per token per layer beyond the GEMVs
    gpu_attn_ctx_us: float = 0.12      # extra per 1k context tokens per MLA layer (indexer, top-2048 cap)
    head_fixed_ms: float = 0.8         # mHC read, norms, router of the first MoE layer
    tail_fixed_ms: float = 1.0         # logits D2H, finite check, argmax, tier planning
    gpu_layer_fixed_us: float = 100.0  # router, mHC, mailbox publish/wait kernels per MoE layer per token group
    draft_fixed_ms: float = 1.3        # per draft step beyond its head and experts (MLA, norms, launches)
    tier_upload_ms: float = 1.0        # exposed GPU time per step from adaptive-tier uploads
    prefill_fixed_s: float = 4.0       # per-chunk GPU work independent of chunk size
    prefill_ms_per_token: float = 2.0  # GEMM + attention + gather per token of chunk
    prefill_sync_ms_per_layer: float = 45.0  # router-before-upload serialization per MoE layer per chunk
    prefill_legacy_factor: float = 1.9  # GPU prefill work before FP16 MLA / dequant-once / KDA row parts
    prefill_lazy_gain: float = 1.0     # SSD reads overlap with PCIe this much (1 = fully serialized)
    ram_tier_wait_ms: float = 1.5      # exposed wait per disk-fetched expert on top of its bytes


def cpu_roofline(hw, params, bytes_, macs, width, workers, kernel_efficiency):
    """CPU expert pass of one layer: (milliseconds, 'memory' | 'compute').

    bytes_: distinct non-resident expert bytes streamed from DRAM; macs: multiply-adds over the non-resident
    routes. The memory side is the DRAM stream the kernel reaches (capped by what the worker count can pull),
    the compute side the quantized-dot rate of the cores for this expert format.
    """
    if bytes_ <= 0:
        return params.cpu_layer_overhead_ms * 0.5, "memory"
    cores = min(workers, hw.cpu.cores)
    bw = min(hw.memory.dram_gbps, cores * hw.memory.per_core_gbps) * params.cpu_bw_scale
    bw *= max(0.5, 1 - params.cpu_width_penalty * (width - 1))
    per_core_gmacs = hw.cpu.peak_gflops() / 2 / hw.cpu.cores * kernel_efficiency * params.cpu_quant_scale
    memory_ms = bytes_ / (bw * 1e9) * 1e3
    compute_ms = macs / (per_core_gmacs * cores * 1e9) * 1e3 if per_core_gmacs > 0 else float("inf")
    bound = "memory" if memory_ms >= compute_ms else "compute"
    return max(memory_ms, compute_ms) + params.cpu_layer_overhead_ms, bound


def cpu_expert_ms(hw, params, bytes_, macs, width, workers, kernel_efficiency):
    return cpu_roofline(hw, params, bytes_, macs, width, workers, kernel_efficiency)[0]


def gpu_width_factor(params, width):
    if width <= 4:
        return 1 + params.gpu_width_factor * (width - 1)
    return 1 + params.gpu_width_factor * 3 + params.gpu_width_factor_tail * (width - 4)


def gpu_gemv_ms(hw, bytes_, width, params):
    bw = hw.gpu.bandwidth_gbps * hw.gpu.gemv_efficiency
    return bytes_ / (bw * 1e9) * 1e3 * gpu_width_factor(params, width)


def gpu_tier_ms(hw, bytes_):
    return bytes_ / (hw.gpu.tier_gbps * 1e9) * 1e3


def gpu_launch_ms(hw, params, layers=1):
    return params.gpu_kernels_per_layer * hw.gpu.launch_us * 1e-3 * layers


def pcie_ms(hw, bytes_, direction="h2d"):
    bw = hw.pcie.h2d() if direction == "h2d" else hw.pcie.d2h()
    return bytes_ / (bw * 1e9) * 1e3


def disk_ms(hw, bytes_, parallel=False):
    rate = hw.disk.fetch_gbps if parallel else hw.disk.read_gbps
    return bytes_ / (rate * 1e9) * 1e3


def handoff_ms(hw):
    return hw.pcie.latency_us * 1e-3
