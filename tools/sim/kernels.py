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
    # prefill (per MoE layer of one chunk): mixer, 18 group uploads, MoE GEMMs; see prefill.py
    prefill_kda_us: float = 36.0       # KDA mixer per token per layer (FP32 chunked recurrence)
    prefill_kda_parts_factor: float = 0.7   # with 4 row parts (P11)
    prefill_mla_us: float = 58.8      # MLA mixer per token per layer with FP32 products
    prefill_mla_f16_factor: float = 0.468     # with FP16 products (P10)
    prefill_gemm_us: float = 21.4      # routed-expert GEMMs (MMQ) per token per layer
    prefill_legacy_gemm: float = 1.87  # dequantize-per-use before P04
    prefill_overlap: float = 0.58      # share of the GEMM time hidden under the ring's remaining uploads
    prefill_group_fixed_ms: float = 0.73     # per group: staging handoff, gather/scatter launches
    prefill_layer_fixed_ms: float = 30.8     # per MoE layer: router, hc, norms, events
    prefill_dense_us: float = 0.62      # dense layers and head per token
    prefill_lazy_gain: float = 1.0     # SSD reads overlap with PCIe this much (1 = fully serialized)
    ram_tier_wait_ms: float = 1.5      # exposed wait per disk-fetched expert on top of its bytes
    prefetch_accuracy: float = 0.9     # share of predicted next-layer routes that are right (FATE: 97 %, DraftExpert: 86-88 %)
    tail_topk_acceptance: float = 0.97  # acceptance multiplier per halving of experts on tapered draft positions (AcceptMoE: -0.27 pt)
    ecospec_union: float = 0.85        # window union multiplier with cost-aware drafts (EcoSpec reuses active experts)
    ecospec_acceptance: float = 0.98   # acceptance multiplier for the cost-aware choice
    selfspec_expert_mib: float = 24.0  # one DraftExpert per layer (about a routed expert's size at Q8)
    tier_expert_us: float = 30.0       # per-expert launch/tile cost of the resident kernel (0.575 ms for 8 x 8 rows)
    cold_tier_tokens: float = 256.0    # tokens after prefill before the adaptive tier is warm
    cold_tier_speed: float = 0.75      # decode speed during that period relative to warm (measured 13-18 vs 21)
    truncation_efficiency: float = 0.5  # share of would-be-rejected draft positions an adaptive window leaves out


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


def gpu_tier_ms(hw, bytes_, experts=0.0, params=None):
    """Resident-expert kernel: bytes at the tier rate plus a per-expert tile cost (GLM_BATCH_DECODE.md table)."""
    per = (params.tier_expert_us if params else 0.0) * 1e-3 * experts
    return bytes_ / (hw.gpu.tier_gbps * 1e9) * 1e3 + per


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
