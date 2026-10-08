"""Decode step and MTP round model of Strata's pipelined GLM decoder.

One step of width `nt` tokens walks 45 layers. Dense layers 0-2 run on the GPU. In each MoE layer the GPU
publishes the routes, runs the shared expert and the tier-resident experts while the CPU computes the other
routed experts; the next layer's mixer waits for the CPU (docs/GLM_Q2_DECODE_REDESIGN.md, run_pipelined_step).
Split verify cuts the window into groups A (nt/2) and B and alternates them per layer, so one group's GPU work
runs under the other group's CPU pass: per layer, max(cpuA, gpuB) + max(cpuB, gpuA).
"""
import dataclasses
import math

import kernels
import model
import routing
import vram
from model import GEOMETRY, FIXED_BYTES


@dataclasses.dataclass
class StepBreakdown:
    width: int
    rows: int
    ms: float
    head_ms: float
    cpu_ms: float
    gpu_ms: float            # all GPU work in MoE layers (serial chain + shared + resident experts)
    gpu_exposed_ms: float    # GPU time the CPU waited for
    cpu_exposed_ms: float    # CPU time the GPU waited for (= cpu_ms unless the GPU is the long pole)
    handoff_ms: float
    tail_ms: float
    tier_upload_ms: float
    disk_ms: float
    pcie_ms: float
    remote_ms: float
    cpu_bytes: float
    gpu_bytes: float
    pcie_bytes: float
    disk_bytes: float
    hit_bytes_share: float
    union_ratio: float
    cpu_bound: str
    bottleneck: str
    groups: int


@dataclasses.dataclass
class DecodeResult:
    tok_s: float
    tokens_per_round: float
    round_ms: float
    draft_ms: float
    verify_ms: float
    resync_ms: float
    step: StepBreakdown
    tier_mib: float
    tier_slots: int
    hit_bytes_share: float
    cpu_gb_per_token: float
    gpu_gb_per_token: float
    disk_gb_per_token: float
    cpu_gbps: float
    bottleneck: str
    lossless: bool
    notes: list
    cpu_ceiling_tok_s: float = 0.0   # tokens per round over the CPU expert time alone (perfect GPU overlap)
    gpu_ceiling_tok_s: float = 0.0   # tokens per round over all GPU work alone (perfect CPU overlap)

    def to_dict(self):
        return dataclasses.asdict(self)


def spec_depth(cfg):
    """Draft tokens per round: 0 for ordinary decode."""
    if cfg.speculation == "mtp":
        return min(cfg.mtp_depth, cfg.max_verify_width - 1)
    if cfg.speculation == "dflash":
        return max(1, min(cfg.draft_block, cfg.max_verify_width) - 1)
    return 0


def make_plan(hw, cfg, pack):
    depth = spec_depth(cfg)
    mtp = depth if cfg.speculation == "mtp" else 0
    drafter = cfg.draft_model_mib if (cfg.speculation == "dflash" and depth > 0) else 0.0
    return vram.plan(hw, pack, cfg.context, mtp, cfg.gpu_draft_experts, cfg.dense_format,
                     cfg.gpu_budget_mib or None, cfg.reserve_mib,
                     None if cfg.decode_cache_mib < 0 else cfg.decode_cache_mib,
                     cfg.prefill_scratch_mib, cfg.prefetch_groups, cfg.gpus, drafter, depth, cfg.tier_compress)


def tier_hit_share(hw, cfg, pack, plan, params, curve=None, trace=None, prior=None):
    """Share of routed bytes served by the GPU tier for this configuration."""
    if plan.slots <= 0 or cfg.placement == "gpu_stream":
        return 0.0
    curve = curve or routing.TierCurve()
    fraction = plan.slots / pack.slots()
    if trace is not None and prior is not None and pack.experts == 288:
        h = routing.trace_tier_hit(trace, prior, plan.slots, cfg.tier_policy == "adaptive")
        if cfg.affinity > 0:
            h = h + (1 - h) * (1 - math.exp(-cfg.affinity / curve.affinity_tau))
        return h
    return curve.hit(fraction, policy=cfg.tier_policy, affinity=cfg.affinity, skew=pack.skew)


def ram_hit_share(hw, cfg, pack, curve=None):
    """ram_tier placement: share of bytes found in the host RAM expert set; 1.0 when the pack fits."""
    curve = curve or routing.TierCurve()
    expert_gib = cfg.ram_expert_gib or max(0.0, hw.memory.gib - 6.0 - 0.04 * cfg.context / 1024)
    # with remote TP the worker keeps its share of every expert's rows in its own RAM
    total_gib = pack.routed_total_bytes() / 2 ** 30 * (1 - cfg.remote_share)
    if expert_gib >= total_gib or cfg.ram_mode == "frozen":
        return 1.0, expert_gib
    h = curve.hit(expert_gib / total_gib, policy="adaptive", skew=pack.skew)
    if cfg.ram_mode == "hybrid":
        h = h + (1 - h) * (1 - math.exp(-cfg.ram_margin / 0.02))
    return min(1.0, h), expert_gib


def _group_widths(width, split):
    if split and width >= 2:
        a = width // 2
        return [a, width - a]
    return [width]


def _union(width, rows, pack, consecutive, override=None):
    g = GEOMETRY
    u = override if override is not None else routing.union_ratio(width, pack.experts, g.top_k, consecutive)
    if rows > 1:
        k_eff = g.top_k * u
        u = pack.experts * (1 - (1 - k_eff / pack.experts) ** rows) / g.top_k
    return u


def step(hw, cfg, params, width, pack=None, plan=None, hit=None, rows=1, consecutive=True, union=None):
    """Time of one decode step over `rows` sequences each contributing `width` consecutive tokens."""
    g = GEOMETRY
    pack = pack or model.pack(cfg.pack)
    plan = plan or make_plan(hw, cfg, pack)
    hit = tier_hit_share(hw, cfg, pack, plan, params) if hit is None else hit
    workers = hw.cpu.workers(cfg.threads)
    split = cfg.split_verify and width >= 2 and cfg.placement == "cpu"
    widths = _group_widths(width, split)
    unions = [_union(w, rows, pack, consecutive, union if len(widths) == 1 else None) for w in widths]
    ram_hit, _ = ram_hit_share(hw, cfg, pack) if cfg.placement == "ram_tier" else (1.0, 0)
    handoff = kernels.handoff_ms(hw)

    cpu_ms = gpu_ms = gpu_exposed = cpu_exposed = handoffs = disk = pcie = remote = 0.0
    cpu_bytes = gpu_bytes = pcie_bytes = disk_bytes = 0.0
    bound = "memory"
    layer_total = 0.0
    dense_scale = model.DENSE_FORMATS[cfg.dense_format]["bytes_scale"]
    keep = 1.0 - cfg.expert_skip
    for layer in range(g.layers):
        fixed = model.layer_fixed_bytes(layer) * dense_scale
        if layer < g.dense_layers:
            t = sum(kernels.gpu_gemv_ms(hw, fixed, rows * w, params) + kernels.gpu_launch_ms(hw, params)
                    for w in widths)
            gpu_ms += t
            layer_total += t
            continue
        one = pack.expert_bytes(layer) * g.top_k
        eff = pack.kernel_efficiency(layer)
        cpu_g, gpu_g = [], []
        for w, u in zip(widths, unions):
            cols = rows * w
            attn = params.gpu_attn_us_per_layer * 1e-3 * cols
            if layer in g.mla_layers:
                attn += params.gpu_attn_ctx_us * 1e-3 * (cfg.context / 1024) * cols
            gpu = (kernels.gpu_gemv_ms(hw, fixed - FIXED_BYTES["shared_expert"], cols, params)
                   + kernels.gpu_gemv_ms(hw, FIXED_BYTES["shared_expert"], cols, params)
                   + kernels.gpu_launch_ms(hw, params) + params.gpu_layer_fixed_us * 1e-3 + attn)
            union_bytes = one * u * keep
            resident = union_bytes * hit / cfg.tier_compress
            nonres = union_bytes * (1 - hit)
            routes = cols * g.top_k * (1 - hit) * keep
            macs = routes * g.expert_macs()
            gpu_bytes += resident
            if cfg.gpus > 1 and plan.tier_mib > 0:
                # the primary's chain only carries its own tier share; the other card's share runs in parallel
                # and its partial sums come back through the host (one extra handoff each way)
                primary_share = min(1.0, plan.primary_tier_mib / plan.tier_mib) if plan.primary_tier_mib else 0.0
                secondary = kernels.gpu_tier_ms(hw, resident * (1 - primary_share)) + 2 * handoff
                gpu = max(gpu + kernels.gpu_tier_ms(hw, resident * primary_share), secondary)
            else:
                gpu += kernels.gpu_tier_ms(hw, resident)
            if cfg.placement == "gpu_stream":
                pcie_bytes += nonres
                stream = kernels.pcie_ms(hw, nonres) + kernels.gpu_tier_ms(hw, nonres)
                pcie += stream
                gpu += stream
                cpu_g.append(0.0)
                gpu_g.append(gpu)
                continue
            if cfg.pcie_share > 0:
                # the GPU pulls a share of this layer's experts over PCIe and computes them while the CPU streams the rest
                streamed = nonres * cfg.pcie_share
                nonres -= streamed
                macs *= 1 - cfg.pcie_share
                pcie_bytes += streamed
                stream_ms = kernels.pcie_ms(hw, streamed) + kernels.gpu_tier_ms(hw, streamed)
                pcie += stream_ms
                gpu += stream_ms
            local_bytes = nonres * (1 - cfg.remote_share)
            t_cpu, bound = kernels.cpu_roofline(hw, params, local_bytes, macs * (1 - cfg.remote_share), w, workers, eff)
            if cfg.remote_share > 0 and hw.remote.expert_gbps > 0:
                remote_bytes = nonres * cfg.remote_share
                t_remote = remote_bytes / (hw.remote.expert_gbps * 1e9) * 1e3 + hw.link.rtt_us * 1e-3
                remote += max(0.0, t_remote - t_cpu) + hw.link.exposed_us * 1e-3
                t_cpu = max(t_cpu, t_remote) + hw.link.exposed_us * 1e-3
            cpu_bytes += local_bytes
            cpu_ms += t_cpu
            if cfg.placement == "ram_tier" and ram_hit < 1.0:
                fetched = nonres * (1 - ram_hit)
                disk_bytes += fetched
                fetch_ms = kernels.disk_ms(hw, fetched, parallel=True) + params.ram_tier_wait_ms * fetched / pack.expert_bytes(layer)
                disk += fetch_ms
                t_cpu += fetch_ms
            cpu_g.append(t_cpu)
            gpu_g.append(gpu)
        gpu_ms += sum(gpu_g)
        if cfg.placement == "gpu_stream":
            t = sum(gpu_g) + handoff
            gpu_exposed += sum(gpu_g)
        elif len(widths) == 2:
            # group B's GPU chain runs under group A's CPU pass and vice versa
            t = max(cpu_g[0], gpu_g[1]) + max(cpu_g[1], gpu_g[0]) + 2 * handoff
            gpu_exposed += max(0.0, gpu_g[1] - cpu_g[0]) + max(0.0, gpu_g[0] - cpu_g[1])
            cpu_exposed += min(cpu_g[0], max(cpu_g[0], gpu_g[1])) + min(cpu_g[1], max(cpu_g[1], gpu_g[0]))
        else:
            # the shared expert and the resident experts overlap the CPU; the mixer/router chain does not
            overlap = kernels.gpu_tier_ms(hw, one * unions[0] * hit) \
                + kernels.gpu_gemv_ms(hw, FIXED_BYTES["shared_expert"], rows * widths[0], params)
            serial = gpu_g[0] - overlap
            t = max(cpu_g[0], overlap) + serial + handoff
            gpu_exposed += serial + max(0.0, overlap - cpu_g[0])
            cpu_exposed += cpu_g[0]
        handoffs += handoff * len(widths)
        layer_total += t

    cols = rows * width
    tail_ms = (kernels.gpu_gemv_ms(hw, FIXED_BYTES["lm_head"], cols, params) + kernels.gpu_launch_ms(hw, params)
               + kernels.pcie_ms(hw, cols * g.vocab * 4, "d2h") + params.tail_fixed_ms)
    head_fixed = params.head_fixed_ms
    tier_up = params.tier_upload_ms if (cfg.tier_policy == "adaptive" and plan.slots > 0
                                        and cfg.placement != "gpu_stream") else 0.0
    total = layer_total + head_fixed + tail_ms + tier_up
    dense_ms = sum(kernels.gpu_gemv_ms(hw, model.layer_fixed_bytes(l) * dense_scale, rows * w, params)
                   + kernels.gpu_launch_ms(hw, params) for l in range(g.dense_layers) for w in widths)
    shares = dict(cpu=cpu_ms, gpu=gpu_ms, pcie=pcie, disk=disk, remote=remote)
    bottleneck = max(shares, key=shares.get)
    if bottleneck == "cpu" and bound == "compute":
        bottleneck = "cpu-compute"
    if cfg.placement == "cpu" and gpu_exposed > 1.5 * cpu_ms:
        bottleneck = "gpu"
    elif cfg.placement == "cpu" and gpu_exposed > 0.4 * cpu_ms:
        bottleneck = "cpu+gpu"      # the serialized GPU chain is a large share of the step
    return StepBreakdown(width=width, rows=rows, ms=total, head_ms=dense_ms + head_fixed, cpu_ms=cpu_ms,
                         gpu_ms=gpu_ms, gpu_exposed_ms=gpu_exposed, cpu_exposed_ms=cpu_exposed, handoff_ms=handoffs,
                         tail_ms=tail_ms, tier_upload_ms=tier_up, disk_ms=disk, pcie_ms=pcie, remote_ms=remote,
                         cpu_bytes=cpu_bytes, gpu_bytes=gpu_bytes, pcie_bytes=pcie_bytes, disk_bytes=disk_bytes,
                         hit_bytes_share=hit, union_ratio=sum(unions), cpu_bound=bound, bottleneck=bottleneck,
                         groups=len(widths))


def draft_step_ms(hw, cfg, params, pack, head=True, plan=None, rows=1):
    """One MTP draft step: embedding, draft MLA, 8 draft experts per row, optional LM head."""
    g = GEOMETRY
    workers = hw.cpu.workers(cfg.threads)
    ms = kernels.gpu_gemv_ms(hw, FIXED_BYTES["mtp_dense"], rows, params) + params.draft_fixed_ms
    expert_bytes = pack.expert_bytes(g.layers - 1) * g.top_k * _union(1, rows, pack, False)
    use_gpu = cfg.gpu_draft_experts and (plan is None or plan.slots == 0)
    if use_gpu:
        ms += kernels.gpu_tier_ms(hw, expert_bytes)
    else:
        eff = pack.kernel_efficiency(g.layers - 1)
        ms += kernels.cpu_expert_ms(hw, params, expert_bytes, rows * g.top_k * g.expert_macs(), 1, workers, eff)
    if head:
        ms += kernels.gpu_gemv_ms(hw, FIXED_BYTES["lm_head"], rows, params) + kernels.pcie_ms(hw, rows * g.vocab * 4, "d2h")
    return ms


def block_draft_ms(hw, cfg, params, rows=1):
    """One block-diffusion drafter forward (DFlash): the drafter's weights stream once per round, conditioned
    on target hidden states that the verify step already produced; no recurrent state, so no resync."""
    weights = cfg.draft_model_mib * 2 ** 20
    launches = cfg.draft_layers * params.gpu_kernels_per_layer * hw.gpu.launch_us * 1e-3
    return kernels.gpu_gemv_ms(hw, weights, 1, params) + launches + params.draft_fixed_ms \
        + kernels.pcie_ms(hw, rows * cfg.draft_block * 4, "d2h")


def simulate(hw, cfg, params=None, trace=None, prior=None):
    """Decode throughput for a configuration; MTP rounds when cfg.mtp_depth > 0, batch rows when cfg.batch > 1."""
    params = params or kernels.Params()
    pack = model.pack(cfg.pack)
    notes = []
    plan = make_plan(hw, cfg, pack)
    if cfg.placement == "gpu_stream":
        plan = dataclasses.replace(plan, slots=0, tier_mib=0.0)
    hit = tier_hit_share(hw, cfg, pack, plan, params, trace=trace, prior=prior)
    local_gib = pack.routed_total_bytes() / 2 ** 30 * (1 - cfg.remote_share)
    if cfg.placement == "cpu" and local_gib + 7 > hw.memory.gib + hw.memory.page_cache_gib:
        notes.append(f"local expert rows ({local_gib:.1f} GiB) do not fit host RAM; use placement=ram_tier")
    if cfg.remote_share > 0 and hw.remote.ram_gib and pack.routed_total_bytes() / 2 ** 30 * cfg.remote_share + 2 > hw.remote.ram_gib:
        notes.append(f"remote rows ({pack.routed_total_bytes() / 2 ** 30 * cfg.remote_share:.1f} GiB) exceed the worker's RAM")
    if cfg.placement == "ram_tier":
        h, gib = ram_hit_share(hw, cfg, pack)
        notes.append(f"RAM expert set {gib:.1f} GiB serves {h * 100:.1f} % of routed bytes ({cfg.ram_mode})")
    lossless = (cfg.affinity == 0 and not (cfg.placement == "ram_tier" and cfg.ram_mode != "exact")
                and not pack.name.startswith("reap") and cfg.expert_skip == 0 and cfg.tier_compress <= 1.0
                and model.DENSE_FORMATS[cfg.dense_format]["lossless"])
    depth = spec_depth(cfg)
    if cfg.speculation == "dflash" and cfg.draft_block > cfg.max_verify_width:
        notes.append(f"draft block {cfg.draft_block} capped at the engine's verify width {cfg.max_verify_width}")
    rows = max(1, cfg.batch)
    union = routing.trace_union_ratio(trace, depth + 1) if (trace is not None and depth > 0) else None
    if depth == 0:
        s = step(hw, cfg, params, 1, pack, plan, hit, rows=rows)
        tokens = float(rows)
        draft = resync = 0.0
        round_ms = verify = s.ms
    elif cfg.speculation == "dflash":
        per_row = routing.tokens_per_round(depth, cfg.draft_acceptance)
        s = step(hw, cfg, params, depth + 1, pack, plan, hit, rows=rows, union=union)
        draft = block_draft_ms(hw, cfg, params, rows)
        resync = 0.0
        verify = s.ms
        round_ms = draft + verify
        tokens = per_row * rows
    else:
        per_row = routing.tokens_per_round(depth, cfg.acceptance)
        s = step(hw, cfg, params, depth + 1, pack, plan, hit, rows=rows, union=union)
        draft = depth * draft_step_ms(hw, cfg, params, pack, True, plan, rows)
        accepted = max(0.0, per_row - (1 if cfg.skip_anchor else 0))
        resync = accepted * draft_step_ms(hw, cfg, params, pack, False, plan, rows)
        verify = s.ms
        round_ms = draft + verify + resync
        tokens = per_row * rows
    tok_s = tokens / round_ms * 1e3
    cpu_gbps = s.cpu_bytes / max(1e-9, s.cpu_ms) / 1e6 if s.cpu_ms else 0.0
    gpu_round = s.gpu_ms + s.head_ms + s.tail_ms + draft + resync
    cpu_ceiling = tokens / max(1e-9, s.cpu_ms + s.disk_ms) * 1e3 if s.cpu_ms else float("inf")
    gpu_ceiling = tokens / max(1e-9, gpu_round) * 1e3
    return DecodeResult(tok_s=tok_s, tokens_per_round=tokens, round_ms=round_ms, draft_ms=draft, verify_ms=verify,
                        resync_ms=resync, step=s, tier_mib=plan.tier_mib, tier_slots=plan.slots,
                        hit_bytes_share=hit, cpu_gb_per_token=s.cpu_bytes / 1e9 / tokens,
                        gpu_gb_per_token=s.gpu_bytes / 1e9 / tokens, disk_gb_per_token=s.disk_bytes / 1e9 / tokens,
                        cpu_gbps=cpu_gbps, bottleneck=s.bottleneck, lossless=lossless, notes=notes,
                        cpu_ceiling_tok_s=cpu_ceiling, gpu_ceiling_tok_s=gpu_ceiling)
