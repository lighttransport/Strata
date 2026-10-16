"""Muse Spark Open 816B-A42B on two nodes (TR16 + B550 over InfiniBand): a per-layer decode and prefill model.

The model is too large for one node at any quantization that keeps quality (UD-IQ1_S is 175.6 GB; TR16 has 125 GiB
and the B550 62 GiB), so the routed experts are split across both nodes' RAM and every decode step crosses the link.
Per layer, a token runs

    GPU (owner of the layer's fixed weights): attention + ffn_latent_down + router + GPU-resident experts
      -> latent (4096 floats) to the CPU expert lanes (local mailbox; the other node over the link)
    CPU expert lanes on both nodes (or one, see `split`), in parallel
      -> partial latents back, GPU: ffn_latent_up + residual

so a decode step is a sum over the 62 layers of the GPU chain plus the slower of the two CPU lanes (and its network
round trip), plus the head and the sampler. Prefill streams each layer's experts to a GPU and is GEMM bound.

Every constant below is either measured (named in the comment: machine, date, tool) or marked as an estimate. The
architecture and every byte count come from the GGUF census (data/census-muse-spark-ud-iq1s.json). Not modelled:
quality, CUDA-graph replay (launch gaps are the measured non-graph values), the multimodal encoder.

Split kinds (where the routed experts live):
  row    every expert's rows are split, node B holds `b_share` of each (balanced work, needs a repack per node);
         ffn_post_exp_norm normalizes each expert's output before the router weights, so the remote node returns
         one partial latent per (token, expert) instead of one sum
  expert whole experts; node B holds the `b_share` hottest after the GPU caches (uneven per-token load, modelled
         exactly as a multinomial over the 8 routes)
  layer  pipeline: a layer's experts live on the node whose GPU owns the layer (no per-layer network trip, but the
         two nodes' CPUs never work at the same time on one token)
"""
import copy
import dataclasses
import functools
import itertools
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
CENSUS = DATA / "census-muse-spark-ud-iq1s.json"
GLM_KERNELS = DATA / "glm_kernels_tr16.json"


# ----------------------------------------------------------------------------- architecture
@dataclasses.dataclass(frozen=True)
class Arch:
    layers: int
    hidden: int
    latent: int
    expert_ff: int
    experts: int
    top_k: int
    q_heads: int
    kv_heads: int
    head_dim: int
    window: int
    full_layers: frozenset
    vocab: int
    expert_bytes: int          # one routed expert (gate + up + down), bytes
    layer_fixed_bytes: int     # attention + latent projections + router + norms of one layer
    lm_head_bytes: int
    token_embd_bytes: int
    layer_params: int          # weight elements of `layer_fixed_bytes` (prefill FLOPs)
    total_bytes: int

    @property
    def expert_macs(self):
        return 3 * self.latent * self.expert_ff

    @property
    def experts_total_bytes(self):
        return self.expert_bytes * self.experts * self.layers

    def kv_bytes(self, layer, context):
        """f16 K and V the decode attention reads for one token at `context` positions."""
        span = context if layer in self.full_layers else min(context, self.window)
        return span * self.kv_heads * self.head_dim * 2 * 2

    def attention_flops(self, layer, tokens, prior):
        """Score + value FLOPs of `tokens` new positions after `prior` cached ones (causal)."""
        span = prior + tokens / 2
        if layer not in self.full_layers:
            span = min(span, self.window)
        return 4 * tokens * span * self.q_heads * self.head_dim


@functools.lru_cache(maxsize=None)
def arch(census=str(CENSUS)):
    data = json.loads(Path(census).read_text())
    hp, k = data["hyperparameters"], data["kinds"]
    layers = hp["block_count"]
    per_layer = {n: v for n, v in k.items() if v["count"] == layers}
    experts = {n: v for n, v in per_layer.items() if "_exps" in n}
    fixed = {n: v for n, v in per_layer.items() if "_exps" not in n}
    return Arch(layers=layers, hidden=hp["embedding_length"], latent=hp["moe_latent_size"],
                expert_ff=hp["expert_feed_forward_length"], experts=hp["expert_count"], top_k=hp["expert_used_count"],
                q_heads=hp["attention.head_count"], kv_heads=hp["attention.head_count_kv"],
                head_dim=hp["attention.key_length"], window=hp["attention.sliding_window"],
                full_layers=frozenset(data["full_attention_layers"]), vocab=data["tokens"]["vocab"],
                expert_bytes=sum(v["bytes"] for v in experts.values()) // (layers * hp["expert_count"]),
                layer_fixed_bytes=sum(v["bytes"] for v in fixed.values()) // layers,
                lm_head_bytes=k["output.weight"]["bytes"], token_embd_bytes=k["token_embd.weight"]["bytes"],
                layer_params=sum(v["elements"] for v in fixed.values()) // layers, total_bytes=data["bytes"])


@functools.lru_cache(maxsize=None)
def fixed_scale(fmt, census=str(CENSUS)):
    """Share of the fixed bytes left once the GPU copies of the Q6_K/Q8_0 trunk are re-quantized to Q4_K (lossy)."""
    if fmt == "orig":
        return 1.0
    data = json.loads(Path(census).read_text())
    rows = [v for n, v in data["kinds"].items() if v["count"] == data["hyperparameters"]["block_count"]
            and "_exps" not in n and set(v["formats"]) - {"F32"}]
    return sum(v["elements"] * 4.5 / 8 for v in rows) / sum(v["bytes"] for v in rows)


@functools.lru_cache(maxsize=None)
def hit_curve():
    """Coverage -> share of routes served, measured on GLM-5.3-Flash Q23 (LRU tier, tools/sim/data/glm_kernels_tr16.json,
    unscaled). Muse Spark has no routing trace yet: this borrows GLM's skew."""
    t = json.loads(GLM_KERNELS.read_text())
    return tuple(tuple(p) for p in t["model"]["hit_curve"]["points"])


def interpolate(points, x):
    if x <= 0:
        return 0.0
    if x <= points[0][0]:
        return points[0][1] * x / points[0][0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1] + (1 - points[-1][1]) * min(1.0, (x - points[-1][0]) / max(1e-9, 1 - points[-1][0]))


def route_share(coverage, scale=1.0):
    """Share of routes that land in the hottest `coverage` fraction of a layer's experts."""
    if coverage >= 1.0:
        return 1.0
    return min(1.0, interpolate(hit_curve(), coverage) * scale)


# ----------------------------------------------------------------------------- hardware
@dataclasses.dataclass
class Gpu:
    name: str
    vram_mib: float
    reserved_mib: float          # desktop + driver + runtime scratch at decode
    gemv_gbps: float             # dense decode GEMV, effective
    expert_gbps: float           # resident routed-expert kernel (8..24 experts per launch), effective
    launch_us: float             # exposed gap per kernel launch in the decode chain
    h2d_gbps: float              # host -> device, measured or estimated per slot
    proj_tflops: float           # prefill dense projections (quantized GEMM), effective
    expert_tflops: float         # prefill routed-expert GEMM, effective
    attn_tflops: float           # prefill attention, effective
    mailbox_us: float = 16.0     # GPU <-> CPU expert lane round trip (16 KiB each way + 3 kernels), measured 15.6-17.1


@dataclasses.dataclass
class Node:
    name: str
    weights_gib: float           # RAM the routed experts may use (total minus OS, desktop, buffers)
    cpu_gbps: float              # expert lane DRAM stream
    cpu_gmacs: dict              # IQ1_S expert lane compute at width 1, per kernel
    request_ms: float            # fixed cost per remote expert request served (barrier, unpack, reply)
    gpus: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Link:
    name: str
    rtt_us: float
    gbps: float


# CPU expert lane rates for IQ1_S, GMAC/s by tokens per distinct expert (1, 2, 4), 15 threads.
# TR16 (Threadripper 1950X, Zen 1), measured 2026-10-16 16:27 on an idle CPU with src/kernels/cpu/iq1s_expert_parity
# --bench --threads 15 --mb 1024 (whole 4096 x 4096 experts through the engine dispatch: gate/up, the hidden's Q8_K,
# down): Strata's AVX2 kernel (iq_avx2.cpp Fmt32<19>, NativeFmt::iq1s_rows) 92.2 / 156.9 / 209.2 GMAC/s (18.0 GB/s of
# expert bytes at 1 token); ggml-cpu's own IQ1_S dot 7.6 / 7.6 / 7.3, because its AVX2 path decodes with BMI2 pdep,
# microcoded on Zen 1 and Zen 2.
# B550 (Ryzen 9 3950X, Zen 2): ggml-style vec_dot bench (1 GiB of rows) on the quiet B550 2026-10-16 15:45: the same
# arithmetic without pdep 89.1 GMAC/s at 16 threads, ggml 6.8. The engine bench could not be timed there (a 26-thread
# miner held the CPU; runs under it measured 14-25 GMAC/s, and iq1s_expert_parity's checks pass on that CPU). Estimate:
# 89.1 x 15/16 x 0.887 (TR16's engine / vec_dot ratio, 92.2 / 104) at 1 token, TR16's width ratios beyond.
TR16_IQ1S_GMACS = dict(ggml={1: 7.6, 2: 7.6, 4: 7.3}, iq256={1: 92.2, 2: 156.9, 4: 209.2}, dram={1: 1e9})
_B550_1 = 89.1 * 15 / 16 * 92.2 / 104.0
B550_IQ1S_GMACS = dict(ggml={1: 6.8 * 15 / 16}, iq256={1: _B550_1, 2: _B550_1 * 156.9 / 92.2, 4: _B550_1 * 209.2 / 92.2},
                       dram={1: 1e9})


def set_lane_rate(node, kernel, one_token_gmacs):
    """Scale a node's per-width table so its 1-token rate is `one_token_gmacs` (e.g. a CPU shared with another job)."""
    table = node.cpu_gmacs[kernel]
    factor = one_token_gmacs / table[1]
    node.cpu_gmacs[kernel] = {w: v * factor for w, v in table.items()}


def gmacs(node, kernel, tokens_per_expert):
    """The lane's GMAC/s at `tokens_per_expert` (linear between the measured widths, flat beyond)."""
    table = sorted(node.cpu_gmacs[kernel].items())
    x = max(1.0, tokens_per_expert)
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def rx9070xt(desktop=True, h2d=14.3):
    """RX 9070 XT: 640 GB/s peak, dense GEMV 0.7 of it (b550 preset); 4.8 us per launch (rocprofv3: 0.313 ms of gaps
    over 65.5 launches per slot, 2026-10-14); 14.3 GB/s H2D in TR16's PCIe 3 x16 slot (measured). Prefill: routed
    experts 17.3 TFLOP/s and projections 25.7 TFLOP/s effective, fitted to the GLM 8K prefill trace (191 ms per MoE
    layer for 3.30 TFLOP; 90 ms per KDA mixer for ~2.3 TFLOP); attention 20 TFLOP/s is an estimate."""
    return Gpu("RX 9070 XT", vram_mib=16304, reserved_mib=(2150 if desktop else 300) + 1024, gemv_gbps=448.0,
               expert_gbps=200.0, launch_us=4.8, h2d_gbps=h2d, proj_tflops=25.7, expert_tflops=17.3, attn_tflops=20.0)


def rtx5060ti(desktop=False, h2d=13.0):
    """RTX 5060 Ti 16 GB: 380 GB/s dense GEMV of 448 (measured on TR16); 5.4 us per launch (nsys: 0.320 ms over 59
    launches); H2D 7.18 GB/s in TR16's PCIe 3 x8 slot (measured), ~13 GB/s in the B550's PCIe 4 slot (estimate).
    Prefill: experts 13.0 TFLOP/s (GLM 8K trace, 254 ms per MoE layer), projections 19 TFLOP/s (122 ms per KDA
    mixer), attention 15 TFLOP/s (estimate)."""
    return Gpu("RTX 5060 Ti", vram_mib=15907, reserved_mib=(2150 if desktop else 300) + 1024, gemv_gbps=380.0,
               expert_gbps=235.0, launch_us=5.4, h2d_gbps=h2d, proj_tflops=19.0, expert_tflops=13.0, attn_tflops=15.0)


def tr16(gpus):
    """TR16: 125 GiB (124 usable), 8 GiB kept for the OS, desktop and buffers; the expert lane streams 65.6 GB/s
    in-run (CPU_EXPERT bytes over flow time of the GLM 8K trace, 15 threads)."""
    return Node("tr16", weights_gib=116.0, cpu_gbps=65.6, cpu_gmacs=dict(TR16_IQ1S_GMACS), request_ms=0.15, gpus=gpus)


def b550(gpus):
    """B550: 62 GiB DDR4-2133, 6 GiB kept for the OS and buffers; expert rows ~26 GB/s under load (measured,
    GLM_REMOTE_TP_COMM.md); 0.15 ms fixed per remote request (0.78 ms per 16 MB request at ~26 GB/s)."""
    return Node("b550", weights_gib=56.0, cpu_gbps=26.0, cpu_gmacs=dict(B550_IQ1S_GMACS), request_ms=0.15, gpus=gpus)


LINKS = {
    # ucomm_bench -b ib between TR16 and the B550, full QDR (2026-10-16): 15.7 us decode-shaped round trip, 3.26 GB/s
    "ib-qdr": Link("ib-qdr", 15.7, 3.26),
    "gbe": Link("1gbe", 401.0, 0.117),        # measured (UDP backend)
}


def scenarios():
    """The two physical arrangements of the two 16 GB cards."""
    return {
        # docs/GLM_MULTINODE.md: TR16 drives its display from the 9070 XT, the 5060 Ti runs headless in the B550
        "split-gpus": (tr16([rx9070xt(desktop=True)]), b550([rtx5060ti(desktop=False, h2d=13.0)])),
        # both cards in TR16 (the 5060 Ti keeps the desktop), the B550 contributes RAM and CPU only
        "gpus-on-tr16": (tr16([rx9070xt(desktop=False), rtx5060ti(desktop=True, h2d=7.18)]), b550([])),
    }


# ----------------------------------------------------------------------------- plan
@dataclasses.dataclass
class Plan:
    split: str = "row"            # row | expert | layer
    b_share: float = 0.375        # share of the expert bytes held by node B (row: whole blocks, 6/16 fits the B550)
    gpu_layers: tuple = ()        # layers per GPU, in order [node A GPUs..., node B GPUs...]; () = balanced by VRAM
    fixed: str = "orig"           # orig (Q6_K/Q8_0 as shipped) | q4 (GPU copies re-quantized to Q4_K, lossy)
    cpu_kernel: str = "iq256"     # ggml (stock ggml-cpu) | iq256 (Strata's AVX2 kernel) | dram (hypothetical: bandwidth bound)
    context: int = 8192
    accept: float = 1.0           # tokens per verify step (1.0: no speculation; Muse Spark ships no MTP head)
    width: int = 1                # verify width (draft + 1)
    hit_scale: float = 1.0        # multiplies the borrowed GLM hit curve
    gpu_experts: bool = True      # fill leftover VRAM with each GPU's hottest experts of its own layers
    top_k: int = 0                # 0 = the model's 8; fewer is a lossy lever


# GLM mixer GEMV time at width 1/2/4 on HIP: 0.31 / 0.34 / 0.38 ms -> +7.5% per extra row
WIDTH_COST = 0.075
# unique experts per route at width w (GLM decode: 0.93 at width 2); wider is an estimate
UNIQUE = {1: 1.0, 2: 0.93, 3: 0.89, 4: 0.86}
LAUNCHES_PER_LAYER = 18           # norm, q/k/v/gate, qk-norm+rope, kv append, attention, gate*o, o, residual, norm,
                                  # router, latent down, pre-norm, resident experts, post-norm, latent up, residual
HOST_STEP_MS = 1.5                # embedding gather, sampler, bookkeeping per step (estimate)
SAMPLER_MS = 0.5


def gpus_of(nodes):
    return [(ni, g) for ni, node in enumerate(nodes) for g in node.gpus]


def layer_owner(plan, nodes, a):
    """GPU index (into gpus_of) of every layer; `plan.gpu_layers` or a split proportional to free VRAM."""
    gl = gpus_of(nodes)
    counts = list(plan.gpu_layers)
    if not counts:
        free = [g.vram_mib - g.reserved_mib for _, g in gl]
        counts = [int(round(a.layers * f / sum(free))) for f in free]
        counts[-1] = a.layers - sum(counts[:-1])
    if len(counts) != len(gl) or sum(counts) != a.layers or min(counts) < 0:
        raise ValueError(f"gpu_layers {counts} must give all {a.layers} layers to the {len(gl)} GPUs")
    owner = []
    for gi, c in enumerate(counts):
        owner += [gi] * c
    return owner


def placement(plan, nodes, a=None):
    """VRAM per GPU, expert cache slots, and the RAM check. Returns dict (feasible flag + notes)."""
    a = a or arch()
    gl = gpus_of(nodes)
    owner = layer_owner(plan, nodes, a)
    scale = fixed_scale(plan.fixed)
    notes = []
    gpus = []
    last = owner[-1]
    for gi, (ni, g) in enumerate(gl):
        own = [l for l in range(a.layers) if owner[l] == gi]
        fixed = len(own) * a.layer_fixed_bytes * scale + (a.lm_head_bytes * scale if gi == last else 0)
        kv = sum(a.kv_bytes(l, plan.context) for l in own)
        free = g.vram_mib * 2 ** 20 - g.reserved_mib * 2 ** 20 - fixed - kv
        slots = max(0, int(free // a.expert_bytes)) if plan.gpu_experts and own else 0
        coverage = min(1.0, slots / (len(own) * a.experts)) if own else 0.0
        gpus.append(dict(node=ni, gpu=g.name, layers=own, fixed_gib=fixed / 2 ** 30, kv_gib=kv / 2 ** 30,
                         free_gib=free / 2 ** 30, slots=slots, coverage=coverage, hit=route_share(coverage, plan.hit_scale)))
        if free < 0:
            notes.append(f"{g.name}: fixed weights + KV exceed VRAM by {-free / 2 ** 30:.1f} GiB")
    total_gib = a.experts_total_bytes / 2 ** 30
    if plan.split == "layer":
        node_gib = [0.0] * len(nodes)
        for l in range(a.layers):
            node_gib[gl[owner[l]][0]] += a.expert_bytes * a.experts / 2 ** 30
    else:
        node_gib = [total_gib * (1 - plan.b_share), total_gib * plan.b_share]
    node_gib[0] += a.token_embd_bytes / 2 ** 30
    for ni, node in enumerate(nodes):
        if node_gib[ni] > node.weights_gib + 1e-9:
            notes.append(f"{node.name}: {node_gib[ni]:.1f} GiB of weights > {node.weights_gib:.0f} GiB available")
    return dict(owner=owner, gpus=gpus, node_gib=node_gib, feasible=not notes, notes=notes)


# ----------------------------------------------------------------------------- decode
def cpu_lane_ms(node, plan, experts, rows, a):
    """One CPU lane: `experts` distinct experts (bytes) over `rows` token x expert rows (MACs), roofline."""
    if experts <= 0:
        return 0.0
    rate = gmacs(node, plan.cpu_kernel, rows / experts)
    return max(experts * a.expert_bytes / (node.cpu_gbps * 1e9), rows * a.expert_macs / (rate * 1e9)) * 1e3


def net_ms(link, nbytes):
    return link.rtt_us / 2 / 1e3 + nbytes / (link.gbps * 1e9) * 1e3


def multinomial(n, probs):
    """All (counts, probability) of n draws over len(probs) categories."""
    k = len(probs)
    for counts in itertools.product(range(n + 1), repeat=k - 1):
        last = n - sum(counts)
        if last < 0:
            continue
        c = counts + (last,)
        p = math.factorial(n)
        for ci, pi in zip(c, probs):
            p = p / math.factorial(ci) * (pi ** ci)
        yield c, p


def decode(plan, nodes, link, a=None):
    """Per-step time and tokens/s. Returns dict with the breakdown (ms per step)."""
    a = a or arch()
    place = placement(plan, nodes, a)
    gl = gpus_of(nodes)
    owner = place["owner"]
    scale = fixed_scale(plan.fixed)
    top_k = plan.top_k or a.top_k
    w = plan.width
    routes = top_k * w * UNIQUE.get(w, 0.85)          # distinct (expert) routes per layer per step
    rows_per_route = top_k * w / routes               # token rows per distinct expert
    latent_msg = w * a.latent * 4                     # f32 latent out, f32 partial back
    totals = dict(gpu=0.0, cpu=0.0, hops=0.0, mailbox=0.0)
    exposed_net = 0.0                                 # part of `cpu` spent waiting on the remote lane + link
    for l in range(a.layers):
        gi = owner[l]
        ni, g = gl[gi]
        info = place["gpus"][gi]
        hit = info["hit"]
        gpu = ((a.layer_fixed_bytes * scale + a.kv_bytes(l, plan.context) * w) / (g.gemv_gbps * 1e9) * 1e3
               * (1 + WIDTH_COST * (w - 1)) + LAUNCHES_PER_LAYER * g.launch_us / 1e3
               + hit * routes * a.expert_bytes / (g.expert_gbps * 1e9) * 1e3)
        cold = routes * (1 - hit)
        local, remote = nodes[ni], nodes[1 - ni]
        if plan.split == "layer":
            cpu, net = cpu_lane_ms(local, plan, cold, cold * rows_per_route, a), 0.0
        elif plan.split == "row":
            s_remote = plan.b_share if ni == 0 else 1 - plan.b_share
            # each node holds a share of every expert's rows: the same experts, a share of the bytes and MACs
            t_local = cpu_lane_ms(local, plan, cold, cold * rows_per_route, a) * (1 - s_remote)
            t_remote = cpu_lane_ms(remote, plan, cold, cold * rows_per_route, a) * s_remote
            # ffn_post_exp_norm normalizes every expert's own output before the router weights apply, so a node
            # holding part of each expert's rows cannot pre-sum: it returns one f32 partial per (token, cold expert)
            reply = cold * rows_per_route * a.latent * 4
            trip = (net_ms(link, latent_msg) + net_ms(link, reply) + remote.request_ms) if s_remote > 0 else 0.0
            cpu = max(t_local, t_remote + trip)
            net = max(0.0, t_remote + trip - t_local)
        else:  # expert: node B holds the b_share hottest experts after the GPU cache; multinomial over the routes
            cov_gpu = info["coverage"]
            p_b = route_share(min(1.0, cov_gpu + plan.b_share), plan.hit_scale) - hit
            p_gpu, p_a = hit, max(0.0, 1 - hit - p_b)
            p_remote = p_b if ni == 0 else p_a
            p_local = p_a if ni == 0 else p_b
            n = max(1, int(round(routes)))
            exp_cpu = exp_net = 0.0
            for (c_gpu, c_local, c_remote), p in multinomial(n, (p_gpu, p_local, p_remote)):
                t_local = cpu_lane_ms(local, plan, c_local, c_local * rows_per_route, a)
                t_remote = cpu_lane_ms(remote, plan, c_remote, c_remote * rows_per_route, a)
                trip = (net_ms(link, latent_msg) * 2 + remote.request_ms) if c_remote else 0.0   # weighted sum back
                t = max(t_local, t_remote + trip)
                exp_cpu += p * t
                exp_net += p * max(0.0, t_remote + trip - t_local)
            cpu, net = exp_cpu * routes / n, exp_net
        totals["gpu"] += gpu
        totals["cpu"] += cpu
        exposed_net += net
        totals["mailbox"] += g.mailbox_us / 1e3 if cold > 0 else 0.0
        if l + 1 < a.layers and owner[l + 1] != gi:      # hidden state (f32) to the next GPU
            hidden = w * a.hidden * 4
            nj = gl[owner[l + 1]][0]
            totals["hops"] += (net_ms(link, hidden) if nj != ni else 0.0) + 2 * g.mailbox_us / 2 / 1e3
    tail_gpu = gl[owner[-1]][1]
    head_tail = a.lm_head_bytes * scale / (tail_gpu.gemv_gbps * 1e9) * 1e3 * (1 + WIDTH_COST * (w - 1)) + SAMPLER_MS
    step = sum(totals.values()) + head_tail + HOST_STEP_MS
    return dict(step_ms=step, tok_s=plan.accept / step * 1e3, head_tail_ms=head_tail, host_ms=HOST_STEP_MS,
                **{f"{k}_ms": v for k, v in totals.items()}, remote_wait_ms=exposed_net, placement=place,
                cpu_bound_share=totals["cpu"] / step)


# ----------------------------------------------------------------------------- prefill
def prefill(plan, nodes, link, tokens=8192, chunk=8192, a=None):
    """Prefill of `tokens` in chunks of `chunk`. Each layer's experts stream to a GPU every chunk (all 256 are routed
    at chunk >= 512), except the ones resident in VRAM. row/expert: the owner GPU runs attention; the expert GEMM is
    shared by the node's GPU(s) in proportion to the expert bytes each node holds (a node with no GPU sends its share
    over the link to the other node's GPUs). layer: each GPU runs its own layers, chunks are pipelined across GPUs.
    """
    a = a or arch()
    place = placement(plan, nodes, a)
    gl = gpus_of(nodes)
    owner = place["owner"]
    top_k = plan.top_k or a.top_k
    layer_expert_bytes = a.expert_bytes * a.experts
    chunks = max(1, math.ceil(tokens / chunk))

    def gpu_rate(gpu_ids, what):
        return sum(getattr(gl[i][1], what) for i in gpu_ids) * 1e12

    # Node B's share of the expert bytes and of the routed rows. row: both equal b_share, and the reply carries one
    # f16 partial per (token, route) because ffn_post_exp_norm needs each expert's full output; expert: B holds the
    # hottest b_share of the experts and gets their share of the rows, and replies with one weighted latent per token.
    if plan.split == "expert":
        rows_b = route_share(plan.b_share, plan.hit_scale)
        reply_per_token = a.latent * 2
    else:
        rows_b = plan.b_share
        reply_per_token = top_k * a.latent * 2

    def node_expert_ms(ni, bytes_share, rows_share, c, l):
        """The expert GEMM + stream of the experts held on node ni (bytes_share of the layer), for c tokens."""
        if bytes_share <= 0:
            return 0.0
        mine = [i for i, (n, _) in enumerate(gl) if n == ni]
        via = mine or [i for i, (n, _) in enumerate(gl) if n != ni]
        stream_bytes = layer_expert_bytes * bytes_share     # prefill scratch replaces the decode expert cache
        h2d = sum(gl[i][1].h2d_gbps for i in via) * 1e9
        stream = stream_bytes / h2d
        if not mine:
            stream = max(stream, stream_bytes / (link.gbps * 1e9))
        compute = 2 * c * top_k * a.expert_macs * rows_share / gpu_rate(via, "expert_tflops")
        if ni != gl[owner[l]][0] and mine:
            trip = (net_ms(link, c * a.latent * 2) + net_ms(link, c * reply_per_token * rows_share)) / 1e3
        else:
            trip = 0.0
        return (max(stream, compute) + trip) * 1e3

    def attention_ms(l, c, prior):
        g = gl[owner[l]][1]
        return (2 * c * a.layer_params / (g.proj_tflops * 1e12) + a.attention_flops(l, c, prior) / (g.attn_tflops * 1e12)) * 1e3

    total = 0.0
    detail = dict(attention=0.0, experts=0.0, hops=0.0)
    if plan.split == "layer":
        # stage per GPU; chunk k on stage s starts after (k, s-1) and (k-1, s)
        stages = sorted(set(owner))
        finish = {}
        for k in range(chunks):
            c = min(chunk, tokens - k * chunk)
            for si, s in enumerate(stages):
                ni = gl[s][0]
                t = 0.0
                for l in (l for l in range(a.layers) if owner[l] == s):
                    att = attention_ms(l, c, k * chunk)
                    exp = node_expert_ms(ni, 1.0, 1.0, c, l)
                    t += att + exp
                    detail["attention"] += att
                    detail["experts"] += exp
                if si + 1 < len(stages):
                    hop = net_ms(link, c * a.hidden * 2) if gl[stages[si + 1]][0] != ni else c * a.hidden * 2 / 10e9 * 1e3
                    t += hop
                    detail["hops"] += hop
                start = max(finish.get((k, si - 1), 0.0), finish.get((k - 1, si), 0.0))
                finish[(k, si)] = start + t
        total = finish[(chunks - 1, len(stages) - 1)]
    else:
        for k in range(chunks):
            c = min(chunk, tokens - k * chunk)
            for l in range(a.layers):
                att = attention_ms(l, c, k * chunk)
                shares = ((1 - plan.b_share, 1 - rows_b), (plan.b_share, rows_b))
                exp = max(node_expert_ms(ni, s, r, c, l) for ni, (s, r) in enumerate(shares))
                total += att + exp
                detail["attention"] += att
                detail["experts"] += exp
                if l + 1 < a.layers and owner[l + 1] != owner[l]:
                    ni, nj = gl[owner[l]][0], gl[owner[l + 1]][0]
                    hop = net_ms(link, c * a.hidden * 2) if nj != ni else c * a.hidden * 2 / 10e9 * 1e3
                    total += hop
                    detail["hops"] += hop
    return dict(seconds=total / 1e3, tok_s=tokens / (total / 1e3), chunk=chunk, chunks=chunks,
                **{f"{k}_s": v / 1e3 for k, v in detail.items()}, placement=place)


# ----------------------------------------------------------------------------- search
def share_grid(split, nodes, a):
    if split == "layer":
        return [0.0]
    cap = min(1.0, nodes[1].weights_gib / (a.experts_total_bytes / 2 ** 30))
    lo = max(0.0, 1 - (nodes[0].weights_gib - a.token_embd_bytes / 2 ** 30) / (a.experts_total_bytes / 2 ** 30))
    if split == "row":
        # whole 256-value blocks of every expert's 4096 rows (strata_muse_row_split): multiples of 1/16
        blocks = a.expert_ff // 256
        return [k / blocks for k in range(blocks + 1) if lo - 1e-9 <= k / blocks <= cap + 1e-9]
    grid = sorted({round(x, 3) for x in [lo + (cap - lo) * i / 8 for i in range(9)]})
    return [s for s in grid if lo - 1e-9 <= s <= cap + 1e-9]


def layer_grids(nodes, a, step=2):
    gl = gpus_of(nodes)
    if len(gl) == 1:
        return [(a.layers,)]
    if len(gl) == 2:
        return [(n, a.layers - n) for n in range(step, a.layers - step + 1, step)]
    raise ValueError("more than two GPUs is not modelled")


def search(nodes, link, base=None, splits=("row", "expert", "layer"), what="decode", tokens=8192, chunks=(2048, 4096, 8192)):
    """Best feasible plan per split kind for decode or prefill."""
    a = arch()
    base = base or Plan()
    best = {}
    for split in splits:
        for share in share_grid(split, nodes, a):
            for gpu_layers in layer_grids(nodes, a):
                plan = dataclasses.replace(base, split=split, b_share=share, gpu_layers=gpu_layers)
                place = placement(plan, nodes, a)
                if not place["feasible"]:
                    continue
                if what == "decode":
                    r = decode(plan, nodes, link, a)
                    score = r["tok_s"]
                else:
                    r = max((prefill(plan, nodes, link, tokens, c, a) for c in chunks), key=lambda x: x["tok_s"])
                    score = r["tok_s"]
                if split not in best or score > best[split][0]:
                    best[split] = (score, plan, r)
    return best
