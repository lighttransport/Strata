"""GLM-5.3-Flash on A64FX nodes: an analytic decode/prefill cost model for 6 and 12 nodes.

No GPU, no PCIe: every stage streams weights from HBM2 on 48 SVE cores, and the nodes talk over Tofu-D. The
model is calibrated on the native 12-node TP12 runner in the gemm repository (a64fx/glm5: 35.46 decode /
412.27 prefill tok/s with UD-Q4_K_XL experts and a Q8_0 trunk) and on the uTofu micro-benchmarks of the same
repository (a64fx/utofu-tests/summary.md). Every constant names its measurement or says it is a hypothesis;
`Profile.hypotheses` lists the unmeasured ones so a report cannot pass them off as measured.

Two efficiency profiles are scored side by side:
- `current`: the TP12 runner as it is (stage times fitted to the 28 ms/token split of
  handoff-decode100-resume-20261004.md, every term `fixed + bytes/rate` with an explicit node-count scaling class);
- `redesign`: streaming at the measured 63 % of the HBM roof of the one-node 270-stage decode chain
  (clair sim-accuracy/glm53f/STATUS.md), one or two collectives per layer, split-phase barriers.

Scaling classes (how a 12-node term moves to 6 nodes): S streamed bytes / trunk_tp, C constant (replicated or
latency), L collective (`ar_steps(ranks)` tree steps + straggler skew), X context-dependent.
"""
import dataclasses
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model    # noqa: E402
import routing  # noqa: E402
from model import GEOMETRY, FIXED_BYTES  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / "data"
_COMM = json.loads((DATA / "measured_a64fx_comm.json").read_text())
_KERN = json.loads((DATA / "measured_a64fx_kernels.json").read_text())


def sdot_q23_mix():
    """The q23 pack (Q2_K gate + up, Q3_K down) through the measured single-row SDOT kernels (v2, one session):
    node GB/s at one token and the time per byte at 1..4 tokens per expert relative to one token."""
    t = _KERN["sdot_kernel_v2"]["node48_gbps_tokens_1_to_4"]
    q2, q3 = t["q2_K_2048x4096"]["v2"], t["q3_K_4096x2048"]["v2"]
    times = [2 * 84 / a + 110 / b for a, b in zip(q2, q3)]     # bytes per 256 weights of gate + up + down
    return (2 * 84 + 110) / times[0], tuple(round(x / times[0], 3) for x in times)


_Q23_RATE, _Q23_COSTS = sdot_q23_mix()
GIB = 2 ** 30

# ---------------------------------------------------------------------------------------------------------------
# Packs. The gemm runner ran UD-Q4_K_XL (Q4_K gate/up, Q5_K down): 15.33 GB per rank of 12 and 426 MB per rank per
# token against the measured 15 GB / 429 MB (handoff-decode100-20261004.md:31). The Strata release pack made from
# the FP8 weights is Q2_K gate/up + Q3_K down on every MoE layer (docs/GLM_RELEASE_Q2Q3.md:44, 110.2 GB).
_MOE = GEOMETRY.moe_layer_ids()
PACKS = dict(model.PACKS)
PACKS["ud_q4kxl"] = model.Pack("ud_q4kxl", 288, {l: ("Q4_K", "Q4_K", "Q5_K") for l in _MOE},
                               description="UD-Q4_K_XL routed experts as run on 12 A64FX nodes (183.9 GB)")
PACKS["q23_fp8"] = model.Pack("q23_fp8", 288, {l: ("Q2_K", "Q2_K", "Q3_K") for l in _MOE},
                              description="experts-q23-fp8.gguf: Q2_K gate/up + Q3_K down on all 42 layers (110.2 GB)")


def pack(name):
    if name not in PACKS:
        raise ValueError(f"unknown pack {name}; choose from {', '.join(PACKS)}")
    return PACKS[name]


# Trunk (non-expert) weight bytes relative to model.FIXED_BYTES, which counts the UD Q5_K/Q6_K mix (~5.7 bpw).
TRUNK_SCALE = dict(q5q6=1.0, q8=8.5 / 5.7, q4=model.DENSE_FORMATS["q4"]["bytes_scale"])
# Bytes every rank reads per token regardless of the trunk split in the current runner (handoff-decode100:31,
# rank logs): the sparse-attention front (q_a/kv_a/indexer of the 11 MLA layers), the F32 routers, the mHC tensors.
REPLICATED_PER_TOKEN = dict(sparse_front=272e6, router=FIXED_BYTES["router"] * GEOMETRY.moe_layers, mhc=66e6)
KDA_LAYERS = 31           # MoE layers with a KDA mixer (the three dense layers' mixers sit in the head/dense term)
MLA_LAYERS = len(GEOMETRY.mla_layers)
KDA_PROJ_BYTES_PER_RANK12 = 10.2e6   # Q8 KDA projections per layer per rank at TP12 (315 MB / 31, handoff :31)
HIDDEN_BYTES_F32 = GEOMETRY.hidden * 4
HIDDEN_BYTES_BF16 = GEOMETRY.hidden * 2
KV_BYTES_PER_TOKEN = 26.1e3          # MLA latent + indexer key, 11 layers + draft (docs/GLM53F_RELEASE.md:90)
KDA_STATE_BYTES = 146 * 2 ** 20
PREFILL_GFLOP_PER_TOKEN = 19.9       # routed 16.9 + shared 2.1 + dense 0.9 (trunk GEMMs are inside the fitted eff)


# ---------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class A64FXNode:
    """One A64FX node. Rates in GB/s, times in microseconds."""
    cores: int = 48
    cmgs: int = 4
    ghz: float = 2.0
    hbm_gib: float = 32.0
    usable_gib: float = 29.0           # after XOS and the runtime (handoff-decode100-resume:90 "29 GB free")
    stream_gbps: float = 915.0         # node read stream, SVE loads, per-thread first-touched 256 MiB: 915.1 GB/s at 48 threads,
                                       # 228.9 per CMG (measured 2026-10-18, tools/a64fx_probe/node_probe.c)
    matvec_q8_gbps: float = 351.0      # standalone cold Q8 matvec, 11.8 MB (handoff-decode100-20261004.md:52)
    chain_efficiency: float = 0.551    # one-node 270-stage decode chain: 2.75 ms vs a 1.74 ms roof at 800 GB/s = 504 GB/s absolute
                                       # (clair glm53f/STATUS.md:266-335); expressed against the measured 915 GB/s stream
    q2k_core_gbps: float = 2.02        # ggml-cpu SVE Q2_K vec_dot, measured 2026-10-18 on one node (data/measured_a64fx_kernels.json);
                                       # Q3_K 1.16, Q4_K 1.6, IQ2_XXS 0.23, IQ1_S 0.17, Q8_0 9.0 GB/s per core; all compute-bound, linear in threads
    expert_q4_gbps: float = 88.0       # in-model Q4_K/Q5_K expert stream: 426 MB in 4.9 ms (the handoff's 107 includes the router)
    int8_tops_m: dict = dataclasses.field(default_factory=lambda: {8: 1.87, 16: 2.60, 32: 3.06, 64: 3.18})  # int8 SDOT GEMM Gop/s/1000 by M (CALIBRATION.md)
    fp_gemm_tflops: float = 1.1        # w8a16 FP GEMM at M=64 (26 % of the 3.07 TFLOPS fp32 FMA peak)
    hw_barrier_us: float = 1.38        # libhwb hierarchical barrier (4 x per-CMG HW + leader combine), 48 threads: 1.38 us;
                                       # 0.12 us inside one CMG (measured 2026-10-18, node_probe.c)
    split_phase_barrier_us: float = 1.38  # = hw_barrier_us (the redesign's split-phase barrier is the libhwb one)
    omp_barrier_us: float = 3.3        # in-model: 1482 OpenMP barriers = 4.9 ms master wait per token (handoff-decode100:29);
                                       # bare flat OpenMP barrier measured 5.29 us at 48 threads, 1.73 at 12 (node_probe.c)
    team_dispatch_us: float = 4.6      # 47-thread warm team dispatch (GLM53F_STRATA.md); 1.6 us at 12 threads;
                                       # an empty `omp parallel` fork-join measures 5.24 us at 48 threads (node_probe.c)
    usable_per_cmg_gib: float = 7.0    # cmg-and-numa/README.md:66-69

    def q2k_core_rate(self):
        """Measured ggml SVE Q2_K rate per core (2.02 GB/s); the old hypothesis (Zen 1 scaled: 7.1) was 3.5x too high."""
        return self.q2k_core_gbps

    def int8_tops(self, m):
        keys = sorted(self.int8_tops_m)
        for k in keys:
            if m <= k:
                return self.int8_tops_m[k]
        return self.int8_tops_m[keys[-1]]

    to_dict = lambda self: dataclasses.asdict(self)

    def set(self, key, value):
        current = getattr(self, key)
        setattr(self, key, type(current)(value) if not isinstance(current, dict) else json.loads(value))
        return self


@dataclasses.dataclass
class TofuD:
    """Tofu-D as measured on the 12-node 2x3x2 torus (a64fx/utofu-tests/summary.md)."""
    put_us: float = 1.23               # per-hop fixed cost of a Put (summary:148: 1.23 us + bytes / 6.36 GB/s)
    link_gbps: float = 6.36
    tree_step_us: float = 4.69         # recursive-doubling allreduce of 16 KiB fp32: 23.47 us over 5 steps at 12 ranks (summary:109)
    bf16_step_factor: float = 13.61 / 23.47
    dc_fixed_us: float = 29.0          # MoE dispatch+combine at B=1 over multi-TNI: 34.6 us (summary:379-422) minus the wire
    a2a_gbps_uniform: float = 86.0     # multi-TNI all-to-all aggregate, uniform routing
    a2a_gbps_skew: float = 36.0        # same under worst-case routing skew
    rsag_8mib_ms: float = 5.7          # 8 MiB (512 tokens x 4096 fp32) mpi-rsag over 12 nodes, 32-token slabs (glm5/README.md:25)
    pp_hop_us: float = 3.2             # pipeline handoff per hop (summary:600)
    tni: int = 6
    # measured 2026-10-18 at 6/8/12 ranks on GLM geometry (data/measured_a64fx_comm.json): [1 token, 3 tokens] in us.
    # Used for those rank counts; other counts (24/48 ranks with 4 ranks per node) fall back to the step model.
    tree_measured_us: dict = dataclasses.field(default_factory=lambda: _COMM["tree_allreduce_us"])
    dc_measured_us: dict = dataclasses.field(default_factory=lambda: _COMM["dispatch_combine_us_bf16"])

    to_dict = lambda self: dataclasses.asdict(self)


def ar_steps(ranks):
    """Recursive-doubling steps: ceil(log2 N), plus a fold and broadcast step when N is not a power of two
    (measured 12 -> 5 steps, not 4)."""
    if ranks <= 1:
        return 0
    steps = math.ceil(math.log2(ranks))
    return steps + (0 if ranks & (ranks - 1) == 0 else 1)


def tree_allreduce_us(tofu, nbytes, ranks, bf16=False, straggler_us=0.0):
    """One allreduce of `nbytes` (fp32 payload size) over `ranks`. At 6, 8 and 12 ranks: the measured 16 KiB and
    48 KiB points, linear in bytes (floor: one Put per step). Elsewhere: per-step cost measured at 16 KiB, scaled by
    bytes above the 1.23 us fixed part, 0.58x for a bf16 payload. Plus the in-model straggler wait."""
    if ranks <= 1:
        return 0.0
    table = tofu.tree_measured_us.get("bf16" if bf16 else "fp32", {}).get(str(ranks)) if tofu.tree_measured_us else None
    if table:
        m = nbytes / 16384.0
        us = table[0] + (table[1] - table[0]) * (m - 1) / 2
        return max(us, ar_steps(ranks) * tofu.put_us) + straggler_us
    step = tofu.put_us + (tofu.tree_step_us - tofu.put_us) * (nbytes / 16384.0)
    if bf16:
        step *= tofu.bf16_step_factor
    return ar_steps(ranks) * step + straggler_us


def distinct_owners(ranks, top_k=8):
    """Expected number of distinct owner ranks of top_k routes: 6.0 at 12 ranks, 4.6 at 6."""
    return ranks * (1 - (1 - 1.0 / ranks) ** top_k)


def dispatch_combine_us(tofu, bytes_per_token, width, ranks, skew=1.0, top_k=8):
    """EP dispatch (hidden to the owners) + combine (partials back) for `width` tokens: the measured B=1 fixed cost
    scaled by owners, plus wire time over the TNIs. 34.6 us measured at 12 ranks, B=1, 8 KB per token."""
    table = tofu.dc_measured_us.get(str(ranks)) if tofu.dc_measured_us and bytes_per_token == HIDDEN_BYTES_BF16 else None
    if table:   # measured at 1 and 3 tokens, uniform routing; linear in width, the wire term carries the skew factor
        base = table[0] + (table[1] - table[0]) * (width - 1) / 2
        return max(base, table[0]) * (1 + (skew - 1) * 0.5)
    owners = distinct_owners(ranks, top_k)
    rounds = max(1.0, owners / tofu.tni)
    wire_bytes = 2 * width * bytes_per_token * owners
    wire_us = wire_bytes / (tofu.link_gbps * min(owners, tofu.tni) * 1e3) * skew
    # the fixed part is mostly per-Put issue/poll latency: half of it scales with the owner count (hypothesis)
    return tofu.dc_fixed_us * (0.5 + 0.5 * owners / distinct_owners(12, top_k)) * rounds + 2 * tofu.put_us * rounds + wire_us


def reduce_scatter_allgather_ms(tofu, tokens, ranks, hidden=GEOMETRY.hidden, bf16=False):
    """Prefill trunk collective (reduce-scatter + all-gather of tokens x hidden): the measured 8 MiB / 5.7 ms row,
    bandwidth-bound, scaled by bytes and by the (N-1)/N share every rank moves."""
    nbytes = tokens * hidden * (2 if bf16 else 4)
    return tofu.rsag_8mib_ms * (nbytes / (8 * 2 ** 20)) * ((ranks - 1) / ranks) / (11 / 12)


def ep_max_load(ranks, top_k=8, parts=1, trials=40000, seed=53, experts=GEOMETRY.experts):
    """Port of glm53f_expert_shard_sim.simulate to N owners: mean over trials of the busiest rank's unit count when
    top_k random experts are each split into `parts` slices on owners (expert % N + p * N // parts) % N.
    Returns (mean_max_units, ratio to the mean load top_k * parts / N)."""
    rng = random.Random(seed)
    total = 0
    for _ in range(trials):
        load = [0] * ranks
        for expert in rng.sample(range(experts), top_k):
            base = expert % ranks
            for p in range(parts):
                load[(base + p * ranks // parts) % ranks] += 1
        total += max(load)
    mean_max = total / trials
    return mean_max, mean_max / (top_k * parts / ranks)


# ---------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Layout:
    nodes: int = 12
    trunk_tp: int = 12                 # ranks that share the trunk (attention, dense, shared expert, head); 1 = replicated
    expert_layout: str = "ep_parts"    # ep: whole experts per rank (parts = 1) | ep_parts: each expert sliced over `parts`
                                       # ranks (the gemm runner: parts = 4 of 12) | tp_parts: sliced over every rank
    parts: int = 4                     # slices per expert for ep_parts; narrower slices cost slice_penalty each
    ranks_per_node: int = 1            # 4 = one rank per CMG (12-thread teams)
    trunk_format: str = "q8"           # q8 | q4 | q5q6
    pack: str = "ud_q4kxl"
    context: int = 8192
    mtp_depth: int = 0
    acceptance: str = "a64fx_best512"
    prefill_chunk: int = 4096
    bf16_collectives: bool = False

    def ranks(self):
        return self.nodes * self.ranks_per_node

    def expert_parts(self):
        if self.expert_layout == "ep":
            return 1
        if self.expert_layout == "tp_parts":
            return self.ranks()
        return max(1, min(self.parts, self.ranks()))

    def threads(self):
        return 47 if self.ranks_per_node == 1 else 12 // self.ranks_per_node * 4 - 1

    def describe(self):
        return (f"{self.nodes}n x{self.ranks_per_node} {self.expert_layout}(parts={self.parts}) trunk_tp={self.trunk_tp} "
                f"{self.trunk_format} {self.pack} ctx={self.context} mtp={self.mtp_depth}")

    to_dict = lambda self: dataclasses.asdict(self)


@dataclasses.dataclass
class Profile:
    """Per-stage constants. `current` is fitted to the 12-node stage split (handoff-decode100-resume-20261004.md:78-85,
    BEST option set at 512 tokens, 28.0 ms/token); `redesign` uses the measured standalone rates of the proposal."""
    name: str = "current"
    # scaling of fitted C-class constants from the BEST-512 binary (28.0 ms) to the qualified binary the validation
    # rows come from (25.2 ms at 512 tokens: 25.06 at 128 + the context slope)
    binary_scale: float = 25.2 / 28.0
    kda_fixed_us: float = 29.0         # per KDA layer fixed part of the projections (60 us total, 10.2 MB at 351 GB/s = 29)
    kda_small_us: float = 115.0        # conv 20 + fb/norm/decay 21 + recurrence 20 + gb/rmsnorm 21 (6 busy threads) + thread-0 wait
                                       # for projection stragglers (j2s: the small-step timers mostly measured that wait)
    kda_oproj_us: float = 14.0
    kda_skew_us: float = 18.0          # 45 us measured allreduce vs 23.5 bare tree + 3.5 wire
    sparse_front_gbps: float = 156.0   # 1.7 ms for 272 MB
    sparse_fixed_us: float = 160.0     # per MLA layer: index/MLA/o_proj fixed work at zero context (fitted to the 4.5 ms stage)
    sparse_skew_us: float = 50.0       # 73 us per sparse allreduce
    ctx_us_per_token_lo: float = 0.39  # (28.2 - 25.06) ms / (8192 - 128) tokens
    ctx_us_per_token_hi: float = 0.136 # (31.55 - 28.2) ms / (32768 - 8192)
    mhc_site_us: tuple = (86.0, 70.0)  # FFN site, end-of-layer site
    router_us: float = 18.0            # 0.76 ms / 42 layers, replicated F32 router
    expert_fixed_us: float = 0.0
    expert_q4_gbps: float = 120.0      # whole-expert Q4_K/Q5_K rate: 88 GB/s measured in-model at parts=4 x the 1.36 slice penalty
    expert_q2k_gbps: float = 78.0      # q23 pack through ggml's SVE kernels, measured 2026-10-18: Q2_K 97 GB/s, Q3_K 55.6 GB/s per
                                       # node (bytes-weighted 2:1 -> 78); the runner's own kernel is what this profile would run
    slice_penalty: float = 0.12        # per extra slice of an expert: measured 0.2109 / 0.1219 x 2 / 0.0724 x 4 ms at parts 1/2/4
                                       # (glm53f_expert_shard_sim.py, job 51040571) = 1.0 / 1.16 / 1.37
    moe_skew_frac: float = 0.72        # straggler wait before the MoE combine = frac x (busiest/mean load - 1) x expert time;
                                       # fitted to 67 us per MoE allreduce at 12 ranks, parts 4 (bare tree 23.5 us)
    moe_collective: str = "allreduce"  # allreduce (the runner today) | auto (dispatch+combine when few ranks own slices)
    head_fixed_ms: float = 0.35        # embed, sampler, finite check (fitted: head/embed/dense stage 0.9 ms)
    phase_us: float = 0.0              # redesign: per phase (split-phase barrier + dispatch)
    phases_per_moe_layer: int = 33
    phases_per_dense_layer: int = 12
    width_slope: float = 0.15          # MTP verify: step(w) = step(1) x (1 + slope (w-1) + slope2 (w-1)^2); BEST binary:
    width_slope2: float = 0.13         # 1.82x at width 3 (3 x 17 / 28 ms), 2.62x at width 4 (47/51 accepted, 47.07 tok/s)
    verify_share: float = 0.92         # verify = 92 % of MTP round time (handoff :88)
    stream_rate_mode: str = "matvec"   # matvec: node.matvec_q8_gbps | chain: node.stream_gbps x chain_efficiency
    sparse_front_sharded: bool = False
    overlap: float = 0.0               # no overlap credit: ring_attn_overlap measured +34 % when overlapping
    prefill_int8_eff: float = 0.40     # fitted: 1/412.27 = 2.43 ms/pos at 12 nodes, chunk 4096
    prefill_layer_fixed_ms: float = 0.8  # per layer per chunk, from the 128-token row (285.31 tok/s)
    prefill_ctx_us_per_1k: float = 21.7  # per position per 1K of mean position (372.39 at 32K)
    prefill_collectives_per_layer: int = 2
    cmg_fixed_scale: float = 1.0       # 4 ranks per node: scale of the fixed per-phase costs (12-thread CMG-local teams)
    expert_width_cost: tuple = ()      # Q2 experts: time per weight byte at 1..4 tokens per expert relative to 1 token (empty:
                                       # bandwidth-bound, extra tokens are free)
    hypotheses: tuple = ("expert_q2k_gbps 78 GB/s is ggml's standalone SVE rate; the runner's in-model Q2 rate is not measured",
                         "context slope applied as one term on the sparse layers",
                         "6-node attention split: 64 heads / 6 is not integral (TP-6 needs a 60+4 or TP-4 split)")

    def stream_rate(self, node):
        return node.matvec_q8_gbps if self.stream_rate_mode == "matvec" else node.stream_gbps * node.chain_efficiency

    to_dict = lambda self: dataclasses.asdict(self)


def current():
    return Profile()


def redesign():
    """The proposed runner: streaming chain at the one-node measured efficiency, fused mHC/norm phases, split-phase
    barriers, one dispatch/combine (EP) or one allreduce (TP) per MoE layer, sparse front sharded, 5 us skew."""
    return Profile(
        name="redesign", binary_scale=1.0,
        kda_fixed_us=5.0, kda_small_us=12.0, kda_oproj_us=3.0, kda_skew_us=5.0,
        sparse_front_gbps=504.0, sparse_fixed_us=25.0, sparse_skew_us=5.0,
        mhc_site_us=(23.0, 23.0),      # standalone local-gram mHC kernel (handoff: ~23 us vs 48 legacy)
        router_us=12.0, expert_fixed_us=3.0,
        expert_q4_gbps=400.0,          # Q4_K gate/up measured 156 GB/s standalone v0 kernel; chain-level streaming assumed
        expert_q2k_gbps=round(_Q23_RATE, 1),   # measured: SDOT v2 Q2_K gate/up 180.3 + Q3_K down 138.0 GB/s per node -> 160.8
                                       # bytes-weighted (job 52438878; the same Q2_K kernel ran 198 on another node, ~9 %
                                       # node spread); standalone, the in-model rate is not measured
        expert_width_cost=_Q23_COSTS,  # same kernels at 2/3/4 tokens per expert (compute-bound, measured): 1.52 / 2.11 / 2.88
        slice_penalty=0.04,            # HYPOTHESIS: 512-bit tiles keep narrow slices efficient
        moe_skew_frac=0.5, moe_collective="auto", head_fixed_ms=0.3,
        phase_us=1.38 + 1.6, phases_per_moe_layer=8, phases_per_dense_layer=4,   # libhwb barrier (measured) + dispatch
        width_slope=-1.0, width_slope2=0.0,   # mechanistic (bytes x union ratio), see decode()
        stream_rate_mode="chain", sparse_front_sharded=True,
        prefill_int8_eff=0.70, prefill_layer_fixed_ms=0.3, prefill_ctx_us_per_1k=12.0,
        prefill_collectives_per_layer=1, cmg_fixed_scale=0.7,
        hypotheses=("expert_q2k_gbps and its width costs are the standalone SDOT kernels (one node); in-model rate unmeasured",
                    "mHC 23 us per site in-model (standalone kernel time; the in-model cost was never reproduced)",
                    "KDA small steps 12 us as one full-team phase (today 82 us on 6 threads)",
                    "prefill int8 efficiency 0.70 (today 0.40 fitted; the 94 % CMG microkernel exists only standalone)",
                    "4 ranks per node: collective costs scaled from the 6/8/12-rank measurements to 24/48 ranks"),
    )


PROFILES = dict(current=current, redesign=redesign)


# ---------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class DecodeResult:
    tok_s: float
    step_ms: float          # one step of `width` tokens
    round_ms: float         # MTP round (= step for plain decode)
    tokens_per_round: float
    width: int
    trunk_ms: float
    experts_ms: float
    comm_ms: float
    phases_ms: float
    mhc_ms: float
    head_ms: float
    kda_ms: float
    sparse_ms: float
    moe_ms: float
    ctx_ms: float
    expert_bytes_per_node: float
    load_ratio: float
    fit: dict
    notes: list

    def to_dict(self):
        return dataclasses.asdict(self)


def trunk_bytes_per_rank(layout, layer):
    """Trunk bytes one rank streams for `layer`: the fixed-weight census scaled to the trunk format and split
    `trunk_tp` ways (class S)."""
    scale = TRUNK_SCALE[layout.trunk_format]
    return model.layer_fixed_bytes(layer) * scale / layout.trunk_tp


def expert_bytes_per_rank(layout, pk, layer, width, profile):
    """Mean routed-expert bytes one rank streams for one layer of a `width`-token step, and the busiest/mean load
    ratio: 1.0 when every rank holds a slice of every expert, else ep_max_load over `parts` slices per expert."""
    g = GEOMETRY
    union = routing.union_ratio(width, pk.experts, g.top_k)
    layer_bytes = pk.expert_bytes(layer) * g.top_k * union
    ranks = layout.ranks()
    parts = layout.expert_parts()
    if parts >= ranks:
        return layer_bytes / ranks, 1.0
    _, ratio = _load_ratio(ranks, g.top_k, parts)
    return layer_bytes / ranks, ratio


_LOAD_CACHE = {}


def _width_cost(profile, pk, width):
    """Compute-bound Q2 expert kernels: time per byte at `width / union` tokens per distinct expert, interpolated in the
    measured 1..4-token table (linear extrapolation beyond 4)."""
    costs = profile.expert_width_cost
    if width <= 1 or not costs or pk.formats[3][0] == "Q4_K":
        return 1.0
    n = width / routing.union_ratio(width, pk.experts, GEOMETRY.top_k)
    if n <= 1:
        return 1.0
    k = int(n)                                   # n in [k, k + 1]; costs[k - 1] is the k-token point
    if k >= len(costs):
        return costs[-1] + (costs[-1] - costs[-2]) * (n - len(costs))
    return costs[k - 1] + (costs[k] - costs[k - 1]) * (n - k)


def _load_ratio(ranks, top_k, parts):
    key = (ranks, top_k, parts)
    if key not in _LOAD_CACHE:
        _LOAD_CACHE[key] = ep_max_load(ranks, top_k, parts, trials=20000)
    return _LOAD_CACHE[key]


def context_ms(layout, profile):
    """Context-dependent decode cost (class X), fitted on the 128 / 8K / 32K rows of the TP12 runner."""
    ctx = layout.context
    lo = min(ctx, 8192) * profile.ctx_us_per_token_lo
    hi = max(0, ctx - 8192) * profile.ctx_us_per_token_hi
    scale = 1.0 if profile.name == "current" else 0.5     # redesign: tiled indexer/MLA at the chain rate (hypothesis)
    return (lo + hi) * 1e-3 * scale


def _step(layout, node, tofu, profile, pk, width):
    """One decode step of `width` tokens: per-layer sum of mixer, mHC, FFN and collectives; returns the stage view."""
    g = GEOMETRY
    R, N = layout.ranks(), layout.nodes
    tp = layout.trunk_tp
    rpn = layout.ranks_per_node
    rate = profile.stream_rate(node) / rpn            # a CMG rank streams from its own HBM quarter
    bs = profile.binary_scale
    fixed_scale = profile.cmg_fixed_scale if rpn > 1 else 1.0
    bf16 = layout.bf16_collectives
    hidden_ar = HIDDEN_BYTES_F32
    notes = []
    trunk = experts = comm = phases = mhc = kda = sparse = moe = dense = 0.0
    expert_bytes_total = 0.0
    load_ratio = 1.0
    width_factor_trunk = 1 + 0.05 * (width - 1)
    width_factor_comm = 1 + 0.10 * (width - 1)
    parts = layout.expert_parts()
    exp_rate = (profile.expert_q4_gbps if pk.formats[3][0] == "Q4_K" else profile.expert_q2k_gbps) / rpn
    exp_rate /= 1 + profile.slice_penalty * (parts - 1)
    owner_ranks = min(R, distinct_owners(R, g.top_k) * parts) if parts < R else R
    for layer in range(g.layers):
        is_mla = layer in g.mla_layers
        is_dense = layer < g.dense_layers
        # ---- mixer
        if is_dense:
            t = trunk_bytes_per_rank(layout, layer) * width_factor_trunk / (rate * 1e6) + 0.03 * bs
            trunk += t
            c = tree_allreduce_us(tofu, hidden_ar * width, tp, bf16, profile.kda_skew_us * bs) * 1e-3 * width_factor_comm
            comm += c
            dense += t + c
            phases += profile.phases_per_dense_layer * (profile.phase_us + node.split_phase_barrier_us) * fixed_scale * 1e-3
            continue
        if is_mla:
            front = REPLICATED_PER_TOKEN["sparse_front"] / MLA_LAYERS
            if profile.sparse_front_sharded:
                front /= tp
            t_front = front * width_factor_trunk / (profile.sparse_front_gbps / rpn * 1e6)
            t_fixed = profile.sparse_fixed_us * bs * fixed_scale * 1e-3 * width
            c = tree_allreduce_us(tofu, hidden_ar * width, tp, bf16, profile.sparse_skew_us * bs) * 1e-3 * width_factor_comm
            sparse += t_front + t_fixed + c
            trunk += t_front
            comm += c
        else:
            proj_bytes = KDA_PROJ_BYTES_PER_RANK12 * 12 / tp * (TRUNK_SCALE[layout.trunk_format] / TRUNK_SCALE["q8"])
            t_proj = profile.kda_fixed_us * bs * fixed_scale * 1e-3 + proj_bytes * width_factor_trunk / (rate * 1e6)
            t_small = (profile.kda_small_us + profile.kda_oproj_us) * bs * fixed_scale * 1e-3 * width
            c = tree_allreduce_us(tofu, hidden_ar * width, tp, bf16, profile.kda_skew_us * bs) * 1e-3 * width_factor_comm
            kda += t_proj + t_small + c
            trunk += t_proj
            comm += c
        # ---- mHC, two sites per layer, replicated, no MPI
        m = sum(profile.mhc_site_us) * bs * fixed_scale * 1e-3 * (1 + 0.3 * (width - 1))
        mhc += m
        # ---- FFN
        router = profile.router_us * bs * fixed_scale * 1e-3 * width
        shared_bytes = FIXED_BYTES["shared_expert"] * TRUNK_SCALE[layout.trunk_format] / tp
        t_shared = shared_bytes * width_factor_trunk / (rate * 1e6)
        eb, ratio = expert_bytes_per_rank(layout, pk, layer, width, profile)
        load_ratio = ratio
        expert_bytes_total += eb
        t_exp = eb / (exp_rate * 1e6) * _width_cost(profile, pk, width) + profile.expert_fixed_us * 1e-3
        # the busiest rank finishes (ratio - 1) x the mean expert time later; everyone waits for it at the combine
        skew = profile.moe_skew_frac * max(0.0, ratio - 1) * t_exp * 1e3
        if profile.moe_collective == "allreduce" or owner_ranks >= R / 2:
            c = tree_allreduce_us(tofu, hidden_ar * width, R, bf16, skew) * 1e-3 * width_factor_comm
        else:
            c = (dispatch_combine_us(tofu, HIDDEN_BYTES_BF16, width, R) + skew) * 1e-3
            if tp > 1 and tp != R:
                c += tree_allreduce_us(tofu, hidden_ar * width, tp, bf16, 0.0) * 1e-3 * width_factor_comm
        moe += router + t_shared + t_exp + c
        trunk += t_shared
        experts += t_exp
        comm += c
        phases += profile.phases_per_moe_layer * (profile.phase_us + node.split_phase_barrier_us) * fixed_scale * 1e-3
    # ---- head: vocab-sharded lm_head, allgather of the logits slice, fixed sampler/embed work
    head_bytes = FIXED_BYTES["lm_head"] * TRUNK_SCALE[layout.trunk_format] / tp
    head = head_bytes * width_factor_trunk / (rate * 1e6) + profile.head_fixed_ms * bs \
        + tree_allreduce_us(tofu, g.vocab // tp * 4 * width, tp, False, 0.0) * 1e-3
    ctx = context_ms(layout, profile) * width
    if profile.name == "current":
        # the fitted stage times already contain the barriers and dispatches: report, do not add
        phases_report = (1482 * node.omp_barrier_us + 258 * node.team_dispatch_us) * 1e-3 * bs
        phases_add = 0.0
    else:
        phases_report = phases
        phases_add = phases
    # stage view (dense + kda + sparse + moe) and resource view (trunk + experts + comm + fixed parts) are the same
    # layer costs; the total is the stage view plus what sits outside the layers
    total = dense + kda + sparse + moe + mhc + head + ctx + phases_add
    if layout.trunk_tp == 1:
        notes.append(f"replicated trunk: every node streams {FIXED_BYTES['total'] * TRUNK_SCALE[layout.trunk_format] / 1e9:.1f} GB "
                     f"per token; bandwidth-infeasible at one token per step")
    return dict(total=total, trunk=trunk, experts=experts, comm=comm, phases=phases_report, mhc=mhc, head=head + dense,
                kda=kda, sparse=sparse + ctx, moe=moe, ctx=ctx, expert_bytes=expert_bytes_total, load_ratio=load_ratio,
                notes=notes)



def decode(layout, node=None, tofu=None, profile=None):
    node = node or A64FXNode()
    tofu = tofu or TofuD()
    profile = profile or current()
    pk = pack(layout.pack)
    fit = memory_fit(layout, node, pk)
    notes = list(fit["notes"])
    depth = layout.mtp_depth
    width = depth + 1
    s1 = _step(layout, node, tofu, profile, pk, 1)
    if depth == 0:
        s = s1
        round_ms = s1["total"]
        tokens = 1.0
    else:
        tokens = routing.tokens_per_round(depth, layout.acceptance)
        if profile.width_slope >= 0:
            step_w = s1["total"] * (1 + profile.width_slope * depth + profile.width_slope2 * depth * depth)
            s = dict(s1)
            s["total"] = step_w
        else:
            s = _step(layout, node, tofu, profile, pk, width)
            step_w = s["total"]
        round_ms = step_w / profile.verify_share
    notes += s["notes"]
    return DecodeResult(tok_s=tokens / round_ms * 1e3, step_ms=s["total"], round_ms=round_ms, tokens_per_round=tokens,
                        width=width, trunk_ms=s["trunk"], experts_ms=s["experts"], comm_ms=s["comm"],
                        phases_ms=s["phases"], mhc_ms=s["mhc"], head_ms=s["head"], kda_ms=s["kda"],
                        sparse_ms=s["sparse"], moe_ms=s["moe"], ctx_ms=s["ctx"],
                        expert_bytes_per_node=s["expert_bytes"] * layout.ranks_per_node, load_ratio=s["load_ratio"],
                        fit=fit, notes=notes)


# ---------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class PrefillResult:
    tok_s: float
    ms_per_pos: float
    compute_ms: float
    comm_ms: float
    fixed_ms: float
    ctx_ms: float
    chunk: int
    prompt: int
    fit: dict
    notes: list

    def to_dict(self):
        return dataclasses.asdict(self)


def prefill(layout, node=None, tofu=None, profile=None, prompt=8192):
    """ms per position = compute (int8 SDOT GEMM roofline over all ranks at the chunk's M per expert) + trunk and
    expert collectives + per-layer fixed cost per chunk + a context term; anchored on 412.27 (8K, chunk 4096),
    285.31 (128 tokens) and 372.39 (32K) tok/s at 12 nodes."""
    node = node or A64FXNode()
    tofu = tofu or TofuD()
    profile = profile or current()
    pk = pack(layout.pack)
    fit = memory_fit(layout, node, pk)
    g = GEOMETRY
    R = layout.ranks()
    chunk = max(1, min(layout.prefill_chunk, prompt))
    chunks = math.ceil(prompt / chunk)
    m_per_expert = chunk * g.top_k / pk.experts
    tops = node.int8_tops(m_per_expert) / layout.ranks_per_node
    compute = PREFILL_GFLOP_PER_TOKEN * 1e9 / (R * tops * 1e12 * profile.prefill_int8_eff) * 1e3
    if pk.formats[3][0] != "Q4_K":
        compute *= 1.08   # Q2_K/Q3_K dequantize-to-int8 once per chunk (hypothesis: 8 % of the GEMM time)
    # collectives per layer per chunk: TP trunk reduce-scatter/all-gather (2 per layer today, 1 with EP + fused
    # combine), expert all-to-all under EP
    rsag = reduce_scatter_allgather_ms(tofu, chunk, layout.trunk_tp, bf16=layout.bf16_collectives) if layout.trunk_tp > 1 else 0.0
    comm_chunk = profile.prefill_collectives_per_layer * g.layers * rsag
    if layout.expert_parts() < R / 2:
        a2a_bytes = 2 * chunk * g.top_k * HIDDEN_BYTES_BF16 * (1 - 1.0 / R)
        comm_chunk += g.moe_layers * a2a_bytes / (tofu.a2a_gbps_uniform * 1e6)
    comm = comm_chunk / chunk
    fixed = profile.prefill_layer_fixed_ms * g.layers / chunk
    mean_pos = prompt / 2
    ctx = profile.prefill_ctx_us_per_1k * 1e-3 * (mean_pos / 1024)
    ms = compute + comm + fixed + ctx
    return PrefillResult(tok_s=1e3 / ms, ms_per_pos=ms, compute_ms=compute, comm_ms=comm, fixed_ms=fixed, ctx_ms=ctx,
                         chunk=chunk, prompt=prompt, fit=fit, notes=list(fit["notes"]))


# ---------------------------------------------------------------------------------------------------------------
def memory_fit(layout, node=None, pk=None):
    """Per-rank bytes: expert share (whole pack / nodes, times the slice duplication of parts), trunk share,
    replicated tensors, KV and KDA state for the context, MTP block, scratch. Refuses what exceeds the usable HBM."""
    node = node or A64FXNode()
    pk = pk or pack(layout.pack)
    R = layout.ranks()
    scale = TRUNK_SCALE[layout.trunk_format]
    experts = pk.routed_total_bytes() / R
    trunk_total = (FIXED_BYTES["total"] - FIXED_BYTES["token_embd"] - FIXED_BYTES["router"] * GEOMETRY.moe_layers) * scale
    trunk = trunk_total / layout.trunk_tp
    replicated = REPLICATED_PER_TOKEN["router"] + REPLICATED_PER_TOKEN["mhc"] + FIXED_BYTES["token_embd"] / layout.trunk_tp
    kv = layout.context * KV_BYTES_PER_TOKEN + KDA_STATE_BYTES
    mtp = FIXED_BYTES["mtp_dense"] * scale / layout.trunk_tp + pk.expert_bytes(GEOMETRY.layers - 1) * pk.experts / R
    scratch = 1.0 * GIB / layout.ranks_per_node
    total = experts + trunk + replicated + kv + mtp + scratch
    usable = node.usable_gib * GIB if layout.ranks_per_node == 1 else node.usable_per_cmg_gib * GIB
    notes = []
    if total > usable:
        notes.append(f"{layout.pack} does not fit: {total / GIB:.1f} GiB per rank exceed the usable {usable / GIB:.1f} GiB "
                     f"({layout.nodes} nodes x {layout.ranks_per_node} ranks)")
    return dict(expert_gib=experts / GIB, trunk_gib=trunk / GIB, replicated_gib=replicated / GIB, kv_gib=kv / GIB,
                mtp_gib=mtp / GIB, scratch_gib=scratch / GIB, total_gib=total / GIB, usable_gib=usable / GIB,
                margin_gib=(usable - total) / GIB, fits=total <= usable, notes=notes)


# ---------------------------------------------------------------------------------------------------------------
def load_records(path=None):
    path = pathlib.Path(path) if path else DATA / "measured_a64fx_12n.json"
    return json.loads(path.read_text())


def layout_from_record(record):
    return Layout(**record.get("layout", {}))


def validate(records=None, node=None, tofu=None):
    """Predicted vs measured for every record; rows carry their own tolerance and profile."""
    data = records if isinstance(records, dict) else load_records(records)
    rows = []
    for rec in data["records"]:
        layout = layout_from_record(rec)
        profile = PROFILES[rec.get("profile", "current")]()
        for key, value in rec.get("profile_set", {}).items():
            setattr(profile, key, value)
        if rec["kind"] == "decode":
            r = decode(layout, node, tofu, profile)
            predicted = r.tok_s
            extra = dict(kda=r.kda_ms, mhc=r.mhc_ms, moe=r.moe_ms, sparse=r.sparse_ms, head=r.head_ms, step_ms=r.step_ms)
        else:
            r = prefill(layout, node, tofu, profile, prompt=rec.get("prompt", 8192))
            predicted = r.tok_s
            extra = dict(ms_per_pos=r.ms_per_pos, compute=r.compute_ms, comm=r.comm_ms, fixed=r.fixed_ms)
        measured = rec["measured"]["tok_s"]
        error = predicted / measured - 1
        rows.append(dict(name=rec["name"], kind=rec["kind"], measured=measured, predicted=predicted, error=error,
                         tolerance=rec.get("tolerance", 0.1), within=abs(error) <= rec.get("tolerance", 0.1),
                         stages=extra, measured_stages=rec["measured"].get("stages", {}), source=rec.get("source", "")))
    return rows
