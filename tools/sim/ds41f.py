"""DeepSeek V4.1 Flash: decode/prefill cost model for four machines and the quantization choice on each.

Model facts come from the gemm repository's native A64FX runner (a64fx/ds41f/ds41f_model.h, a64fx/doc/ds41f.md:2516-2521)
and the Hugging Face card; machine constants from the GLM simulators in this directory (hw.py, a64fx.py) and the muse
worktree's measured IQ1_S kernel tables. Every constant names its measurement or says it is a hypothesis; a result
carries the hypothesis list of the profile that produced it.

What the model is, in bytes:
- 40 MoE layers, hidden 5120, 384 routed experts top-6 + 1 shared, expert rows 2304: 35.4M params per expert,
  543.6B routed params = 288.78 GB at the released MXFP4 (4.25 bpw), 4.5 GB of experts per decoded token;
- trunk (attention CSA2, shared experts, router, mHC, Engram projections, norms) 7.20 GB FP8 read per token, BF16
  embed 1.32 GB (lookup) and head 1.32 GB;
- Engram: two hashed n-gram tables (layers 1 and 14), 384.0M + 384.0M rows x 264 B (256 FP8 + 8 E8M0 scales) =
  202,758,032,400 B, 24 rows per layer per token = 48 random 264 B reads, released in FP8 and left there (on SSD);
- KV 890 B per token (compressed FP4 latent + indexer keys, four source layers), 0.94 GB at 1M tokens.
"""
import dataclasses
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import a64fx    # noqa: E402
import model    # noqa: E402
import routing  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / "data"
GIB = 2 ** 30

# ggml block sizes not in model.BLOCK_BYTES (elements per block, bytes per block); MXFP4 as released: 32 FP4 + E8M0.
BLOCK_BYTES = dict(model.BLOCK_BYTES, **{"IQ1_S": (256, 50), "IQ1_M": (256, 56), "IQ2_XXS": (256, 66), "IQ2_S": (256, 82),
                                        "IQ3_S": (256, 110), "MXFP4": (32, 17), "FP8": (1, 1)})


def tensor_bytes(elements, fmt):
    block, nbytes = BLOCK_BYTES[fmt]
    return int(elements / block * nbytes)


def bpw(fmt):
    block, nbytes = BLOCK_BYTES[fmt]
    return 8.0 * nbytes / block


@dataclasses.dataclass(frozen=True)
class Geometry:
    name: str = "ds41f"
    layers: int = 40
    hidden: int = 5120
    experts: int = 384
    top_k: int = 6
    shared: int = 1
    expert_ff: int = 2304
    vocab: int = 129_280
    engram_layers: tuple = (1, 14)
    engram_rows_per_layer: int = 24           # 3 n-gram orders x 8 heads (ds41f_engram.c:197-211)
    engram_row_bytes: int = 264               # 256 FP8 + 8 E8M0 (ds41f_engram.c:232-241)
    engram_table_rows: tuple = (384_006_168, 384_016_682)
    kv_bytes_per_token: float = 890.0         # arXiv 2609.19969: global KV 890 B/token
    kv_source_layers: int = 4
    trunk_fp8_bytes: float = 7_199_094_208.0  # "dense text weights" 9,846,748,608 minus embed and head (ds41f.md:2516-2521)
    shared_params: int = 3 * 5120 * 2304 * 40 # inside trunk_fp8_bytes (1.415 GB FP8)
    embed_bytes: float = 129_280 * 5120 * 2.0
    head_bytes: float = 129_280 * 5120 * 2.0
    dspark_bytes: float = 7_932_874_632.0     # 3-stage drafter (ds41f.md:1425-1427)
    prefill_active_fraction: float = 1.0      # the card says 8B active in prefill vs 16B in decode; not exploited here (hypothesis)
    # config.json (identical for DeepSeek-V4.1-Flash and -1017, checked 2026-10-18; total_size 510,286,023,000 B, 48 shards)
    mtp_layers: int = 3                       # num_nextn_predict_layers: the DSpark drafter stages
    compress_ratios: tuple = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)   # 40 layers + 3 MTP; KV sources 2, 8, 14, 20
    candidate_blocks: int = 2048              # candidate_topk_blocks x candidate_block_size 8 from layer 20
    candidate_block: int = 8
    index_topk: int = 512
    release: str = "DeepSeek-V4.1-Flash / -1017 (weights-only refresh, same config and size)"

    def expert_params(self):
        return 3 * self.hidden * self.expert_ff

    def expert_bytes(self, fmt3):
        n = self.hidden * self.expert_ff
        return sum(tensor_bytes(n, f) for f in fmt3)

    def engram_bytes(self):
        return sum(self.engram_table_rows) * self.engram_row_bytes

    def engram_rows_per_token(self):
        return len(self.engram_layers) * self.engram_rows_per_layer

    def routed_flops_per_token(self):
        return 2.0 * self.top_k * self.expert_params() * self.layers

    def trunk_flops_per_token(self):
        # FP8 trunk bytes = params (1 B each) + shared experts at 1 B; head BF16
        return 2.0 * (self.trunk_fp8_bytes + self.head_bytes / 2)


GEOMETRY = Geometry()
# DeepSeek V4-Flash 0731 (the older runner's model, for the TR16 cross-check): 43 layers, 256 experts top-6, rows 2048,
# hidden 4096, no Engram, dense FP8 5.682 GB + BF16 head 1.06 GB (gemm hetero/ds4f/README.md:330-331, common/ds4f.h)
V4FLASH = Geometry(name="v4flash", layers=43, hidden=4096, experts=256, top_k=6, expert_ff=2048, engram_layers=(),
                   engram_table_rows=(0, 0), trunk_fp8_bytes=5_682e6, shared_params=3 * 4096 * 2048 * 43,
                   embed_bytes=129_280 * 4096 * 2.0, head_bytes=129_280 * 4096 * 2.0, kv_bytes_per_token=3_200.0)


@dataclasses.dataclass(frozen=True)
class Pack:
    name: str
    formats: tuple                 # (gate, up, down)
    experts: int = 384             # kept experts per layer (REAP prune below 384)
    description: str = ""
    lossless: bool = False
    geometry: Geometry = GEOMETRY

    def expert_bytes(self):
        return self.geometry.expert_bytes(self.formats)

    def total_bytes(self):
        return self.expert_bytes() * self.experts * self.geometry.layers

    def bytes_per_token(self):
        return self.expert_bytes() * self.geometry.top_k * self.geometry.layers

    def bpw(self):
        return 8.0 * self.expert_bytes() / self.geometry.expert_params()

    def skew(self):
        return 1.0 if self.experts == self.geometry.experts else 0.8


def _packs(g):
    base = {
        "mxfp4": (("MXFP4",) * 3, "released MXFP4 experts, bit-exact repack", True),
        "iq3xxs": (("IQ3_XXS",) * 3, "IQ3_XXS gate/up/down (3.06 bpw)", False),
        "q23": (("Q2_K", "Q2_K", "Q3_K"), "Q2_K gate/up + Q3_K down, Strata's GLM pack recipe", False),
        "iq2xxs_q2k": (("IQ2_XXS", "IQ2_XXS", "Q2_K"), "IQ2_XXS gate/up + Q2_K down (the public MixedQ2 GGUF, 2.25 bpw)", False),
        "iq2s_q2k": (("IQ2_S", "IQ2_S", "Q2_K"), "IQ2_S gate/up + Q2_K down", False),
        "iq2xxs": (("IQ2_XXS",) * 3, "IQ2_XXS gate/up/down (2.06 bpw)", False),
        "tetra": (("EXL3", "EXL3", "EXL3"), "llvq Tetra, 2.06 bpw (Strata native)", False),
        "iq1m": (("IQ1_M",) * 3, "IQ1_M gate/up/down (1.75 bpw)", False),
        "iq1s": (("IQ1_S",) * 3, "IQ1_S gate/up/down (1.56 bpw)", False),
        "iq1m_iq2xxs": (("IQ1_M", "IQ1_M", "IQ2_XXS"), "IQ1_M gate/up + IQ2_XXS down", False),
    }
    out = {}
    for name, (fmts, desc, lossless) in base.items():
        for keep in (384, 320, 288, 256):
            key = name if keep == 384 else f"{name}_reap{keep}"
            d = desc if keep == 384 else f"{desc}, REAP-pruned to {keep} experts"
            out[key] = Pack(key, fmts, keep, d, lossless and keep == 384, g)
    return out


PACKS = _packs(GEOMETRY)
PACKS["v4flash_mxfp4"] = Pack("v4flash_mxfp4", ("MXFP4",) * 3, 256, "V4-Flash released MXFP4", True, V4FLASH)
# llvq Tetra: 24 weights in 48 bits
BLOCK_BYTES["EXL3"] = (24, 6)


def pack(name):
    if name not in PACKS:
        raise ValueError(f"unknown pack {name}; choose from {', '.join(sorted(PACKS))}")
    return PACKS[name]


TRUNK_SCALE = dict(fp8=1.0, q8=8.5 / 8, q6=6.5625 / 8, q5=5.5 / 8, q4=4.5 / 8, bf16=2.0)   # bytes vs the FP8 release
HEAD_SCALE = dict(bf16=1.0, q8=0.53, q6=0.41, q4=0.28)

# CPU expert kernel rates. Memory side: the machine's expert stream. Compute side: GMAC/s on TR16's 15 Zen 1 cores
# at width 1 (muse worktree, 2026-10-16: Strata Fmt32<19> IQ1_S 92.2 GMAC/s, 157/209 at width 2/4; Q2_K/Q3_K
# 17 GMAC/s per core = 255; IQ2_XXS codebook-bound ~0.15 of peak = 60; MXFP4 from hetero/ds4f: 41.4 GB/s on TR16,
# "expert reads cap decode at about 12 tok/s", resume-ds4f.md:191). Scaled to other CPUs by core count x clock.
TR16_GMACS_WIDTH1 = {"IQ1_S": 92.2, "IQ1_M": 83.0, "IQ2_XXS": 120.0, "IQ2_S": 200.0, "IQ3_XXS": 126.0, "Q2_K": 255.0,
                     "Q3_K": 255.0, "Q4_K": 255.0, "EXL3": 120.0, "MXFP4": 41.4e9 * (35.4e6 / 18.8e6) / 1e9}
# IQ2_XXS/IQ3_XXS: model.KERNEL_EFFICIENCY notes 8 / 8.4 GMAC/s per Zen 1 core (codebook decode bound) x 15 cores.
TR16_PEAK_GMACS = 15 * 3.4 * 8    # 408 GMAC/s FP32 FMA on 15 cores
HYPOTHESIS_RATES = {"IQ1_M": "IQ1_M CPU rate 0.9 x IQ1_S (no Strata kernel; ggml's is pdep-bound)",
                    "IQ2_S": "IQ2_S rate 200 GMAC/s from the GLM IQ2_S CPU kernel's 17.5 vs 19.6 tok/s (Q2_K) on TR16",
                    "IQ3_XXS": "IQ3_XXS rate from GLM's measured 3.2 GB/s per core",
                    "EXL3": "llvq Tetra CPU rate (AVX2 decode) assumed 120 GMAC/s on 15 Zen 1 cores"}


@dataclasses.dataclass
class Gpu:
    name: str = "none"
    count: int = 0
    vram_gib: float = 0.0
    gemv_gbps: float = 0.0        # dense GEMV rate (5060 Ti 380 of 448; V100 0.8 x 900)
    tier_gbps: float = 0.0        # routed-expert tier kernel
    tflops: float = 0.0           # prefill GEMM (int8/MMQ) throughput
    h2d_gbps: float = 0.0
    launch_us: float = 3.0
    desktop_gib: float = 0.0


@dataclasses.dataclass
class Machine:
    name: str = "tr16"
    cores: int = 15
    ghz: float = 3.4
    ram_gib: float = 124.0
    stream_gbps: float = 65.6      # in-model expert stream (TR16 8K trace, muse.py:224)
    gpu: Gpu = dataclasses.field(default_factory=Gpu)
    page_us: float = 83.0          # 4 KiB QD1 random read (hw_tr16_runner.json storage_io, the x4 NVMe work mount)
    queue_depth: int = 16
    disk_gbps: float = 3.68        # sequential (prefill Engram rows sorted, expert streaming from disk)
    worker: "Machine" = None       # second node for the two-node layouts
    link_rtt_us: float = 0.0
    link_gbps: float = 0.0
    notes: str = ""

    def gmacs(self, fmt, width=1):
        base = TR16_GMACS_WIDTH1[fmt] * (self.cores * self.ghz) / (15 * 3.4)
        return base * {1: 1.0, 2: 1.7, 3: 2.0, 4: 2.27}.get(width, 2.3)

    to_dict = lambda self: dataclasses.asdict(self)


def tr16():
    return Machine(name="tr16", cores=15, ghz=3.4, ram_gib=124.0, stream_gbps=65.6,
                   gpu=Gpu("RTX 5060 Ti 16 GB", 1, 15.5, 380.0, 235.0, 47.0, 7.18, 3.0, 2.1),
                   notes="hw.py tr16: 72 GB/s plain stream, 65.6 in-model; 5060 Ti 380 GB/s GEMV, 235 GB/s tier, PCIe 3 x8 7.18 GB/s; "
                         "x4 NVMe work mount 3.68 GB/s, 4 KiB QD1 83 us")


def b550():
    return Machine(name="b550", cores=16, ghz=3.5, ram_gib=56.0, stream_gbps=26.0,
                   gpu=Gpu("RTX 5060 Ti 16 GB", 1, 15.5, 380.0, 235.0, 47.0, 13.0, 3.0, 0.0),
                   page_us=90.0, disk_gbps=2.0, notes="hw.py b550: expert rows 26 GB/s (DDR4-2133), 5060 Ti since 2026-10-16")


def tr16_b550():
    m = tr16()
    m.name = "tr16+b550"
    m.worker = b550()
    m.link_rtt_us, m.link_gbps = 15.7, 3.26   # ib-qdr measured 2026-10-16 (hw.py LINKS)
    return m


def xeon_v100():
    return Machine(name="xeon_v100", cores=36, ghz=2.6, ram_gib=181.0, stream_gbps=150.0,
                   gpu=Gpu("2x Tesla V100 32 GB", 2, 31.0, 720.0, 400.0, 50.0, 12.0, 4.0, 0.0),
                   page_us=100.0, disk_gbps=2.0,
                   notes="hw.py xeon_v100 (AVX-512, 150 GB/s, V100 0.8 x 900 GB/s GEMV, tier 400 with the current kernel, PCIe 3 x16 12 GB/s); "
                         "RAM 181 GiB per the user (preset said 160); disk unmeasured 2 GB/s")


def xeon_tesla():
    """The xeon+tesla GPU node seen through Slurm `gpu1` (36 CPUs, 2 GPUs, 180 min; memory limit 160 GiB observed,
    181 per the user): shared storage at ~100 MB/s with no measured random-read latency, so the Engram term uses a
    2 ms page latency hypothesis (what a 4 GB RAM row cache has to absorb) until the node is probed."""
    m = xeon_v100()
    m.name = "xeon_tesla"
    m.disk_gbps = 0.1
    m.page_us = 2000.0
    m.notes = ("xeon+tesla GPU node, Slurm partition gpu1: 36 Xeon Gold 6240 cores, 2x V100 32 GB, Slurm limit 160 GiB (181 per the user: --set ram_gib=181), 180-minute jobs; "
               "storage ~100 MB/s shared filesystem, random 4 KiB latency UNMEASURED (2 ms assumed)")
    return m


MACHINES = dict(tr16=tr16, b550=b550, **{"tr16+b550": tr16_b550}, xeon_v100=xeon_v100, xeon_tesla=xeon_tesla)


@dataclasses.dataclass
class Plan:
    machine: str = "tr16"
    pack: str = "iq1s"
    trunk: str = "q4"              # fp8 | q8 | q6 | q5 | q4 | bf16 (GPU copy, or RAM when no GPU)
    head: str = "q4"
    context: int = 8192
    width: int = 1                 # tokens per step (DSpark verify window); tokens_per_round from `acceptance`
    acceptance: str = ""           # routing.ACCEPTANCE profile for DSpark; "" = plain decode
    tier_gib: float = -1.0         # GPU expert tier; -1 = what VRAM leaves after trunk, head, KV, scratch
    engram_cache_hit: float = 0.4  # row-cache hit share (gemm: 30-50 % with 8 MiB, prompt dependent)
    engram_prefetch: bool = True   # issue the 48 reads at token id, collect before layer 1 / 14
    worker_share: float = -1.0     # two-node: share of expert bytes on the worker; -1 = balance by stream rate within RAM
    prefill_chunk: int = 4096
    cpu_rate_override_gbps: float = 0.0   # what-if: a DRAM-speed kernel for the pack's format

    to_dict = lambda self: dataclasses.asdict(self)


@dataclasses.dataclass
class Result:
    tok_s: float
    step_ms: float
    round_ms: float
    tokens_per_round: float
    cpu_ms: float
    cpu_bound: str
    gpu_ms: float
    gpu_exposed_ms: float
    engram_ms: float
    engram_exposed_ms: float
    remote_ms: float
    head_ms: float
    expert_gb_per_token: float
    tier_hit: float
    fit: dict
    hypotheses: list
    notes: list

    def to_dict(self):
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------------------------------------------
def memory_fit(plan, m=None):
    """RAM and VRAM per machine for the plan; refuses what does not fit. Engram lives on SSD (plus an optional
    scale cache); the trunk sits in VRAM when there is a GPU, else in RAM."""
    m = m or MACHINES[plan.machine]()
    pk = pack(plan.pack)
    g = pk.geometry
    notes = []
    trunk = g.trunk_fp8_bytes * TRUNK_SCALE[plan.trunk]
    head = g.head_bytes * HEAD_SCALE[plan.head]
    kv = g.kv_bytes_per_token * plan.context
    experts = pk.total_bytes()
    gpu = m.gpu
    vram_total = gpu.count * gpu.vram_gib * GIB
    gpu_fixed = (trunk + head + kv + 1.0 * GIB * max(1, gpu.count) + gpu.desktop_gib * GIB) if gpu.count else 0.0
    tier = 0.0
    if gpu.count:
        tier = max(0.0, vram_total - gpu_fixed) if plan.tier_gib < 0 else plan.tier_gib * GIB
        if gpu_fixed + tier > vram_total + 1e6:
            notes.append(f"VRAM: trunk {trunk / GIB:.1f} + head {head / GIB:.2f} + KV {kv / GIB:.2f} + tier {tier / GIB:.1f} GiB exceed "
                         f"{vram_total / GIB:.0f} GiB; use a smaller trunk format or tier")
    ram_need = experts + g.embed_bytes + 4.0 * GIB            # embed table, activations, page cache floor
    if not gpu.count:
        ram_need += trunk + head + kv
    worker_bytes = 0.0
    if m.worker:
        share = plan.worker_share if plan.worker_share >= 0 else worker_share(plan, m)
        worker_bytes = experts * share
        ram_need -= worker_bytes
        if worker_bytes + 2 * GIB > m.worker.ram_gib * GIB:
            notes.append(f"worker RAM: {worker_bytes / GIB:.1f} GiB of expert rows exceed {m.worker.ram_gib:.0f} GiB")
    if ram_need > m.ram_gib * GIB:
        notes.append(f"RAM: experts {(experts - worker_bytes) / GIB:.1f} GiB + embed/activations do not fit {m.ram_gib:.0f} GiB")
    ssd = g.engram_bytes() + experts + trunk + head
    return dict(expert_gib=experts / GIB, worker_gib=worker_bytes / GIB, trunk_gib=trunk / GIB, head_gib=head / GIB, kv_gib=kv / GIB,
                tier_gib=tier / GIB, ram_need_gib=ram_need / GIB, ram_gib=m.ram_gib, vram_gib=vram_total / GIB,
                engram_ssd_gib=g.engram_bytes() / GIB, ssd_total_gib=ssd / GIB, fits=not notes, notes=notes)


def worker_share(plan, m):
    """Two-node split that equalizes the two nodes' expert pass times within the worker's RAM."""
    pk = pack(plan.pack)
    w = m.worker
    by_rate = w.stream_gbps / (m.stream_gbps + w.stream_gbps)
    by_ram = max(0.0, (w.ram_gib - 2) * GIB / pk.total_bytes())
    by_local = max(0.0, 1 - (m.ram_gib - 6) * GIB / pk.total_bytes())   # at least what does not fit locally
    return min(0.75, max(by_rate, by_local, 0.0), by_ram) if by_ram > 0 else 0.0


def tier_hit(plan, pk, tier_bytes):
    """Share of routed bytes served by the GPU expert tier: the GLM-fitted static-prior curve h = 1.16 f^0.545 x1.08
    adaptive (routing.TierCurve), applied to this model's slot count (hypothesis: DS routing is as skewed as GLM's)."""
    if tier_bytes <= 0:
        return 0.0
    slots = tier_bytes / pk.expert_bytes()
    fraction = min(1.0, slots / (pk.experts * pk.geometry.layers))
    return min(1.0, routing.TierCurve().hit(fraction, policy="adaptive", skew=pk.skew()))


def cpu_expert_ms(m, pk, bytes_, width, override_gbps=0.0):
    fmt = pk.formats[0]
    memory_ms = bytes_ / (m.stream_gbps * 1e6)
    if override_gbps:
        # what-if: a kernel that streams this format at `override_gbps` (bounded by DRAM)
        return bytes_ / (min(override_gbps, m.stream_gbps) * 1e6), "memory"
    macs = bytes_ / pk.expert_bytes() * pk.geometry.expert_params() * width / max(1.0, routing.union_ratio(width, pk.experts, pk.geometry.top_k))
    gm = m.gmacs(fmt, width) * 1e9
    compute_ms = macs / gm * 1e3 if gm > 0 else 0.0
    return max(memory_ms, compute_ms), ("compute" if compute_ms > memory_ms else "memory")


def engram_ms(plan, m, g, prev_layer_ms):
    """One Engram layer's cost: 24 random 264 B rows = one 4 KiB page each, queue-depth limited; with prefetch the
    reads overlap the preceding layer and only the remainder is exposed."""
    if not g.engram_layers:
        return 0.0, 0.0
    rows = g.engram_rows_per_layer * (1 - plan.engram_cache_hit)
    total = math.ceil(rows / m.queue_depth) * m.page_us * 1e-3 + rows * 0.002   # + decode of the rows
    exposed = max(0.0, total - prev_layer_ms) if plan.engram_prefetch else total
    return total, exposed


def decode(plan, m=None):
    m = m or MACHINES[plan.machine]()
    pk = pack(plan.pack)
    g = pk.geometry
    fit = memory_fit(plan, m)
    notes = list(fit["notes"])
    hyps = ["GPU tier hit curve taken from GLM (1.16 f^0.545)", "per-layer mixer fixed cost 40 us + 0.12 us per 1K context per source layer (GLM-fitted)"]
    if pk.formats[0] in HYPOTHESIS_RATES:
        hyps.append(HYPOTHESIS_RATES[pk.formats[0]])
    if plan.cpu_rate_override_gbps:
        hyps.append(f"CPU expert kernel assumed to stream at {plan.cpu_rate_override_gbps:.0f} GB/s")
    width = plan.width
    union = routing.union_ratio(width, pk.experts, g.top_k)
    gpu = m.gpu
    gemv = gpu.gemv_gbps * gpu.count if gpu.count else 0.0
    trunk_layer = g.trunk_fp8_bytes * TRUNK_SCALE[plan.trunk] / g.layers
    shared_layer = g.shared_params * TRUNK_SCALE[plan.trunk] / g.layers
    width_factor = 1 + 0.185 * min(3, width - 1) + 0.3 * max(0, width - 4)
    hit = tier_hit(plan, pk, fit["tier_gib"] * GIB) if gpu.count else 0.0
    share = plan.worker_share if plan.worker_share >= 0 else (worker_share(plan, m) if m.worker else 0.0)
    cpu_ms = gpu_ms = gpu_exposed = remote = engram_total = engram_exposed = 0.0
    expert_bytes_total = 0.0
    bound = "memory"
    prev = 0.0
    for layer in range(g.layers):
        # Engram before layers 1 and 14
        if layer in g.engram_layers:
            t, e = engram_ms(plan, m, g, prev)
            engram_total += t
            engram_exposed += e
        layer_bytes = pk.expert_bytes() * g.top_k * union
        expert_bytes_total += layer_bytes
        resident = layer_bytes * hit
        nonres = layer_bytes * (1 - hit)
        if gpu.count:
            attn = 40e-3 * width + 0.12e-3 * (plan.context / 1024) * (g.kv_source_layers / g.layers) * width
            t_gpu_serial = ((trunk_layer - shared_layer) * width_factor / (gemv * 1e6) + 44 * gpu.launch_us * 1e-3 + attn)
            t_gpu_overlap = shared_layer * width_factor / (gemv * 1e6) + resident / (gpu.tier_gbps * gpu.count * 1e6)
        else:
            t_gpu_serial = trunk_layer * width_factor / (m.stream_gbps * 1e6) + 0.1
            t_gpu_overlap = shared_layer / (m.stream_gbps * 1e6)
        local_bytes = nonres * (1 - share)
        t_cpu, bound = cpu_expert_ms(m, pk, local_bytes, width, plan.cpu_rate_override_gbps)
        t_cpu += 0.12   # per-layer pool overhead (GLM fit)
        if m.worker and share > 0:
            t_w, _ = cpu_expert_ms(m.worker, pk, nonres * share, width)
            t_w += 0.15 + m.link_rtt_us * 1e-3 + (4800 + width * 2 * g.hidden) / (m.link_gbps * 1e6)
            remote += max(0.0, t_w - t_cpu)
            t_cpu = max(t_cpu, t_w)
        cpu_ms += t_cpu
        gpu_ms += t_gpu_serial + t_gpu_overlap
        t_layer = max(t_cpu, t_gpu_overlap) + t_gpu_serial + 0.016
        gpu_exposed += t_gpu_serial + max(0.0, t_gpu_overlap - t_cpu)
        prev = t_layer
    head_bytes = g.head_bytes * HEAD_SCALE[plan.head]
    head = (head_bytes * width_factor / (gemv * 1e6) if gpu.count else head_bytes / (m.stream_gbps * 1e6)) + 1.0
    step = cpu_ms + gpu_exposed + engram_exposed + head + 0.016 * g.layers
    tokens = routing.tokens_per_round(width - 1, plan.acceptance) if plan.acceptance and width > 1 else 1.0
    round_ms = step / 0.92 if width > 1 else step    # DSpark draft/replay overhead as the A64FX MTP measurement (8 %)
    if width > 1:
        hyps.append("DSpark: 8 % drafting overhead and the GLM width curve; verify bytes grow with the union ratio")
    return Result(tok_s=tokens / round_ms * 1e3, step_ms=step, round_ms=round_ms, tokens_per_round=tokens, cpu_ms=cpu_ms,
                  cpu_bound=bound, gpu_ms=gpu_ms, gpu_exposed_ms=gpu_exposed, engram_ms=engram_total, engram_exposed_ms=engram_exposed,
                  remote_ms=remote, head_ms=head, expert_gb_per_token=expert_bytes_total / 1e9 / tokens, tier_hit=hit, fit=fit,
                  hypotheses=hyps, notes=notes)


@dataclasses.dataclass
class PrefillResult:
    tok_s: float
    seconds: float
    ms_per_pos: float
    upload_ms: float
    gemm_ms: float
    mixer_ms: float
    engram_ms: float
    cpu_ms: float
    bottleneck: str
    fit: dict
    hypotheses: list

    def to_dict(self):
        return dataclasses.asdict(self)


def prefill(plan, m=None, prompt=8192):
    """Strata's streamed-expert prefill: per chunk the GPU runs the mixers while every touched expert is uploaded
    over PCIe and multiplied on the GPU (prefill.py); experts already in the tier skip the upload. Engram: 48 rows
    per token read in sorted page order (bandwidth-bound). Without a GPU the CPU GEMM roofline applies."""
    m = m or MACHINES[plan.machine]()
    pk = pack(plan.pack)
    g = pk.geometry
    fit = memory_fit(plan, m)
    hyps = ["prefill mixer cost 36 us per token per layer and 21.4 us per token per layer of MMQ GEMM scaled by TFLOPS (GLM-fitted on the 5060 Ti)",
            "8B-active prefill (encoder half) not exploited: full 40-layer expert traffic charged"]
    chunk = max(1, min(plan.prefill_chunk, prompt))
    chunks = math.ceil(prompt / chunk)
    gpu = m.gpu
    distinct = pk.experts * (1 - (1 - g.top_k / pk.experts) ** chunk)
    upload_bytes = distinct * pk.expert_bytes() * g.layers * (1 - (fit["tier_gib"] * GIB / max(1.0, pk.total_bytes()) if gpu.count else 0))
    engram_pages = g.engram_rows_per_token() * chunk
    engram = engram_pages * 4096 / (m.disk_gbps * 1e6) if g.engram_layers else 0.0
    if gpu.count:
        upload = upload_bytes / (gpu.h2d_gbps * gpu.count * 1e6)
        gemm = chunk * 21.4e-3 * g.layers * (47.0 / gpu.tflops) * (g.routed_flops_per_token() / 16.9e9) / gpu.count
        mixer = chunk * 36e-3 * g.layers * (448.0 / (gpu.gemv_gbps / 0.85)) * (g.hidden / 4096)
        layer_ms = max(mixer + gemm, upload) + 30.8 * g.layers   # deep streaming: uploads overlap the GPU
        cpu = 0.0
        bottleneck = "pcie" if upload > mixer + gemm else "gpu"
    else:
        upload = 0.0
        flops = (g.routed_flops_per_token() + g.trunk_flops_per_token()) * chunk
        gemm = flops / (m.cores * m.ghz * 32 * 0.5 * 1e9) * 1e3
        mixer = 0.0
        cpu = gemm
        layer_ms = gemm
        bottleneck = "cpu"
    total_ms = (layer_ms + engram) * chunks
    return PrefillResult(tok_s=prompt / (total_ms / 1e3), seconds=total_ms / 1e3, ms_per_pos=total_ms / prompt, upload_ms=upload * chunks,
                         gemm_ms=gemm * chunks, mixer_ms=mixer * chunks, engram_ms=engram * chunks, cpu_ms=cpu * chunks,
                         bottleneck=bottleneck, fit=fit, hypotheses=hyps)


# ---------------------------------------------------------------------------------------------------------------
# A64FX nodes: the stage structure of tools/sim/a64fx.py with this model's bytes. `current` is fitted to the gemm
# ds41f runner's best run (24.56 tok/s = 40.7 ms: attention 17.34, routed + shared experts 9.68, the rest 13.7 ms;
# ds41f.md:160,2650-2652), `redesign` uses the GLM-plan redesign constants.
@dataclasses.dataclass
class A64FXPlan:
    nodes: int = 12
    pack: str = "mxfp4"
    trunk: str = "fp8"
    context: int = 8192
    parts: int = 12                  # expert slices per expert over the ranks (gemm: expert id % 12 whole experts -> parts 1)
    expert_layout: str = "ep"        # ep (whole experts per rank, the runner today) | ep_parts | tp_parts
    profile: str = "current"
    width: int = 1
    acceptance: str = ""


A64FX_CURRENT = dict(attn_us_per_layer=433.0,     # 17.34 ms / 40 (current runner: INT8 approximations, BF16 reductions)
                     expert_gbps=39.0,            # 376 MB per node per token in 9.68 ms, in-model
                     comm_ms=6.0,                 # of the 13.7 ms "rest": ~90 collectives at the measured ~67 us (hypothesis split)
                     other_ms=5.0,                # mHC, router, head, waits: the rest after trunk 1.7, Engram 0.56, head 0.3, context
                     mhc_site_us=0.0, phase_us=0.0, prefill_per_pos_factor=29.9 / 40.7)
# ggml-cpu SVE vec_dot per node (48 threads), measured 2026-10-18 (tools/sim/data/measured_a64fx_kernels.json); all compute-bound
A64FX_GGML_NODE_GBPS = {"Q2_K": 97.0, "Q3_K": 55.6, "Q4_K": 76.0, "IQ2_XXS": 11.2, "IQ2_S": 10.5, "IQ1_S": 8.2, "Q8_0": 432.1}
# The A64FX SDOT kernels per node, 48 threads, one token, measured 2026-10-18 in one session (job 52438878,
# tools/sim/data/measured_a64fx_kernels.json "sdot_kernel_v2"; tools/a64fx_probe/q23_sdot_sve.c, iq_sdot_sve.c).
# gate/up on the DS4.1F 2304 x 5120 shape, down on 5120 x 2304 where measured; Q3_K from GLM's shapes; an IQ format used
# as down takes its gate/up rate (not measured on the down shape).
A64FX_SDOT_GATEUP_GBPS = {"Q2_K": 192.2, "Q3_K": 171.8, "IQ2_XXS": 84.8, "IQ1_S": 60.3}
A64FX_SDOT_DOWN_GBPS = {"Q2_K": 133.4, "Q3_K": 138.0, "IQ2_XXS": 84.8, "IQ1_S": 60.3}
# time per weight byte at 1..4 tokens per expert relative to one token (GLM shapes, same session): compute-bound kernels
A64FX_SDOT_WIDTH_COST = {"Q2_K": (1.0, 1.508, 2.265, 3.158), "Q3_K": (1.0, 1.538, 1.936, 2.551),
                         "IQ2_XXS": (1.0, 1.438, 1.683, 2.324), "IQ1_S": (1.0, 1.381, 1.698, 2.139)}


def a64fx_sdot_cost(fmt, tpe):
    """Relative time per byte of the SDOT kernel for `fmt` at `tpe` tokens per distinct expert (linear between the
    measured 1..4-token points, extrapolated beyond 4)."""
    c = A64FX_SDOT_WIDTH_COST[fmt]
    if tpe <= 1:
        return 1.0
    k = int(tpe)
    if k >= len(c):
        return c[-1] + (c[-1] - c[-2]) * (tpe - len(c))
    return c[k - 1] + (c[k] - c[k - 1]) * (tpe - k)


A64FX_REDESIGN = dict(attn_us_per_layer=60.0,     # trunk stream at the chain rate + one fused phase per sub-layer (hypothesis)
                      expert_gbps=250.0, comm_ms=0.0, other_ms=0.0, mhc_site_us=23.0, phase_us=3.0, prefill_int8_eff=0.7)


def a64fx_memory_fit(ap):
    pk = pack(ap.pack)
    g = pk.geometry
    node = a64fx.A64FXNode()
    R = ap.nodes
    experts = pk.total_bytes() / R
    trunk = g.trunk_fp8_bytes * TRUNK_SCALE[ap.trunk] / R + (g.embed_bytes + g.head_bytes) / R
    kv = g.kv_bytes_per_token * ap.context + 0.5 * GIB
    engram_scales = g.engram_bytes() * 8 / 264 / R      # E8M0 scale cache in HBM (gemm --engram-scale-cache, ~512 MB/node)
    total = experts + trunk + kv + engram_scales + 1.0 * GIB
    usable = node.usable_gib * GIB
    notes = [] if total <= usable else [f"{ap.pack} does not fit {R} nodes: {total / GIB:.1f} GiB per node exceed {usable / GIB:.0f}"]
    return dict(expert_gib=experts / GIB, trunk_gib=trunk / GIB, kv_gib=kv / GIB, engram_scale_gib=engram_scales / GIB,
                total_gib=total / GIB, usable_gib=usable / GIB, engram_ssd_gib=g.engram_bytes() / R / GIB, fits=not notes, notes=notes)


def a64fx_decode(ap):
    pk = pack(ap.pack)
    g = pk.geometry
    node, tofu = a64fx.A64FXNode(), a64fx.TofuD()
    prof = A64FX_CURRENT if ap.profile == "current" else A64FX_REDESIGN
    R = ap.nodes
    fit = a64fx_memory_fit(ap)
    width = ap.width
    union = routing.union_ratio(width, pk.experts, g.top_k)
    parts = 1 if ap.expert_layout == "ep" else (R if ap.expert_layout == "tp_parts" else min(ap.parts, R))
    _, ratio = a64fx.ep_max_load(R, g.top_k, parts, trials=20000) if parts < R else (0, 1.0)
    rate = prof["expert_gbps"] / (1 + 0.12 * (parts - 1)) if ap.profile == "current" else prof["expert_gbps"] / (1 + 0.04 * (parts - 1))
    if pk.formats[0] != "MXFP4":
        if ap.profile == "current":
            # today's kernels = ggml's SVE dots, bytes-weighted over the pack's three projections (measured per node)
            n = g.hidden * g.expert_ff
            bytes_ = [tensor_bytes(n, f) for f in pk.formats]
            rate = sum(bytes_) / sum(b / A64FX_GGML_NODE_GBPS.get(f, 20.0) for b, f in zip(bytes_, pk.formats)) / (1 + 0.12 * (parts - 1))
        elif all(f in A64FX_SDOT_GATEUP_GBPS for f in pk.formats):
            # redesign with K-quant or IQ2_XXS/IQ1_S experts: the measured SDOT kernels per projection, bytes-weighted,
            # each at its measured cost for `width / union` tokens per distinct expert
            n = g.hidden * g.expert_ff
            tpe = width / union
            rates = [A64FX_SDOT_GATEUP_GBPS[pk.formats[0]], A64FX_SDOT_GATEUP_GBPS[pk.formats[1]], A64FX_SDOT_DOWN_GBPS[pk.formats[2]]]
            bytes_ = [tensor_bytes(n, f) for f in pk.formats]
            secs = sum(b / r * a64fx_sdot_cost(f, tpe) for b, r, f in zip(bytes_, rates, pk.formats))
            rate = sum(bytes_) / secs / (1 + 0.04 * (parts - 1))
        # other IQ formats (IQ1_M, IQ2_S, IQ3_XXS), Tetra and MXFP4 keep the redesign's SDOT-tile hypothesis rate
    per_layer_experts = pk.expert_bytes() * g.top_k * union / R
    t_exp = per_layer_experts / (rate * 1e6) * g.layers
    skew = (0.72 if ap.profile == "current" else 0.5) * max(0.0, ratio - 1) * t_exp / g.layers * 1e3
    trunk_per_layer = g.trunk_fp8_bytes * TRUNK_SCALE[ap.trunk] / R / g.layers
    stream = node.matvec_q8_gbps if ap.profile == "current" else node.stream_gbps * node.chain_efficiency
    t_trunk = trunk_per_layer * width / (stream * 1e6) * g.layers
    t_attn = prof["attn_us_per_layer"] * 1e-3 * width * g.layers
    ar = a64fx.tree_allreduce_us(tofu, g.hidden * 4 * width, R, False, 5.0 if ap.profile == "redesign" else 20.0) * 1e-3
    owners = a64fx.distinct_owners(R, g.top_k) * parts
    moe_c = (a64fx.tree_allreduce_us(tofu, g.hidden * 4 * width, R, False, skew) if owners >= R / 2 else
             a64fx.dispatch_combine_us(tofu, g.hidden * 2, width, R) + skew) * 1e-3
    if ap.profile == "current":
        # the fitted collective time, moved to N ranks by tree steps and the whole-expert load imbalance
        _, ratio12 = a64fx.ep_max_load(12, g.top_k, 1, trials=20000)
        t_comm = prof["comm_ms"] * (a64fx.ar_steps(R) / a64fx.ar_steps(12)) * (0.5 + 0.5 * ratio / ratio12)
    else:
        t_comm = (ar + moe_c) * g.layers
    engram = 2 * (math.ceil(g.engram_rows_per_layer / R) * 100e-3 + 0.05) + a64fx.tree_allreduce_us(tofu, 24 * 256 * 4, R) * 1e-3 * 2
    engram = max(0.3, engram)
    other = prof["other_ms"] + (2 * prof["mhc_site_us"] + 8 * prof["phase_us"]) * 1e-3 * g.layers
    head = g.head_bytes * HEAD_SCALE["bf16"] / R / (stream * 1e6) + 0.3
    ctx = 0.39e-3 * min(ap.context, 8192) * (0.5 if ap.profile == "redesign" else 1.0) * (g.kv_source_layers / 11)
    step = t_exp + t_trunk + t_attn + t_comm + engram + other + head + ctx
    tokens = routing.tokens_per_round(width - 1, ap.acceptance) if ap.acceptance and width > 1 else 1.0
    round_ms = step / 0.92 if width > 1 else step
    return dict(tok_s=tokens / round_ms * 1e3, step_ms=step, experts_ms=t_exp, trunk_ms=t_trunk, attn_ms=t_attn, comm_ms=t_comm,
                engram_ms=engram, other_ms=other, head_ms=head, ctx_ms=ctx, load_ratio=ratio, fit=fit, notes=fit["notes"],
                hypotheses=[f"a64fx {ap.profile} profile constants (see A64FX_CURRENT / A64FX_REDESIGN)",
                            "current: ggml SVE dots measured per node (Q2_K 97, IQ2_XXS 11, IQ1_S 8 GB/s); redesign: measured SDOT kernels for Q2_K/Q3_K/IQ2_XXS/IQ1_S (192/172/85/60 GB/s gate-up, standalone), SDOT-tile hypothesis for other formats"] if pk.formats[0] != "MXFP4" else
                           [f"a64fx {ap.profile} profile constants (see A64FX_CURRENT / A64FX_REDESIGN)"])


def a64fx_prefill(ap, prompt=8192):
    pk = pack(ap.pack)
    g = pk.geometry
    node, tofu = a64fx.A64FXNode(), a64fx.TofuD()
    R = ap.nodes
    if ap.profile == "current":
        d = a64fx_decode(ap)
        ms = d["step_ms"] * A64FX_CURRENT["prefill_per_pos_factor"]
        return dict(tok_s=1e3 / ms, ms_per_pos=ms, note="unbatched per-position prefill as the gemm runner today (fitted 29.9 vs 40.7 ms)")
    chunk = min(4096, prompt)
    flops = (g.routed_flops_per_token() + g.trunk_flops_per_token()) * g.prefill_active_fraction
    m_per_expert = chunk * g.top_k / pk.experts
    compute = flops / (R * node.int8_tops(m_per_expert) * 1e12 * A64FX_REDESIGN["prefill_int8_eff"]) * 1e3
    comm = 1 * g.layers * a64fx.reduce_scatter_allgather_ms(tofu, chunk, R, hidden=g.hidden, bf16=True) / chunk
    engram = g.engram_rows_per_token() * 4096 / (R * 80e6) * 1e3     # 48 pages per token over R nodes' /local at 80 MB/s, sorted
    ms = compute + comm + engram + 0.3 * g.layers / chunk
    return dict(tok_s=1e3 / ms, ms_per_pos=ms, compute_ms=compute, comm_ms=comm, engram_ms=engram,
                note="int8 SDOT GEMM roofline at 0.7 efficiency, one bf16 reduce-scatter/all-gather per layer, Engram pages from /local at 80 MB/s per node")


# ---------------------------------------------------------------------------------------------------------------
def load_records(path=None):
    import json
    return json.loads(pathlib.Path(path or DATA / "measured_ds41f.json").read_text())


def validate(records=None):
    data = records if isinstance(records, dict) else load_records(records)
    rows = []
    for rec in data["records"]:
        if rec["kind"] == "a64fx_decode":
            r = a64fx_decode(A64FXPlan(**rec["plan"]))
            predicted = r["tok_s"]
        elif rec["kind"] == "a64fx_prefill":
            r = a64fx_prefill(A64FXPlan(**rec["plan"]), rec.get("prompt", 8192))
            predicted = r["tok_s"]
        elif rec["kind"] == "decode":
            m = MACHINES[rec["plan"]["machine"]]()
            for key, value in rec.get("machine_set", {}).items():
                if key.startswith("gpu."):
                    setattr(m.gpu, key[4:], value)
                else:
                    setattr(m, key, value)
            predicted = decode(Plan(**rec["plan"]), m).tok_s
        else:
            m = MACHINES[rec["plan"]["machine"]]()
            predicted = prefill(Plan(**rec["plan"]), m, rec.get("prompt", 1024)).tok_s
        measured = rec["measured"]["tok_s"]
        error = predicted / measured - 1
        tol = rec.get("tolerance", 0.15)
        rows.append(dict(name=rec["name"], kind=rec["kind"], measured=measured, predicted=predicted, error=error, tolerance=tol,
                         within=abs(error) <= tol, fit=rec.get("fit", True), source=rec.get("source", "")))
    return rows
