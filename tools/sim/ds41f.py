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
import json
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
                                        "IQ3_S": (256, 110), "IQ4_NL": (32, 18), "MXFP4": (32, 17), "FP8": (1, 1)})


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
# The A64FX SDOT kernels per node, 48 threads, as the runner would build them: clang 21 -mcpu=a64fx, the best kernel per
# width, one session (job 52440185; tools/sim/data/measured_a64fx_kernels.json "sdot_kernel_v3"). gate/up on the DS4.1F
# 2304 x 5120 shape, down on 5120 x 2304 where measured; Q3_K from GLM's shapes; an IQ format used as down takes its
# gate/up rate (not measured on the down shape).
def _sdot_tables():
    data = json.loads((pathlib.Path(__file__).resolve().parent / "data" / "measured_a64fx_kernels.json").read_text())
    c = data["sdot_kernel_v3"]["chosen_llvm_best_per_width"]
    gateup = {"Q2_K": c["q2_K_2304x5120"][0], "Q3_K": c["q3_K_2048x4096"][0], "IQ2_XXS": c["iq2_xxs_2304x5120"][0], "IQ1_S": c["iq1_s_2304x5120"][0]}
    down = {"Q2_K": c["q2_K_5120x2304"][0], "Q3_K": c["q3_K_4096x2048"][0], "IQ2_XXS": gateup["IQ2_XXS"], "IQ1_S": gateup["IQ1_S"]}
    curve = lambda s: tuple(round(s[0] / v, 3) for v in s)
    width = {"Q2_K": curve(c["q2_K_2048x4096"]), "Q3_K": curve(c["q3_K_4096x2048"]), "IQ1_S": curve(c["iq1_s_2048x4096"])}
    # IQ2_XXS: the best equal-size kernel per width across sessions (the in-register svtbl decode won at 2-4 tokens,
    # job 52439675 "iq_gather_free"), so a width-dispatching runner gets the lower of the two ratios
    gf = data["iq_gather_free"]["node48_gbps_tokens_1_to_4"]["iq2_xxs_2048x4096"]
    env_gf = [max(gf["gather"][i], gf["tbl"][i]) for i in range(4)]
    width["IQ2_XXS"] = tuple(min(a, b) for a, b in zip(curve(c["iq2_xxs_2048x4096"]), curve(env_gf)))
    return gateup, down, width


A64FX_SDOT_GATEUP_GBPS, A64FX_SDOT_DOWN_GBPS, A64FX_SDOT_WIDTH_COST = _sdot_tables()


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
                      expert_gbps=250.0, comm_ms=0.0, other_ms=0.0, mhc_site_us=a64fx.MHC_LOCAL_GRAM_INMODEL_US,
                      phase_us=3.0, prefill_int8_eff=0.7)   # mHC: GLM's in-model local-gram site (42.2 us; was the 23 us standalone)


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
# The Strata DS4.1F runner (src/program/ds41_decode.cpp) on TR16 + RX 9070 XT, Unsloth UD-IQ1_S (experts IQ1_S /
# IQ2_XXS 115.2 GB, Engram IQ4_NL 110.6 GB on SSD, trunk Q6_K/Q5_K/Q8_0 6.3 GB on the GPU). Measured 2026-10-20/21 on
# an idle TR16 (out-of-sample wikitext, 1063 tokens). Exact routing. Two constant sets: `now` (what the runner
# measures) and `target` (the plan's tuned kernels: GEMV at 75 % of 640 GB/s, batched trunk prefill, prefetch).
@dataclasses.dataclass
class RunnerHw:
    name: str = "tr16_9070xt"
    trunk_gb: float = 6.30            # trunk + head read per decoded token (the GGUF's own quant types)
    expert_mb: float = 7.5            # mean expert [gate|up|down] (115.2 GB / 15360)
    resident_gib: float = 7.4         # GPU expert tier (one arena, ~1100 slots)
    gemv_gbps: float = 320.0          # Q6_K MMVQ, rocprof 2026-10-21: 12.8 ms for 4.1 GB
    small_ms: float = 8.5             # quantize / norms / hc mixes / adds / f32 GEMVs and their launch gaps per token
    attn_ms_layer: float = 0.06       # head-grouped MQA kernel, <= 640 rows (window 128 + top-512)
    other_ms: float = 7.5             # host syncs (router, indexer), Engram gather, head, sampling
    cpu_lane_gbps: float = 17.5       # IQ1_S lane, one token (15 Zen 1 threads, Fmt32<19>)
    cpu_gmacs_multi: float = 180.0    # the lane on a 256-token chunk (~25 ms/token for 54 % of routes)
    h2d_gbps: float = 13.8            # pinned DMA (hipMemcpy, PCIe 3.0 x16)
    staged_gbps: float = 21.0         # host copy mapping -> pinned (MAP_SHARED, 8 threads)
    stream_eff_gbps: float = 9.0      # streamed prefill copies as measured (~15 % of the experts not page-cached)
    hit_adaptive: float = 0.61        # tier hit out of sample, 2 promotions/token (static profile: 0.39)
    hit_prefill: float = 0.46         # the same tier during chunked prefill (promotions once per pass)
    prefetch_gain: float = 0.0        # next-layer prediction: hit += gain (Xolotl: +~0.08)
    split_misses: bool = False        # misses split between the CPU lane and PCIe uploads
    dense_ms_token: float = 9.0       # prefill dense half per token (GEMM trunk, per-token attention, host top-k), 1K context
    mmq_ms_token: float = 3.1         # prefill MMQ experts per token not hidden under the copies (measured)
    chunk_ms_layer: float = 1.0       # per layer per pass: syncs, batch setup
    overlap: bool = False             # stream: expert copies under the dense/MMQ work (now: within a layer only, summed)
    stream_min: int = 512             # passes shorter than this keep the CPU lane (STRATA_DS41_STREAM_MIN)
    engram_prefill_ms_token: float = 0.25
    ssd_gbps: float = 1.7             # Gen3 x2 NVMe: experts not in the page cache
    # two nodes (STRATA_DS41_REMOTE_TP): a B550 worker (Ryzen 3950X, DDR4-2133) holds FFN rows of every expert
    remote_lane_gbps: float = 17.0    # its IQ1_S lane (decode requests: ~10 MB in ~0.6 ms, 2026-10-21)
    remote_layer_ms: float = 0.02     # per layer: the request/reply round trip not hidden (measured wait 0.02-0.07 ms)


RUNNER_PROFILES = {
    "now": RunnerHw(),
    "target": RunnerHw(gemv_gbps=480.0, small_ms=2.5, attn_ms_layer=0.05, other_ms=2.5, prefetch_gain=0.09, split_misses=True,
                       stream_eff_gbps=13.8, dense_ms_token=2.3, mmq_ms_token=0.4, chunk_ms_layer=0.5, overlap=True),
}


@dataclasses.dataclass
class RunnerPlan:
    profile: str = "now"
    mode: str = "stream"              # prefill: "cpu_experts" (GPU trunk, CPU lane + GPU tier) or "stream" (all experts on the GPU)
    chunk: int = 256                  # prefill tokens per pass (stream: the layer-major sweep is chunk = prompt)
    enc_skip: int = 0                 # prefill: layers past the last KV source run only the last `enc_skip` tokens (0 off)
    uncached_gb: float = 0.0          # experts read from the SSD per pass (not page-cached)
    remote_share: float = 0.0         # two nodes: the worker's share of every expert's FFN rows (k/9)
    set: dict = dataclasses.field(default_factory=dict)   # RunnerHw overrides

    def hw(self):
        h = dataclasses.replace(RUNNER_PROFILES[self.profile])
        for k, v in self.set.items():
            setattr(h, k, v)
        return h


def runner_decode(rp, layers=40, top_k=6):
    """ms per token = trunk (GEMV + small kernels + attention) + experts not on the GPU + other. Misses go to the CPU
    lane, or split with PCIe uploads computed on the GPU (both streams in parallel)."""
    h = rp.hw()
    s = rp.remote_share
    # the tier holds this node's slices: 1 / (1 - s) as many experts; the hit rate takes the same share of the misses
    hit = min(0.98, h.hit_adaptive + h.prefetch_gain)
    if s:
        hit = min(0.98, hit + (1 - hit) * s * 0.5)
    trunk = h.trunk_gb / h.gemv_gbps * 1e3 + h.small_ms + h.attn_ms_layer * layers
    miss_gb = layers * top_k * (1 - hit) * h.expert_mb / 1e3 * (1 - s)
    experts = miss_gb / ((h.cpu_lane_gbps + h.h2d_gbps) if h.split_misses else h.cpu_lane_gbps) * 1e3
    remote = 0.0
    if s:   # the worker computes its rows of every route, in parallel with this node's experts, layer by layer
        remote = layers * top_k * h.expert_mb / 1e3 * s / h.remote_lane_gbps * 1e3 + layers * h.remote_layer_ms
        experts = max(experts, remote)
    step = trunk + experts + h.other_ms
    return dict(tok_s=1e3 / step, step_ms=step, trunk_ms=trunk, experts_ms=experts, other_ms=h.other_ms, hit=hit, miss_gb=miss_gb,
                remote_ms=remote)


def runner_prefill(rp, prompt, layers=40, top_k=6, experts=384, last_source=20):
    """Seconds for a prompt. cpu_experts: per pass, the GPU trunk then the CPU lane on the non-resident routes (no
    overlap). stream: per pass and layer, max(expert uploads, dense + MMQ) - every touched expert crosses PCIe once
    per pass; enc_skip shrinks the dense work of the layers past the last KV source."""
    h = rp.hw()
    chunk = max(1, min(rp.chunk, prompt))
    passes = math.ceil(prompt / chunk)
    resident = h.resident_gib * GIB / 1e9
    total = 0.0
    parts = {}
    for i in range(passes):
        n = min(chunk, prompt - i * chunk)
        ctx_scale = 1.0 + 0.15 * ((i * chunk + n / 2) / 1024 - 1) if (i * chunk + n / 2) > 1024 else 1.0   # attention/indexer grow
        dense_layers = layers if not rp.enc_skip else last_source + 1 + (layers - last_source - 1) * min(1.0, rp.enc_skip / n)
        dense = n * h.dense_ms_token * ctx_scale * dense_layers / layers / 1e3
        if rp.mode == "cpu_experts" or n < h.stream_min:
            routes = n * top_k * layers * (1 - h.hit_prefill)
            cpu = routes * 3 * 5120 * 2304 / (h.cpu_gmacs_multi * 1e9)
            t = dense + cpu + layers * h.chunk_ms_layer / 1e3
            parts = dict(dense_s=dense, cpu_s=cpu)
        else:
            distinct = experts * (1 - (1 - top_k / experts) ** n)
            upload_gb = max(0.0, distinct * layers * h.expert_mb / 1e3 - resident)
            upload = upload_gb / h.stream_eff_gbps + rp.uncached_gb / h.ssd_gbps
            gpu = dense + n * h.mmq_ms_token / 1e3
            t = (max(upload, gpu) if h.overlap else upload + gpu) + layers * h.chunk_ms_layer / 1e3
            parts = dict(dense_s=dense, upload_s=upload, mmq_s=n * h.mmq_ms_token / 1e3)
        total += t + n * h.engram_prefill_ms_token / 1e3
    return dict(tok_s=prompt / total, seconds=total, passes=passes, **parts)


# ---------------------------------------------------------------------------------------------------------------
# The Strata DS4.1F runner on the x64 GPU node (2x Tesla V100 PCIe 32 GB, 2x Xeon Gold 6240, 181 GiB, Lustre),
# IQ2_XXSE4 / IQ2_XXSE3, every decode option on (docs/DS41F_PLAN.md 8e), measured 2026-10-22. A decode step:
#   plain:  one pass of T = 1 token;
#   DSpark: draft (3-layer drafter on GPU0) + one verify pass of T = 1 + drafted tokens + the drafter's injection;
# a pass = A + B T + C (CPU experts) + Engram stall + head/host overhead, where the CPU experts are
# 40 layers x T x 6 routes x (1 - GPU tier hit) and the Engram stall is the file reads of rows not in RAM:
# the first table's before layer 1 (not hidden), the second table's under the draft and layers 0-13.
# A, B, C come from the full-node points with no Engram reads (IQ2_XXSE3, plain IQ2_XXSE4): C is the CPU lane's
# effective cost per expert once its work overlaps the GPU's (the lane alone measures ~0.12 ms per IQ2_XXS expert
# per layer, LANE_BENCH); B the per-token dense kernels (the phase timings: ~3.6 ms per extra token).
@dataclasses.dataclass
class V100Hw:
    name: str = "xeon_v100"
    pass_ms: float = 23.0             # per pass: fixed dense kernels, 40 router syncs, merges (fit, less the peer wait; the
                                      # lumped fit was 32.9 for one- and 3.6-token passes alike)
    token_ms: float = 6.1             # per token in a pass: dense kernels, attention, MoE bookkeeping (fit)
    cpu_expert_ms: float = 0.06       # per CPU-lane expert evaluation, net of overlap (fit; the lane alone 0.12)
    peer_wait_ms: float = 9.3         # per pass: the host's wait for the peer GPU after the CPU lane (measured 9.3 at one
    peer_wait_token_ms: float = 0.8   # token, 13.1 at 3.6, 14.4 at 5.2; fit with pass_ms, 3 %; 0 with STRATA_DS41_PEER_THREAD)
    draft_ms: float = 6.6             # DSpark draft less its router syncs: Q8_0 trunk, device Markov chain (measured 14.5-15.9 whole)
    draft_route_ms: float = 8.4       # the draft's per-layer host syncs for routing (0 with DSPARK_GPU_ROUTE)
    lane_gbps_per_core: float = 5.0   # the IQ2_XXS CPU kernel's codebook ceiling (measured 3.9 in decode): the lane cannot
                                      # be asked to go below lane_mb x experts / (36 cores x this)
    lane_mb: float = 9.0              # the lane's bytes per expert evaluation (IQ2_XXSE4; other packs scale it and cpu_expert_ms
                                      # by their bytes-weighted expert size over 9.95 MB, less a per-format kernel efficiency)
    prefill_ms_token: float = 2.05    # a streamed prompt pass: MMQ + dense per token (fit to IQ2_XXSE4 4K 389 / 8K 412 tok/s with
    h2d_gbps: float = 24.0            # the RAM experts uploaded once per pass over both GPUs' links, half of it overlapped)
    upload_overlap: float = 0.5
    inject_ms: float = 0.4            # the drafter's keys of the accepted positions
    host_ms: float = 1.5              # per pass: logits download, argmax (+ 0.7 ms per token, in token_ms)
    engram_iops: float = 800.0        # effective random 4 KiB Lustre reads/s for Engram misses (raw buffered: ~1,550;
                                      # fit to the full-node IQ2_XXSE4 points, varies by day: 600-1,000)
    engram_rows: int = 24             # rows per table per token
    table_gib: tuple = (51.5, 51.5)   # IQ2_XXSE4 tables (Q4_K)
    expert_gib: float = 142.4         # all routed experts
    gpu_expert_gib: float = 47.4      # resident on the two GPUs with the drafter on GPU0 (55.3 without it)
    hit_exponent: float = 0.40        # tier hit = resident fraction ^ exponent (profile-ranked residency):
                                      # 55.3 GiB -> 0.684, 47.4 GiB -> 0.644 (measured)
    index_gbps: float = 800.0         # long context: indexer key reads (V100 HBM2)
    index_tflops: float = 12.0        # long context: indexer scoring, FP32 (FP16 tensor cores: ~4x)
    topk_gelem_s: float = 10.0        # long context: top-k selection, elements per second


@dataclasses.dataclass(frozen=True)
class V100Pack:
    """A pack on the V100 runner: its Engram tables and routed experts in bytes (the per-layer formats from a census)."""
    name: str
    table_gib: tuple                  # the two Engram tables (GiB)
    expert_gib: float = 142.4         # all routed experts
    expert_mb: float = 9.95           # bytes-weighted mean expert (MB; IQ2_XXS gate/up + Q2_K down)
    layers: tuple = ()                # per layer (gate_fmt, down_fmt, expert_bytes)
    description: str = ""

    def lane_mb_eff(self, lane_eff=None):
        """The mean expert's bytes at the IQ2_XXS kernel's rate: a layer's bytes over its formats' efficiency (a dict
        format -> rate relative to IQ2_XXS; 1.0 where unknown), gate/up and down weighted by their bytes."""
        if not self.layers:
            return self.expert_mb
        eff = lane_eff or {}
        tot = 0.0
        for gate, down, nbytes in self.layers:
            gu = 2 * tensor_bytes(GEOMETRY.hidden * GEOMETRY.expert_ff, gate)
            d = nbytes - gu
            tot += gu / eff.get(gate, 1.0) + d / eff.get(down, 1.0)
        return tot / len(self.layers) / 1e6


def v100_pack_from_census(path, name, table_gib, description=""):
    """A V100Pack from tools/sim/data/ds41f-census-*.json (per layer gate/up/down formats and bytes; see tools/gguf_reader.py)."""
    import json
    d = json.loads(pathlib.Path(path).read_text())
    layers = []
    for l in sorted(d["layers"], key=int):
        t = d["layers"][l]
        layers.append((t["gate"]["format"], t["down"]["format"], (t["gate"]["bytes"] + t["up"]["bytes"] + t["down"]["bytes"]) / 384))
    total = sum(b for _, _, b in layers) * 384
    return V100Pack(name, table_gib, total / GIB, total / len(layers) / 384 / 1e6, tuple(layers), description)


V100_PACKS = {   # the experts are the same IQ2_XXS / Q2_K in the first three; GSQ-RCO's differ per layer (IQ1_M / IQ2_XXS / IQ2_XS)
    "IQ2_XXSE4": V100Pack("IQ2_XXSE4", (51.5, 51.5), description="Engram Q4_K, KL 0.289"),
    "IQ2_XXSE43": V100Pack("IQ2_XXSE43", (51.5, 39.3), description="layer-14 table Q3_K, KL 0.307"),
    "IQ2_XXSE3": V100Pack("IQ2_XXSE3", (39.3, 39.3), description="Engram Q3_K, KL 0.314"),
    "GSQ-IQ2_XXS": v100_pack_from_census(DATA / "ds41f-census-gsq-iq2xxs.json", "GSQ-IQ2_XXS", (51.5, 51.5),
                                         "ISTA-DASLab GSQ-RCO 2.00 bpw, Engram IQ4_NL"),
}
KV_BYTES = dict(fp32=6400, fp16=3200, q8=1700)   # per token: 640-float rows, layers 2/8/14 one per 2 positions, 20 one each


@dataclasses.dataclass
class V100Plan:
    pack: str = "IQ2_XXSE4"
    engram_gib: float = 76.0          # Engram in RAM (STRATA_DS41_ENGRAM_RAM_GIB): table 1 whole first
    spec: bool = True                 # DSpark
    drafted: float = 2.6              # drafted tokens per step (N=5, PMIN 0.4: wikitext 2.5-2.6, code 4.2-4.3)
    acceptance: float = 0.68          # accepted / drafted (wikitext 0.67-0.74, code 0.63-0.65)
    drafter_on_gpu: bool = True       # the drafter's 8 GB on GPU0 (fewer resident experts)
    extra_gpu_gib: float = 0.0        # more resident experts (GROW, a smaller drafter, ...)
    sessions: int = 1                 # batched sessions per pass (each its own tokens); 1 = today's runner
    context: int = 600                # positions before the generated tokens (long context: indexer cost, KV)
    kv: str = "fp32"                  # compressed KV rows: fp32 (now), fp16, q8
    index_tc: bool = False            # FP16 tensor-core index scoring
    engram_hit1: float = -1.0         # table 1's RAM hit fraction (-1: its leading-rows share; a frequency-ranked row
                                      # cache, STRATA_DS41_ENGRAM_ROWS, reaches more at the same bytes)
    lane_eff: dict = dataclasses.field(default_factory=dict)   # CPU lane rate per expert format relative to IQ2_XXS
    set: dict = dataclasses.field(default_factory=dict)   # V100Hw overrides

    def hw(self):
        pk = V100_PACKS[self.pack]
        scale = pk.lane_mb_eff(self.lane_eff) / 9.95   # the fit's IQ2_XXSE4 expert
        h = dataclasses.replace(V100Hw(), table_gib=pk.table_gib, expert_gib=pk.expert_gib, lane_mb=9.0 * scale,
                                cpu_expert_ms=0.06 * scale)
        for k, v in self.set.items():
            setattr(h, k, v)
        return h


def v100_decode(vp):
    """tok/s of the V100 runner (aggregate over the sessions): see V100Hw."""
    h = vp.hw()
    t0, t1 = h.table_gib
    share0 = min(1.0, vp.engram_gib / t0)
    share1 = min(1.0, max(0.0, vp.engram_gib - t0) / t1)
    if vp.engram_hit1 >= 0: share1 = min(1.0, vp.engram_hit1)
    kv_gib = KV_BYTES[vp.kv] * vp.context * vp.sessions / GIB
    resident = h.gpu_expert_gib + (0.0 if vp.spec or vp.drafter_on_gpu else 7.9) + vp.extra_gpu_gib - kv_gib   # no drafter: +7.9 GiB
    hit = min(0.98, max(0.0, resident / h.expert_gib) ** h.hit_exponent)
    per = 1 + vp.drafted if vp.spec else 1.0          # tokens of one session in a pass
    T = per * vp.sessions
    cpu = 40 * T * 6 * (1 - hit)
    lane_floor = cpu * h.lane_mb * 1e-3 / (36 * h.lane_gbps_per_core) * 1e3 * 0.5   # GB over the cores' ceiling, half overlapped as the fit's
    dense = h.pass_ms + h.token_ms * T + max(h.cpu_expert_ms * cpu, lane_floor) + h.host_ms
    dense += h.peer_wait_ms + h.peer_wait_token_ms * (T - 1)
    # long context: every token's 8 index layers score 3 pos/2 + 5 pos rows of 128-dim keys, then a top-k over them
    rows = 6.5 * vp.context
    key_b = dict(fp32=512, fp16=256, q8=136)[vp.kv]          # one 128-dim key
    flops = rows * 32 * 128 * 2
    index_ms = T * (max(rows * key_b / (h.index_gbps * 1e9), flops / (h.index_tflops * 1e12 * (4 if vp.index_tc else 1)))
                    + rows / (h.topk_gelem_s * 1e9)) * 1e3
    dense += index_ms
    draft = ((h.draft_ms + h.draft_route_ms) * vp.sessions + h.inject_ms) if vp.spec else 0.0
    hide = draft + 14 / 40 * dense
    m0 = h.engram_rows * T * (1 - share0)
    m1 = h.engram_rows * T * (1 - share1)
    engram = m0 / h.engram_iops * 1e3 + max(0.0, m1 / h.engram_iops * 1e3 - hide)
    step = draft + dense + engram
    tokens = (1 + vp.acceptance * vp.drafted if vp.spec else 1.0) * vp.sessions
    return dict(tok_s=tokens / step * 1e3, step_ms=step, draft_ms=draft, dense_ms=dense, engram_ms=engram, index_ms=index_ms,
                cpu_experts=cpu, hit=hit, tokens_per_step=tokens, kv_gib=kv_gib, per_session_tok_s=tokens / step * 1e3 / vp.sessions)


def v100_prefill(vp, prompt):
    """tok/s of a streamed prompt pass (STRATA_DS41_RAM_PINNED + PEER_STREAM): the RAM experts uploaded once over both
    links, partly under the per-token MMQ and dense work; a crude two-point fit, see V100Hw."""
    h = vp.hw()
    resident = h.gpu_expert_gib + (0.0 if vp.spec or vp.drafter_on_gpu else 7.9) + vp.extra_gpu_gib
    upload_s = max(0.0, h.expert_gib - resident) * GIB / (h.h2d_gbps * 1e9)
    pass_s = upload_s * (1 - h.upload_overlap) + prompt * h.prefill_ms_token * 1e-3
    return dict(tok_s=prompt / pass_s, pass_s=pass_s, upload_s=upload_s)


def v100_packs_compare(packs=("IQ2_XXSE4", "GSQ-IQ2_XXS"), lane_effs=({}, {"IQ1_M": 0.6})):
    """(label, decode wiki600, decode code600, decode 106K q8, prefill 4K, prefill 8K) per pack and lane-rate hypothesis,
    with Engram RAM at 76 GiB and at the RAM the smaller experts free (+ expert GiB saved vs IQ2_XXSE4)."""
    out = []
    e4 = V100_PACKS["IQ2_XXSE4"].expert_gib
    for name in packs:
        pk = V100_PACKS[name]
        for eff in (lane_effs if pk.layers else ({},)):
            for egib in sorted({76.0, min(103.0, 76.0 + (e4 - pk.expert_gib))}):
                r = lambda **kw: V100Plan(**dict(V100_BASE, pack=name, engram_gib=egib, lane_eff=eff, **kw))   # noqa: E731
                label = f"{name} Engram {egib:.0f} GiB" + (f" lane {eff}" if eff else "")
                out.append((label, v100_decode(r())["tok_s"], v100_decode(r(drafted=4.2, acceptance=0.65))["tok_s"],
                            v100_decode(r(context=106000, kv="q8", index_tc=True, drafted=1.7, acceptance=0.6))["tok_s"],
                            v100_prefill(r(), 4096)["tok_s"], v100_prefill(r(), 8192)["tok_s"]))
    return out


V100_BASE = dict(pack="IQ2_XXSE4", engram_gib=76.0, spec=True, drafted=2.6, acceptance=0.68)


def v100_levers(base=None):
    """(lever, plan) pairs on top of the full-node IQ2_XXSE4 DSpark configuration (wikitext acceptance)."""
    b = dict(V100_BASE, **(base or {}))
    r = lambda **kw: V100Plan(**dict(b, **kw))   # noqa: E731
    sets = lambda **kw: V100Plan(**dict(b, set=kw))   # noqa: E731
    return [
        ("baseline", r()),
        ("Engram whole in RAM: IQ2_XXSE3 pack (KL 0.314)", r(pack="IQ2_XXSE3", engram_gib=79)),
        ("IQ2_XXSE43 pack (layer-14 table Q3_K, KL 0.307)", r(pack="IQ2_XXSE43")),
        ("Engram: Lustre at its raw 1,550 reads/s", sets(engram_iops=1550.0)),
        ("peer GPU on its own thread (PEER_THREAD; the host no longer waits for it)", sets(peer_wait_ms=0.0, peer_wait_token_ms=0.0)),
        ("draft routed on the device (DSPARK_GPU_ROUTE: no per-layer host sync)", sets(draft_route_ms=0.0)),
        ("Engram rows by frequency (ENGRAM_ROWS: table 1 hit 0.8 at the same bytes)", r(engram_hit1=0.8)),
        ("CPU lane 1.25x (placement; 2x is past its codebook ceiling)", sets(cpu_expert_ms=0.048)),
        ("CPU lane 2x faster (IQ2_XXS kernel)", sets(cpu_expert_ms=0.03)),
        ("GPU tier +6.3 GiB (GROW after a 4K prompt)", r(extra_gpu_gib=6.3)),
        ("drafter experts off GPU0 (+7.2 GiB of target experts)", r(extra_gpu_gib=7.2)),
        ("dense kernels -25 % (FUSE: launch fusion)", sets(pass_ms=23.0 * 0.75, token_ms=6.1 * 0.75)),
        ("draft time -50 %", sets(draft_ms=7.5)),
        ("acceptance +10 % (0.75)", r(acceptance=0.75)),
        ("code-like text (4.2 drafted, 0.65 accepted)", r(drafted=4.2, acceptance=0.65)),
        ("Engram in RAM + lane 2x + dense -25 %", V100Plan(**dict(b, pack="IQ2_XXSE3", engram_gib=79,
                                                             set=dict(cpu_expert_ms=0.03, pass_ms=23.0 * 0.75, token_ms=6.1 * 0.75)))),
        ("the 40+ stack: peer thread + FUSE -25 % + device draft (IQ2_XXSE3)", V100Plan(**dict(b, pack="IQ2_XXSE3", engram_gib=79,
                                                             set=dict(peer_wait_ms=0.0, peer_wait_token_ms=0.0, draft_route_ms=0.0,
                                                                      pass_ms=23.0 * 0.75, token_ms=6.1 * 0.75)))),
        ("the 40+ stack + Engram rows 0.8 (IQ2_XXSE4)", V100Plan(**dict(b, engram_hit1=0.8,
                                                             set=dict(peer_wait_ms=0.0, peer_wait_token_ms=0.0, draft_route_ms=0.0,
                                                                      pass_ms=23.0 * 0.75, token_ms=6.1 * 0.75)))),
    ]


def v100_sessions(counts=(1, 2, 4, 8), **kw):
    """Aggregate and per-session tok/s for S batched sessions (plain and DSpark)."""
    out = []
    for s in counts:
        for spec in (False, True):
            out.append((s, spec, v100_decode(V100Plan(**dict(V100_BASE, spec=spec, sessions=s, **kw)))))
    return out


def v100_context(contexts=(600, 131072, 262144, 524288, 1048576), kvs=("fp32", "fp16", "q8"), **kw):
    """DSpark decode at long context per KV row format (the indexer scores every visible row)."""
    out = []
    for ctx in contexts:
        for kv in kvs:
            out.append((ctx, kv, v100_decode(V100Plan(**dict(V100_BASE, context=ctx, kv=kv, **kw)))))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Levers from the 2026-10-26 research (docs/DS41F_PLAN.md 8k) on a phase model of the V100 runner, calibrated to the
# 2026-10-25/26 runs at the node-safe Engram RAM 56 GiB (8j). A DSpark step = draft + verify pass of T = 1 + k tokens +
# Engram stall; the verify pass = dense(T) + FFN(T) + host. FFN(T) per pass = GPU-0 experts (G) + its wait for the
# CPU lane and GPU 1 (W): MMQ G 36.4 / W 23.2 ms, S27 G 26.0 / W 21.5 ms at T ~ 5 (s72: experts= and lane_wait=),
# scaled sub-linearly in T (more tokens, more distinct experts, not proportionally). Draft acceptance as conditional
# probabilities per draft position, fitted to the measured tokens per step (code600 4.0-4.27, wiki600 2.56-2.63).
@dataclasses.dataclass
class LeverHw:
    draft_ms: float = 14.0             # DSpark draft per step (5 positions; 8.4 of it router syncs)
    dense0_ms: float = 24.0            # dense phases per pass (pre, q, kv/index, attention, out, ffn_pre, router, tail)
    dense_tok_ms: float = 1.94         #   + per token (33.7 ms at T = 5)
    host_ms: float = 4.0               # per pass: logits, argmax, rewind, injection
    ffn_g: dict = dataclasses.field(default_factory=lambda: {"mmq": 36.4, "s27": 26.0})   # GPU-0 experts per pass at T = 5
    ffn_w: dict = dataclasses.field(default_factory=lambda: {"mmq": 23.2, "s27": 21.5})   # its wait for the lane / GPU 1
    lane_ms: float = 27.5              # the CPU lane per pass at T = 5 (lane= in decode_gpu)
    ffn_a: float = 0.118               # FFN(T) = FFN(T=5) x (a + b T): fitted to E3 MMQ code600 (T ~ 5.3, verify 94 ms
    ffn_b: float = 0.154               #   without Engram) and wiki600 (T ~ 3, 68.5 ms); S27 then within 4 ms of both
    attn_share: float = 0.80           # share of the dense phases a deferred lane job can run under (the next layer's attention)
    link_gbps: float = 11.0            # one GPU's PCIe link for DECODE_STREAM copies
    expert_mb: float = 9.95            # IQ2_XXS gate/up + Q2_K down
    lane_route_ms: float = 0.065       # lane time per route per layer (27.5 ms / 40 layers / ~10.6 CPU routes)
    peer_layer_ms: float = 0.39        # GPU 1 busy per layer with S27 (15.5 ms / 40)
    # Engram stall per step at Engram RAM 56 GiB (host engram seconds / steps; it varies 2-3x between runs - Lustre)
    engram_ms: dict = dataclasses.field(default_factory=lambda: {
        ("IQ2_XXSE4", "code"): 54.6, ("IQ2_XXSE4", "wiki"): 17.5,
        ("IQ2_XXSE43", "code"): 24.8, ("IQ2_XXSE43", "wiki"): 16.4,
        ("IQ2_XXSE3", "code"): 15.0, ("IQ2_XXSE3", "wiki"): 3.0})
    accept: dict = dataclasses.field(default_factory=lambda: {   # P(draft position j accepted | earlier accepted)
        "code": (0.94, 0.88, 0.82, 0.77, 0.70), "wiki": (0.85, 0.75, 0.55, 0.45, 0.40)})
    drafted: dict = dataclasses.field(default_factory=lambda: {"code": 4.3, "wiki": 2.0})   # mean drafted per step (PMIN 0.4)
    # per-pack verify-pass scale on top of the E3 calibration (E4: same experts, the Engram stall carries the difference)
    pack_scale: dict = dataclasses.field(default_factory=lambda: {"IQ2_XXSE4": 1.0, "IQ2_XXSE43": 1.0, "IQ2_XXSE3": 1.0})
    kl: dict = dataclasses.field(default_factory=lambda: {"IQ2_XXSE4": 0.290, "IQ2_XXSE43": 0.307, "IQ2_XXSE3": 0.314})


@dataclasses.dataclass
class LeverPlan:
    pack: str = "IQ2_XXSE3"
    workload: str = "code"            # code / wiki
    kernel: str = "s27"               # mmq / s27
    draft_n: int = 5                  # drafted positions per step (the drafter's block is 5)
    sched: bool = False               # DSPARK_SCHED: the verified prefix by expected tokens per ms
    defer_rank: int = 6               # DEFER_RANK (6 = off): lane routes of rank >= R overlap the next layer
    decode_stream: int = 0            # DECODE_STREAM 0,b: experts per layer from the lane to GPU 1 over its link
    lane_speedup: float = 1.0         # LANE_NUMA or other lane gains (upstream: 1.10 on a dual Xeon)
    set: dict = dataclasses.field(default_factory=dict)

    def hw(self):
        h = LeverHw()
        for k, v in self.set.items():
            setattr(h, k, v)
        return h


def lever_verify_ms(lp, T, h=None):
    """One verify pass of T tokens (ms) and its parts."""
    h = h or lp.hw()
    scale = h.ffn_a + h.ffn_b * T
    dense = h.dense0_ms + h.dense_tok_ms * T
    g = h.ffn_g[lp.kernel] * scale
    lane = h.lane_ms * scale / lp.lane_speedup
    w = h.ffn_w[lp.kernel] * scale * (h.lane_ms * scale / lane) ** -1 if lp.lane_speedup != 1.0 else h.ffn_w[lp.kernel] * scale
    if lp.lane_speedup != 1.0:   # the wait shrinks by what the lane saves
        w = max(0.0, h.ffn_w[lp.kernel] * scale - (h.lane_ms * scale - lane))
    # DECODE_STREAM: b experts per layer leave the lane (b x ~1.1 routes) and cross GPU 1's link (copy + compute)
    if lp.decode_stream:
        saved = 40 * lp.decode_stream * 1.1 * h.lane_route_ms * scale
        peer = 40 * (h.peer_layer_ms + lp.decode_stream * h.expert_mb / h.link_gbps)
        w = max(0.0, w - saved, peer - g)
    # deferral: the share of lane routes with rank >= R finishes under the next layer's attention
    hidden = 0.0
    if lp.defer_rank < 6:
        share = (6 - lp.defer_rank) / 6.0
        hidden = min(share * lane, h.attn_share * dense)
        w = max(0.0, w - hidden)
    v = (dense + g + w + h.host_ms) * h.pack_scale[lp.pack]
    return v, dict(dense=dense, g=g, w=w, hidden=hidden)


def lever_decode(lp):
    """tok/s of a DSpark step: the expected tokens of the verified prefix over draft + verify + Engram stall; with
    sched, the prefix length k <= draft_n that maximizes that ratio (per step it would use the drafter's confidences;
    in expectation the best fixed k is the bound computed here)."""
    h = lp.hw()
    p = h.accept[lp.workload]
    eng = h.engram_ms[(lp.pack, lp.workload)]
    def rate(k):
        exp_tok, run = 1.0, 1.0
        for j in range(int(math.ceil(k))):
            run *= p[j]
            exp_tok += run * min(1.0, k - j)
        step = h.draft_ms + lever_verify_ms(lp, k + 1, h)[0] + eng
        return exp_tok / step * 1e3, exp_tok, step
    # without sched: today's PMIN cut (the measured mean drafted); with it, the best prefix length
    ks = range(0, min(lp.draft_n, len(p)) + 1) if lp.sched else [min(float(lp.draft_n), h.drafted[lp.workload])]
    best = max(ks, key=lambda k: rate(k)[0])
    r, tok, step = rate(best)
    kl = h.kl[lp.pack] + {6: 0.0, 5: 0.008, 4: 0.018, 3: 0.029}.get(lp.defer_rank, 0.05)   # dsx: +0.029 at R=3 (T16)
    return dict(tok_s=r, tokens_per_step=tok, step_ms=step, k=best, engram_ms=eng, kl=kl)


V100_PREFILL = dict(upload_gib=94.8, link_gbps=24.0, overlap=0.5, mmq_ms_token=2.05, f16_gain=1.0)


def lever_prefill(prompt, batch=2048, f16_gain=1.0, peer_split=True, set=None):
    """Streamed prompt passes on the V100 node: each pass uploads every RAM expert once (both links when the GPU-1
    split is on), partly under the per-token work (MMQ experts + dense). f16_gain: the expert compute's speed-up from
    an FP16 tensor-core GEMM path (C2; set from the C1 bench)."""
    c = dict(V100_PREFILL, **(set or {}))
    passes = math.ceil(prompt / batch)
    link = c["link_gbps"] if peer_split else c["link_gbps"] / 2
    upload_s = c["upload_gib"] * GIB / (link * 1e9)
    expert_share = 0.6                 # of the per-token work, the experts' (the rest: dense, attention, indexer)
    per_tok = c["mmq_ms_token"] * ((1 - expert_share) + expert_share / f16_gain)
    total = 0.0
    for i in range(passes):
        n = min(batch, prompt - i * batch)
        ctx = i * batch + n / 2
        grow = 1.0 + 0.004 * max(0.0, ctx / 8192 - 1)      # attention / indexer growth (INDEX_FAST, 2-stage top-k: 377 at 106K)
        total += upload_s * (1 - c["overlap"]) + n * per_tok * grow * 1e-3
    return dict(tok_s=prompt / total, seconds=total, passes=passes)


def lever_table():
    """Rows (label, plan) for the V100 node: the measured baseline, each lever alone, and the stacks."""
    rows = []
    for pack in ("IQ2_XXSE4", "IQ2_XXSE3"):
        for wl in ("code", "wiki"):
            b = dict(pack=pack, workload=wl)
            rows += [
                (f"{pack} {wl}: MMQ (measured base)", LeverPlan(kernel="mmq", **b)),
                (f"{pack} {wl}: S27", LeverPlan(**b)),
                (f"{pack} {wl}: S27 + sched", LeverPlan(sched=True, **b)),
                (f"{pack} {wl}: S27 + defer 4", LeverPlan(defer_rank=4, **b)),
                (f"{pack} {wl}: S27 + defer 3", LeverPlan(defer_rank=3, **b)),
                (f"{pack} {wl}: S27 + DECODE_STREAM 1", LeverPlan(decode_stream=1, **b)),
                (f"{pack} {wl}: S27 + lane NUMA 1.10", LeverPlan(lane_speedup=1.10, **b)),
                (f"{pack} {wl}: S27 + sched + defer 3 + NUMA", LeverPlan(sched=True, defer_rank=3, lane_speedup=1.10, **b)),
            ]
    return rows


# TR16 (Threadripper 1950X, 128 GB DDR4, RX 9070 XT 16 GB), Unsloth UD-IQ1_S with the Engram tables on the local NVMe.
# The runner model above (RunnerHw `now`: 12.5 tok/s measured 2026-10-21) plus the same levers: deferral hides part of
# the CPU lane under the next layer's attention; DSpark needs its 7.7 GB of drafter weights on the GPU, which with the
# 6.3 GB trunk and the 3 GiB desktop reserve leaves no expert tier on 16 GB, so speculation there is the history-lookup
# drafter (no weights) or a drafter with its experts on the CPU.
def tr16_levers(profile="now"):
    out = []
    base = runner_decode(RunnerPlan(profile=profile))
    h = RUNNER_PROFILES[profile]
    trunk = base["trunk_ms"]
    for label, defer_share, spec_tokens, spec_cost in (
            ("plain", 0.0, 1.0, 0.0),
            ("defer 3 (lossy, ~+0.03 KL)", 0.5, 1.0, 0.0),
            ("lookup drafter on code (~1.6 tokens/step, verify ~1.35x)", 0.0, 1.6, 0.35),
            ("defer 3 + lookup drafter", 0.5, 1.6, 0.35)):
        hidden = min(defer_share * base["experts_ms"], 0.8 * (trunk - h.small_ms))
        step = base["step_ms"] - hidden
        step = step * (1 + spec_cost)
        out.append((label, spec_tokens / step * 1e3, step))
    vram_gib = 16 - 3 - h.trunk_gb * 1e9 / GIB
    return out, dict(base_tok_s=base["tok_s"], hit=base["hit"], vram_for_experts_and_drafter_gib=vram_gib, drafter_gib=7.7)


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# The measured two-node runner: TR16 (RX 9070 XT) + B550, UD-IQ1_S, worker owning 48 GiB of whole experts over IB, DSpark N=5.
# Fitted to tools/sim/data/measured_ds41f_two_node.json (ten 2026-10-29 runs, c_osd 1000-token prompt, 128 generated). The generic
# decode() above predicts 15.0 tok/s plain and 33.2 with DSpark for this pair; the runner's measured breakdown differs in four places:
# the expert pass is a flat ~37 ms per unit of union x non-canonical share and gains nothing from width (decode() assumed a 2.3x
# width gain: measured width-5 / width-1 expert time is 3.9x against a union ratio of 3.6), the GPU-to-router wait grows to ~50 ms at
# width 5 (the per-token attention), Engram is 6 ms per width-5 step, and a DSpark round also pays ~25 ms drafting.
TWO_NODE = dict(
    cpu_ms_per_union=37.3,        # LS fit through the origin, grid8 runs: width 1 23.3 ms and width 5 90.8 ms at canon hit 0.325 (sim union ratio 1 / 3.585)
    grid8_off_factor=1.13,        # worker CPU 1.580 -> 1.435 ms/request; cpu_experts 103.5 / 90.8 ms at width 5, 26.1 / 23.3 at width 1
    g2r_ms=(2.95, 40.4),          # gpu_to_router wait at width 1 / width 5 before the CPU coupling below
    g2r_per_cpu_ms=0.127,         # ... plus this per ms of expert pass at width 5 (LS over the four configuration means: 46.8-55.6 ms wait for 55-104 ms of experts)
    engram_ms=(0.37, 6.25),       # per step at width 1 / 5 (cache hit and async both unchanged: STRATA_DS41_ENGRAM_ASYNC=0)
    head_ms=(0.077, 2.1),
    other_ms=5.8,                 # step minus the four terms above: 6.1 at width 1, 5.7 at width 5
    draft_ms_per_token=5.0,       # draft_s / steps = 23.0-27.2 ms at N=5 (mean of the ten runs 25.3; flat: its slope against the expert pass is within noise)
    inject_ms=0.4,
    first_step_ms=30.5,           # the plain step that produces the first generated token (profile_prompt_s step, width 1)
    hit_ref=(2302.0, 0.3097, 0.1816),   # canonical-tier hit h = 0.3097 (MiB / 2302) ^ 0.1816, from 2302 MiB (0.3097) and 3033 MiB (0.3253)
    trim_cut_ref=(0.125, 0.232, 2.64),  # CPU expert time removed by STRATA_DS41_CPU_TRIM t: 0.232 (t / 0.125) ^ 2.64, from t = 0.125 and 0.15 (0.375)
    vram_free_ref=(3033.0, 3130449920.0),   # minimum free VRAM at 3033 MiB loaded; each loaded MiB costs one MiB
    vram_floor_bytes=2.5 * GIB,
    canon_loaded_mib_default=3033.0,    # what a 3328 MiB request loads on the desktop as of 2026-10-29 (3254 on an earlier day)
    accept_profile="ds41_osd_n5",
)


@dataclasses.dataclass
class TwoNodePlan:
    width: int = 5                       # verify width (DSpark N); only 1 and 5 are measured, others interpolate/extrapolate linearly
    canon_mib: float = -1.0              # canonical tier loaded MiB; -1 = TWO_NODE default
    trim: float = 0.0                    # STRATA_DS41_CPU_TRIM
    grid8: bool = True
    cpu_speed: float = 1.0               # what-if: expert-pass speedup (worker kernel / threads)
    draft_overlap: float = 0.0           # what-if: share of the drafting time hidden behind the verify
    engram_hidden: float = 0.0           # what-if: share of the Engram time hidden (async)
    generated: int = 128
    acceptance: str = ""                 # "" = the fitted profile

    to_dict = lambda self: dataclasses.asdict(self)


def two_node_hit(canon_mib):
    ref, h, a = TWO_NODE["hit_ref"]
    return min(0.95, h * (canon_mib / ref) ** a)


def two_node_trim_cut(trim):
    if trim <= 0:
        return 0.0
    t, c, k = TWO_NODE["trim_cut_ref"]
    return min(0.6, c * (trim / t) ** k)


def two_node_vram_free_gib(canon_mib):
    mib, free = TWO_NODE["vram_free_ref"]
    return (free - (canon_mib - mib) * 2 ** 20) / GIB


def two_node_decode(plan):
    """Width-`plan.width` DSpark decode on the measured TR16 + B550 runner, ms per component. Returns a dict like a64fx_decode."""
    c = TWO_NODE
    mib = plan.canon_mib if plan.canon_mib > 0 else c["canon_loaded_mib_default"]
    w = plan.width
    frac = (w - 1) / 4.0                                     # 0 at the measured width 1, 1 at width 5
    lerp = lambda pair: pair[0] + (pair[1] - pair[0]) * frac
    hit = two_node_hit(mib)
    union = routing.union_ratio(w, 384, 6)
    cut = two_node_trim_cut(plan.trim)
    cpu = c["cpu_ms_per_union"] * union * (1 - hit) * (1 - cut) * (1.0 if plan.grid8 else c["grid8_off_factor"]) / plan.cpu_speed
    g2r, engram, head = lerp(c["g2r_ms"]) + c["g2r_per_cpu_ms"] * cpu * frac, lerp(c["engram_ms"]) * (1 - plan.engram_hidden), lerp(c["head_ms"])
    verify = g2r + cpu + engram + head + c["other_ms"]
    out = dict(width=w, canon_hit=hit, cpu_ms=cpu, g2r_ms=g2r, engram_ms=engram, head_ms=head, other_ms=c["other_ms"], verify_ms=verify,
               vram_free_gib=two_node_vram_free_gib(mib), vram_ok=two_node_vram_free_gib(mib) * GIB >= c["vram_floor_bytes"])
    if w == 1:
        out.update(step_ms=verify, tok_s=1e3 / verify, tokens_per_round=1.0, draft_ms=0.0, round_ms=verify)
        return out
    profile = plan.acceptance or c["accept_profile"]
    tokens = routing.tokens_per_round(w - 1, profile)
    draft = c["draft_ms_per_token"] * w * (1 - plan.draft_overlap)
    round_ms = verify + draft + c["inject_ms"]
    rounds = (plan.generated - 1) / tokens                  # expected rounds; measured runs are 27 or 28
    total = c["first_step_ms"] + rounds * round_ms
    out.update(tokens_per_round=tokens, draft_ms=draft, round_ms=round_ms, rounds=rounds, step_ms=verify, tok_s=plan.generated / total * 1e3)
    return out


def two_node_levers(base=None):
    """What-ifs on the calibrated runner. Each row changes one thing and says what it would take; none is measured."""
    base = base or TwoNodePlan()
    rows = [("measured configuration", base)]
    for label, kw in (("canonical tier at the VRAM floor (3458 MiB; +425 MiB)", dict(canon_mib=3458.0)),
                      ("worker + local expert kernels 15 % faster", dict(cpu_speed=1.15)),
                      ("worker + local expert kernels 30 % faster", dict(cpu_speed=1.30)),
                      ("Engram fully hidden by async prefetch", dict(engram_hidden=1.0)),
                      ("drafting half hidden behind the verify", dict(draft_overlap=0.5)),
                      ("drafting fully hidden (drafter on its own stream)", dict(draft_overlap=1.0)),
                      ("trim 0.125 (lossy; fails the KL gate)", dict(trim=0.125)),
                      ("grid8 off (the pre-2026-10-29 kernel)", dict(grid8=False))):
        rows.append((label, dataclasses.replace(base, **kw)))
    return [(label, two_node_decode(p)) for label, p in rows]


# ---------------------------------------------------------------------------------------------------------------
# Two-node whole-expert prefill (resume-ds41f.md): UNCALIBRATED against a two-node run, none exists yet. Anchored on the single-node
# Strata runner at 8192 tokens on the RX 9070 XT (docs/DS41F_PLAN.md, 2026-10-27/28):
#   - 15.22 s warm pass with PREFILL_TIMING: attention 1.83 + output projection 1.49 + experts 9.71 (2.66 s of it GPU copy waits), 116.1 GB
#     crossing the copy stream in 9.28 s (12.5 GB/s); the rest of the pass (Engram, mixers, router) 2.2 s;
#   - 12.7 s warm pass (645 tok/s, four paired passes 12.65-13.08 s) without that profiling, same prompt and model, no breakdown.
# A layer-major pass: per layer the trunk (root only), then the FFN over all rows. Expert copies run on their own stream and hide behind the
# compute only partly: exposed wait = max(0, DMA - f x compute); f = 0.527 from the 15.22 s breakdown (DMA 9.28, compute 12.55, wait 2.66).
# The 12.7 s pass is explained either by faster compute at the same f ("compute") or by better overlap at the same compute ("overlap", f = 0.727).
TWO_NODE_PREFILL = dict(
    prompt=8192,
    dma_gb=116.1, root_dma_gbps=12.5,        # copy stream bytes and rate of the 15.22 s pass
    breakdown_s=dict(trunk=5.52, ffn=7.05),   # 1.83 + 1.49 + 2.2 other; experts 9.71 - 2.66 waits
    wait_s=2.66, target_s=12.7,
    ib_gbps=2.44,                            # resume-ds41f.md: interconnect around 2.44 GB/s (sim LINKS ib-qdr measured 3.26 on 2026-10-16)
    worker_dma_gbps=14.38,                   # measured 2026-10-29 on the B550: 5060 Ti, PCIe 4.0 x8 link (sysfs 16 GT/s x8), 7.5 MiB blobs, 8 s sustained, 14.38 GB/s flat (the earlier 7.2 GB/s assumption was wrong)
    owned_share=0.445,                       # 6834 of 15360 experts, 48 GiB (UD-IQ1_S, 7.2 MiB each)
    route_curve=(1.16, 0.545),               # share of routes on the hottest fraction p of experts: 1.16 p^0.545 (GLM-fitted tier curve; hypothesis for DS)
    q8_bytes_per_value=144 / 128,            # logical block: 16 metadata + 128 quant bytes
)


@dataclasses.dataclass
class TwoNodePrefillPlan:
    prompt: int = 8192
    calibration: str = "compute"        # compute | overlap: how the 12.7 s single-node pass is explained
    worker_share: float = -1.0          # share of expert bytes executed on the worker; -1 = DMA-balanced, capped by what it owns
    ib_gbps: float = -1.0
    worker_dma_gbps: float = -1.0
    root_dma_gbps: float = -1.0
    worker_speed: float = 1.0           # worker GPU MMQ speed relative to the root's (hypothesis: equal)
    trunk_speed: float = 1.0            # what-if: root trunk (attention, output projection, mixers) speedup
    ffn_speed: float = 1.0              # what-if: expert kernels on both GPUs
    overlap_f: float = -1.0             # -1 = the calibration's
    slabs: int = 8                      # row slabs per layer pipelined through quantize/send, worker MMQ, return
    return_bytes: float = 4.0           # bytes per hidden value in the worker's reply (FP32 partial sums; 2 = FP16)
    dma_scale: float = 1.0              # what-if: expert bytes copied per pass (e.g. a higher tier hit)

    to_dict = lambda self: dataclasses.asdict(self)


def two_node_prefill_base(plan):
    """Single-node anchor: (trunk_s, ffn_s, f) such that the model reproduces the 12.7 s pass."""
    c = TWO_NODE_PREFILL
    d0 = c["dma_gb"] / c["root_dma_gbps"]
    k_prof = c["breakdown_s"]["trunk"] + c["breakdown_s"]["ffn"]
    f_prof = (d0 - c["wait_s"]) / k_prof
    if plan.calibration == "overlap":
        k, f = k_prof, 1 - (c["target_s"] - d0) / k_prof     # (1-f) K + D = target
    else:
        f = f_prof
        k = (c["target_s"] - d0) / (1 - f)
    share = c["breakdown_s"]["trunk"] / k_prof
    return k * share, k * (1 - share), f


def two_node_prefill(plan, single=False):
    c = TWO_NODE_PREFILL
    trunk, ffn, f = two_node_prefill_base(plan)
    scale = plan.prompt / c["prompt"]
    trunk, ffn = trunk * scale / plan.trunk_speed, ffn * scale / plan.ffn_speed
    if plan.overlap_f >= 0:
        f = plan.overlap_f
    layers, hidden = 40, 5120
    rate_r = plan.root_dma_gbps if plan.root_dma_gbps > 0 else c["root_dma_gbps"]
    rate_w = plan.worker_dma_gbps if plan.worker_dma_gbps > 0 else c["worker_dma_gbps"]
    ib = plan.ib_gbps if plan.ib_gbps > 0 else c["ib_gbps"]
    dma = c["dma_gb"] * scale * plan.dma_scale
    if single:
        share = 0.0
    elif plan.worker_share >= 0:
        share = min(plan.worker_share, c["owned_share"])
    else:
        share = min(c["owned_share"], rate_w / (rate_r + rate_w))
    a, b = c["route_curve"]
    q = min(1.0, a * share ** b) if share > 0 else 0.0            # share of the routes on the worker
    d_root, d_work = dma * (1 - share) / rate_r, dma * share / rate_w
    f_root, f_work = ffn * (1 - q), ffn * q / plan.worker_speed
    t_out = plan.prompt * hidden / 128 * 144 / ib / 1e9 * 1e3      # ms per layer, Q8 blocks
    t_back = plan.prompt * hidden * plan.return_bytes / ib / 1e9 * 1e3
    n = max(1, plan.slabs)
    stage = [t_out / n, f_work / layers * 1e3 / n, t_back / n]
    chain = (sum(stage) + (n - 1) * max(stage)) * 1e-3 if q > 0 else 0.0   # s per layer: pipelined slabs
    window = trunk + max(f_root, chain * layers)                 # compute the copy stream can hide behind, waits excluded (as in the anchor)
    wait_r, wait_w = max(0.0, d_root - f * window), max(0.0, d_work - f * window)
    p = trunk + max(f_root + wait_r, (chain * layers + wait_w) if q > 0 else 0.0)
    total = p + 0.05                                           # final norm, head and logits for the last position
    bound = "worker chain" if q > 0 and chain * layers + wait_w > f_root + wait_r else ("root DMA" if wait_r > 0 else "root compute")
    return dict(tok_s=plan.prompt / total, seconds=total, bound=bound, share=share, route_share=q, trunk_s=trunk, root_ffn_s=f_root,
                worker_ffn_s=f_work, chain_s=chain * layers, root_dma_s=d_root, worker_dma_s=d_work, root_wait_s=wait_r, worker_wait_s=wait_w,
                link_out_s=t_out * layers * 1e-3, link_back_s=t_back * layers * 1e-3, overlap_f=f, meets_1200=plan.prompt / total >= 1200)


def two_node_prefill_levers(base=None):
    base = base or TwoNodePrefillPlan()
    rows = [("single node (the anchor)", None)]
    rows.append(("two nodes, DMA-balanced share", base))
    for label, kw in (("IB 3.26 GB/s (the 2026-10-16 measurement)", dict(ib_gbps=3.26)),
                      ("FP16 worker reply (changes the activation precision of the reduction)", dict(return_bytes=2.0)),
                      ("16 row slabs", dict(slabs=16)),
                      ("root PCIe at today's 9.6 GB/s sustained (miner running)", dict(root_dma_gbps=9.6)),
                      ("worker GPU 1.5x the root's MMQ speed", dict(worker_speed=1.5)),
                      ("copies fully overlapped (f = 1)", dict(overlap_f=1.0)),
                      ("trunk 1.5x faster (WMMA attention)", dict(trunk_speed=1.5)),
                      ("expert kernels 1.5x faster", dict(ffn_speed=1.5)),
                      ("overlap calibration (12.7 s explained by overlap)", dict(calibration="overlap")),
                      ("overlap calibration + trunk and experts 1.5x faster", dict(calibration="overlap", trunk_speed=1.5, ffn_speed=1.5))):
        rows.append((label, dataclasses.replace(base, **kw)))
    return [(label, two_node_prefill(p if p else base, single=p is None)) for label, p in rows]


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
        elif rec["kind"] == "runner_decode":
            predicted = runner_decode(RunnerPlan(**rec["plan"]))["tok_s"]
        elif rec["kind"] == "two_node_decode":
            predicted = two_node_decode(TwoNodePlan(**rec["plan"]))["tok_s"]
        elif rec["kind"] == "v100_decode":
            predicted = v100_decode(V100Plan(**rec["plan"]))["tok_s"]
        elif rec["kind"] == "runner_prefill":
            predicted = runner_prefill(RunnerPlan(**rec["plan"]), rec.get("prompt", 1024))["tok_s"]
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
