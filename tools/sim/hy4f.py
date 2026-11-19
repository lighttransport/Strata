"""Hy4-Flash (GGUF arch hy_v4_flash, "512x71B" A12B) decode on TR16: Threadripper 1950X (15 worker threads, 128 GB DDR4)
+ RX 9070 XT 16 GB, the Hy4-flash-SPARK-4Bit-Fast GGUF (routed experts NVFP4, MTP block Q8_0, Engram table Q8_0).

Model: the DS4.1F runner model (ds41f.RunnerHw, measured 2026-10-21 on the same box: 12.5 tok/s) with this model's
bytes. Per token = GPU trunk (dense weights of every block + head, GEMV) + small kernels + attention + the routed
expert bytes not in the GPU tier on the CPU lane (optionally split with PCIe uploads) + host syncs/Engram.

Census: tools/sim/data/hy4f_census.json (tools/hy4f_census.py). Every constant names its measurement or is listed in
`HYPOTHESES`.
"""
import dataclasses
import json
import math
import pathlib

import routing

DATA = pathlib.Path(__file__).resolve().parent / "data"
GIB = 2 ** 30


@dataclasses.dataclass
class Geometry:
    trunk_layers: int
    moe_layers: int
    experts: int
    top_k: int
    expert_bytes: int            # gate+up+down of one routed expert (NVFP4: 3 x 1536 x 1536 x 0.5625 B)
    trunk_bytes: int             # non-expert bytes of the trunk blocks (read on the GPU every token)
    head_bytes: int
    mtp_dense_bytes: int
    mtp_expert_bytes: int
    engram_layers: tuple
    engram_rows_per_layer: int   # (ngram - 1) x heads per n-gram
    engram_row_bytes: int
    engram_table_bytes: int
    swa_layers: int
    window: int
    kv_row_bytes: int            # MLA latent 512 + rope 64, f16

    def routed_bytes_per_token(self):
        return self.moe_layers * self.top_k * self.expert_bytes

    def expert_total_bytes(self):
        return self.moe_layers * self.experts * self.expert_bytes


def load_geometry(path=DATA / "hy4f_census.json"):
    c = json.loads(pathlib.Path(path).read_text())
    m = c["meta"]
    nextn = int(m.get("nextn_predict_layers", 0))
    blocks = c["blocks"]
    trunk = blocks[:len(blocks) - nextn]
    mtp = blocks[len(blocks) - nextn:]
    moe = [b for b in trunk if b["experts"]]
    assert len({b["expert_bytes"] for b in moe}) == 1
    ple = c["globals"]["per_layer_token_embd.weight"]
    rows = (int(m["ple.ngram_size"]) - 1) * int(m["ple.heads_per_ngram"])
    swa = m["attention.sliding_window_pattern"][:len(trunk)]
    return Geometry(
        trunk_layers=len(trunk), moe_layers=len(moe), experts=int(m["expert_count"]), top_k=int(m["expert_used_count"]),
        expert_bytes=moe[0]["expert_bytes"], trunk_bytes=sum(b["dense_bytes"] for b in trunk),
        head_bytes=c["globals"]["output.weight"][1],
        mtp_dense_bytes=sum(b["dense_bytes"] for b in mtp), mtp_expert_bytes=mtp[0]["expert_bytes"] if mtp else 0,
        engram_layers=tuple(m["ple.layers"]), engram_rows_per_layer=rows,
        engram_row_bytes=int(m["embedding_length_per_layer_input"]) // 32 * 34, engram_table_bytes=ple[1],
        swa_layers=sum(bool(s) for s in swa), window=int(m["attention.sliding_window"]),
        kv_row_bytes=(int(m["attention.kv_lora_rank"]) + int(m["rope.dimension_count"])) * 2)


GEOMETRY = load_geometry()

HYPOTHESES = [
    "GPU constants (GEMV 320 GB/s, small kernels 8.5 ms, attention 0.06 ms/layer, host syncs 7.5 ms) are the DS4.1F "
    "runner's on the same 9070 XT (ds41f.RunnerHw 'now'); Hy4 has the same block shape (2 HC mixes, MLA, indexer)",
    "GPU tier NVFP4 expert kernel 300 GB/s on gfx1201 (nvfp4_decode_bench is unmeasured on HIP)",
    "tier hit: 'glm' = GLM-fitted adaptive curve 1.16 f^0.545 x 1.08; 'ds41f' = DS4.1F measured 0.61 at 7.2 % of slots "
    "scaled by f^0.545 (code text, adaptive with 2 promotions/token); Hy4 routing skew unmeasured",
    "MTP verify (width 2): trunk x1.185 (GLM width curve), expert union routing.union_ratio(2, 512, 16), CPU lane 27 GB/s "
    "for the union (measured 2-token rate), draft = MTP block (dense on GPU, its Q8_0 experts on the CPU lane) + head",
]


@dataclasses.dataclass
class Hw:
    name: str = "tr16_9070xt"
    vram_gib: float = 15.9
    desktop_gib: float = 3.0          # ds41f-tr16-decode-gate: 12-13 GB usable
    scratch_gib: float = 0.8
    gemv_gbps: float = 320.0          # DS4.1F Q6_K MMVQ rocprof 2026-10-21
    tier_gbps: float = 300.0          # hypothesis
    small_ms: float = 8.5
    attn_ms_layer: float = 0.06
    other_ms: float = 7.5
    cpu_lane_gbps: float = 35.0       # nvfp4_avx512_parity --bw 15 1 avx2, TR16 2026-11-19 (one other job running)
    cpu_lane2_gbps: float = 27.3      # same, 2 tokens
    h2d_gbps: float = 13.8            # pinned DMA, measured for DS4.1F
    layer_ms: float = 0.12            # per-layer CPU pool / sync overhead (GLM fit)
    page_us: float = 83.0             # 4 KiB QD1 random read, work NVMe
    queue_depth: int = 16
    ram_gib: float = 124.0


PROFILES = {
    "now": Hw(),
    # the DS4.1F plan's tuned trunk (GEMV at 75 % of 640 GB/s, fused small kernels, fewer host syncs)
    "target": Hw(gemv_gbps=480.0, small_ms=2.5, attn_ms_layer=0.05, other_ms=2.5),
}


@dataclasses.dataclass
class Plan:
    profile: str = "now"
    hit_model: str = "ds41f"          # ds41f | glm | fixed
    hit: float = 0.0                  # hit_model == "fixed"
    tier_gib: float = -1.0            # -1: what VRAM leaves
    split_misses: bool = False        # misses split between the CPU lane and PCIe uploads computed on the GPU
    mtp: str = ""                     # acceptance profile (routing.ACCEPTANCE) for MTP depth 1; "" = plain decode
    context: int = 4096
    engram_cache_hit: float = 0.3     # rows found in the page cache / row cache
    set: dict = dataclasses.field(default_factory=dict)

    def hw(self):
        h = dataclasses.replace(PROFILES[self.profile])
        for k, v in self.set.items():
            setattr(h, k, type(getattr(h, k))(v))
        return h


def memory(plan, g=GEOMETRY):
    h = plan.hw()
    trunk = g.trunk_bytes + g.head_bytes + g.mtp_dense_bytes * bool(plan.mtp)
    kv = g.kv_row_bytes * (g.swa_layers * min(plan.context, g.window) + (g.trunk_layers - g.swa_layers) * plan.context)
    free = (h.vram_gib - h.desktop_gib - h.scratch_gib) * GIB - trunk - kv
    tier = max(0.0, free) if plan.tier_gib < 0 else plan.tier_gib * GIB
    ram = g.expert_total_bytes() + (g.mtp_expert_bytes * g.experts if plan.mtp else 0) + 4 * GIB
    notes = []
    if tier > max(0.0, free) + 1e6:
        notes.append(f"VRAM: tier {tier / GIB:.1f} GiB > {free / GIB:.1f} free")
    if ram > h.ram_gib * GIB:
        notes.append(f"RAM: {ram / GIB:.1f} GiB of experts do not fit {h.ram_gib:.0f}")
    return dict(trunk_gib=trunk / GIB, kv_gib=kv / GIB, tier_gib=tier / GIB, slots=int(tier // g.expert_bytes),
                slot_fraction=tier / g.expert_bytes / (g.moe_layers * g.experts), ram_experts_gib=ram / GIB,
                engram_table_gib=g.engram_table_bytes / GIB, notes=notes)


def tier_hit(plan, fraction):
    if plan.hit_model == "fixed":
        return plan.hit
    if fraction <= 0:
        return 0.0
    if plan.hit_model == "glm":
        return routing.TierCurve().hit(fraction, policy="adaptive")
    return min(0.98, 0.61 * (fraction / 0.0716) ** 0.545)


def engram_ms(plan, h, g=GEOMETRY):
    """All Engram rows are hashed at token id (the n-gram is known), so the reads are issued before layer 0 and hide
    under the first layers; exposed only if they exceed the time to the first Engram layer."""
    rows = len(g.engram_layers) * g.engram_rows_per_layer * (1 - plan.engram_cache_hit)
    return math.ceil(rows / h.queue_depth) * h.page_us * 1e-3


def decode(plan, g=GEOMETRY):
    h = plan.hw()
    mem = memory(plan, g)
    hit = tier_hit(plan, mem["slot_fraction"])
    width = 2 if plan.mtp else 1
    union = routing.union_ratio(width, g.experts, g.top_k)
    wf = 1 + 0.185 * (width - 1)
    trunk = (g.trunk_bytes + g.head_bytes) * wf / (h.gemv_gbps * 1e6) + h.small_ms + h.attn_ms_layer * g.trunk_layers * width
    routed = g.routed_bytes_per_token() * union
    miss = routed * (1 - hit)
    lane = h.cpu_lane_gbps if width == 1 else h.cpu_lane2_gbps
    rate = lane + h.h2d_gbps if plan.split_misses else lane
    experts = miss / (rate * 1e6) + h.layer_ms * g.moe_layers
    tier = routed * hit / (h.tier_gbps * 1e6)   # overlaps the CPU lane
    experts = max(experts, tier)
    eng = engram_ms(plan, h, g)
    eng_exposed = max(0.0, eng - trunk / g.trunk_layers * g.engram_layers[0])
    step = trunk + experts + h.other_ms + eng_exposed
    draft = 0.0
    tokens = 1.0
    if plan.mtp:
        draft = (g.mtp_dense_bytes + g.head_bytes) / (h.gemv_gbps * 1e6) + h.small_ms / g.trunk_layers * 2 \
            + g.top_k * g.mtp_expert_bytes / (h.cpu_lane_gbps * 1e6) + 1.0
        tokens = routing.tokens_per_round(1, plan.mtp)
    round_ms = step + draft
    return dict(tok_s=tokens / round_ms * 1e3, round_ms=round_ms, trunk_ms=trunk, experts_ms=experts, tier_ms=tier,
                other_ms=h.other_ms, engram_ms=eng, engram_exposed_ms=eng_exposed, draft_ms=draft, tokens_per_round=tokens,
                hit=hit, routed_gb=routed / 1e9, miss_gb=miss / 1e9, memory=mem)


def table(contexts=(4096,)):
    """The forecast grid: profile x hit model x misses x MTP."""
    rows = []
    for profile in ("now", "target"):
        for hit_model in ("glm", "ds41f"):
            for split in (False, True):
                for mtp in ("", "coding", "mixed"):
                    for ctx in contexts:
                        p = Plan(profile=profile, hit_model=hit_model, split_misses=split, mtp=mtp, context=ctx)
                        r = decode(p)
                        rows.append(dict(profile=profile, hit_model=hit_model, split=split, mtp=mtp or "-", context=ctx,
                                         hit=r["hit"], tok_s=r["tok_s"], round_ms=r["round_ms"], trunk_ms=r["trunk_ms"],
                                         experts_ms=r["experts_ms"], draft_ms=r["draft_ms"]))
    return rows
