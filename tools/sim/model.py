"""GLM-5.3-Flash geometry and the byte sizes of its weight packs.

Numbers come from the GGUF headers (parsed once with tools/gguf_reader.py) and docs/GLM53_FLASH_PERFORMANCE.md,
docs/GLM_Q2_DECODE_REDESIGN.md and docs/GLM_DUAL_128G_64G.md. `from_gguf()` can replace the tables with exact
per-layer sizes when the model files are present, but nothing here needs hardware or the files.
"""
import dataclasses
import json
import pathlib
import re
import sys

TOOLS = pathlib.Path(__file__).resolve().parents[1]

# ggml block geometry: elements per block, bytes per block
BLOCK_BYTES = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q8_0": (32, 34), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "IQ2_XS": (256, 74), "IQ3_XXS": (256, 98), "IQ4_XS": (256, 136),
    "IQ1_S": (256, 50), "IQ1_M": (256, 56), "IQ2_XXS": (256, 66), "IQ2_S": (256, 82),
    "EXL3": (1, 0.25153),  # 6.035 MiB per expert / 3 x 4096 x 2048 elements (bytes-only format)
}

# Effective quantized-dot throughput of one core relative to its FP32 FMA peak (MAC/s over MAC/s), for the
# expert kernels on the 1950X: Q2_K/Q3_K ~17 GMAC/s per core out of 27 peak (6 GB/s from L2, 2.76 MAC/byte);
# IQ2_XS 2.6 GB/s x 3.12 MAC/byte ~ 8 GMAC/s; IQ3_XXS 3.2 GB/s x 2.61 ~ 8.4 (codebook decode bound).
KERNEL_EFFICIENCY = {
    "Q2_K": 0.63, "Q3_K": 0.63, "Q4_K": 0.63, "Q5_K": 0.6, "Q6_K": 0.6, "Q8_0": 0.7,
    "IQ2_XS": 0.30, "IQ3_XXS": 0.31, "IQ4_XS": 0.45, "EXL3": 0.0,
    # IQ1_S: ggml's AVX2 kernel without the BMI2 pdep path, 89 GMAC/s on 16 Zen 2 cores (B550, 2026-10-16) of a
    # 16 x 3.5 x 8 = 448 GMAC/s peak (tools/sim/muse.py); with pdep (microcoded on Zen 1/2) it is 6.8 GMAC/s.
    "IQ1_S": 0.20, "IQ1_M": 0.14, "IQ2_XXS": 0.15,
}


def tensor_bytes(elements, fmt):
    block, nbytes = BLOCK_BYTES[fmt]
    return int(elements / block * nbytes)


@dataclasses.dataclass(frozen=True)
class Geometry:
    layers: int = 45                 # main blocks 0..44; block 45 is the MTP draft
    dense_layers: int = 3            # blocks 0..2 have a dense FFN
    hidden: int = 4096
    dense_ff: int = 12288
    expert_ff: int = 2048
    experts: int = 288
    top_k: int = 8
    shared_experts: int = 1
    vocab: int = 154880
    mla_layers: tuple = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)
    mtp_layers: int = 1
    expert_hidden: int = 0           # input/output width of a routed expert; 0 = hidden (Muse Spark: 4096 latent)

    @property
    def expert_in(self):
        return self.expert_hidden or self.hidden

    @property
    def moe_layers(self):
        return self.layers - self.dense_layers

    def moe_layer_ids(self):
        return list(range(self.dense_layers, self.layers))

    def expert_elements(self):
        return 3 * self.expert_in * self.expert_ff

    def expert_macs(self):
        """Multiply-adds of one expert on one token."""
        return self.expert_elements()

    def routed_macs_per_token(self):
        return self.top_k * self.moe_layers * self.expert_macs()


GEOMETRY = Geometry()

# Fixed (non routed-expert) weights of the original UD-Q2_K_XL file, bytes. The GPU reads about 6.45 GB of them
# per decoded token (token_embd rows are read on the CPU only). Per-layer values from the GGUF header.
FIXED_BYTES = dict(
    total=6_888_746_232, q5k=3_190_000_000, q6k=2_350_000_000, q8_0=770_000_000, output_q4k=357_000_000,
    f32=220_000_000, token_embd=436_000_000,
    dense_layer=216_600_000, kda_moe_layer=129_200_000, mla_layer=129_600_000, blk11_extra=14_000_000,
    shared_expert=18_415_616,      # Q5_K gate/up (2 x 5,767,168) + Q6_K down (6,881,280), inside the layer figures
    router=4_718_592,              # F32 4096 x 288
    mtp_dense=164_458_624,
    lm_head=357_000_000,
)

# VRAM accounting constants in MiB (docs/GLM_Q2_DECODE_REDESIGN.md, GPU_LIVE lines)
# GPU copies of the fixed weights: VRAM in MiB and the per-token read bytes relative to the original Q5_K/Q6_K
# (Q8_0 copies of Q5_K/Q6_K tensors cost 551 MiB less VRAM in the engine; Q4_K copies would shrink the Q5_K/Q6_K
# 5.54 GB to 4.22 GB).
DENSE_FORMATS = dict(
    q8=dict(vram_mib=6154, bytes_scale=1.0, lossless=True),
    orig=dict(vram_mib=6705, bytes_scale=1.0, lossless=True),
    q4=dict(vram_mib=5312, bytes_scale=0.86, lossless=False),
)

VRAM = dict(
    dense_q8=6154, dense=6705, mtp_dense_q8=157, mtp_dense=221, draft_bank=2502,
    state_fixed=145.6, state_per_token=0.0228, verify_slot=145.6, staging_slot=208, staging_slots=2,
    prefetch_group=146, misc=635,
    exl3_dense=6694,
)


@dataclasses.dataclass
class Pack:
    """Routed-expert pack: per-layer (gate, up, down) formats; `experts` per layer."""
    name: str
    experts: int
    formats: dict                   # layer -> (gate_fmt, up_fmt, down_fmt)
    description: str = ""
    dense_vram_mib: float = VRAM["dense"]
    skew: float = 1.0               # scales the tier hit curve (fewer, broader experts hit less)
    geometry: Geometry = GEOMETRY

    def expert_bytes(self, layer):
        g = self.geometry
        gate, up, down = self.formats[layer]
        n = g.expert_in * g.expert_ff
        return tensor_bytes(n, gate) + tensor_bytes(n, up) + tensor_bytes(n, down)

    def mean_expert_bytes(self):
        ids = self.geometry.moe_layer_ids()
        return sum(self.expert_bytes(l) for l in ids) / len(ids)

    def routed_total_bytes(self):
        return sum(self.expert_bytes(l) for l in self.geometry.moe_layer_ids()) * self.experts

    def routed_bytes_per_token(self):
        """Bytes of the eight routed experts of every MoE layer with no reuse (3.09 GB for q23)."""
        return sum(self.expert_bytes(l) for l in self.geometry.moe_layer_ids()) * self.geometry.top_k

    def slots(self):
        return self.experts * self.geometry.moe_layers

    def kernel_efficiency(self, layer):
        """Weighted by bytes: the compute efficiency of the mix of formats in this layer's expert."""
        g = self.geometry
        n = g.expert_in * g.expert_ff
        parts = [(tensor_bytes(n, f), KERNEL_EFFICIENCY[f]) for f in self.formats[layer]]
        total = sum(b for b, _ in parts)
        return sum(b * e for b, e in parts) / total



def _uniform(fmt3, experts=288, exceptions=None, **kw):
    formats = {l: fmt3 for l in GEOMETRY.moe_layer_ids()}
    formats.update(exceptions or {})
    return dict(experts=experts, formats=formats, **kw)


PACKS = {
    "q2_orig": Pack("q2_orig", description="UD-Q2_K_XL as shipped: IQ2_XS gate/up + IQ3_XXS down (99.0 GB)",
                    **_uniform(("IQ2_XS", "IQ2_XS", "IQ3_XXS"), exceptions={
                        11: ("IQ3_XXS", "IQ3_XXS", "IQ4_XS"), 12: ("IQ2_XS", "IQ2_XS", "IQ4_XS"),
                        44: ("IQ2_XS", "IQ2_XS", "IQ4_XS")})),
    "q23": Pack("q23", description="experts-q23.gguf: Q2_K gate/up + Q3_K down, blk.11/12/44 kept (111.2 GB)",
                **_uniform(("Q2_K", "Q2_K", "Q3_K"), exceptions={
                    11: ("IQ3_XXS", "IQ3_XXS", "IQ4_XS"), 12: ("Q2_K", "Q2_K", "IQ4_XS"),
                    44: ("Q2_K", "Q2_K", "IQ4_XS")})),
    "q22": Pack("q22", description="experts-q22.gguf: Q2_K gate/up/down, exceptions as q23 (untimed, ppl 4.867)",
                **_uniform(("Q2_K", "Q2_K", "Q2_K"), exceptions={
                    11: ("IQ3_XXS", "IQ3_XXS", "IQ4_XS"), 12: ("Q2_K", "Q2_K", "IQ4_XS"),
                    44: ("Q2_K", "Q2_K", "IQ4_XS")})),
    "reap224_q23": Pack("reap224_q23", description="Strata REAP-style prune to 224 experts, Q23 (95.5 GB assembled)",
                        **_uniform(("Q2_K", "Q2_K", "Q3_K"), experts=224, exceptions={
                            11: ("IQ3_XXS", "IQ3_XXS", "IQ4_XS"), 12: ("Q2_K", "Q2_K", "IQ4_XS"),
                            44: ("Q2_K", "Q2_K", "IQ4_XS")}), skew=0.8),
    "reap224_q22": Pack("reap224_q22", description="Strata REAP-style prune to 224 experts, Q22 (88.1 GB assembled)",
                        **_uniform(("Q2_K", "Q2_K", "Q2_K"), experts=224, exceptions={
                            11: ("IQ3_XXS", "IQ3_XXS", "IQ4_XS"), 12: ("Q2_K", "Q2_K", "IQ4_XS"),
                            44: ("Q2_K", "Q2_K", "IQ4_XS")}), skew=0.8),
    "reap50_q23": Pack("reap50_q23", description="REAP-50 (144 experts) in Q2_K/Q3_K, all 42 layers (55.1 GB)",
                       **_uniform(("Q2_K", "Q2_K", "Q3_K"), experts=144), skew=0.8),
    "reap50_q22": Pack("reap50_q22", description="REAP-50 Q23 assembly with Q2_K down sidecar",
                       **_uniform(("Q2_K", "Q2_K", "Q2_K"), experts=144), skew=0.8),
    "reap50_q4km": Pack("reap50_q4km", description="REAP-50 Q4_K_M as downloaded (~15.1 MB per expert)",
                        **_uniform(("Q4_K", "Q4_K", "Q6_K"), experts=144), skew=0.8),
    "q8": Pack("q8", description="hypothetical Q8_0 experts (26.7 MB each, 323 GB)",
               **_uniform(("Q8_0", "Q8_0", "Q8_0"))),
    "exl3": Pack("exl3", description="EXL3 3-bit experts, 6.035 MiB each (bytes only; GPU-side format)",
                 **_uniform(("EXL3", "EXL3", "EXL3")), dense_vram_mib=VRAM["exl3_dense"]),
}


def from_census(path):
    data = json.loads(pathlib.Path(path).read_text())
    formats = {int(l): tuple(row[p]["format"] for p in ("gate", "up", "down"))
               for l, row in data["layers"].items()}
    result = Pack(data["name"], data["experts"], formats, description="GGUF census: " + data["source"],
                  geometry=dataclasses.replace(GEOMETRY, experts=data["experts"]), skew=0.8)
    if result.routed_total_bytes() != data["bytes"]["main_routed"]:
        raise ValueError("census does not match supported tensor block geometry")
    return result


for _path in sorted((pathlib.Path(__file__).parent / "data").glob("census-*.json")):
    if _path.name.startswith("census-muse-"):
        continue                    # Muse Spark census (strata.muse.census.v1), loaded below
    _pack = from_census(_path)
    PACKS[_pack.name] = _pack


def pack(name):
    if name not in PACKS:
        raise ValueError(f"unknown pack {name}; choose from {', '.join(PACKS)}")
    return PACKS[name]


def from_gguf(paths, name="gguf"):
    """Exact per-layer expert formats from GGUF files (model shards and/or an expert pack)."""
    sys.path.insert(0, str(TOOLS))
    from gguf_reader import GGUFFile  # noqa: E402
    formats = {}
    experts = GEOMETRY.experts
    for path in paths:
        for tensor in GGUFFile(pathlib.Path(path)).tensors:
            m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
            if not m:
                continue
            layer, which = int(m.group(1)), m.group(2)
            if layer >= GEOMETRY.layers:
                continue
            experts = tensor.shape[-1] if len(tensor.shape) == 3 else experts
            formats.setdefault(layer, {})[which] = tensor.type_name
    table = {l: (f["gate"], f["up"], f["down"]) for l, f in formats.items() if len(f) == 3}
    missing = [l for l in GEOMETRY.moe_layer_ids() if l not in table]
    if missing:
        raise ValueError(f"no expert tensors for layers {missing}")
    return Pack(name, experts=experts, formats=table, description=f"from {', '.join(map(str, paths))}")


def dense_gpu_bytes_per_token(geometry=None):
    """Fixed-weight bytes the GPU streams for one token: dense layers, MoE layer mixers, head."""
    geometry = geometry or GEOMETRY
    f = FIXED_BYTES
    kda_moe = geometry.moe_layers - len(geometry.mla_layers)
    return (geometry.dense_layers * f["dense_layer"] + kda_moe * f["kda_moe_layer"]
            + len(geometry.mla_layers) * f["mla_layer"] + f["blk11_extra"] + f["lm_head"])


def layer_fixed_bytes(layer, geometry=None):
    geometry = geometry or GEOMETRY
    f = FIXED_BYTES
    if layer < geometry.dense_layers:
        return f["dense_layer"]
    base = f["mla_layer"] if layer in geometry.mla_layers else f["kda_moe_layer"]
    return base + (f["blk11_extra"] if layer == 11 else 0)


# ---------------------------------------------------------------------------------------------------------------
# GLM-5.3 (non-Flash, 743B): `glm_moe_dsa` = DeepSeek-V3.2-style MLA with a lightning-indexer sparse attention (DSA)
# on 21 of 78 layers (the others reuse the previous top-k), plain residual (no mHC), no KDA. From
# huggingface.co/zai-org/GLM-5.3 config.json: 78 layers + 1 MTP layer, first 3 dense (ff 12288), hidden 6144,
# 256 routed experts of 2048 rows, top-8, 1 shared expert, vocab 154880, q_lora 2048, kv_lora 512, 64 heads with
# qk 256 (192 nope + 64 rope) and v 256. Routed experts total 724.8B parameters (37.75M each).
# Fixed-weight bytes are estimates (no GGUF header was parsed): attention 165.0M params per layer, shared expert
# 37.75M, router 256 x 6144 f32, indexer 9.4M on the 21 full layers, at an assumed 5.5 bits per weight for the
# Unsloth UD mixes ("orig"), 8.5 for Q8_0 and 4.5 for Q4_K GPU copies.
GLM53_GEOMETRY = Geometry(layers=78, dense_layers=3, hidden=6144, dense_ff=12288, expert_ff=2048, experts=256,
                          top_k=8, shared_experts=1, vocab=154880, mla_layers=tuple(range(3, 78)), mtp_layers=1)
_B = 5.5 / 8
GLM53_FIXED_BYTES = dict(
    total=int((3 * (165.0e6 + 226.5e6) + 75 * (165.0e6 + 37.75e6) + 21 * 9.38e6) * _B + 75 * 6.29e6 + 2 * 951.6e6 * 6.5625 / 8),
    token_embd=int(951.6e6 * 6.5625 / 8),
    dense_layer=int((165.0e6 + 226.5e6) * _B),
    kda_moe_layer=int((165.0e6 + 37.75e6 + 21 / 75 * 9.38e6) * _B + 6.29e6),   # no KDA layers; kept for the table shape
    mla_layer=int((165.0e6 + 37.75e6 + 21 / 75 * 9.38e6) * _B + 6.29e6),
    blk11_extra=0,
    shared_expert=int(37.75e6 * _B),
    router=6_291_456,
    mtp_dense=int((165.0e6 + 75.5e6 + 37.75e6) * _B),
    lm_head=int(951.6e6 * 6.5625 / 8),
)
_dense_mib = (3 * GLM53_FIXED_BYTES["dense_layer"] + 75 * GLM53_FIXED_BYTES["mla_layer"] + GLM53_FIXED_BYTES["lm_head"]) / 2 ** 20
GLM53_DENSE_FORMATS = dict(
    q8=dict(vram_mib=_dense_mib * 8.5 / 5.5, bytes_scale=8.5 / 5.5, lossless=True),
    orig=dict(vram_mib=_dense_mib, bytes_scale=1.0, lossless=True),
    q4=dict(vram_mib=_dense_mib * 4.5 / 5.5, bytes_scale=4.5 / 5.5, lossless=False),
)
GLM53_VRAM = dict(
    dense_q8=GLM53_DENSE_FORMATS["q8"]["vram_mib"], dense=_dense_mib,
    mtp_dense_q8=GLM53_FIXED_BYTES["mtp_dense"] * 8.5 / 5.5 / 2 ** 20, mtp_dense=GLM53_FIXED_BYTES["mtp_dense"] / 2 ** 20,
    draft_bank=3600, state_fixed=64.0,
    state_per_token=(576 * 2 * 78 + 21 * 128 * 2) / 2 ** 20,      # MLA latent cache + indexer keys, bf16
    verify_slot=16.0, staging_slot=219, staging_slots=2, prefetch_group=150, misc=635, exl3_dense=_dense_mib,
)


def _glm53(name, fmt3, experts, description, skew=1.0):
    formats = {l: fmt3 for l in GLM53_GEOMETRY.moe_layer_ids()}
    return Pack(name, experts=experts, formats=formats, description=description, dense_vram_mib=_dense_mib, skew=skew,
                geometry=dataclasses.replace(GLM53_GEOMETRY, experts=experts))


GLM53_PACKS = {}
for _keep, _tag in ((256, ""), (192, "_reap25"), (160, "_reap37"), (128, "_reap50")):
    _skew = 1.0 if _keep == 256 else 0.8
    for _suffix, _fmt, _what in (("q23", ("Q2_K", "Q2_K", "Q3_K"), "Q2_K gate/up + Q3_K down"),
                                  ("q22", ("Q2_K", "Q2_K", "Q2_K"), "Q2_K gate/up/down"),
                                  ("iq2", ("IQ2_XS", "IQ2_XS", "IQ3_XXS"), "IQ2_XS gate/up + IQ3_XXS down (UD-Q2_K_XL-like)"),
                                  ("q4", ("Q4_K", "Q4_K", "Q4_K"), "Q4_K")):
        _name = f"glm53{_tag}_{_suffix}"
        GLM53_PACKS[_name] = _glm53(_name, _fmt, _keep, f"GLM-5.3 {_keep} experts/layer, {_what}", _skew)

MODELS = dict(
    flash=dict(geometry=GEOMETRY, fixed=dict(FIXED_BYTES), vram=dict(VRAM), dense=dict(DENSE_FORMATS)),
    glm53=dict(geometry=GLM53_GEOMETRY, fixed=GLM53_FIXED_BYTES, vram=GLM53_VRAM, dense=GLM53_DENSE_FORMATS, packs=GLM53_PACKS),
)
# ---------------------------------------------------------------------------------------------------------------
# MiMo-V2.6 (`mimo_v2`, MIT): grouped-query attention with 128-token sliding windows on most layers (attention-sink
# bias), sigmoid routing (noaux_tc), no shared expert, 1 dense layer; native MXFP4 experts in the release.
# From huggingface.co/XiaomiMiMo/MiMo-V2.6-{Pro,Flash}-RL config.json (vision/audio encoders not counted):
#   Pro:   70 layers (69 MoE), hidden 6144, 384 experts x 2048 rows, top-8, 128 q / 8 kv heads, qk 192, v 128,
#          10 full-attention layers; 1,000B routed (37.75M per expert), 1.02T total, 42B active; no MTP in the config.
#   Flash: 48 layers (47 MoE), hidden 4096, 256 experts x 2048, top-8, 64 q / 4 kv heads (8 on SWA layers), 9 full
#          layers, 3 MTP layers; 302.8B routed (25.17M per expert), 309B total, 15B active.
# Fixed weights are estimated at 5.5 bits per weight ("orig", a Q8_0/Q6_K-mix GGUF trunk would be higher).

def _attention_params(hidden, q_heads, kv_heads, qk, v):
    return hidden * q_heads * qk + hidden * kv_heads * (qk + v) + q_heads * v * hidden


def _register(key, geometry, attn, dense_ffn, mtp_params, kv_bytes_per_token, prefix, keeps, description):
    lm = geometry.vocab * geometry.hidden
    router = geometry.experts * geometry.hidden * 4
    moe_layer = int(attn * _B + router)
    fixed = dict(total=int((geometry.dense_layers * (attn + dense_ffn) + geometry.moe_layers * attn) * _B + 2 * lm * 6.5625 / 8),
                 token_embd=int(lm * 6.5625 / 8), dense_layer=int((attn + dense_ffn) * _B), kda_moe_layer=moe_layer,
                 mla_layer=moe_layer, blk11_extra=0, shared_expert=0, router=router, mtp_dense=int(mtp_params * _B),
                 lm_head=int(lm * 6.5625 / 8))
    dense_mib = (geometry.dense_layers * fixed["dense_layer"] + geometry.moe_layers * moe_layer + fixed["lm_head"]) / 2 ** 20
    dense = dict(q8=dict(vram_mib=dense_mib * 8.5 / 5.5, bytes_scale=8.5 / 5.5, lossless=True),
                 orig=dict(vram_mib=dense_mib, bytes_scale=1.0, lossless=True),
                 q4=dict(vram_mib=dense_mib * 4.5 / 5.5, bytes_scale=4.5 / 5.5, lossless=False))
    vram = dict(dense_q8=dense["q8"]["vram_mib"], dense=dense_mib, mtp_dense_q8=fixed["mtp_dense"] * 8.5 / 5.5 / 2 ** 20,
                mtp_dense=fixed["mtp_dense"] / 2 ** 20, draft_bank=2502, state_fixed=64.0, state_per_token=kv_bytes_per_token / 2 ** 20,
                verify_slot=16.0, staging_slot=219, staging_slots=2, prefetch_group=150, misc=635, exl3_dense=dense_mib)
    packs = {}
    for keep, tag in keeps:
        for suffix, fmt, what in (("q23", ("Q2_K", "Q2_K", "Q3_K"), "Q2_K gate/up + Q3_K down"),
                                  ("q22", ("Q2_K", "Q2_K", "Q2_K"), "Q2_K gate/up/down"),
                                  ("iq2", ("IQ2_XS", "IQ2_XS", "IQ3_XXS"), "IQ2_XS gate/up + IQ3_XXS down"),
                                  ("q4", ("Q4_K", "Q4_K", "Q4_K"), "Q4_K (about the native MXFP4 size)")):
            name = f"{prefix}{tag}_{suffix}"
            g = dataclasses.replace(geometry, experts=keep)
            packs[name] = Pack(name, experts=keep, formats={l: fmt for l in g.moe_layer_ids()},
                               description=f"{description} {keep} experts/layer, {what}", dense_vram_mib=dense_mib,
                               skew=1.0 if keep == geometry.experts else 0.8, geometry=g)
    MODELS[key] = dict(geometry=geometry, fixed=fixed, vram=vram, dense=dense, packs=packs)
    return packs


MIMO_PRO_GEOMETRY = Geometry(layers=70, dense_layers=1, hidden=6144, dense_ff=16384, expert_ff=2048, experts=384, top_k=8,
                             shared_experts=0, vocab=152576, mla_layers=tuple(range(1, 70)), mtp_layers=0)
MIMO_FLASH_GEOMETRY = Geometry(layers=48, dense_layers=1, hidden=4096, dense_ff=16384, expert_ff=2048, experts=256, top_k=8,
                               shared_experts=0, vocab=152576, mla_layers=tuple(range(1, 48)), mtp_layers=3)
MIMO_PACKS = {}
MIMO_PACKS.update(_register("mimo_pro", MIMO_PRO_GEOMETRY, _attention_params(6144, 128, 8, 192, 128), 3 * 6144 * 16384,
                            _attention_params(6144, 128, 8, 192, 128) + 2 * 6144 * 6144,
                            # KV per token: 10 full layers x 8 kv heads x (192 + 128) x bf16; SWA layers hold 128 tokens
                            10 * 8 * 320 * 2, "mimop", ((384, ""), (288, "_reap25"), (192, "_reap50"), (144, "_reap62")),
                            "MiMo-V2.6-Pro"))
MIMO_PACKS.update(_register("mimo_flash", MIMO_FLASH_GEOMETRY, _attention_params(4096, 64, 4, 192, 128), 3 * 4096 * 16384,
                            3 * (_attention_params(4096, 64, 4, 192, 128) + 2 * 4096 * 4096),
                            9 * 4 * 320 * 2, "mimof", ((256, ""), (192, "_reap25"), (128, "_reap50")), "MiMo-V2.6-Flash"))

# ---------------------------------------------------------------------------------------------------------------
# Muse Spark Open 816B-A42B (`muse-spark`, Meta): 62 layers, all MoE, no dense layer, no shared expert, no MTP head.
# Gated GQA (128 q / 16 kv heads of 64, sigmoid output gate, QK-norm), 2048-token sliding window on 46 layers (full
# attention on 1, 5, ..., 61). The MoE runs in a 4096-wide latent: ffn_latent_down 8192 -> 4096, 256 experts of
# 4096 -> 4096 -> 4096 (top-8, sigmoid router on the 8192 input), ffn_latent_up 4096 -> 8192. Sizes come from the
# GGUF census (tools/muse_gguf_census.py, data/census-muse-spark-*.json); `mla_layers` lists the full-attention
# layers so the generic tables have a long-context set. The two-node model is tools/sim/muse.py.
MUSE_GEOMETRY = Geometry(layers=62, dense_layers=0, hidden=8192, dense_ff=16384, expert_ff=4096, experts=256, top_k=8,
                         shared_experts=0, vocab=202048, mla_layers=tuple(range(1, 62, 4)), mtp_layers=0,
                         expert_hidden=4096)


def muse_census(path):
    """(pack, fixed-weight table, VRAM table, dense formats) from a strata.muse.census.v1 file."""
    data = json.loads(pathlib.Path(path).read_text())
    k = data["kinds"]
    layers = MUSE_GEOMETRY.layers
    expert_total = sum(v["bytes"] for name, v in k.items() if "_exps" in name)
    per_layer = sum(v["bytes"] for name, v in k.items() if v["count"] == layers and "_exps" not in name)
    lm = k["output.weight"]["bytes"]
    fixed = dict(total=data["bytes"] - expert_total, token_embd=k["token_embd.weight"]["bytes"], dense_layer=0,
                 kda_moe_layer=per_layer // layers, mla_layer=per_layer // layers, blk11_extra=0, shared_expert=0,
                 router=k["ffn_gate_inp.weight"]["bytes"] // layers, mtp_dense=0, lm_head=lm)
    dense_mib = (per_layer + lm) / 2 ** 20
    # q4: GPU copies of the Q6_K/Q8_0 trunk re-quantized to Q4_K (lossy), bytes from the census mix
    q4 = sum(v["elements"] * 4.5 / 8 for name, v in k.items()
             if name != "token_embd.weight" and "_exps" not in name and v["formats"].keys() - {"F32"})
    q4_scale = q4 / sum(v["bytes"] for name, v in k.items()
                        if name != "token_embd.weight" and "_exps" not in name and v["formats"].keys() - {"F32"})
    dense = dict(orig=dict(vram_mib=dense_mib, bytes_scale=1.0, lossless=True),
                 q8=dict(vram_mib=dense_mib, bytes_scale=1.0, lossless=True),
                 q4=dict(vram_mib=dense_mib * q4_scale, bytes_scale=q4_scale, lossless=False))
    full = len(data["full_attention_layers"])
    kv_per_token = data["hyperparameters"]["attention.head_count_kv"] * 64 * 2 * 2 * full     # f16 K and V
    vram = dict(dense_q8=dense_mib, dense=dense_mib, mtp_dense_q8=0.0, mtp_dense=0.0, draft_bank=0,
                state_fixed=(layers - full) * data["hyperparameters"]["attention.sliding_window"] * 4096 / 2 ** 20,
                state_per_token=kv_per_token / 2 ** 20, verify_slot=16.0, staging_slot=219, staging_slots=2,
                prefetch_group=150, misc=635, exl3_dense=dense_mib)
    formats = {int(l): tuple(f) for l, f in data["expert_formats"].items()}
    result = Pack(f"muse_{data['name'].replace('-', '_')}", experts=MUSE_GEOMETRY.experts, formats=formats,
                  description=f"Muse Spark Open {data['name']} ({data['bytes'] / 1e9:.1f} GB), GGUF census",
                  dense_vram_mib=dense_mib, geometry=MUSE_GEOMETRY)
    if result.routed_total_bytes() != expert_total:
        raise ValueError("Muse census does not match the expert block geometry")
    return result, fixed, vram, dense


MUSE_PACKS = {}
for _path in sorted((pathlib.Path(__file__).parent / "data").glob("census-muse-spark-*.json")):
    _pack, _fixed, _vram, _dense = muse_census(_path)
    MUSE_PACKS[_pack.name] = _pack
if MUSE_PACKS:
    MODELS["muse"] = dict(geometry=MUSE_GEOMETRY, fixed=_fixed, vram=_vram, dense=_dense, packs=MUSE_PACKS)


ACTIVE_MODEL = "flash"


def use_model(name):
    """Switch every simulator module to `name`'s geometry and fixed-weight tables (packs carry their own geometry)."""
    global GEOMETRY, ACTIVE_MODEL
    spec = MODELS[name]
    old = GEOMETRY
    for module in list(sys.modules.values()):
        if getattr(module, "GEOMETRY", None) is old:
            module.GEOMETRY = spec["geometry"]
    for table, new in ((FIXED_BYTES, spec["fixed"]), (VRAM, spec["vram"]), (DENSE_FORMATS, spec["dense"])):
        table.clear(); table.update(new)
    PACKS.update(spec.get("packs", {}))
    ACTIVE_MODEL = name
    return spec["geometry"]


def packs_for(name):
    return [k for k, v in PACKS.items() if (v.geometry.hidden == 6144) == (name == "glm53")] if name == "glm53" else \
        [k for k in PACKS if not k.startswith("glm53")]
