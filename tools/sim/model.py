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
    "EXL3": (1, 0.25153),  # 6.035 MiB per expert / 3 x 4096 x 2048 elements (bytes-only format)
}

# Effective quantized-dot throughput of one core relative to its FP32 FMA peak (MAC/s over MAC/s), for the
# expert kernels on the 1950X: Q2_K/Q3_K ~17 GMAC/s per core out of 27 peak (6 GB/s from L2, 2.76 MAC/byte);
# IQ2_XS 2.6 GB/s x 3.12 MAC/byte ~ 8 GMAC/s; IQ3_XXS 3.2 GB/s x 2.61 ~ 8.4 (codebook decode bound).
KERNEL_EFFICIENCY = {
    "Q2_K": 0.63, "Q3_K": 0.63, "Q4_K": 0.63, "Q5_K": 0.6, "Q6_K": 0.6, "Q8_0": 0.7,
    "IQ2_XS": 0.30, "IQ3_XXS": 0.31, "IQ4_XS": 0.45, "EXL3": 0.0,
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

    @property
    def moe_layers(self):
        return self.layers - self.dense_layers

    def moe_layer_ids(self):
        return list(range(self.dense_layers, self.layers))

    def expert_elements(self):
        return 3 * self.hidden * self.expert_ff

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
        n = g.hidden * g.expert_ff
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
        n = g.hidden * g.expert_ff
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
    "reap50_q23": Pack("reap50_q23", description="REAP-50 (144 experts) in Q2_K/Q3_K, all 42 layers (55.1 GB)",
                       **_uniform(("Q2_K", "Q2_K", "Q3_K"), experts=144), skew=0.8),
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


def dense_gpu_bytes_per_token(geometry=GEOMETRY):
    """Fixed-weight bytes the GPU streams for one token: dense layers, MoE layer mixers, head."""
    f = FIXED_BYTES
    kda_moe = geometry.moe_layers - len(geometry.mla_layers)
    return (geometry.dense_layers * f["dense_layer"] + kda_moe * f["kda_moe_layer"]
            + len(geometry.mla_layers) * f["mla_layer"] + f["blk11_extra"] + f["lm_head"])


def layer_fixed_bytes(layer, geometry=GEOMETRY):
    f = FIXED_BYTES
    if layer < geometry.dense_layers:
        return f["dense_layer"]
    base = f["mla_layer"] if layer in geometry.mla_layers else f["kda_moe_layer"]
    return base + (f["blk11_extra"] if layer == 11 else 0)
