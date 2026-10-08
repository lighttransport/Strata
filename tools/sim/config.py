"""Run configuration: the engine knobs the simulator understands (names follow configs/glm53f-*.json)."""
import dataclasses


@dataclasses.dataclass
class RunConfig:
    pack: str = "q23"
    threads: int = 0                   # expert workers; 0 = physical cores minus one (host thread)
    context: int = 4096
    prompt: int = 1024
    generate: int = 512
    speculation: str = "mtp"           # none | mtp | dflash (block-diffusion drafter) | selfspec (DraftExpert: target
                                       # attention + one GPU draft expert per layer, sequential drafts)
    mtp_depth: int = 2                 # mtp: drafts per round; 0 = ordinary greedy decode
    acceptance: str = "mixed"          # mtp: routing.ACCEPTANCE profile
    draft_block: int = 8               # dflash: verify width per round (block of 7 drafts + anchor), DFlash2 default 8
    draft_acceptance: str = "dflash_code"  # dflash: routing.ACCEPTANCE profile (dflash_chat / dflash_code / dflash_math)
    draft_model_mib: float = 1536      # dflash: drafter weights resident on the GPU (read once per round)
    draft_layers: int = 5              # dflash: drafter depth (launch overhead per forward)
    max_verify_width: int = 8          # engine cap on tokens per step (cpu::MAXT = 8); raise to explore
    expert_skip: float = 0.0           # share of routed expert bytes skipped by gate thresholds (lossy)
    pcie_share: float = 0.0            # share of each layer's non-resident expert bytes the GPU streams over PCIe
                                       # and computes itself, concurrently with the CPU pass (architecture option)
    pcie_prefetch: bool = False        # predict next layer's routes (cross-layer gate) and upload those experts
                                       # during this layer's CPU pass; the GPU computes them (lossless, FATE/SpecPrefetch)
    expert_deferral: bool = False      # add part of a layer's routed output one layer late so the GPU chain can run
                                       # ahead of the CPU (KTransformers expert deferral; lossy approximation)
    deferral_share: float = 0.3        # share of each layer's routed work that may be deferred (quality cost grows with it)
    adaptive_window: bool = False      # stop drafting at low confidence (STRATA_GLM_MTP_CONTINUATION_MARGIN / DraftExpert
                                       # truncation): fewer rejected positions are verified; lossless
    spec_tail_topk: int = 8            # experts per route for draft positions >= 2 in a verify window (AcceptMoE
                                       # style verifier sizing; lossy below 8)
    cost_aware_drafts: bool = False    # EcoSpec: prefer draft tokens whose experts the window already uses (lossless)
    draft_prefetch: bool = False       # drafts' routes known a round ahead: RAM-tier disk fetches overlap the CPU pass
    cold_share: float = 0.0            # share of experts (coldest by prior) stored in a smaller format (lossy)
    cold_scale: float = 0.6            # bytes of a cold expert relative to the pack's format (0.6: ~1.6 bpw)
    tail_affinity: float = 0.0         # route affinity applied only to draft positions >= 2 (our variant: the
                                       # routing bias lands on the tokens most likely to be rejected anyway)
    tier_compress: float = 1.0         # GPU tier stores experts in a denser format: slots x this (lossy above 1)
    dense_format: str = "q8"           # GPU copies of the fixed weights: q8 (default) | orig (Q5_K/Q6_K) | q4 (lossy)
    split_verify: bool = True          # STRATA_GLM_SPLIT_VERIFY (needs mtp_depth >= 1)
    skip_anchor: bool = True           # STRATA_GLM_MTP_SKIP_ANCHOR
    decode_cache_mib: float = -1       # GPU expert tier; -1 = fill what the budget leaves, 0 = none
    tier_policy: str = "adaptive"      # static | static_prior | adaptive (STRATA_GLM_TIER_ADAPT) | lru (recency)
    affinity: float = 0.0              # STRATA_GLM_ROUTE_AFFINITY (lossy above 0)
    gpu_draft_experts: bool = False    # draft experts on the GPU (forced to CPU when a tier exists)
    gpu_budget_mib: float = 0          # --gpu-budget-mib; 0 = usable VRAM minus desktop
    reserve_mib: float = 512           # STRATA_GLM_GPU_RESERVE_MIB
    prefill_scratch_mib: float = 1024  # STRATA_GLM_PREFILL_SCRATCH_CAP_MIB
    prefill_chunk: int = 4096          # prefill_batch
    prefetch_groups: int = 8           # STRATA_GLM_STAGE_PREFETCH groups; 0 = off
    prefill_expert_cache_mib: float = 0
    prefill_legacy: bool = False       # expert weights dequantized per use (before P04 dequant-once)
    prefill_mla_f16: bool = True       # STRATA_GLM_MLA_F16 in prefill (P10)
    prefill_kda_parts: bool = True     # STRATA_GLM_PREFILL_KDA_ROW_PARTS=4 (P11)
    prefill_stream_depth: int = 0      # expert groups in flight beyond the 2-slot ring; >= 18 streams continuously
                                       # across layers (uploads never wait for the GPU; architecture option)
    prefill_gemm_scale: float = 1.0    # relative speed of the MoE GEMM path (what-if for a better kernel)
    prefill_experts: str = "gpu"       # gpu (stream over PCIe) | cpu (CPU computes the union of experts) | auto
    prefill_cpu_assist: bool = False   # CPU computes experts for part of the chunk while the GPU streams the rest
    batch: int = 1                     # independent sequences decoded per step (<= 8)
    placement: str = "cpu"             # cpu | gpu_stream (decode_experts=gpu) | ram_tier (64 GB modes)
    ram_mode: str = "exact"            # ram_tier: exact | frozen | hybrid
    ram_margin: float = 0.10           # hybrid routing margin
    ram_expert_gib: float = 0          # RAM budget for experts in ram_tier; 0 = what hw.memory leaves
    remote_share: float = 0.0          # share of each expert's rows computed by hw.remote (0.25/0.5/0.75)
    remote_reply: str = "f16"
    gpus: int = 1
    trace: str = ""                    # optional routing trace CSV for exact union / hit rates
    prior: str = ""                    # coverage.json for trace replay
    numa_placement: bool = True

    def copy(self, **changes):
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_dict(cls, data):
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)
