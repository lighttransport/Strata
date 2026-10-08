"""VRAM budget accounting: what is left for the decode expert tier after the fixed residents."""
import dataclasses

from model import VRAM, DENSE_FORMATS


@dataclasses.dataclass
class VramPlan:
    budget_mib: float
    items: dict
    tier_mib: float
    slots: int
    expert_bytes: float
    primary_tier_mib: float = 0.0

    def table(self):
        rows = [(k, round(v, 1)) for k, v in self.items.items()]
        rows.append(("tier", round(self.tier_mib, 1)))
        return rows


def plan(hw, pack, context, mtp_depth=0, gpu_draft_experts=False, dense_format="q8", budget_mib=None,
         reserve_mib=512, decode_cache_mib=None, prefill_scratch_mib=1024, prefetch_groups=0, gpus=1,
         draft_model_mib=0.0, verify_width=None, tier_compress=1.0):
    """Returns the VramPlan. `decode_cache_mib` caps the tier (the engine's --decode-cache-mib); None = fill.

    mtp_depth > 0 adds GLM's draft block; draft_model_mib > 0 adds a separate drafter (block diffusion);
    verify_width sets the history slots (defaults to mtp_depth)."""
    budget = budget_mib if budget_mib else hw.gpu.vram_mib - hw.gpu.desktop_mib
    items = {}
    q8 = dense_format == "q8"
    items["dense"] = DENSE_FORMATS[dense_format]["vram_mib"] if pack.name != "exl3" else VRAM["exl3_dense"]
    history = verify_width if verify_width is not None else mtp_depth
    if mtp_depth > 0:
        items["mtp_dense"] = VRAM["mtp_dense_q8"] if q8 else VRAM["mtp_dense"]
        if gpu_draft_experts:
            items["draft_bank"] = VRAM["draft_bank"]
    if draft_model_mib > 0:
        items["draft_model"] = draft_model_mib
    if history > 0:
        items["verify_history"] = VRAM["verify_slot"] * history
    items["state"] = VRAM["state_fixed"] + VRAM["state_per_token"] * context
    items["staging"] = VRAM["staging_slot"] * VRAM["staging_slots"]
    if prefetch_groups:
        items["prefetch"] = VRAM["prefetch_group"] * 2
    items["prefill_scratch"] = prefill_scratch_mib
    items["misc"] = VRAM["misc"]
    items["reserve"] = reserve_mib
    used = sum(items.values())
    primary_free = max(0.0, budget - used)
    secondary_free = (gpus - 1) * max(0.0, hw.gpu.vram_mib - reserve_mib - items["staging"] - prefill_scratch_mib)
    free = primary_free + secondary_free
    tier = free if decode_cache_mib is None else min(free, decode_cache_mib)
    expert_bytes = pack.mean_expert_bytes() / tier_compress
    slots = int(tier * 2 ** 20 / expert_bytes)
    tier_mib = slots * expert_bytes / 2 ** 20
    primary = min(primary_free, tier_mib) if gpus == 1 else tier_mib * primary_free / max(1e-9, free)
    return VramPlan(budget_mib=budget, items=items, tier_mib=tier_mib, slots=slots, expert_bytes=expert_bytes,
                    primary_tier_mib=primary)
