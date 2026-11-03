"""MiniMax M3.1 Flash step-time model, calibrated on the measured strata-m3 path (tools/sim/data/minimax_m31_measured.json).

The GLM-derived decode model in this directory (decode.py, kernels.Params) was fitted on GLM; this module is the MiniMax one: every term
is either a measurement (dense matvec table, step profile, routing union law, cache hit rates) or a small number of fitted
constants (the CPU expert lane), and `check_anchors` prints how far the fitted model is from the six measured single-stream decode
speeds.  It models one verify step of `B` sequences of `T` rows each (speculative window), then tok/s = B * tokens per step / step time.

Scope and honesty: the batched-decode and prefill rows are PROJECTIONS from this calibrated model, not measurements; the engine has
no batched decode yet (one sequence, at most 8 rows), so the batch columns assume the engine grows that support.
"""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib

DATA_PATH = pathlib.Path(__file__).resolve().parent / "data" / "minimax_m31_measured.json"
DATA = json.loads(DATA_PATH.read_text())
DENSE = DATA["dense_matvec_us"]
PROF = DATA["step_profile_max_mode_one_gpu"]
ANCH = DATA["decode_anchors_tok_s"]
LAYERS = PROF["layers"]
TOP_K = {"exact": 4, "fast": 3, "max": 2}
BYTES_PER_WEIGHT = {"Q4_K": 144 / 256, "Q5_K": 176 / 256, "Q6_K": 210 / 256, "Q8_0": 34 / 32}


def matrix_bytes(key):
    d = DENSE[key]
    return d["n_in"] * d["n_out"] * BYTES_PER_WEIGHT[d["type"]]


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])


def rho(T):
    u = DATA["union_law"]
    return interp(u["T"], u["rho"], T)


def cross_ratio(B):
    """Distinct experts of B independent windows relative to B times one window; a fit r = 1 / (1 + a (B-1)^b) to the measured 0.83 / 0.75 / 0.66."""
    return 1.0 if B <= 1 else 1.0 / (1.0 + 0.205 * (B - 1) ** 0.84)


def hit_degradation(B, slots):
    """Hit rate of the distinct experts at B sequences relative to one sequence (cache sim at 1100 and 2800 slots, interpolated in log(slots))."""
    c = DATA["cache_hit_unique_experts"]

    def f(tab):
        ratios = [h / tab["hit"][0] for h in tab["hit"]]
        val = interp(tab["B"], ratios, min(B, 8))
        return val * (0.93 ** math.log2(B / 8) if B > 8 else 1.0)   # beyond the simulated range: keep falling

    lo, hi = f(c["slots_1100"]), f(c["slots_2800"])
    t = (math.log(max(slots, 1100)) - math.log(1100)) / (math.log(2800) - math.log(1100))
    t = min(1.0, max(0.0, t))
    return lo + t * (hi - lo)


def cols_us(key, n):
    """Dense matvec time for n (possibly fractional) columns with the engine's launches of at most 8 columns each (every launch re-reads the weights)."""
    us = DENSE[key]["us"]
    full = int(n // 8)
    rest = n - 8 * full
    t = full * us[7]
    if rest > 1e-9:
        t += interp(list(range(1, 9)), us, max(1.0, rest))
    return t


def gemm_us(key, n, dequant_gbs=300.0, tflops=30.0):
    """The tensor-core path for many columns (prototype: dequantize-bound, flat in the columns): weights read once."""
    flops = 2.0 * DENSE[key]["n_in"] * DENSE[key]["n_out"] * n
    return max(matrix_bytes(key) / (dequant_gbs * 1e9), flops / (tflops * 1e12)) * 1e6 + 12.0


def best_us(key, n, tensor_core):
    t = cols_us(key, n)
    if tensor_core and n > 8:
        t = min(t, gemm_us(key, n))
    return t


@dataclasses.dataclass
class Params:
    # measured decomposition of the one-GPU max-mode step (docs table), per layer, microseconds
    dense_scale: float = 1.41          # (measured qkv + o time) / (matvec bench at the same columns): quantize, launches, gaps
    shared_scale: float = 1.12         # same for the shared expert's three matvecs
    msa_seq_us: float = 32.0           # block scores + select of the first sequence of a step
    msa_extra_seq_us: float = 90.0     # each further sequence of a batch step: its own attention launches (fitted on the measured batches; 30 with one launch for all)
    draft_batched: bool = False        # the draft pass of all sequences as one pass (the engine runs one pass per sequence)
    msa_row_us: float = 23.2           # sparse attention per row (5.4 ms / 57 / 4.1)
    layer_fix_us: float = 100.0        # norms, rope, router, indexer, layer start
    job_us_seq: float = 62.0           # one resident expert job (5 kernels), one stream
    job_us_stream: float = 23.0        # effective, 4 side streams
    combine_us: float = 56.0           # CPU wait slack + accumulate + residual add
    # fitted on the six anchors
    cpu_job_us: float = 290.0          # CPU lane: one expert job (weights read once)
    cpu_layer_us: float = 70.0         # CPU lane: fixed per layer with at least one job (quantize, barriers)
    cpu_token_us: float = 20.0         # CPU lane: every further token of a job
    host_pre_us: float = 0.0           # host time per layer before any expert work starts: stream sync wake-up, routing filter, queueing (fitted)
    unique_hit: float = 0.85           # hit of a distinct expert / hit of an expert use
    # draft head, per sequence
    draft_rows: int = 6
    host_ms: float = 2.1
    max_p2: dict = dataclasses.field(default_factory=lambda: {"one_gpu": 0.486, "two_gpu": 0.757})
    max_heavy_miss: dict = dataclasses.field(default_factory=lambda: {"one_gpu": 0.242, "two_gpu": 0.118})
    tokens_accept: float = 0.94        # per-position acceptance giving 3.8 tokens at 4.1 rows
    dev1_latency_us: float = 25.0


TOKENS_BY_ROWS = ([1.0, 2.0, 3.0, 4.1], [1.0, 1.9, 2.76, 3.8])   # measured: 3 rows 2.76 tokens (88 % of 2 drafts), 4.1 rows 3.8 tokens (draft 6, pmin 0.7); 2 rows interpolated


def tokens_per_step(T, p=None):
    """Tokens one sequence gains per verify step of T rows; not extrapolated beyond the measured 4.1 rows."""
    return interp(TOKENS_BY_ROWS[0], TOKENS_BY_ROWS[1], T)


Q8_DRAFT_US = [67.4, 70.7, 79.6, 82.8, 93.3, 101.6, 111.7, 122.8]


def draft_ms(rows, B, p, tensor_core):
    """Draft pass for B sequences of `rows` rows: layers part scales as the Q8_0 matvec bench, head as the lm_head bench, Markov chain per sequence."""
    d = PROF["draft_pass_ms"]
    n = rows * B
    full = int(n // 8)
    rest = n - 8 * full
    lay6 = Q8_DRAFT_US[5]
    lay = full * Q8_DRAFT_US[7] + (interp(list(range(1, 9)), Q8_DRAFT_US, max(1.0, rest)) if rest > 1e-9 else 0.0)
    layers = d["layers"] * lay / lay6
    if tensor_core and n > 8:   # the draft layers (about 1.9 GB of Q8_0) through the dequantizing tensor-core path: flat in the rows
        layers = min(layers, 1.9e9 / 300e9 * 1e3 + 0.3)
    head = d["norm_lm_head"] * best_us("head", n, tensor_core) / DENSE["head"]["us"][5]
    markov = d["markov_argmax_conf_after"] * (1 + 0.15 * (B - 1))
    return layers + head + markov + 0.5


KV_BYTES_PER_TOKEN = 60 * (2 * 4 * 128 * 2 + 128 * 4)      # FP16 K/V and the F32 indexer key, all 60 layers (tools/sim/model.py)
SLOT_MIB = 17.3
GPU0_SLOT_BUDGET_MIB = 1100 * SLOT_MIB + 8192 * KV_BYTES_PER_TOKEN / (1 << 20)   # what the measured one-GPU engine had for slots + K/V (8192 context)
GPU1_SLOTS = 1800


def slots_for(B, ctx, gpus):
    s0 = int(max(0.0, GPU0_SLOT_BUDGET_MIB - B * ctx * KV_BYTES_PER_TOKEN / (1 << 20)) / SLOT_MIB)
    return s0 + (GPU1_SLOTS if gpus == 2 else 0)


def hit_for_slots(mode, slots):
    """Hit rate of expert uses vs slots: log-linear through the two measured points (1100 slots one GPU, 2900 slots two GPUs), clamped."""
    h1, h2 = ANCH["use_hit_rate"]["one_gpu"][mode], ANCH["use_hit_rate"]["two_gpu"][mode]
    t = (math.log(max(slots, 50)) - math.log(1100)) / (math.log(2900) - math.log(1100))
    return min(0.985, max(0.05, h1 + t * (h2 - h1)))


def hits(mode, gpus, B, p, ctx=8192):
    slots = slots_for(B, ctx, gpus)
    return hit_for_slots(mode, slots) * hit_degradation(B, slots), slots


def step_ms(mode, gpus, B, T, p, tensor_core=False, streams=True, detail=False, ctx=8192):
    N = B * T
    L = LAYERS
    k = TOP_K[mode]
    # --- GPU chain to the router
    dense = p.dense_scale * (best_us("q", N, tensor_core) + 2 * best_us("kv", N, tensor_core) + best_us("o", N, tensor_core))
    chain = dense + p.msa_seq_us + (B - 1) * p.msa_extra_seq_us + p.msa_row_us * N + p.layer_fix_us
    # --- experts
    h, slots = hits(mode, gpus, B, p, ctx)
    key = "two_gpu" if gpus == 2 else "one_gpu"
    uni = rho(T) * cross_ratio(B)
    if mode == "max":
        # top-2 with every missed expert dropped except a token's heaviest: CPU jobs are the heaviest experts that miss
        pm = min(0.95, p.max_heavy_miss[key] * (1 - hit_for_slots(mode, slots)) / (1 - ANCH["use_hit_rate"][key][mode]) * (1 + 2.0 * (1 - hit_degradation(B, slots))))
        uses_cpu = N * pm
        uses_gpu = N * (1 - pm + p.max_p2[key])
    else:
        uses = N * k
        uses_cpu = uses * (1 - h)
        uses_gpu = uses * h
    jobs_cpu = uses_cpu * uni
    jobs_gpu = uses_gpu * uni
    shared = p.shared_scale * (2 * best_us("shg", N, tensor_core) + 1.19 * best_us("shg", N, tensor_core))
    jt = p.job_us_stream if streams else p.job_us_seq
    if gpus == 2:
        gpu_jobs_us = jobs_gpu * jt * 0.5 + p.dev1_latency_us
    else:
        gpu_jobs_us = jobs_gpu * jt
    cpu_us = 0.0
    if jobs_cpu > 0.05:
        cpu_us = p.cpu_layer_us + jobs_cpu * p.cpu_job_us + max(0.0, uses_cpu - jobs_cpu) * p.cpu_token_us
    post = p.host_pre_us + max(shared + gpu_jobs_us, cpu_us) + p.combine_us
    head = best_us("head", N, tensor_core)
    draft = draft_ms(p.draft_rows, B, p, tensor_core) if p.draft_batched else B * draft_ms(p.draft_rows, 1, p, False)
    ms = L * (chain + post) / 1000.0 + head / 1000.0 + draft + p.host_ms
    if detail:
        return ms, dict(chain_us=chain, dense_us=dense, post_us=post, shared_us=shared, gpu_jobs_us=gpu_jobs_us, cpu_us=cpu_us,
                        jobs_cpu=jobs_cpu, jobs_gpu=jobs_gpu, hit=h, slots=slots, draft_ms_total=draft, head_ms=head / 1000.0,
                        draft_ms=draft, union=uni)
    return ms


def decode_tok_s(mode, gpus, B, T, p, **kw):
    ms = step_ms(mode, gpus, B, T, p, **kw)
    return B * tokens_per_step(T, p) / ms * 1000.0, ms


def anchors():
    out = []
    for gpus, key in ((1, "one_gpu"), (2, "two_gpu")):
        for mode in ("exact", "fast", "max"):
            meas = sum(ANCH[key][mode]) / 2.0
            out.append((gpus, mode, meas))
    return out


def check_anchors(p, T=4.1):
    rows = []
    for gpus, mode, meas in anchors():
        tok, ms = decode_tok_s(mode, gpus, 1, T, p)
        rows.append((gpus, mode, meas, tok, ms))
    return rows


def fit(p, T=4.1):
    """Grid search of the CPU-lane constants, the host term and the effective stream job time on the six anchors plus the measured
    post-router time of the one-GPU max-mode profile (29.9 ms per step, 4.1 rows), by squared relative error."""
    post_meas = 29.9
    best = None
    for hp in [0, 40, 80, 120, 160]:
        for cj in [220, 250, 280, 310, 340, 380]:
            for cl in [0, 40, 80, 130, 190]:
                for ct in [0, 20, 40]:
                    for js in [23, 35, 50]:
                        q = dataclasses.replace(p, host_pre_us=hp, cpu_job_us=cj, cpu_layer_us=cl, cpu_token_us=ct, job_us_stream=js)
                        err = 0.0
                        for gpus, mode, meas, tok, ms in check_anchors(q, T):
                            err += ((tok - meas) / meas) ** 2
                        _, d = step_ms("max", 1, 1, T, q, detail=True)
                        err += 0.5 * ((LAYERS * d["post_us"] / 1000.0 - post_meas) / post_meas) ** 2
                        if best is None or err < best[0]:
                            best = (err, q)
    return best[1], best[0]


# ---------------------------------------------------------------- prefill
@dataclasses.dataclass
class PrefillParams:
    h2d_gbs: float = 12.3
    pack_gib: float = 124.7
    compute_s_per_ktok: float = 1.43    # GPU work per 1000 tokens with --pf-gemm: cuBLAS 0.78 + dequant 0.24 (measured) + attention and the rest 0.41 (assumed: the rest of the 14.2 s row)
    overlap: float = 0.81               # share of the GPU work hidden under the pack stream with --pf-prefetch (fitted on the 14.2 s row)
    warm_s: float = 1.4                 # warm-up of the expert cache from the prompt's routing (measured ~1.4 s), once per prompt batch
    touch_tokens: float = 150.0         # experts touched by n tokens: 1 - exp(-n / touch_tokens) of the pack (fitted on the 210 / 487-token prompts)
    gpu_scaling_2: float = 1.7          # compute speedup of the second GPU (attention stays on GPU 0)
    resident_share: float = 0.15        # share of the pack that is already in VRAM slots (1100 of 7296 experts)


def prefill_s(tokens, chunk, gpus, pp: PrefillParams, resident_share=None, overlap=None, compute_scale=1.0):
    """Seconds to prefill `tokens` tokens in chunks of `chunk` tokens: each chunk streams the experts it touches over PCIe (one link per GPU)
    while the GPU computes; `overlap` of the compute hides under the stream, the rest adds."""
    res = pp.resident_share if resident_share is None else resident_share
    ov = pp.overlap if overlap is None else overlap
    total = pp.warm_s if tokens > 0 else 0.0
    left = tokens
    while left > 0:
        n = min(chunk, left)
        left -= n
        touched = 1.0 - math.exp(-n / pp.touch_tokens)
        stream = pp.pack_gib * (1 << 30) * touched * (1.0 - res) / (pp.h2d_gbs * 1e9 * gpus)
        comp = n / 1000.0 * pp.compute_s_per_ktok * compute_scale / (pp.gpu_scaling_2 if gpus == 2 else 1.0)
        total += max(stream, comp) + (1.0 - ov) * min(stream, comp)
    return total


def fit_prefill(pp):
    """Grid search of warm_s, touch_tokens and overlap on the three measured prefill points (one GPU)."""
    best = None
    for w in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        for t in [60, 90, 120, 150, 200, 260]:
            for ov in [0.5, 0.6, 0.7, 0.8, 0.9]:
                q = dataclasses.replace(pp, warm_s=w, touch_tokens=t, overlap=ov)
                err = sum(((mod - meas) / meas) ** 2 for _, _, meas, mod in prefill_anchor_errors(q))
                if best is None or err < best[0]:
                    best = (err, q)
    return best[1], best[0]


def prefill_anchor_errors(pp):
    """Model vs the measured points: 6960 tokens in one chunk 14.2 s, 210 tokens 22.8 tok/s, 487 tokens 38.6 tok/s, one GPU."""
    pts = [(6960, 7168, 14.2), (210, 512, 210 / 22.8), (487, 512, 487 / 38.6)]
    return [(n, c, m, prefill_s(n, c, 1, pp)) for n, c, m in pts]
