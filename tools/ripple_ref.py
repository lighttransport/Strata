#!/usr/bin/env python3
"""Ripple-65B numpy reference (fp32; no torch): the first N layers + final norm + lm_head on a short token list, or an NVFP4 check.

  ripple_ref.py --model DIR --ids ids.txt --layers 5 --out logits.f32   # logits of every position, float32 [T][vocab]
  ripple_ref.py --model DIR --fp4 DIR_FP4 --check-fp4                  # NVFP4 nibble order / scale decode vs the BF16 weights

Mirrors modeling_ricursive.py (PeriNorm, gated attention, value residual, global NoPE layers, sliding RoPE layers)."""
import argparse, json, struct, sys
import numpy as np


class St:
    def __init__(self, d):
        self.d = d
        wm = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
        self.t = {}
        for f in sorted(set(wm.values())):
            with open(f"{d}/{f}", "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            for k, v in h.items():
                if k != "__metadata__":
                    self.t[k] = (f, v["dtype"], v["shape"], 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0])

    def raw(self, k):
        f, dt, sh, off, ln = self.t[k]
        with open(f"{self.d}/{f}", "rb") as fh:
            fh.seek(off)
            b = fh.read(ln)
        return dt, sh, b

    def get(self, k, rows=None):
        dt, sh, b = self.raw(k)
        if dt == "BF16":
            a = (np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(sh)
        elif dt == "F32":
            a = np.frombuffer(b, dtype=np.float32).reshape(sh)
        elif dt == "U8":
            a = np.frombuffer(b, dtype=np.uint8).reshape(sh)
        elif dt == "F8_E4M3":
            a = e4m3(np.frombuffer(b, dtype=np.uint8)).reshape(sh)
        else:
            raise SystemExit(f"dtype {dt}")
        return a


def e4m3(u):
    u = u.astype(np.int32)
    s = np.where(u >> 7, -1.0, 1.0)
    e = (u >> 3) & 15
    m = u & 7
    v = np.where(e == 0, m * 2.0 ** -9, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    return (s * v).astype(np.float32)


E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def nvfp4(q, sc, s2, low_first=True):
    lo, hi = q & 15, q >> 4
    a, b = (lo, hi) if low_first else (hi, lo)
    w = np.empty((q.shape[0], q.shape[1] * 2), dtype=np.float32)
    w[:, 0::2] = E2M1[a]
    w[:, 1::2] = E2M1[b]
    return (w.reshape(q.shape[0], -1, 16) * (sc[:, :, None] * s2)).reshape(q.shape[0], -1)


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, pos, theta):  # x [T, h, d]
    d = x.shape[-1]
    inv = 1.0 / theta ** (np.arange(0, d, 2) / d)
    a = pos[:, None] * inv[None, :]
    c, s = np.cos(a)[:, None, :], np.sin(a)[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def sig(x):
    return 1 / (1 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fp4")
    ap.add_argument("--ids")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--check-fp4", action="store_true")
    ap.add_argument("--use-fp4", action="store_true", help="forward from the NVFP4 checkpoint (--fp4 DIR): MLP weights dequantized")
    a = ap.parse_args()
    cfg = json.load(open(f"{a.model}/config.json"))
    S = St(a.model)
    if a.check_fp4:
        F = St(a.fp4)
        for name in ("gate_proj", "down_proj"):
            ref = S.get(f"model.layers.0.mlp.{name}.weight")[:64]
            b = f"model.layers.0.mlp.{name}"
            q, sc, s2 = F.get(b + ".weight")[:64], F.get(b + ".weight_scale")[:64], float(F.get(b + ".weight_scale_2"))
            for lf in (True, False):
                w = nvfp4(q, sc, s2, lf)
                err = np.abs(w - ref).mean() / np.abs(ref).mean()
                print(f"{name} low_nibble_first={lf} mean_rel_err={err:.4f}")
        return
    F = St(a.fp4) if a.use_fp4 else None
    H, nh, nkv, hd, eps = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
    ids = np.array([int(t) for t in open(a.ids).read().split()])
    T = len(ids)
    pos = np.arange(T, dtype=np.float64)
    x = S.get("model.embed_tokens.weight")[ids] * np.float32(cfg.get("embed_scale") or 1.0)
    emb0 = x.copy()
    mask = np.tril(np.ones((T, T), bool))
    swa = mask & (np.arange(T)[None, :] >= np.arange(T)[:, None] - (cfg["sliding_window"] - 1))
    clamp = cfg["ffn_clamp_threshold"]
    for l in range(a.layers):
        p = f"model.layers.{l}."
        def g(n):
            if a.use_fp4 and n.startswith("mlp.") and n.endswith(".weight"):
                b = p + n[:-len(".weight")]
                return nvfp4(F.get(b + ".weight"), F.get(b + ".weight_scale"), float(F.get(b + ".weight_scale_2")))
            return (F if a.use_fp4 else S).get(p + n)
        xn = rms(x, g("input_layernorm.weight"), eps)
        b = sig(float(g("self_attn.value_res_lambda")[0]))
        vin = emb0 + b * (xn - emb0)
        q = (xn @ g("self_attn.q_proj.weight").T).reshape(T, nh, hd)
        k = (xn @ g("self_attn.k_proj.weight").T).reshape(T, nkv, hd)
        v = (vin @ g("self_attn.v_proj.weight").T).reshape(T, nkv, hd)
        gate = xn @ g("self_attn.g_proj.weight").T
        q, k = rms(q, g("self_attn.q_norm.weight"), eps), rms(k, g("self_attn.k_norm.weight"), eps)
        sliding = cfg["layer_types"][l] == "sliding_attention"
        if sliding:
            q, k = rope(q, pos, cfg["rope_theta"]), rope(k, pos, cfg["rope_theta"])
        k, v = np.repeat(k, nh // nkv, 1), np.repeat(v, nh // nkv, 1)
        s = np.einsum("thd,uhd->htu", q, k) / np.sqrt(hd)
        s = np.where((swa if sliding else mask)[None], s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("htu,uhd->thd", s, v).reshape(T, nh * hd) * sig(gate)
        x = x + rms(o @ g("self_attn.o_proj.weight").T, g("post_attention_layernorm.weight"), eps)
        xn = rms(x, g("pre_feedforward_layernorm.weight"), eps)
        gt = np.minimum(xn @ g("mlp.gate_proj.weight").T, clamp)
        up = np.clip(xn @ g("mlp.up_proj.weight").T, -clamp, clamp)
        x = x + rms((gt * sig(gt) * up) @ g("mlp.down_proj.weight").T, g("post_feedforward_layernorm.weight"), eps)
        print(f"layer {l} done", file=sys.stderr)
    x = rms(x, S.get("model.norm.weight"), eps)
    logits = (x @ S.get("lm_head.weight").T).astype(np.float32)
    logits.tofile(a.out)
    print("top1", logits.argmax(-1).tolist())


main()
