"""Differentiable DFlash2 calibration core with FP32 activation gradients.

Forward linear products use the serving FP32-output helper on CUDA. Frozen
half-weight values are promoted once for backward products. Half attention
products retain serving forward rounding while passing FP32 gradients.
"""
import torch
from torch.nn import functional as F
from glm_dflash2_worker import Draft


class HalfOperand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.half().float()

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, weight32, cuda_helper):
        ctx.save_for_backward(weight32)
        scale = (x.abs().amax(-1, keepdim=True) / 16384).clamp_min(1)
        operand = (x / scale).half().contiguous()
        result = (torch.ops.strata_dflash2.linear_f32(operand, weight) if cuda_helper else
                  F.linear(operand.float(), weight32))
        return result * scale

    @staticmethod
    def backward(ctx, gradient):
        weight, = ctx.saved_tensors
        # Straight-through operand rounding; row scaling cancels in dx.
        return gradient.matmul(weight), None, None, None


class HalfBmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ah, bh = a.half(), b.half()
        ctx.save_for_backward(ah.float(), bh.float())
        return torch.bmm(ah, bh).float()

    @staticmethod
    def backward(ctx, gradient):
        a, b = ctx.saved_tensors
        return torch.bmm(gradient, b.transpose(1, 2)), torch.bmm(a.transpose(1, 2), gradient)


class TrainingDraft(Draft):
    # Serving deliberately disables autograd; calibration must retain the
    # complete feature -> FC -> committed K/V path.
    append = Draft.append.__wrapped__

    def __init__(self, config, weights, device, adapter, *, cuda_helper=False):
        for weight in weights.values():
            weight.requires_grad_(False)
        super().__init__(config, weights, device, torch.float32, gemm_f32=True)
        self.adapter = adapter
        self.cuda_helper = cuda_helper
        self.backward_weights = {}

    def project(self, x, weight, packed=False):
        if packed:
            raise ValueError('calibration needs row-major frozen weights')
        key = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()))
        if key not in self.backward_weights:
            # select() also calls this projection under inference_mode during
            # evaluation. Its cached frozen operand must remain usable by the
            # next differentiated training step.
            with torch.inference_mode(False), torch.no_grad():
                self.backward_weights[key] = weight.to(dtype=torch.float32, copy=True)
        return Linear.apply(x.float(), weight, self.backward_weights[key], self.cuda_helper)

    def linear(self, x, name):
        output = super().linear(x, name)
        return output + self.adapter(x) if name == 'fc' else output

    def forward(self, embedding):
        if not self.cache or len(embedding) != 8:
            raise ValueError('calibration needs committed context and eight embedding slots')
        positions = torch.arange(self.end, self.end + 8, device=self.device)
        context_positions = torch.arange(self.start, self.end + 8, device=self.device)
        visible = (positions[:, None] - context_positions[None, :] < self.window)
        visible &= (context_positions[None, :] - positions[:, None] < self.window)
        x = embedding.float()
        for layer in range(self.c['num_hidden_layers']):
            p = f'layers.{layer}'
            residual = x
            y, delta = self.conv_prepare(self.norm(x, p + '.input_layernorm'), p + '.attention_conv')
            combined = self.project(y, self.qkv[layer])
            kvwidth = self.kvheads * self.hd
            q, k, v = combined.split((self.heads * self.hd, kvwidth, kvwidth), dim=-1)
            q = self.rope(self.norm(q.reshape(8, self.heads, self.hd), p + '.self_attn.q_norm'), positions)
            k = self.rope(self.norm(k.reshape(8, self.kvheads, self.hd), p + '.self_attn.k_norm'), positions)
            v = v.reshape(8, self.kvheads, self.hd)
            old_k, old_v = self.cache[layer]
            k = torch.cat((old_k, k)).transpose(0, 1)
            v = torch.cat((old_v, v)).transpose(0, 1)
            groups = self.heads // self.kvheads
            q = q.reshape(8, self.kvheads, groups, self.hd).permute(1, 2, 0, 3).reshape(self.kvheads, groups * 8, self.hd)
            scores = HalfBmm.apply(q, k.transpose(1, 2)) * (self.hd ** -0.5)
            scores = scores.reshape(self.kvheads, groups, 8, -1).masked_fill(~visible, float('-inf'))
            probabilities = scores.softmax(-1).reshape(self.kvheads, groups * 8, -1)
            attention = HalfBmm.apply(probabilities, v).reshape(self.kvheads, groups, 8, self.hd)
            attention = attention.permute(2, 0, 1, 3).reshape(8, self.h)
            y = self.linear(attention, p + '.self_attn.o_proj')
            x = residual + self.conv(y, delta, self.w[p + '.attention_conv.base_kernel'][1])
            residual = x
            y, delta = self.conv_prepare(self.norm(x, p + '.post_attention_layernorm'), p + '.mlp_conv')
            y = F.silu(self.linear(y, p + '.mlp.gate_proj')) * self.linear(y, p + '.mlp.up_proj')
            y = self.linear(y, p + '.mlp.down_proj')
            x = residual + self.conv(y, delta, self.w[p + '.mlp_conv.base_kernel'][1])
        self.last_hidden = self.norm(x, 'norm')[1:]
        if not torch.isfinite(self.last_hidden).all():
            raise ValueError('nonfinite calibration activation')
        return self.last_hidden


class LinearTrainingDraft(TrainingDraft):
    """Opt-in residual updates to FC and each MLP output projection."""
    def linear(self, x, name):
        output = Draft.linear(self, x, name)
        return output + self.adapter(x, name) if name in self.adapter.by_name else output
