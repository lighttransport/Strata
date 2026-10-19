"""GLM-5.3-Flash DFlash2 evaluation worker for CUDA SM70.

Protocol 2 transports committed target features and the target's raw embeddings.
The engine evaluates its own compressed LM head. No duplicate embedding or head
weights are loaded here. This implementation uses PyTorch FP16 GEMMs and ordinary
attention; it requires neither BF16 instructions nor FlashAttention/Triton.
Architecture reference: https://github.com/z-lab/dflash (MIT), dflash/model.py.
Checkpoint: incoai/GLM-5.3-Flash-DFlash2; its separate license applies to weights.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F


TARGET_LAYERS = [5, 14, 24, 33, 42]


def validate_config(c):
    d = c.get('dflash_config', {})
    expected = {'hidden_size': 4096, 'vocab_size': 154880, 'num_hidden_layers': 5,
                'num_attention_heads': 32, 'num_key_value_heads': 8, 'head_dim': 128,
                'intermediate_size': 12288, 'sliding_window': 2048,
                'rms_norm_eps': 1e-5, 'attention_bias': False, 'is_causal': False,
                'hidden_act': 'silu', 'layer_types': ['sliding_attention'] * 5}
    draft = {'block_size': 8, 'mask_token_id': 154856, 'target_layer_ids': TARGET_LAYERS,
             'conv_group_size': 16, 'conv_kernel_size': 2, 'selector_rank': 256,
             'selector_top_k': 16}
    if c.get('architectures') != ['DFlash2DraftModel'] or any(c.get(k) != v for k, v in expected.items()):
        raise ValueError('unsupported DFlash2 target architecture')
    if any(d.get(k) != v for k, v in draft.items()):
        raise ValueError('unsupported DFlash2 feature, convolution or selector geometry')
    if c.get('rope_parameters') != {'rope_theta': 10000.0, 'rope_type': 'default'}:
        raise ValueError('unsupported DFlash2 rotary embedding')
    return c


def rms(x, weight, eps):
    # Float reductions avoid FP16 sum/variance overflow on Volta.
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def convolve(x, delta, base, group):
    n, hidden = x.shape
    values = x.reshape(n, hidden // group, group)
    coefficients = base.reshape(1, 2, hidden // group, group) + delta.unsqueeze(-1)
    shifted = F.pad(values[:-1], (0, 0, 0, 0, 1, 0))
    return (coefficients[:, 0] * values + coefficients[:, 1] * shifted).reshape_as(x)


def select_path(hidden, candidates, unary, anchor, projection, predecessor, successor):
    projected = F.linear(hidden, projection)
    selected = []
    for row in range(candidates.shape[0]):
        ids = candidates[row]
        scores = unary[row].float() + ((predecessor[anchor] * projected[row]).unsqueeze(0)
                                      * successor[ids]).float().sum(-1)
        # Keep the selection on device, including the predecessor dependency.
        anchor = ids[scores.argmax()]
        selected.append(anchor)
    return torch.stack(selected)


class Draft:
    def __init__(self, config, weights, device, dtype=torch.float16, gemm_f32=False, round_bf16=False, selector_scale=1.0):
        self.c = config
        self.gemm_f32 = gemm_f32
        self.round_bf16 = round_bf16
        self.selector_scale = selector_scale
        self.skinny = False
        self.graph = None
        self.capturing = False
        self.skinny_names = set()
        self.prefill_weights = {}
        self.prefill_kv = []
        self.w = weights
        self.device, self.dtype = device, dtype
        self.h = config['hidden_size']
        self.hd = config['head_dim']
        self.heads, self.kvheads = config['num_attention_heads'], config['num_key_value_heads']
        self.window = config['sliding_window']
        self.eps = config['rms_norm_eps']
        self.group = config['dflash_config']['conv_group_size']
        self.inv = 1.0 / (10000 ** (torch.arange(0, self.hd, 2, device=device).float() / self.hd))
        self.end = 0
        self.start = 0
        self.cache = []
        self.last_hidden = None
        self.qkv = []
        for layer in range(config['num_hidden_layers']):
            prefix = f'layers.{layer}.self_attn.'
            self.qkv.append(torch.cat([self.w.pop(prefix + name + '.weight')
                                       for name in ('q_proj', 'k_proj', 'v_proj')]))

    def enable_skinny(self, library):
        if not self.gemm_f32:
            raise ValueError('SM70 skinny requires the FP32-output fallback')
        torch.ops.load_library(str(library))
        # Keep row-major weights only for the large initial context projection.
        # Iterate keys, not tensor pairs, so replaced originals are released.
        for name in list(self.w):
            weight=self.w[name]
            if name.endswith('.weight') and weight.ndim==2:
                stem=name[:-7]
                if stem=='fc':self.prefill_weights[stem]=weight
                self.w[name]=torch.ops.strata_dflash2_sm70.pack(weight)
                self.skinny_names.add(stem)
        for layer in range(len(self.qkv)):
            weight=self.qkv[layer]
            self.prefill_kv.append(weight[self.heads*self.hd:].clone())
            self.qkv[layer]=torch.ops.strata_dflash2_sm70.pack(weight)
        self.skinny=True

    @torch.inference_mode()
    def enable_cuda_graph(self, embedding):
        if self.graph is not None or not self.cache or any(len(k)!=self.window for k,v in self.cache):
            raise ValueError('CUDA graph requires a complete prepared sliding window')
        # Stable inputs are separate from append()'s committed K/V. A replay
        # copies the current window; rejected draft rows never enter that cache.
        self.graph_cache=[(k.clone(),v.clone()) for k,v in self.cache]
        self.graph_embedding=embedding.clone()
        self.graph_end=torch.full((),self.end,device=self.device,dtype=torch.long)
        self.graph_offsets=torch.arange(8,device=self.device)
        self.graph_context_offsets=torch.arange(-self.window,8,device=self.device)
        original=self.cache
        self.cache=self.graph_cache
        self.capturing=True
        stream=torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):self.forward(self.graph_embedding)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                self.graph_hidden=self.forward(self.graph_embedding)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.graph=graph
        finally:
            self.cache=original
            self.capturing=False

    @torch.inference_mode()
    def replay(self, embedding):
        for (k,v),(target_k,target_v) in zip(self.cache,self.graph_cache):
            target_k.copy_(k);target_v.copy_(v)
        self.graph_end.fill_(self.end)
        self.graph_embedding.copy_(embedding)
        self.graph.replay()
        self.last_hidden=self.graph_hidden
        if not torch.isfinite(self.last_hidden).all():
            raise ValueError('DFlash2 nonfinite graph activation')
        return self.last_hidden

    def rounded(self, x):
        return x.to(torch.bfloat16).float() if self.round_bf16 else x

    def project(self, x, weight, packed=False):
        if not self.gemm_f32:
            return self.rounded(F.linear(x, weight))
        # Preserve wide residuals and down-projection outputs in FP32. Scale
        # unusually large inputs per row before conversion to tensor operands.
        scale = (x.abs().amax(-1, keepdim=True) / 16384).clamp_min(1)
        operand = (x / scale).to(torch.float16).contiguous()
        result=(torch.ops.strata_dflash2_sm70.linear(operand,weight) if packed
                else torch.ops.strata_dflash2.linear_f32(operand,weight))
        return self.rounded(result * scale)

    def linear(self, x, name):
        if self.skinny and name in self.skinny_names:
            if len(x)<=8:return self.project(x,self.w[name+'.weight'],packed=True)
            return self.project(x,self.prefill_weights[name])
        return self.project(x, self.w[name + '.weight'])

    def norm(self, x, name):
        if self.round_bf16:
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)
            return self.rounded(self.rounded(normalized)*self.w[name+'.weight'])
        return rms(x, self.w[name + '.weight'], self.eps)

    def rope(self, x, positions):
        phase = positions.float().unsqueeze(-1) * self.inv
        phase = torch.cat((phase, phase), -1).unsqueeze(1)
        rotated = torch.cat((-x[..., self.hd // 2:], x[..., :self.hd // 2]), -1)
        cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        if self.round_bf16:
            return self.rounded(self.rounded(x*self.rounded(cos))+self.rounded(rotated*self.rounded(sin)))
        return x * cos + rotated * sin

    def kv(self, x, layer, positions):
        p = f'layers.{layer}.self_attn.'
        if self.skinny and len(x)>8:
            combined=self.project(x,self.prefill_kv[layer])
        else:
            combined = self.project(x, self.qkv[layer][self.heads * self.hd:],packed=self.skinny)
        k, v = combined.chunk(2, dim=-1)
        k = self.rope(self.norm(k.reshape(-1, self.kvheads, self.hd), p + 'k_norm'), positions)
        return k, v.reshape(-1, self.kvheads, self.hd)

    @torch.inference_mode()
    def append(self, features, start, end):
        if start < 0 or end < start or end - start != len(features) or len(features) > self.window:
            raise ValueError('invalid DFlash2 context span')
        # A reset/repetition supplies a complete sliding window. Normal rounds
        # append only accepted target rows. No draft-token KV enters this cache.
        reset = start != self.end or end < self.end or not self.cache
        if reset:
            if start != max(0, end - self.window):
                raise ValueError('DFlash2 reset is missing committed context')
            self.cache = []
            self.start = start
        if len(features):
            context = self.norm(self.linear(self.rounded(features), 'fc'), 'hidden_norm')
            positions = torch.arange(start, end, device=self.device)
            cache = []
            for layer in range(self.c['num_hidden_layers']):
                k, v = self.kv(context, layer, positions)
                if not reset:
                    old_k, old_v = self.cache[layer]
                    k, v = torch.cat((old_k, k)), torch.cat((old_v, v))
                cache.append((k[-self.window:].contiguous(), v[-self.window:].contiguous()))
            self.cache = cache
        self.end = end
        self.start = end - (len(self.cache[0][0]) if self.cache else 0)

    def conv(self, x, delta, base):
        if not self.round_bf16:
            return convolve(x,delta,base,self.group)
        values=x.reshape(len(x),self.h//self.group,self.group)
        output=torch.zeros_like(values)
        for tap in range(2):
            shifted=values if not tap else F.pad(values[:-1],(0,0,0,0,1,0))
            product=self.rounded(base[tap].reshape(1,self.h//self.group,self.group)*shifted)
            output=self.rounded(output+product)
            output=self.rounded(torch.addcmul(output,delta[:,tap,:,None],shifted))
        return output.reshape_as(x)

    def conv_prepare(self, x, name):
        groups = self.h // self.group
        delta = self.linear(x, name + '.kernel_projection').reshape(len(x), 2, 2, groups)
        return self.conv(x, delta[:, 0], self.w[name + '.base_kernel'][0]), delta[:, 1]

    @torch.inference_mode()
    def forward(self, embedding):
        if not self.cache or len(embedding) != 8:
            raise ValueError('DFlash2 needs prepared context and a complete eight-slot block')
        if self.graph is not None and not self.capturing and all(len(k)==self.window for k,v in self.cache):
            return self.replay(embedding)
        positions = (self.graph_offsets+self.graph_end if self.capturing else
                     torch.arange(self.end, self.end + 8, device=self.device))
        context_positions = (self.graph_context_offsets+self.graph_end if self.capturing else
                             torch.arange(self.start, self.end + 8, device=self.device))
        # Bidirectional draft block, with each query's trained sliding window.
        visible = (positions[:, None] - context_positions[None, :] < self.window)
        visible &= (context_positions[None, :] - positions[:, None] < self.window)
        x = self.rounded(embedding)
        for layer in range(self.c['num_hidden_layers']):
            p = f'layers.{layer}'
            residual = x
            y, delta = self.conv_prepare(self.norm(x, p + '.input_layernorm'), p + '.attention_conv')
            combined = self.project(y, self.qkv[layer],packed=self.skinny)
            kvwidth = self.kvheads * self.hd
            q, k, v = combined.split((self.heads * self.hd, kvwidth, kvwidth), dim=-1)
            q = self.rope(self.norm(q.reshape(8, self.heads, self.hd), p + '.self_attn.q_norm'), positions)
            k = self.rope(self.norm(k.reshape(8, self.kvheads, self.hd), p + '.self_attn.k_norm'), positions)
            v = v.reshape(8, self.kvheads, self.hd)
            old_k, old_v = self.cache[layer]
            k = torch.cat((old_k, k)).transpose(0, 1)
            v = torch.cat((old_v, v)).transpose(0, 1)
            groups = self.heads // self.kvheads
            # Fold the query groups into rows; read each cached K/V head once.
            q = q.reshape(8, self.kvheads, groups, self.hd).permute(1, 2, 0, 3).reshape(self.kvheads, groups * 8, self.hd).to(torch.float32 if self.round_bf16 else self.w['fc.weight'].dtype)
            k, v = k.to(q.dtype), v.to(q.dtype)
            scores = torch.bmm(q, k.transpose(1, 2)).float() * (self.hd ** -0.5)
            scores = scores.reshape(self.kvheads, groups, 8, -1)
            scores.masked_fill_(~visible, float('-inf'))
            probabilities = scores.softmax(-1).reshape(self.kvheads, groups * 8, -1).to(q.dtype)
            attention = torch.bmm(probabilities, v).reshape(self.kvheads, groups, 8, self.hd)
            attention = attention.permute(2, 0, 1, 3).reshape(8, self.h)
            y = self.linear(self.rounded(attention), p + '.self_attn.o_proj')
            x = self.rounded(residual + self.conv(y, delta, self.w[p + '.attention_conv.base_kernel'][1]))
            residual = x
            y, delta = self.conv_prepare(self.norm(x, p + '.post_attention_layernorm'), p + '.mlp_conv')
            y = self.rounded(self.rounded(F.silu(self.linear(y, p + '.mlp.gate_proj'))) * self.linear(y, p + '.mlp.up_proj'))
            y = self.linear(y, p + '.mlp.down_proj')
            x = self.rounded(residual + self.conv(y, delta, self.w[p + '.mlp_conv.base_kernel'][1]))
        self.last_hidden = self.norm(x, 'norm')[1:]
        if not self.capturing and not torch.isfinite(self.last_hidden).all():
            raise ValueError('DFlash2 nonfinite draft activation')
        return self.last_hidden

    @torch.inference_mode()
    def select(self, candidates, unary, anchor):
        p = 'candidate_selector.'
        if self.selector_scale == 0:
            return candidates.gather(-1,unary.argmax(-1,keepdim=True))[:,0]
        if not self.gemm_f32 and not self.round_bf16 and self.selector_scale == 1:
            return select_path(self.last_hidden, candidates, unary, anchor,
                               self.w[p + 'hidden_projection.weight'],
                               self.w[p + 'predecessor_codebook'], self.w[p + 'successor_codebook'])
        projected = self.linear(self.last_hidden, p + 'hidden_projection')
        selected = []
        for row in range(len(candidates)):
            predecessor = self.rounded(self.w[p + 'predecessor_codebook'][anchor].float() * projected[row])
            transition = (predecessor.unsqueeze(0) * self.w[p + 'successor_codebook'][candidates[row]].float()).sum(-1)
            scores = unary[row] + self.selector_scale * self.rounded(transition)
            anchor = candidates[row, scores.argmax()]
            selected.append(anchor)
        return torch.stack(selected)


def read_exact(stream, size):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        n = stream.readinto(view[offset:])
        if not n:
            raise EOFError('truncated DFlash2 frame')
        offset += n
    return data


def tensor_frame(stream, rows, columns, device, dtype=torch.float16):
    if rows == 0:
        return torch.empty((0, columns), device=device, dtype=dtype)
    raw = read_exact(stream, rows * columns * 4)
    return torch.frombuffer(raw, dtype=torch.float32).reshape(rows, columns).to(device=device, dtype=dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--gemm-f32', type=Path, required=True, help='compiled FP32-output cuBLAS helper; keeps half weights')
    p.add_argument('--skinny-library', type=Path, help='optional SM70 half-QPN backend; large contexts retain cuBLAS')
    p.add_argument('--cuda-graph', action='store_true', help='replay fixed-window draft forward on CUDA; shorter windows use eager mode')
    p.add_argument('--round-bf16', action='store_true', help='software BF16 activation rounding with FP16 tensor operands')
    p.add_argument('--selector-scale', type=float, default=1.0, help='candidate transition scale; zero tests unary proposals')
    a = p.parse_args()
    if not 0 <= a.selector_scale <= 2:
        p.error('selector scale must be finite and in 0..2')
    config = validate_config(json.loads((a.model / 'config.json').read_text()))
    if sys.byteorder != 'little' or not a.device.startswith('cuda:'):
        p.error('little-endian CUDA host required')
    torch.set_num_threads(1)
    torch.cuda.set_device(a.device)
    # Volta has no BF16 tensor instructions. Convert per tensor, without a second
    # complete model in host RAM or another persistent copy in device memory.
    from safetensors import safe_open
    weights = {}
    with safe_open(a.model / 'model.safetensors', framework='pt', device='cpu') as f:
        for name in f.keys():
            weights[name] = f.get_tensor(name).to(device=a.device, dtype=torch.float16)
            if not torch.isfinite(weights[name]).all():
                raise ValueError('DFlash2 weight cannot be represented in FP16: ' + name)
    if a.gemm_f32:
        torch.ops.load_library(str(a.gemm_f32.resolve()))
    activation_dtype = torch.float32 if a.gemm_f32 else torch.float16
    model = Draft(config, weights, a.device, activation_dtype, gemm_f32=bool(a.gemm_f32), round_bf16=a.round_bf16, selector_scale=a.selector_scale)
    if a.skinny_library:
        model.enable_skinny(a.skinny_library.resolve())
        torch.cuda.synchronize()
        packing_peak=torch.cuda.max_memory_reserved()/2**20
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f'DFLASH_WORKER_PACKED transient_peak_reserved_mib={packing_peak}',file=sys.stderr,flush=True)
    if a.gemm_f32:
        # Materialize the maximum trained context and draft workspaces before
        # the target admits its expert cache into the remaining VRAM.
        model.append(torch.zeros((2048,20480),device=a.device,dtype=activation_dtype),0,2048)
        model.forward(torch.zeros((8,4096),device=a.device,dtype=activation_dtype))
        if a.cuda_graph:
            model.enable_cuda_graph(torch.zeros((8,4096),device=a.device,dtype=activation_dtype))
        model.cache=[];model.end=model.start=0;model.last_hidden=None
    torch.cuda.synchronize()
    reserved = torch.cuda.max_memory_reserved() / 2**20
    print(json.dumps(dict(protocol=2, ready=True, vram_mib=reserved, hidden_size=4096,
                          vocab_size=154880, block_size=8, mask_token_id=154856)), flush=True)
    last_id = 0
    waiting_select = False
    source, destination = sys.stdin.buffer, sys.stdout.buffer
    while True:
        line = source.readline(8193)
        if not line:
            break
        if len(line) > 8192:
            raise ValueError('oversized DFlash2 header')
        r = json.loads(line)
        identity = r.get('request_id')
        if r.get('protocol') != 2 or type(identity) is not int or identity <= last_id:
            raise ValueError('invalid DFlash2 request identity')
        last_id = identity
        op = r.get('op')
        if op in ('prepare', 'draft'):
            if waiting_select:
                raise ValueError('DFlash2 selector reply missing')
            start, end = r['start'], r['end']
            if type(start) is not int or type(end) is not int or not 0 <= start <= end <= 131072 or end - start > 2048:
                raise ValueError('invalid DFlash2 context bounds')
            features = tensor_frame(source, end - start, 20480, a.device, activation_dtype)
            if not torch.isfinite(features).all():
                raise ValueError('nonfinite DFlash2 target features')
            model.append(features, start, end)
            if op == 'prepare':
                print(json.dumps(dict(protocol=2, request_id=identity, prepared=True)), flush=True)
                continue
            embedding = tensor_frame(source, 8, 4096, a.device, activation_dtype)
            hidden = model.forward(embedding).float().cpu().contiguous()
            print(json.dumps(dict(protocol=2, request_id=identity, rows=7, columns=4096)), flush=True)
            destination.write(hidden.numpy().tobytes());destination.flush()
            waiting_select = True
        elif op == 'select':
            ids, logits, anchor = r['ids'], r['logits'], r['anchor']
            if not waiting_select or type(anchor) is not int or not 0 <= anchor < 154880:
                raise ValueError('unexpected DFlash2 selector request')
            if len(ids) != 7 or any(len(row) != 16 for row in ids) or any(type(t) is not int or not 0 <= t < 154880 for row in ids for t in row):
                raise ValueError('invalid DFlash2 candidate IDs')
            if len(logits) != 7 or any(len(row) != 16 for row in logits):
                raise ValueError('invalid DFlash2 candidate logits')
            values = torch.tensor(logits, dtype=torch.float32, device=a.device)
            if not torch.isfinite(values).all():
                raise ValueError('nonfinite DFlash2 candidate logits')
            tokens = model.select(torch.tensor(ids, device=a.device), values, anchor).tolist()
            waiting_select = False
            print(json.dumps(dict(protocol=2, request_id=identity, tokens=tokens)), flush=True)
        else:
            raise ValueError('unknown DFlash2 operation')


if __name__ == '__main__':
    main()
