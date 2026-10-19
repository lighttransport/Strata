"""Optional FP32 residual adapters for the FC bridge and draft MLP outputs."""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from glm_dflash2_adapter import file_sha

SCHEMA = 'strata.dflash2.linear_adapters.v1'


def target_shapes(config, weights):
    names = ['fc'] + [f'layers.{i}.mlp.down_proj' for i in range(config['num_hidden_layers'])]
    shapes = []
    for name in names:
        weight = weights[name + '.weight']
        if weight.ndim != 2:
            raise ValueError('linear adapters require unpacked base weights')
        output, inputs = weight.shape
        shapes.append(dict(name=name, inputs=inputs, outputs=output))
    return shapes


class LinearAdapters(nn.Module):
    def __init__(self, shapes, rank, alpha, *, device='cpu'):
        super().__init__()
        if (type(rank) is not int or not 1 <= rank <= 64 or
                not math.isfinite(alpha) or not 0 < alpha <= 64 or
                not isinstance(shapes, list) or not 1 <= len(shapes) <= 33):
            raise ValueError('invalid linear adapter limits')
        self.shapes = [dict(s) for s in shapes]
        self.rank, self.alpha = rank, float(alpha)
        self.by_name = {}
        parameters = 0
        for i, shape in enumerate(self.shapes):
            if (set(shape) != {'name', 'inputs', 'outputs'} or
                    not isinstance(shape['name'], str) or shape['name'] in self.by_name or
                    any(type(shape[k]) is not int or not 1 <= shape[k] <= 65536 for k in ('inputs', 'outputs'))):
                raise ValueError('invalid linear adapter target geometry')
            self.by_name[shape['name']] = i
            parameters += rank * (shape['inputs'] + shape['outputs'])
        if parameters > 16 * 1024**2:
            raise ValueError('linear adapter parameter budget exceeded')
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for shape in self.shapes:
            down = nn.Linear(shape['inputs'], rank, bias=False, device=device, dtype=torch.float32)
            up = nn.Linear(rank, shape['outputs'], bias=False, device=device, dtype=torch.float32)
            nn.init.normal_(down.weight, std=shape['inputs'] ** -.5)
            nn.init.zeros_(up.weight)
            self.down.append(down)
            self.up.append(up)

    def forward(self, x, name):
        index = self.by_name[name]
        return F.linear(F.linear(x.float(), self.down[index].weight), self.up[index].weight) * (self.alpha / self.rank)


def apply_linear_adapters(draft, adapters):
    if draft.skinny or adapters.shapes != target_shapes(draft.c, draft.w):
        raise ValueError('linear adapters do not match unpacked draft projections')
    if getattr(draft, 'fc_adapter', None) is not None or getattr(draft, 'linear_adapters', None) is not None:
        raise ValueError('draft already has residual adapters')
    original = draft.linear
    def linear(x, name):
        output = original(x, name)
        return output + adapters(x, name) if name in adapters.by_name else output
    draft.linear = linear
    draft.linear_adapters = adapters


def load_linear_adapters(directory, draft, checkpoint):
    from safetensors.torch import load_file
    directory = Path(directory)
    metadata = json.loads((directory / 'adapter.json').read_text())
    if (metadata.get('schema') != SCHEMA or not metadata.get('complete') or
            metadata.get('base_checkpoint_sha256') != file_sha(checkpoint)):
        raise ValueError('linear adapter checkpoint identity or manifest is invalid')
    # Resolve targets from the actual base matrices before allocating any
    # adapter. Manifest dimensions cannot request arbitrary device memory.
    if draft.skinny or metadata.get('targets') != target_shapes(draft.c, draft.w):
        raise ValueError('linear adapter targets do not match the base draft')
    weights = directory / 'linear.safetensors'
    if weights.stat().st_size > 65 * 1024**2 or file_sha(weights) != metadata.get('weights_sha256'):
        raise ValueError('linear adapter weight size or checksum mismatch')
    adapters = LinearAdapters(metadata['targets'], metadata['rank'], metadata['alpha'], device=draft.device)
    state = load_file(str(weights), device=draft.device)
    expected = adapters.state_dict()
    if (set(state) != set(expected) or any(state[k].shape != expected[k].shape or
            state[k].dtype != torch.float32 or not torch.isfinite(state[k]).all() for k in state)):
        raise ValueError('linear adapter weights have invalid shapes, dtype or values')
    adapters.load_state_dict(state, strict=True)
    adapters.requires_grad_(False)
    apply_linear_adapters(draft, adapters)
    return metadata
