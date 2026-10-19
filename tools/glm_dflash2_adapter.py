"""Optional FP32 low-rank correction to DFlash2's target-feature bridge."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'strata.dflash2.fc_adapter.v1'


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


class FcAdapter(nn.Module):
    def __init__(self, hidden, features, rank, alpha, *, device='cpu'):
        super().__init__()
        if (type(hidden) is not int or type(features) is not int or type(rank) is not int or
                hidden < 1 or features != 5 * hidden or not 1 <= rank <= 64 or
                not math.isfinite(alpha) or not 0 < alpha <= 64):
            raise ValueError('invalid FC adapter geometry or scale')
        self.hidden, self.features, self.rank, self.alpha = hidden, features, rank, float(alpha)
        self.down = nn.Linear(features, rank, bias=False, device=device, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden, bias=False, device=device, dtype=torch.float32)
        nn.init.normal_(self.down.weight, std=features**-0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return F.linear(F.linear(x.float(), self.down.weight), self.up.weight) * (self.alpha / self.rank)


def apply_fc_adapter(draft, adapter):
    if adapter.hidden != draft.h or adapter.features != draft.w['fc.weight'].shape[1]:
        raise ValueError('FC adapter does not match the draft feature bridge')
    if getattr(draft, 'fc_adapter', None) is not None:
        raise ValueError('an FC adapter is already attached')
    original = draft.linear
    def linear(x, name):
        output = original(x, name)
        return output + adapter(x) if name == 'fc' else output
    draft.linear = linear
    draft.fc_adapter = adapter


def load_fc_adapter(directory, draft, checkpoint):
    from safetensors.torch import load_file
    directory = Path(directory)
    metadata = json.loads((directory / 'adapter.json').read_text())
    if (metadata.get('schema') != SCHEMA or not metadata.get('complete') or
            metadata.get('base_checkpoint_sha256') != file_sha(checkpoint)):
        raise ValueError('FC adapter checkpoint identity or manifest is invalid')
    weights = directory / 'fc.safetensors'
    if file_sha(weights) != metadata.get('weights_sha256'):
        raise ValueError('FC adapter weight checksum mismatch')
    if metadata.get('hidden') != draft.h or metadata.get('features') != draft.w['fc.weight'].shape[1]:
        raise ValueError('FC adapter geometry does not match the draft feature bridge')
    adapter = FcAdapter(metadata['hidden'], metadata['features'], metadata['rank'], metadata['alpha'],
                        device=draft.device)
    state = load_file(str(weights), device=draft.device)
    expected = adapter.state_dict()
    if (set(state) != set(expected) or any(state[k].shape != expected[k].shape or
            state[k].dtype != torch.float32 or not torch.isfinite(state[k]).all() for k in state)):
        raise ValueError('FC adapter weights have invalid shapes, dtype or values')
    adapter.load_state_dict(state, strict=True)
    adapter.requires_grad_(False)
    apply_fc_adapter(draft, adapter)
    return metadata
