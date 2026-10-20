"""Real-weight CUDA checks for the optional DFlash2 FC calibration core."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from safetensors import safe_open

from glm_dflash2_adapter import FcAdapter, apply_fc_adapter, file_sha
from glm_dflash2_calibrate import BASE_SHA, restore_weights
from glm_dflash2_training import TrainingDraft
from glm_dflash2_worker import Draft, validate_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--gemm-f32', type=Path, required=True)
    p.add_argument('--skinny-library', type=Path)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--adapter-targets', choices=('fc', 'fc-mlp-down', 'fc-mlp-down-selector'), default='fc')
    a = p.parse_args()
    if not torch.cuda.is_available():
        return 77
    torch.set_num_threads(1)
    torch.manual_seed(150)
    torch.cuda.set_device(a.device)
    torch.ops.load_library(str(a.gemm_f32.resolve()))
    checkpoint = a.model/'model.safetensors'
    if file_sha(checkpoint) != BASE_SHA:
        raise ValueError('wrong base checkpoint for real-weight calibration check')
    config = validate_config(json.loads((a.model/'config.json').read_text()))
    weights = {}
    with safe_open(checkpoint, framework='pt', device='cpu') as source:
        for name in source.keys():
            weights[name] = source.get_tensor(name).to(device=a.device, dtype=torch.float16)
    apply_adapter = apply_fc_adapter
    if a.adapter_targets == 'fc':
        adapter = FcAdapter(4096, 20480, 8, 8, device=a.device)
        model = TrainingDraft(config, weights, a.device, adapter, cuda_helper=True)
    else:
        from glm_dflash2_linear_adapter import LinearAdapters, apply_linear_adapters, target_shapes
        from glm_dflash2_training import LinearTrainingDraft
        adapter = LinearAdapters(target_shapes(config, weights,
                                 include_selector=a.adapter_targets == 'fc-mlp-down-selector'), 8, 8, device=a.device)
        model = LinearTrainingDraft(config, weights, a.device, adapter, cuda_helper=True)
        apply_adapter = apply_linear_adapters
    reference = Draft(config, restore_weights(model), a.device, torch.float32, gemm_f32=True)
    frozen_names = ['fc.weight']
    if a.adapter_targets != 'fc':
        frozen_names += [f'layers.{i}.mlp.down_proj.weight' for i in range(config['num_hidden_layers'])]
    if a.adapter_targets == 'fc-mlp-down-selector':
        frozen_names.append('candidate_selector.hidden_projection.weight')
    frozen = {name: model.w[name].clone() for name in frozen_names}
    embedding = torch.randn((8, 4096), device=a.device)
    head = torch.randn((64, 4096), device=a.device)*.01
    errors, gradient_norms = [], []
    def loss_for(hidden):
        logits = F.linear(hidden, head)
        if a.adapter_targets == 'fc-mlp-down-selector':
            from glm_dflash2_selector_training import calibration_loss
            labels = logits.detach().argmax(-1)  # Every synthetic label is available to top16.
            return calibration_loss(logits, labels, model=model, anchor=42,
                                    selector_weight=1, prefix_decay=.8)[0]
        return F.cross_entropy(logits, torch.arange(7, device=a.device))
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.0003, weight_decay=0)
    for rows in (8, 2048):
        features = torch.randn((rows, 20480), device=a.device)*.1
        features[0] *= 1000000
        reference.cache = []
        reference.append(features, 0, rows)
        expected = reference.forward(embedding)
        model.cache = []
        model.append(features, 0, rows)
        actual = model.forward(embedding)
        errors.append((actual.detach()-expected).abs().max().item())
        if not torch.equal(actual.detach(), expected):
            raise ValueError('calibration forward differs from serving core at context '+str(rows))
        if a.adapter_targets == 'fc-mlp-down-selector':
            # Check the real evaluate()->train() transition: serving select()
            # is inference-only even when called on the differentiated model.
            with torch.no_grad():
                logits = F.linear(actual, head)
                ids = logits.argsort(dim=-1, descending=True, stable=True)[:, :16]
                model.select(ids, logits.gather(1, ids), 42)
            if any(weight.is_inference() for weight in model.backward_weights.values()):
                raise ValueError('evaluation cached an inference-only backward operand')
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(actual)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1, error_if_nonfinite=True)
        if float(gradient) <= 0:
            raise ValueError('feature bridge has no real-weight calibration gradient')
        gradient_norms.append(float(gradient))
        if a.adapter_targets != 'fc':
            for module in adapter.up:
                if module.weight.grad is None or not torch.isfinite(module.weight.grad).all() or module.weight.grad.abs().sum() <= 0:
                    raise ValueError('a linear adapter has no finite output gradient')
    optimizer.step()
    # A nonzero adapter must produce the same forward in training and serving.
    apply_adapter(reference, adapter)
    reference.cache = []
    reference.append(features, 0, 2048)
    expected = reference.forward(embedding)
    model.cache = []
    model.append(features, 0, 2048)
    actual = model.forward(embedding)
    if not torch.equal(actual.detach(), expected):
        raise ValueError('trained adapter forward differs between training and serving')
    optimizer.zero_grad(set_to_none=True)
    loss_for(actual).backward()
    down = [adapter.down] if a.adapter_targets == 'fc' else adapter.down
    if any(module.weight.grad is None or not torch.isfinite(module.weight.grad).all() or
           module.weight.grad.abs().sum() <= 0 for module in down):
        raise ValueError('nonzero adapters did not retain complete gradients')
    if any(not torch.equal(weight, model.w[name]) for name, weight in frozen.items()) or any(w.grad is not None for w in model.w.values()):
        raise ValueError('calibration changed frozen base weights')
    if a.adapter_targets == 'fc-mlp-down-selector':
        from glm_dflash2_selector_training import selector_scores
        with torch.inference_mode():
            logits = F.linear(expected, head)
            ids = logits.argsort(dim=-1, descending=True, stable=True)[:, :16]
            unary = logits.gather(1, ids)
            chosen = reference.select(ids, unary, 42)
            previous = torch.cat((chosen.new_tensor([42]), chosen[:-1]))
            scores = selector_scores(model, actual, ids, unary, previous)
            if not torch.equal(chosen, ids.gather(1, scores.argmax(-1)[:, None])[:, 0]):
                raise ValueError('differentiated selector scores changed serving choices')
    skinny_relative_error = None
    if a.skinny_library:
        reference.enable_skinny(a.skinny_library.resolve())
        reference.cache = []
        reference.append(features, 0, 2048)
        narrow = reference.forward(embedding)
        skinny_relative_error = ((actual.detach()-narrow).float().norm()/actual.detach().float().norm()).item()
        if not torch.isfinite(narrow).all() or skinny_relative_error > .005:
            raise ValueError('skinny serving core diverges from calibrated reference')
        if a.adapter_targets == 'fc-mlp-down-selector':
            from glm_dflash2_worker import warm_protocol, warm_incremental
            reference.enable_cuda_graph(embedding)
            warm_protocol(reference)
            cache, first, end = reference.cache, reference.start, reference.end
            warm_incremental(reference)
            if reference.cache is not cache or (reference.start, reference.end) != (first, end):
                raise ValueError('incremental warmup committed synthetic context rows')
            if not torch.equal(narrow, reference.replay(embedding)):
                raise ValueError('protocol or incremental warmup changed packed CUDA graph outputs')
            from glm_dflash2_selector_lookahead import transition_tables, path_indices, install_lookahead
            candidates = torch.arange(7*16, device=a.device).reshape(7, 16)
            unary = torch.zeros((7, 16), device=a.device)
            original = reference.select(candidates, unary, 42)
            tables = transition_tables(reference, candidates, unary, 42)
            if not torch.equal(original, candidates.gather(1, path_indices(tables, 0.)[:, None])[:, 0]):
                raise ValueError('zero-discount lookahead changed sequential selection')
            install_lookahead(reference, .25)
            selected = reference.select(candidates, unary, 42)
            expected = candidates.gather(1, path_indices(tables, .25)[:, None])[:, 0]
            if not torch.equal(selected, expected) or not torch.equal(narrow, reference.replay(embedding)):
                raise ValueError('lookahead selector or graph state differs')
    print(json.dumps(dict(passed=True, gpu=torch.cuda.get_device_name(a.device), adapter_targets=a.adapter_targets,
                          zero_adapter_context_rows=[8, 2048], zero_adapter_max_abs_errors=errors,
                          nonzero_adapter_serving_equal=True, gradient_norms=gradient_norms,
                          base_weights_unchanged=True, skinny_relative_l2_error=skinny_relative_error,
                          selector_lookahead_checked=a.adapter_targets == 'fc-mlp-down-selector',
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(), fit=False,
                          performance_anchor=False, checkpoint_sha256=BASE_SHA)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
