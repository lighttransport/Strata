"""Offline FC-adapter calibration from committed DFlash2 protocol captures.

Artifacts are experimental model data. A calibration loss or token agreement
does not establish inference throughput or coding quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from glm_dflash2_adapter import FcAdapter, SCHEMA, file_sha
from glm_dflash2_training import HalfOperand, TrainingDraft
from glm_dflash2_worker import Draft, validate_config

BASE_SHA = 'b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac'


def child_path(root, name):
    if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
        raise ValueError('capture manifest contains an invalid relative filename')
    return root / name


class CalibrationData:
    def __init__(self, directory, hidden=4096, window=2048):
        self.directory, self.hidden, self.window = Path(directory), hidden, window
        self.samples = []
        result = json.loads((self.directory/'result.json').read_text())
        if (not result.get('complete') or result.get('decoder_returncode') != 0 or
                result.get('fresh_holdout_scored') or result.get('performance_anchor')):
            raise ValueError('teacher generation is incomplete or is not calibration data')
        prompts = self.directory/'prompts.json'
        if file_sha(prompts) != result['prompt_manifest_sha256']:
            raise ValueError('teacher prompt checksum mismatch')
        prompt_data = json.loads(prompts.read_text())
        if not prompt_data.get('frozen_before_generation') or prompt_data.get('fresh_holdout_scored'):
            raise ValueError('teacher prompts were not frozen independently')
        index = {p['index']: p for p in prompt_data['prompts']}
        capture = self.directory/'captures/capture.json'
        if file_sha(capture) != result['capture_manifest_sha256']:
            raise ValueError('capture completion checksum mismatch')
        completion = json.loads(capture.read_text())
        if not completion.get('complete') or completion.get('child_returncode') != 0 or completion['requests'] != len(result['tasks']):
            raise ValueError('capture transport is incomplete')
        for row in result['tasks']:
            prompt = index[row['index']]
            ids = child_path(self.directory, prompt['ids_file'])
            generated = child_path(self.directory, row['generated_file'])
            if file_sha(ids) != prompt['prompt_sha256'] or file_sha(generated) != row['tokens_sha256']:
                raise ValueError('teacher token checksum mismatch')
            prefix = [int(t) for t in ids.read_text().split(',') if t]
            suffix = [int(t) for t in generated.read_text().split(',') if t]
            if len(prefix) != row['prompt_tokens'] or len(suffix) != row['generated_tokens']:
                raise ValueError('teacher token count mismatch')
            tokens = prefix + suffix
            root = child_path(self.directory/'captures', row['capture_directory'])
            manifest_file = root/'manifest.json'
            if file_sha(manifest_file) != row['capture_manifest_sha256']:
                raise ValueError('request capture checksum mismatch')
            manifest = json.loads(manifest_file.read_text())
            if not manifest.get('complete') or manifest['feature_width'] != hidden*5 or manifest['embedding_width'] != hidden or manifest['embedding_rows'] != 8:
                raise ValueError('capture geometry mismatch')
            first, end = manifest['logical_first'], manifest['logical_end']
            if first != max(0, len(prefix)-window) or not len(prefix) <= end <= len(tokens):
                raise ValueError('capture absolute positions do not match teacher tokens')
            for name in ('features.f32', 'embeddings.f32'):
                path = root/name
                info = manifest['files'][name]
                if path.stat().st_size != info['bytes'] or file_sha(path) != info['sha256']:
                    raise ValueError('captured tensor checksum mismatch')
            if manifest['files']['features.f32']['bytes'] != (end-first)*hidden*5*4:
                raise ValueError('committed feature file span mismatch')
            position, embedding_offset, drafts = first, 0, []
            with (root/'features.f32').open('rb') as features, (root/'embeddings.f32').open('rb') as embeddings:
                for i, frame in enumerate(manifest['frames']):
                    start, stop = frame['start'], frame['end']
                    if (start != position or stop < start or frame['feature_byte_offset'] != (start-first)*hidden*5*4 or
                            frame['feature_bytes'] != (stop-start)*hidden*5*4 or (i == 0) != (frame['op'] == 'prepare')):
                        raise ValueError('noncontiguous committed feature frames')
                    payload = features.read(frame['feature_bytes'])
                    if hashlib.sha256(payload).hexdigest() != frame['feature_sha256']:
                        raise ValueError('feature frame checksum mismatch')
                    position = stop
                    if frame['op'] == 'draft':
                        if frame['embedding_byte_offset'] != embedding_offset:
                            raise ValueError('embedding frame offset mismatch')
                        data = embeddings.read(8*hidden*4)
                        if len(data) != 8*hidden*4 or hashlib.sha256(data).hexdigest() != frame['embedding_sha256']:
                            raise ValueError('embedding frame checksum mismatch')
                        embedding_offset += len(data)
                        if stop >= len(tokens) or frame.get('anchor') != tokens[stop]:
                            raise ValueError('draft anchor is not the next teacher token')
                        selected = frame.get('selected')
                        if not isinstance(selected, list) or len(selected) != 7:
                            raise ValueError('missing selector provenance')
                        drafts.append(frame)
                        if stop+7 < len(tokens) and stop-window >= first:
                            self.samples.append(dict(prompt=row['index'], root=root, first=first,
                                                     end=stop, embedding_offset=frame['embedding_byte_offset'],
                                                     anchor=frame['anchor'],
                                                     labels=tokens[stop+1:stop+8]))
                if position != end or features.read(1) or embeddings.read(1):
                    raise ValueError('capture contains unexplained tensor bytes')
            for previous, following in zip(drafts, drafts[1:]):
                s, t = previous['end'], following['end']
                if not 1 <= t-s <= 8 or tokens[s:t] != [previous['anchor']] + previous['selected'][:t-s-1]:
                    raise ValueError('committed teacher span does not match strict accepted proposals')
        self.result_sha256 = file_sha(self.directory/'result.json')
        if not self.samples:
            raise ValueError('capture has no fully labelled causal draft samples')

    def tensors(self, sample, device):
        h = self.hidden
        features = np.memmap(sample['root']/'features.f32', mode='r', dtype='<f4').reshape(-1, h*5)
        stop = sample['end'] - sample['first']
        context = np.array(features[stop-self.window:stop], copy=True)
        embeddings = np.memmap(sample['root']/'embeddings.f32', mode='r', dtype='<f4',
                               offset=sample['embedding_offset'], shape=(8, h))
        embedding = np.array(embeddings, copy=True)
        if not np.isfinite(context).all() or not np.isfinite(embedding).all():
            raise ValueError('nonfinite captured input')
        return (torch.from_numpy(context).to(device), torch.from_numpy(embedding).to(device),
                torch.tensor(sample['labels'], dtype=torch.long, device=device))


def restore_weights(model):
    result = dict(model.w)
    for i, qkv in enumerate(model.qkv):
        sizes = (model.heads*model.hd, model.kvheads*model.hd, model.kvheads*model.hd)
        for name, weight in zip(('q_proj', 'k_proj', 'v_proj'), qkv.split(sizes)):
            result[f'layers.{i}.self_attn.{name}.weight'] = weight
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'head', 'head-proof', 'model', 'gemm-f32', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=8)
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--seed', type=int, default=140)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--adapter-targets', choices=('fc', 'fc-mlp-down', 'fc-mlp-down-selector'), default='fc')
    p.add_argument('--selector-loss-weight', type=float, default=0.)
    p.add_argument('--prefix-decay', type=float, default=1.)
    p.add_argument('--selection-metric', choices=('token-agreement', 'selector-prefix'), default='token-agreement')
    a = p.parse_args()
    if not 1 <= a.steps <= 4096 or not 0 < a.learning_rate <= .01 or not a.device.startswith('cuda:'):
        p.error('invalid calibration limits or device')
    if not 0 <= a.selector_loss_weight <= 4 or not 0 < a.prefix_decay <= 1:
        p.error('invalid selector loss weight or prefix decay')
    if list(a.dataset.parent.parent.glob('fresh20-frozen-policy-*.json')):
        raise ValueError('final evaluation policy already frozen; no further calibration')
    checkpoint = a.model/'model.safetensors'
    if file_sha(checkpoint) != BASE_SHA:
        raise ValueError('calibration needs the recorded unmodified base draft checkpoint')
    config = validate_config(json.loads((a.model/'config.json').read_text()))
    data = CalibrationData(a.dataset)
    proof = json.loads(a.head_proof.read_text())
    if not proof.get('complete') or proof.get('performance_anchor'):
        raise ValueError('head export validation is incomplete')
    for name, info in proof['files'].items():
        if file_sha(child_path(a.head, name)) != info['sha256']:
            raise ValueError('head operand checksum mismatch')
    head_meta = json.loads((a.head/'metadata.json').read_text())
    if head_meta['valid_head_bitwise_widths'] != [1, 8] or head_meta['columns'] != 4096 or head_meta['rows'] != 154880:
        raise ValueError('unsupported calibration head')
    train = [s for s in data.samples if s['prompt'] % 4 != 3]
    evaluation = []
    for prompt in sorted({s['prompt'] for s in data.samples if s['prompt'] % 4 == 3}):
        choices = [s for s in data.samples if s['prompt'] == prompt]
        evaluation += [choices[0], choices[len(choices)//2]]
    if not train or not evaluation:
        raise ValueError('calibration needs independent training/evaluation prompt groups')
    a.output.mkdir(exist_ok=False)
    summary = dict(complete=False, schema='strata.dflash2.calibration.v1', fit=False,
                   performance_anchor=False, fresh_holdout_scored=False, seed=a.seed, steps=a.steps,
                   rank=a.rank, alpha=a.alpha, learning_rate=a.learning_rate,
                   training_samples=len(train), evaluation_samples=len(evaluation),
                   evaluation_prompt_rule='prompt_index modulo4 equals3; first and middle usable frames',
                   dataset_result_sha256=data.result_sha256, base_checkpoint_sha256=BASE_SHA,
                   head_proof_sha256=file_sha(a.head_proof), helper_sha256=file_sha(a.gemm_f32),
                   training_source_sha256=file_sha(Path(__file__)), records=[])
    def save():
        temp = a.output/'training.json.tmp'
        temp.write_text(json.dumps(summary, indent=2)+'\n')
        temp.replace(a.output/'training.json')
    save()
    if a.selector_loss_weight or a.prefix_decay != 1 or a.selection_metric != 'token-agreement':
        summary.update(selector_loss_weight=a.selector_loss_weight, prefix_decay=a.prefix_decay,
                       selection_metric=a.selection_metric)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.cuda.set_device(a.device)
    torch.ops.load_library(str(a.gemm_f32.resolve()))
    from safetensors import safe_open
    weights = {}
    with safe_open(checkpoint, framework='pt', device='cpu') as source:
        for name in source.keys():
            weights[name] = source.get_tensor(name).to(device=a.device, dtype=torch.float16)
            if not torch.isfinite(weights[name]).all():
                raise ValueError('nonfinite frozen draft weight')
    adapter_file, adapter_schema = 'fc.safetensors', SCHEMA
    geometry = dict(hidden=4096, features=20480)
    if a.adapter_targets == 'fc':
        adapter = FcAdapter(4096, 20480, a.rank, a.alpha, device=a.device)
        model = TrainingDraft(config, weights, a.device, adapter, cuda_helper=True)
    else:
        from glm_dflash2_linear_adapter import LinearAdapters, SCHEMA as LINEAR_SCHEMA, target_shapes
        from glm_dflash2_training import LinearTrainingDraft
        shapes = target_shapes(config, weights, include_selector=a.adapter_targets == 'fc-mlp-down-selector')
        adapter = LinearAdapters(shapes, a.rank, a.alpha, device=a.device)
        model = LinearTrainingDraft(config, weights, a.device, adapter, cuda_helper=True)
        adapter_file, adapter_schema = 'linear.safetensors', LINEAR_SCHEMA
        geometry = dict(targets=shapes)
        summary['adapter_targets'] = a.adapter_targets
    head_array = np.memmap(a.head/'weights.f16', mode='c', dtype='<f2', shape=(154880, 4096))
    head = torch.from_numpy(head_array).to(device=a.device, dtype=torch.float32)
    mask = torch.from_numpy(np.fromfile(a.head/'allowed.u8', dtype=np.uint8)).to(device=a.device, dtype=torch.bool)
    if int(mask.sum()) != head_meta['valid_rows'] or not torch.isfinite(head).all():
        raise ValueError('invalid head mask or operand values')
    # Compare the differentiated core to the serving FP32-output core before
    # any optimizer step. Packed skinny inference is qualified separately.
    sample = evaluation[0]
    features, embedding, _ = data.tensors(sample, a.device)
    reference = Draft(config, restore_weights(model), a.device, torch.float32, gemm_f32=True)
    reference.append(features, sample['end']-2048, sample['end'])
    expected = reference.forward(embedding)
    model.cache = []
    model.append(features, sample['end']-2048, sample['end'])
    actual = model.forward(embedding)
    error = (actual.detach()-expected).abs().max().item()
    if not torch.equal(actual.detach(), expected):
        raise ValueError('zero-adapter calibration core changed serving forward values: '+str(error))
    summary['zero_adapter_serving_max_abs_error'] = error
    model.cache = []
    model.last_hidden = None
    del reference, expected, actual, features, embedding
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=a.learning_rate, weight_decay=0)
    rng = random.Random(a.seed)
    torch.cuda.reset_peak_memory_stats()
    best = None
    def prediction(sample):
        features, embedding, labels = data.tensors(sample, a.device)
        model.cache = []
        model.append(features, sample['end']-2048, sample['end'])
        hidden = model.forward(embedding)
        logits = F.linear(HalfOperand.apply(hidden), head).masked_fill(~mask[None, :], float('-inf'))
        return logits, labels
    def evaluate(step):
        from safetensors.torch import save_file
        losses, correct, selector_correct = [], [], []
        with torch.no_grad():
            for item in evaluation:
                logits, labels = prediction(item)
                losses.append(F.cross_entropy(logits, labels).item())
                correct.append((logits.argmax(-1) == labels).float().cpu().tolist())
                if a.selection_metric == 'selector-prefix':
                    ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :16]
                    selected = model.select(ids, logits.gather(1, ids), item['anchor'])
                    selector_correct.append((selected == labels).cpu().tolist())
        record = dict(step=step, evaluation_loss=sum(losses)/len(losses),
                      evaluation_agreement=sum(sum(row) for row in correct)/(7*len(correct)),
                      position_agreement=[sum(row[i] for row in correct)/len(correct) for i in range(7)],
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if selector_correct:
            from glm_dflash2_selector_training import correct_prefix
            record.update(selector_mean_prefix=sum(correct_prefix(row) for row in selector_correct)/len(selector_correct),
                          selector_agreement=sum(map(sum, selector_correct))/(7*len(selector_correct)))
        dest = a.output/('step-%06d' % step)
        dest.mkdir(exist_ok=False)
        state = {n: w.detach().cpu().contiguous() for n, w in adapter.state_dict().items()}
        save_file(state, str(dest/adapter_file))
        metadata = dict(schema=adapter_schema, complete=True, **geometry, rank=a.rank, alpha=a.alpha,
                        base_checkpoint_sha256=BASE_SHA, weights_sha256=file_sha(dest/adapter_file),
                        dataset_result_sha256=data.result_sha256, head_proof_sha256=file_sha(a.head_proof),
                        calibration=record, performance_anchor=False, fresh_holdout_scored=False)
        (dest/'adapter.json').write_text(json.dumps(metadata, indent=2)+'\n')
        summary['records'].append(record)
        save()
        print(json.dumps(record), flush=True)
        return record
    started = time.monotonic()
    best = evaluate(0)
    from glm_dflash2_selector_training import calibration_loss, selection_key
    for step in range(1, a.steps+1):
        optimizer.zero_grad(set_to_none=True)
        sample = rng.choice(train)
        logits, labels = prediction(sample)
        loss, loss_diagnostics = calibration_loss(logits, labels, model=model, anchor=sample['anchor'],
                                                selector_weight=a.selector_loss_weight, prefix_decay=a.prefix_decay)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite calibration loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if step % 16 == 0:
            print(json.dumps(dict(step=step, training_loss=loss.item(), gradient_norm=float(grad), **loss_diagnostics)), flush=True)
        if step % 64 == 0 or step == a.steps:
            record = evaluate(step)
            if selection_key(record, a.selection_metric) > selection_key(best, a.selection_metric):
                best = record
    summary.update(complete=True, selected_step=best['step'], training_seconds=time.monotonic()-started,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   scope='Independent synthetic calibration prompt split. Agreement/loss do not prove tok/s or coding quality; target checkpoint is unchanged.')
    save()


if __name__ == '__main__':
    main()
