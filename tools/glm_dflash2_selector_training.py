"""Optional teacher-forced selector loss and free-prefix calibration selection."""
import math

import torch
from torch.nn import functional as F


def selector_scores(model, hidden, candidates, unary, predecessors):
    """Serving selector scores given each row's explicit predecessor token."""
    p = 'candidate_selector.'
    projected = model.linear(hidden, p + 'hidden_projection')
    predecessor = model.rounded(model.w[p + 'predecessor_codebook'][predecessors].float() * projected)
    successor = model.w[p + 'successor_codebook'][candidates].float()
    transition = (predecessor[:, None, :] * successor).sum(-1)
    return unary + model.selector_scale * model.rounded(transition)


def calibration_loss(logits, labels, *, model=None, anchor=None, selector_weight=0., prefix_decay=1.):
    if (not math.isfinite(selector_weight) or not 0 <= selector_weight <= 4 or
            not math.isfinite(prefix_decay) or not 0 < prefix_decay <= 1):
        raise ValueError('invalid selector loss or prefix weighting')
    if selector_weight == 0 and prefix_decay == 1:
        return F.cross_entropy(logits, labels), {}
    weights = prefix_decay ** torch.arange(len(labels), device=logits.device, dtype=torch.float32)
    unary = (F.cross_entropy(logits, labels, reduction='none') * weights).sum() / weights.sum()
    if selector_weight == 0:
        return unary, {}
    if model is None or anchor is None:
        raise ValueError('selector loss needs the draft and its input anchor')
    candidates = logits.detach().topk(16, dim=-1).indices
    values = logits.gather(1, candidates)
    predecessors = torch.cat((labels.new_tensor([anchor]), labels[:-1]))
    scores = selector_scores(model, model.last_hidden, candidates, values, predecessors)
    matches = candidates == labels[:, None]
    valid = matches.any(-1)
    coverage = valid.float().mean().item()
    # If a teacher token is outside the available candidates, its unary loss
    # still trains the proposal head; it cannot supervise a selector choice.
    if not valid.any():
        return unary, dict(selector_target_coverage=coverage, selector_loss=None)
    target = matches[valid].long().argmax(-1)
    selected = (F.cross_entropy(scores[valid], target, reduction='none') * weights[valid]).sum() / weights[valid].sum()
    return (unary + selector_weight * selected) / (1 + selector_weight), dict(
        selector_target_coverage=coverage, selector_loss=selected.detach().item())


def correct_prefix(correct):
    return next((i for i, value in enumerate(correct) if not value), len(correct))


def selection_key(record, metric):
    if metric == 'token-agreement':
        return record['evaluation_agreement'], -record['evaluation_loss']
    if metric == 'selector-prefix':
        return record['selector_mean_prefix'], record['selector_agreement'], -record['evaluation_loss']
    raise ValueError('unknown checkpoint selection metric')
