"""Optional discounted path selection for the DFlash2 candidate codebooks."""
import math
import torch


def path_indices(tables, discount):
    if not math.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError('discount must be finite and within zero to one')
    if not tables or tables[0].ndim != 2 or tables[0].shape[0] != 1:
        raise ValueError('first transition table needs one predecessor')
    for before, after in zip(tables, tables[1:]):
        if after.ndim != 2 or before.shape[1] != after.shape[0]:
            raise ValueError('incompatible transition-table widths')
    future = tables[-1].new_zeros(tables[-1].shape[1])
    choices = [None]*len(tables)
    for row in range(len(tables)-1, 0, -1):
        values = tables[row] + discount*future[None, :]
        future, choices[row] = values.max(-1)
    index = (tables[0][0] + discount*future).argmax()
    result = [index]
    for row in range(1, len(tables)):
        index = choices[row][index]
        result.append(index)
    return torch.stack(result)


@torch.inference_mode()
def transition_tables(model, candidates, unary, anchor):
    """Preserve serving multiplication/reduction order for each predecessor."""
    if candidates.ndim != 2 or unary.shape != candidates.shape:
        raise ValueError('candidate and unary dimensions must agree')
    p = 'candidate_selector.'
    projected = model.linear(model.last_hidden, p+'hidden_projection')
    tables = []
    for row in range(len(candidates)):
        previous = (torch.tensor([anchor], device=candidates.device) if row == 0
                    else candidates[row-1])
        predecessor = model.rounded(model.w[p+'predecessor_codebook'][previous].float()*projected[row])
        successor = model.w[p+'successor_codebook'][candidates[row]].float()
        transition = (predecessor[:, None, :]*successor[None, :, :]).sum(-1)
        tables.append(unary[row][None, :] + model.selector_scale*model.rounded(transition))
    return tables


@torch.inference_mode()
def select_candidates(model, candidates, unary, anchor, discount):
    tables = transition_tables(model, candidates, unary, anchor)
    return candidates.gather(1, path_indices(tables, discount)[:, None])[:, 0]


def install_lookahead(model, discount):
    """Replace proposal selection only when a positive discount is requested."""
    if not math.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError('lookahead discount must be finite and within zero to one')
    if discount == 0:
        return
    if not model.gemm_f32:
        raise ValueError('selector lookahead requires the FP32-output serving core')

    @torch.inference_mode()
    def select(candidates, unary, anchor):
        return select_candidates(model, candidates, unary, anchor, discount)
    model.select = select
