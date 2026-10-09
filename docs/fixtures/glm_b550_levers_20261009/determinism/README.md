# B550 cross-process determinism (2026-10-09)

Run by the tr16 session over SSH with the step-1 BASE15 configuration and the same harness
(`tools/glm_low_memory_bench.py`, 60 GiB cgroup, decoder sha256 8c2c7542...). Each variant runs as two separate
processes for 64 tokens; `bisect-results.jsonl` lists whether the ids and the final logits are identical.

| Variant | Identical across processes |
|---|---|
| BASE15 (prefill_experts f16-batched), two separate pairs | no (first difference at token 12, then 1) |
| floating Q8 off, Q23 prefetch off, static tier, no tier, no NUMA weights, no tier + no floating Q8, layer flow off, graphs off, no tier + layer flow off | no |
| **prefill_experts mmq** | **yes, ids and final logits bit-identical** |
| prefill_experts mmq, no tier | yes |
| prefill_experts mmq, MTP1 | yes, and equal to the mmq BASE ids |

The batched FP16 prefill expert path is the only source of the nondeterminism; the MTP1 mismatch in step 1 came
from it, not from MTP. The mmq prefill run takes 38 s against 30 s for f16-batched at a 2048-token prompt.

`affinity-mmq-summary.json`: affinity re-measured with mmq prefill (512 tokens, 3 trials, base between levers):
base 10.95 / 10.98 / 10.97 tok/s (bit-identical to each other), affinity 0.06 16.49 tok/s, affinity 0.08
18.02 tok/s; both diverge from base within 4-6 tokens, so token matching does not measure their quality.
`bisect.py` and `aff_mmq.py` are the drivers.
