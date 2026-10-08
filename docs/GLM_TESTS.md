# GLM-5.3-Flash tests

Three ctest labels cover the GLM engine. Model tests need the files listed in `tests/glm/regression.json`
(override with `STRATA_GLM_TEST_MODEL`, `STRATA_GLM_TEST_PACK`, `STRATA_GLM_TEST_EVAL_REFERENCE`) and report
"skipped" when they are missing. They share one resource lock, so `ctest -j` never loads the weights twice.

```sh
cmake -S . -B build-glm -DSTRATA_GLM_MODEL_TESTS=ON      # model tests are off by default
ctest --test-dir build-glm -L '^glm$'      # kernels and logic on synthetic data, ~1 min (-L is a regex)
ctest --test-dir build-glm -L glm_model    # real model, short context, ~3 min
ctest --test-dir build-glm -L glm_quality  # held-out quality gate, ~9 min
```

| Label | Tests | What fails it |
|---|---|---|
| `glm` | glm_parity, glm_prefill_parity, glm_kda_column_tiles_parity, glm_q23_parity, glm_mailbox_test, glm_q8_test, prefill_mmq_q23_test, q23_test, q23_pool_test, canon_expert_test, flow_pool_test, numa_pool_test, glm_regression_unit | kernel or pool output differs from its reference |
| `glm_model` | decode, check_replay, check_verify, check_decode_graphs, mtp_split_verify, mtp_adaptive_tier (48-token prompt, 16 greedy tokens, q23 pack, canonical mode); glm_native_expert_parity, glm_quant_parity, glm_cache_parity on the real shard | any of the 16 tokens differs from the golden list, or a decoder self-check throws |
| `glm_quality` | glm_quality_gate: 16 held-out sequences (8,192 tokens) teacher-forced against the BF16 baseline logits | KL above baseline x 1.002 + 0.0002, top-1 below baseline - 0.001 (about 8 tokens), perplexity above baseline x 1.002 |

Each case runs about 25 s, nearly all of it loading. The golden tokens and the quality baseline (KL 0.1367, top-1
87.68 %, perplexity 4.825) were recorded on tr16 on 2026-10-09 and match the documented q23 numbers. In
canonical mode the evaluation is bit-for-bit repeatable (three runs gave identical KL, nll and differing-logit
counts), so the limits are tight: they absorb intended rounding changes and catch route affinity 0.05 (KL 0.143).
The gate also prints whether a run is bit-identical to the baseline. After an
intended quality change, re-record the baseline with
`python3 tools/glm_regression.py quality --decoder build-glm/strata-glm-decode --update-baseline` and the golden
tokens by editing `tests/glm/regression.json`.

Known gap: `--check-prefill` is not in the suite. Batched GPU prefill differs from sequential execution by 11-14 %
relative L2 in the KDA state of 124 buffers (the check's limit is 2 %), with the greedy token equal; the pre-merge
decoder behaves the same.
