# Ripple-65B on A64FX (experimental)

`ripple_runner` runs Ripple-65B Preview (Ricursive, a dense 80-layer transformer) on A64FX nodes, one rank per node,
with the weights split across ranks (tensor parallel). It reads the original safetensors checkpoints directly: BF16
(`ripple`) and ModelOpt NVFP4 (`ripple-fp4`; only the MLP is FP4, attention stays BF16). It takes token-id files and
decodes greedily. It has no tokenizer, sampler or server. `tools/ripple_tok.py` turns text into ids and back.

## What was measured

Fugaku, 6 nodes of a 12-node interactive job, 47 threads per rank, uTofu collectives, one run of each (not repeated).

| Checkpoint | Weights per rank | Peak RSS per rank | Cold load | Replay load | Decode | Prompt (batches of 8) |
|---|---|---|---|---|---|---|
| BF16 (`ripple`) | 22.9 GB | 23.3 GB | 1014 s | 19.8 s | 69 to 73 ms/token | 34 ms/token (short), 57 ms/token (2600 tokens) |
| NVFP4 MLP (`ripple-fp4`) | 11.3 GB | 11.9 GB | 596 s | 19 s | 57 ms/token | 35 ms/token (short), 58 ms/token (2600 tokens) |

- Cold load reads the shards from the shared file system; replay reads the per-rank image the cold run wrote to
  `/local` (`--image DIR`).
- NVFP4 uses an int8 dot-product kernel: the loader repacks each row so that one 64-byte load yields two 64-lane nibble
  vectors, `svtbl` maps them to doubled-e2m1 int8 weights, `sdot` multiplies them with int8 activations (one scale per
  64 elements), and the F16 block scales are applied to the integer sums. This took NVFP4 decode from 82 (float kernel)
  to 57 ms/token; the MLP matmuls went from 46 to 22 ms/token. NVFP4 decode time now (ms/token, rank 0): MLP gate+up 15,
  MLP down 8, attention projections 13 (BF16 in this checkpoint), communication 10, attention core 7, output projection 4.
  The repacked weights are 11.3 GB per rank rather than 10.8 because the scales are F16 instead of E4M3. The loader's
  scratch buffers are kept between calls on purpose: freeing them every call made the attention matmuls 50% slower
  (their pages were recycled from the scratch, already homed in one memory group).
- Where BF16 decode time goes (ms/token, rank 0): MLP gate+up 23, MLP down 12, attention projections 13, communication
  10, attention core 7, output projection 4. About 50 ms of the 70 is the weight stream, roughly 460 GB/s.
- How decode went from 190 to 70 ms/token: weights are first-touched by the thread that later reads them, so they
  spread over the four memory groups of the node (190 to 101, with the same tokens); uTofu instead of MPI (101 to 95);
  the MLP activation loop run in parallel and one parallel region for short-context attention (95 to 81); RoPE tables
  built once per token and the QK-norm loop in parallel (81 to 70). Prompt processing in batches of 8 tokens took the
  prompt from 92 to 34 ms/token.

### Output quality

- Prompts through the chat template, greedy, up to 300 tokens, BF16: a correct `fib` function with the right time and
  space complexity; "210 ÷ 70 = 3 hours. 3:15 pm + 3 hours = 6:15 pm."; a three-paragraph explanation of Rayleigh
  scattering; "日本の首都は東京都です。". Each ends with `<|im_end|>` and stops there. NVFP4 gives the same code answer,
  the same arithmetic and Japanese answers, and an explanation with different wording.
- Perplexity on 2600 tokens of this repository's text (`AGENTS.md` + `docs/DETAILS.md`): BF16 9.999, NVFP4 10.085
  (+0.9%); the float NVFP4 kernel gave 10.080, so quantizing the activations to int8 costs 0.04% here. Three of four
  300-token NVFP4 generations are token-identical between the float and the int8 kernel; the explanation prompt differs
  in wording. Batches of 8 and batches of 1 give identical perplexity (9.99865) over the sliding window and the global
  layers. This text is not a standard benchmark; it only compares the two checkpoints and the code paths.
- Logits against the float32 numpy reference (`tools/ripple_ref.py`) for the first layers, BF16: 5 layers, relative L2
  error 1.8e-4 to 4.2e-4 on 8 positions; 20 layers, 4.7e-5 to 1.3e-4 on 20 positions. Top-1 equal at every position.
  NVFP4 against the same reference with dequantized FP4 MLP weights, 5 layers: 1.9e-4 to 4.1e-4, top-1 equal. The FP4
  checkpoint differs from the BF16 one by 2 to 5% relative L2 in the same 5 layers.
- Not checked: logits of all 80 layers against a reference (none fits in reach), and any standard benchmark.
- The optimisations did not change the output: the four BF16 prompts above give byte-identical token streams to the
  first, unoptimised version (hundreds of tokens each).

## Layout

Two all-reduces per layer. q and gate projections are split by heads (80 heads over the ranks), `o_proj` by the matching
columns. `k_proj`, `v_proj` and the KV cache (fp16) are replicated on every rank (8 kv heads do not divide 6). The MLP
width is split in rows for gate/up and in columns for down. `lm_head` and the token embedding are split by vocabulary.
Any rank count works. BF16 needs about 23 GB per rank at 6 ranks, close to the 25 GiB watchdog; use more ranks if the
context is long.

Model details the runner follows (from the checkpoint's `modeling_ricursive.py`): embedding scaled by `embed_scale`;
four RMSNorms per layer; per-head q/k RMSNorm; sigmoid-gated attention output; a value residual
(`v_proj(lerp(embedding, normed input, sigmoid(lambda)))`); sliding window 1024 with RoPE on 64 layers and global
attention without RoPE on every fifth layer; MLP with silu and clamp at 20.

## Run it

The scripts live in `priv/a64fx-ripple/` (`priv/` is not tracked). `stage.sh` uploads the source with tar to a fresh
directory (nodes cache directory listings). `build_node.sh` builds on every node and stops unless the kernel test prints
`RIPPLEK fails=0`. `run6.sh NAME [args]` starts the ranks with the RSS watchdog.

```sh
tools/jobctl/jobctl --config interactive.json launch ripple --interactive --nodes 12 --time 06:00:00   # rscgrp=int
T=$(priv/a64fx-ripple/stage.sh)
# on the job (jobctl run ripple ...):
mpiexec -np 12 -std-proc OUT/b bash ~/work/$T/priv/a64fx-ripple/build_node.sh
COMM=utofu NP=6 bash priv/a64fx-ripple/run6.sh NAME --context 1024 --image /local/ripple/bin \
  --eval ids.txt --generate 48        # MODEL=$HOME/models/ripple-fp4 for NVFP4
```

Options: `--layers N`, `--context N`, `--eval IDS[,IDS] --tokens N`, `--generate N`, `--batch N` (prompt batch, 1 to 8,
default 8), `--no-score` (prompt only: no head for the prompt tokens), `--no-eos`, `--steps N --token T`, `--no-head`,
`--dump-logits FILE`, `--image DIR`.

## Limits

- Generation stops at the checkpoint's end-of-sequence tokens (`--no-eos` keeps going).
- NVFP4 activations are quantized to int8 per 64 elements, and `input_scale` (the checkpoint's static activation scale) is
  not applied. The kernel test bounds the quantization error on synthetic data with outliers at 1.3% relative L2 per
  matrix; the perplexity above is the end-to-end check. Prompt batches reuse the same kernel, one token at a time per row.
- Attention for contexts over 384 keys uses plain loops over three parallel regions; fine to 2600 tokens, not tuned for
  longer contexts. The KV cache is fp16 and replicated, 0.54 GB per rank at 4096 context.
- Sampling is greedy only; no server.
- A kernel test (`tools/a64fx_probe/ripple_kernel_test.c`) checks the NVFP4 rows against a double reference
  (relative error below 2e-5 on random rows).
