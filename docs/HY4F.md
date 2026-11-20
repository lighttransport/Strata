# Hy4-Flash (`strata-hy4f`)

`strata-hy4f` runs Tencent's Hy4-Flash (GGUF architecture `hy_v4_flash`, 512 experts, about 12B active parameters per
token) on one GPU and the CPU. It is experimental: it reads token ids, writes token ids, and serves nothing yet.
The default Strata paths are unchanged by it.

The reference it is checked against is llama.cpp pull request #32517 (`src/models/hy-v4-flash.cpp`).

## What runs where

- **GPU (about 4.2 GB):** everything except the routed experts and the Engram table. That is the 4-stream
  hyper-connections, the attention (MLA with sinks and an output gate; sliding-window or full layers), the
  token selector of the full layers past 2048 positions, the router, the shared experts and the output head.
- **GPU expert cache (`--tier-gib`):** the routed experts used most often. By default it takes the free VRAM minus
  3 GiB, and at most 4 GiB.
- **CPU:** every other routed expert (NVFP4, 16 per token), and the Engram rows, read from the memory-mapped GGUF.

Each token runs as HIP graphs between the points where the GPU waits for the CPU's experts. Past 2048 positions
the token selector's top-k is computed on the CPU, and the step runs without graphs.

## Build and run

```sh
cmake -B build-hy4f -DSTRATA_ENABLE_HIP=ON -DGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
cmake --build build-hy4f --target strata-hy4f

build-hy4f/strata-hy4f MODEL.gguf --ids 1,2,3 --n 32 --threads 15
```

`tools/hy4f_run.py` takes text instead: it tokenizes with llama.cpp's `llama-tokenize` (vocabulary only) and
decodes the output with the GGUF's vocabulary.

| option | meaning |
|---|---|
| `--ids a,b,c` | the prompt as token ids (required) |
| `--n N` | tokens to generate (default 16); generation stops after EOS unless `--ignore-eos` |
| `--ctx N` | context length (default 2048); the prompt plus `--n` must fit |
| `--threads N` | CPU expert threads (default 15) |
| `--tier-gib G` | GPU expert cache size; `0` turns it off |
| `--tier-reserve-gib R`, `--tier-uploads U` | VRAM left free by the default size (3); cache replacements per token (8) |
| `--layers L` | run only the first L blocks (no output head) |
| `--dump DIR` | write every block's four streams and the logits of every prompt position (llama.cpp dump format) |
| `--inject REF` | with `--dump`: after each block, continue from the reference dump's streams |

| variable | meaning |
|---|---|
| `STRATA_HY4F_GRAPH=0` | run each step eagerly instead of as HIP graphs |
| `STRATA_HY4F_UNFUSED=1` | the unfused kernels (for A/B checks) |
| `STRATA_HY4F_PROF=1` | time attention, routing, experts and the rest (adds synchronizations) |
| `STRATA_HY4F_CKSUM=1` | print hashes of the logits and of every block's streams, per step |
| `STRATA_DS41_LANE_BANDS=1` | place the CPU experts' pages on the NUMA node of the threads that read them |

## Checking it against llama.cpp

The full model is 148.6 GiB, so most checks use cut models. `tools/hy4f_truncate_gguf.py` writes a GGUF with the
first L blocks (`--layers L`) or a chosen set of blocks (`--blocks 0,2,21,31`), with the output head kept and,
with `--keep-engram`, the Engram modules of the kept blocks. `tools/hy4f_compare.py` compares two dumps.
`tools/hy4f_full_check.sh MODEL.gguf LLAMA_DUMP OUT_DIR` runs the whole full-model check on an idle machine: the
llama.cpp reference, Strata end to end and per block, the comparisons, and decode speed with and without the GPU
cache, NUMA bands, 16 threads and eager steps, twice each. It stops if the kernel log shows trouble.

Measured on a Threadripper 1950X with 125 GB of RAM and an RX 9070 XT, with other jobs running on the CPU and
GPU. The reference is llama.cpp's CPU build on the same cut model and the same prompt.

| check | result |
|---|---|
| 3 blocks, 947 tokens (past the 512-token window) | mean KL 3.4e-4 |
| 3 blocks, 3761 tokens (token selector active past 2048) | mean KL 3.6e-4, top-1 97.5 % |
| 12 blocks with Engram modules 0 and 1, 405 tokens | mean KL 1.0e-3, top-1 94.8 %, stream error about 2 % |
| 12 blocks, each block on llama.cpp's input (`--inject`), 405 tokens | error per block 0.2-0.5 % (block 0: 1.1 %) |
| the same, 3750 tokens | 0.24-0.65 % per block, the same below 512, at 512-2048 and past 2048 |
| HIP graphs against eager steps | cache off: identical logits and streams; cache on (2 GiB, 12 blocks, 48 tokens): identical tokens |

Block 0's 1.1 % is the attention output projection: against a float64 computation, llama.cpp and Strata are each
0.5-0.6 % off (both quantize that projection's input), and their errors are independent.

Over many blocks the per-block differences grow. Two Strata runs that differ only in how the experts are rounded
(CPU against GPU cache) differ by about as much as Strata and llama.cpp do, so the end-to-end drift is expected and
not a single faulty block.

## Not done

- The full 42-block run, its llama.cpp comparison and its speed. The simulator (`tools/hy4f_sim.py`) forecasts
  12-18 tok/s with today's kernels on this machine; a rough estimate from the cut models on the busy machine is
  about 5.5 tok/s without the GPU cache. Neither has been measured on the full model.
- Multi-token prediction (the model's draft block), a prompt pass faster than one token at a time, and a tokenizer
  inside the runner.
- The token selector's top-k on the GPU, so that steps past 2048 positions can run as graphs too.
