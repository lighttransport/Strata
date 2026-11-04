# DeepSeek Coder with Engram on another machine

This opt-in Linux path keeps the Coder model shards on the GPU machine and reads its IQ4_NL Engram table from a second machine over InfiniBand. It uses `strata-dsv41`, the production DeepSeek engine. It needs no filesystem mount and does not copy the table to the GPU machine.

The first version accepts the six-shard Coder layout: five local model shards and the combined `per_layer_token_embd.weight` IQ4_NL table. Other remote table layouts are rejected. Other models and local Engram reads keep their existing path.

## Build

Build the GPU engine with `STRATA_BUILD_DSV41=ON`, `STRATA_DSV41_REMOTE_ENGRAM=ON`, and the usual backend settings. For an RX 9070 XT, select HIP architecture `gfx1201`. Build targets `strata-dsv41` and `strata-dsv41-engram`.

The table machine needs C and C++ compilers, CMake, pthreads, OpenSSL development files, and the runtime IB verbs library. It needs no GPU toolkit:

```sh
cmake -S tools/ucomm -B build-engram -DUCOMM_BUILD_ENGRAM=ON -DBUILD_TESTING=OFF
cmake --build build-engram --target strata-dsv41-engram
```

## Configure and run

Copy `configs/experimental/dsv41-coder-remote-engram.example.json` into `priv/` and edit it. Use an SSH alias. Keep connection details and real model paths in this ignored file. Run commands from the repository root. Point `engine_exe`, `helper`, and `remote_exe` at the binaries just built. The endpoint is the table machine's reachable bootstrap address and a free port. `ssh_config` optionally names an SSH configuration file.

Use a Python environment with the server's runtime dependencies. Add the HIP runtime library directory to the config's `env.LD_LIBRARY_PATH` if the engine needs it.

```sh
python tools/dsv41/remote_engram.py preflight --config priv/coder-remote.json
python tools/dsv41/remote_engram.py start --config priv/coder-remote.json
python tools/dsv41/remote_engram.py status --config priv/coder-remote.json
python tools/dsv41/remote_engram.py stop --config priv/coder-remote.json
```

Preflight checks the local Coder files against the pinned SHA-256 values and reports the owned remote process. Start launches the remote service, verifies its descriptor against the pinned table, creates a small pack, then launches the API on `127.0.0.1:8080`. Check `api.log` for readiness; a running process does not establish readiness. The service hashes the entire table once before accepting connections, so the first start can take several minutes. It keeps the file open and uses a fresh child for each client connection. Stop signals only process groups whose saved process identity still matches.

The generated profile uses 32K total context, one active request, six experts per token, 15 CPU threads, 8192-token prefill chunks, a 3072 MiB GPU reserve, a read-only Coder expert profile, mmap local experts, and a 256 MiB Engram row cache. DSpark and vision are disabled for this initial setup. The normal engine placement and prefill logic handle the remaining GPU memory.

The optional `remote_readers` setting selects 1–64 file reader threads on the table machine. Without it, the service keeps its environment setting or its default of eight. Status reports the saved reader count; an older process may report an unknown count. Stop the owned service before changing this setting. More readers are an experiment: compare exposed wait tails on the intended prompts before keeping a change.

## Manual use

On the table machine:

```sh
strata-dsv41-engram serve /models/coder/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf 192.0.2.2:29576
```

On the GPU machine:

```sh
strata-dsv41-engram describe 192.0.2.2:29576 > priv/engram.json
python tools/dsv41/pack_gguf.py --model models/coder/IQ2_XXS --out priv/coder-pack --engram-remote-manifest priv/engram.json
strata-dsv41 --pack priv/coder-pack --engram-remote 192.0.2.2:29576 --experts 6 --dspark none --vision off --serve
```

The descriptor contains the source size, full-file SHA-256, and the first bytes needed to parse the GGUF header. It contains no remote paths. The engine checks that the source identity still matches at every connection. It never opens the remote entry as a local file. `--pack-info` checks that entry's descriptor; startup checks the actual service.

Remote batches preserve exact row bytes, order, and duplicates. A failed or incomplete batch terminates the engine, so the API cannot continue a sequence with zeroed Engram rows. Idle connections send heartbeat reads on the communication worker. An interrupted client requires a new connection and fresh sequence; the service's child timeout is 60 seconds by default. In-place source changes are unsupported; stop the service before replacing weights.

Production uses IB explicitly. `UCOMM_BACKEND=udp` is reserved for explicit tests. The wire protocol is versioned and uses the native little-endian ABI on both hosts. Serve it only on the intended private network.

## Validation

CPU tests need no GPU or model:

```sh
python -m unittest discover -s tests/terminal -p test_dsv41_remote_engram.py
python -m unittest discover -s tests/terminal -p test_pack_gguf_engram.py
cmake -S tools/ucomm -B build-engram-tests -DUCOMM_BUILD_ENGRAM=ON -DBUILD_TESTING=ON
cmake --build build-engram-tests --target strata-dsv41-engram ucomm_test_engram ucomm_test_engram_native
ctest --test-dir build-engram-tests -R '^ucomm_engram$' --output-on-failure
```

The transport test checks ordered and duplicate reads, invalid bounds, the production C++ reader across the batch limit, source mismatch, and successive sessions. Hardware validation must separately record IB delivery, short and long prompts, six-route decode, cancellation, recovery, memory use, and throughput. This document makes no speed claim.


## Decode screens

With the API and other GPU work stopped, `tools/dsv41/bench_decode.py` runs an existing engine config against token IDs. It leaves the config unchanged and disables adaptive expert movement for the comparison. Keep its inputs and output in `priv/` because the evidence records actual file paths and environment overrides.

```sh
python tools/dsv41/bench_decode.py --config priv/run.json --variants priv/variants.json \
  --prompt-ids priv/prompt.ids --output priv/decode-screen --tokens 128
```

The variants file is a JSON list, for example `[{"name":"baseline"},{"name":"threads8","flags":{"--cpu-threads":8}}]`. Each result records decode throughput, available wait counters, token-output SHA-256, and equality to the first successful run. Raw output is retained. Equality on one screen is a regression check, not proof of exactness for other prompts or contexts. Qualify a speed change with repeated runs, representative prompts, and the intended context length.

Experimental decode controls remain opt-in:

- `STRATA_IQ1M_DIRECT=1` consumes IQ1_M decoded vectors immediately for single-token CPU rows. Other widths keep their existing implementation.
- `DS4_IQ_PF_COLD=1` checks Linux IQ expert mappings for residency before issuing prefetch hints. The default keeps the existing hints. Residency affects scheduling only; a cold mapping still receives the hint.
- `--dspark-vram-mib N` caps resident drafter expert bytes per GPU group and divides that allowance across the drafter blocks. Dense drafter weights and buffers still consume additional VRAM. Remaining experts use the existing host path. Without this flag, placement keeps its existing drafter-first policy.
- `--resident-route-bias X` subtracts a nonnegative value from nonresident experts' selection scores. Zero keeps the original path. This is a lossy experiment affecting both prompt reading and decode, and can severely reduce answer quality. It requires one GPU, fixed placement (`--no-expert-cache`), `--dspark none`, and `--vision off`. Each layer must have enough resident experts for its route count. The selected experts keep their original gate weights before normalization. Measure correctness as well as speed; no bias is recommended or enabled by this document.

A changed CPU/GPU placement can change greedy tokens through arithmetic rounding, even with drafting disabled. Compare target placement and token IDs against the frozen reference before describing a speculative configuration as exact. A drafter budget is a memory control, not a speed or exactness guarantee.

For controlled comparisons, `--target-vram-counts N,N,...` pins one resident expert count per target layer. Counts must match the target's layer count, fit each layer, and fit the selected GPU group budget alongside the drafter. Startup fails if they do not fit. Keep the expert profile fixed and disable adaptive movement when comparing initial placements. `DS4_PLACE_HASH=1` prints both the complete placement hash and the target-only hash, plus the target counts; identical counts alone do not establish identical expert identities. Without the flag, placement keeps its existing budget policy.

`DS4_TOKLOG` also reports exposed Engram wait quantiles. Speculative rounds divide the wait across the tokens they commit; these are round-amortized waits, not individual token delivery latencies. `DS4_CPU_HASH=1` prints compact per-call hashes of CPU expert inputs and outputs for exactness diagnosis. Hashing adds host work and should be disabled for throughput qualification.

`--lookup-draft N` with `--parallel 1` enables greedy suffix proposals from the existing GLM lookup drafter. It needs no drafter weights. `--lookup-min N` sets the minimum matching suffix (default six tokens). Every proposal passes through the target verifier; sampled requests use the existing path. Restored or replaced histories rebuild the lookup index. Both controls are off unless requested.

For a decode-only parity screen, add `--lookup-check` with `--dspark none`, one sequence, token IDs, and a positive `--n`. It prefills once, keeps that state, then runs serial decode, lookup decode, and serial decode again from that same state. It reports parity and timings for all three passes. The fixed-length check continues through EOS; measure normal response completion and coding quality separately. This isolates decode from prefill variation and does not establish repeatable prefill or exactness on other prompts.

`tools/dsv41/bench_decode.py` accepts `--reference-ids FILE` to compare against frozen engine token output rather than its first successful run. The reference must contain at least the requested number of tokens.
