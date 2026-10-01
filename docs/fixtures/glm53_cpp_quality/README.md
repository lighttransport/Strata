# GLM Q2 C++ output validation

The user accepted the earlier best median of **7.54 tok/s** and requested
quality validation instead of further decode optimization. That earlier
benchmark rate is workload-specific; this new task ran more slowly.

The model received a **4,096-token** prompt using its embedded chat template
with `reasoning_effort=low`. The task was to implement a strict C++17
`parse_u64` function in a telemetry service, with representative protocol
records, overflow rejection, ASCII-only digits, leading zeroes, and preservation
of the caller's output on failure. Generation used a **512-token cap**.

| Mode | Prefill tok/s | Decode tok/s | Output tokens including stop | Peak owned GPU MiB |
| --- | ---: | ---: | ---: | ---: |
| Single | 84.48 | 2.15 | 393 | 9055.21 |
| GPU MTP, depth 1 | 84.21 | 2.12 | 398 | 10149.90 |

Both answers reached their stop token before the cap, contain complete functions,
and provide coherent explanations. Each generated function compiled as C++17
with `-Wall -Wextra -Werror` and passed **400,532 cases** against a `std::from_chars`
oracle, including UINT64_MAX boundaries, overflow, all 256 byte values, embedded
NUL, long inputs, leading zeroes, and deterministic random inputs. ASan/UBSan
reported no errors; leak detection was disabled because the sandbox uses ptrace.

**Quality passes for this task. Exact greedy token equality does not pass.**
Outputs first differ at token index 96 (zero-based): `HI_REM` versus `LO_DIGIT`.
Other differences are comments and explanatory text. Both use the same correct
algorithm. This does not establish lossless speculative decoding on general
prompts, nor broad coding quality. No inference-kernel fix was made in this
validation task, and no cause is assigned to the token divergence without
further controlled investigation.

Configuration: Q2_K_XL four-shard model, 15 worker threads with automatic physical
core affinity, CPU main experts, GPU fixed layers and GPU MTP draft experts,
2048-token prefill batches, 8192 context, 4096 MiB dense cache,
12288 MiB total GPU budget, 2 GiB physical display-headroom guard,
no main expert cache, no CPU weight prepacking, NUMA scheduling disabled.
MTP priming took 10.23 seconds and is excluded from its decode rate; startup,
weight prefaulting, and prefill are also excluded. One run per mode was used.

Artifacts: [prompt](prompt.txt), [rendered chat](chat_prompt.txt),
[single answer](single_output.md), [speculative answer](mtp_output.md),
[single C++](generated.cpp), [speculative C++](generated_mtp.cpp),
[test harness](test_generated.cpp), and [measurement](measurement.json).

To regenerate outputs from the repository root (Python dependencies: regex and
jinja2):

```sh
python3 tools/glm_cpp_quality.py "$GLM_Q2_MODEL" --decoder build-glm/strata-glm-decode
```

The runner reports token equality separately. To extract fresh generated functions,
copy only the fenced C++ blocks into `generated.cpp` and `generated_mtp.cpp` before
rerunning the harness. The saved harness uses independent parsing as its oracle.

```sh
g++ -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined \
  docs/fixtures/glm53_cpp_quality/test_generated.cpp -o /tmp/glm-quality-test
ASAN_OPTIONS=detect_leaks=0 /tmp/glm-quality-test
g++ -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined \
  '-DGENERATED_SOURCE="generated_mtp.cpp"' \
  docs/fixtures/glm53_cpp_quality/test_generated.cpp -o /tmp/glm-quality-test-mtp
ASAN_OPTIONS=detect_leaks=0 /tmp/glm-quality-test-mtp
```
