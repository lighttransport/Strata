# GLM Q2 C++ output validation: initial results

The model received a **4096-token** prompt using its embedded chat template with low reasoning effort. The task was to implement a strict C++17 `parse_u64` function with overflow rejection, ASCII-only digits, leading zeroes and unchanged output on failure. Generation used a **512-token cap** and ended naturally at **393 tokens including stop**.

| Mode | Decode trial tok/s | Decode median |
| --- | --- | ---: |
| Single | 7.263, 7.287, 7.256 | **7.26 tok/s** |
| GPU MTP depth 1 | 8.362, 8.347, 8.305 | **8.35 tok/s** |

All six outputs are token-identical. The generated parser compiled with C++17 `-Wall -Wextra -Werror` and passed **400,532 cases** against an independent `std::from_chars` oracle under ASan/UBSan. Tests cover UINT64_MAX boundaries, overflow, all 256 byte values, embedded NUL, leading zeroes, long inputs and deterministic random data. Leak detection was disabled for sandbox compatibility. This validates one coding task, not broad coding quality or universal numerical equivalence.

Competing CPU activity was sampled once per second without stopping other processes. Decode averages were 0.59 core for single and 0.38 core for MTP, with no individual competing process consuming a full core. Within-mode rate spreads were below 0.7%. Each mode prefills once and repeats decode three times from the same state; prefill and MTP priming are excluded from decode timing. Separate three-trial warm prefill medians are **98.29 tok/s at batch 2048** and **158.25 tok/s at batch 4096**, using this same prompt. See the [prefill benchmark and raw results](../../README_GLM53_FLASH.md#initial-warm-4k-prefill).

Configuration: Q2_K_XL four-shard model, Threadripper 1950X with 15 worker threads plus the host and automatic physical-core affinity, 160 GB DDR4, RTX 5060 Ti 16 GB, CPU main experts, GPU fixed layers, optional GPU MTP draft, prefill batch 2048, context 8192, dense cache 4096 MiB, total GPU budget 12288 MiB and physical free-memory guard. No main expert cache, CPU prepacking or experimental NUMA scheduling is enabled. MTP priming took 8.87 seconds.

Artifacts: [prompt](prompt.txt), [rendered chat](chat_prompt.txt), [single answer](single_output.md), [MTP answer](mtp_output.md), [single C++](generated.cpp), [MTP C++](generated_mtp.cpp), [test harness](test_generated.cpp), [quality measurement](measurement.json), and [initial decode measurements](../../glm53_flash_q2_initial_decode_measurement.json).

To repeat the measurements from the repository root on Linux (Python dependencies: `regex` and `jinja2`):

```sh
python3 tools/glm_decode_stability.py "$GLM_Q2_MODEL" --output /tmp/glm-stability.json
```

Runtime answers and logs remain in the printed temporary directory. Extract the fenced C++ blocks into `generated.cpp` and `generated_mtp.cpp` before testing freshly generated output.

```sh
g++ -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined \
  docs/fixtures/glm53_cpp_quality/test_generated.cpp -o /tmp/glm-quality-test
ASAN_OPTIONS=detect_leaks=0 /tmp/glm-quality-test
g++ -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined \
  '-DGENERATED_SOURCE="generated_mtp.cpp"' \
  docs/fixtures/glm53_cpp_quality/test_generated.cpp -o /tmp/glm-quality-test-mtp
ASAN_OPTIONS=detect_leaks=0 /tmp/glm-quality-test-mtp
```
