# Q2 coding checks on the two-node Threadripper

These answers came from the ordinary Q2 coding trials described in the
[measurement report](../../GLM_Q2_NUMA_DECODE.md). All MTP and repetition IDs matched.
The prompt definitions are in `tools/glm_q2_coding_bench.py`.

Each answer contains the generated reasoning and one complete C++17 code block.
The independent checks compiled the code with `-Wall -Wextra -Werror` and passed
400,538 primality, 2052 JSON and 2058 CSV cases. Reproduce from the repository root:

```sh
python3 tools/glm_q2_coding_check.py docs/fixtures/glm53_q2_numa_decode
```

The JSON checker uses Python's JSON parser, including all byte values and raw-byte
preservation. CSV tests use Python's writer/reader for valid records and explicit
invalid records to check failure and preservation of the caller's output.
These fixtures do not establish broad model quality.
