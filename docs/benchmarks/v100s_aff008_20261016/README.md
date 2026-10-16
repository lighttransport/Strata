# V100s affinity 0.08 comparison, 2026-10-16 JST

The earlier **50.3616 decode / 850.218 prefill tok/s** result used affinity
**0.14**, with repeat guard 3. This comparison changes only
`STRATA_GLM_ROUTE_AFFINITY` between 0.08 and 0.14 and keeps repeat guard 3.
All six cases ran on the same allocation with the same verified binaries,
model, socket placement, cache sizes and MTP settings.

| Prompt | Decode 0.08, tok/s | Decode 0.14, tok/s | Prefill 0.08, tok/s | Prefill 0.14, tok/s |
| --- | ---: | ---: | ---: | ---: |
| Training code | 49.6759 | 50.1858 | 852.568 | 856.209 |
| Held-out code | 50.0360 | 49.9529 | 854.747 | 854.020 |
| Held-out chat | 52.3209 | 52.4433 | 852.182 | 853.083 |

These are warm-cache medians at 8,192 prompt tokens and 512 generated tokens.
Affinity 0.08 remains near 50 decode / 850 prefill tok/s; its training decode
median is slightly below 50. Small differences between these cases do not
establish a precise speedup, since affinity and adaptive learning can change
the generated token streams.

Hardware: **v100s node**, two Tesla V100 PCIe 32 GB GPUs, two Xeon Gold 6240
CPUs, 36 physical cores / 72 logical CPUs, four NUMA domains and a 160 GiB RAM
allocation. RAM speed was unavailable. The build uses CUDA 12.9, GNU 12.2.1
and sm70, as recorded in the [original validation](../v100s_opt_20261016/validation.json).
Weights are GLM-5.3-Flash REAP-224 Q23, 95,538,419,872 bytes, SHA-256
`3ef5726ae4a9768fa8311453927674401f3958a8c3884effddac92fc41262003`.
The engine code is `0c1598183878b989e50b540b2625a78f8b355829`.

[Results](results.json) preserve all 24 individual decode measurements, 18
prefill measurements, stream-health checks, and source, binary, prompt and raw
output hashes. [Profiles](profiles.json) contain the complete environments for
`v100s_affinity_008` and its `v100s_affinity_014` control. The recorded
per-case environment was checked against the corresponding profile; the expert
prior's connection-specific path is represented by its repository-relative path.
[Report validation](validation.json) records integrity, privacy and shutdown checks.

Each case uses one prefill warmup followed by three timed prefills, then one
complete 512-token decode warmup followed by three timed 512-token repeats from
the same prefill state. Decode rates count generated tokens after the initial
token. Staging, checksum verification, initialization and initial cache setup are
untimed. Adaptive cache learning continues across decode repeats. No profiler
or CPU test overlaps the accepted timings. The fixed case order is in
`results.json`.

All 24 streams completed 512 tokens and passed the early-stop and consecutive
repetition-cycle checks. Every timed prefill matched its warmup's last-prompt
logits and aggregate route counts. These checks do not establish coding
correctness. **HumanEval and the C++ coding task were not run in this throughput
comparison.** The stricter identical-output timing gate in
`tools/glm_v100_optimize.py` remains unmet. Soft MTP acceptance, routing affinity,
FP16 prefill operations and dense Q4 remain experimental quality settings.

For reproduction, use the [previous launcher and resident-model procedure](../v100s_opt_20261016/README.md#reproduction)
with `v100s_affinity_008` from this report's profile file. Keep a 2,048 MiB
primary expert cache, 28,672 MiB worker cache, 30,720 MiB budgets on both GPUs,
primary CPUs 0-34, worker CPU 35, 34 primary workers, MTP depth 3 and prefetch
groups 14. CPU masks must follow the actual socket and PCI topology. Use the
socket-row plan named in the profile. The timed runs reused the original sealed
95 GB socket-placement prototype; the public copier's complete 95 GB path was
not exercised in this comparison.

No engine source or production default changed. The existing backend validation
is linked above; builds were not repeated for this environment-only comparison.
The six resident-model tests were rerun locally and passed.
