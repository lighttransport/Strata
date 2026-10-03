# GLM-5.3-Flash V100 prefill and decode validation

See [the Xeon Gold and Tesla V100 measurement report](GLM53_V100.md) for
hardware, measured throughput, memory limits and reproduction commands.

The validated 16K single-batch prefill result is 790.47 tok/s with KDA
column tiling. Regular 512-token decode reaches 28.91 tok/s; MTP depth 1
reaches 29.06 tok/s. The 40 / 1000 tok/s targets remain unmet.

The prefill bucket fix uses keyed storage rather than indexing a fixed
array, covering expert row counts above 8192. KDA column tiling preserves
the original input-row reduction order while splitting independent output
columns across blocks. Its component checks compare exact output and
retained state bits; model memcheck and repeated 512-token IDs also pass.

Native two-token expert kernels passed real-weight component parity and
model verification/rollback checks, but their component speedup gave little
end-to-end MTP gain. They remain opt-in. Dense two-row packing and fused
RMS integration were rejected and removed. The unqualified stable peer-buffer
and lookup-snapshot prototypes were removed during source cleanup.

All reported coding responses pass the strict C++17 build and 400,532
independent oracle cases. Instrumented profiles are diagnostics and are
excluded from throughput results.
