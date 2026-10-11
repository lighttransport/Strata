"""Prepare default-inactive Q3 interleaved MMVQ in the isolated Q3 experiment."""
from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
p=r/'renorm-source/src/kernels/cuda/native_mmvq.cu';s=p.read_text()
anchor='struct IlQ4K {'
assert s.count(anchor)==1
s=s.replace(anchor,'''// Q3 keeps the exact native dot expression and reduction order.
// No default table entry: selected only by an explicit measured rows override.
struct IlQ3K {
    using F = Q3KTraits;
    static constexpr bool PM = false;
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        int u[4][NC];
        float d8[4][NC];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            x.u(kby + r.bq8_offset + i, iqs % 8, u[i]);
            x.scales(kby + r.bq8_offset + i, d8[i]);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int uc[4] = {u[0][c], u[1][c], u[2][c], u[3][c]};
            const float dc[4] = {d8[0][c], d8[1][c], d8[2][c], d8[3][c]};
            out[c] = q3_q8_dot_impl(r.vl, r.vh, uc, r.scales, r.scale_offset, r.d, dc);
        }
    }
};
'''+anchor)
s=s.replace('(ggml_type == 23 || ggml_type == 12 || ggml_type == 13 || ggml_type == 14);','(ggml_type == 23 || ggml_type == 11 || ggml_type == 12 || ggml_type == 13 || ggml_type == 14);')
anchor='    case 12: launch_il_rows<IlQ4K>'
assert s.count(anchor)==1
s=s.replace(anchor,'    case 11: launch_il_rows<IlQ3K>(r, weights, x_il, y, n_in, n_out, ncols, s); break;\n'+anchor)
(r/'dense-q3-native-mmvq.cu').write_text(s)
p=r/'renorm-source/src/kernels/mmvq_il_parity.cpp';s=p.read_text()
anchor='const Case GLM_CASES[] = {';assert s.count(anchor)==1
shapes=[(256,7),(2048,4096),(4096,1536),(4096,2048),(4096,8192),(4096,12288),(4096,2051),(8192,4096),(12288,4096),(16384,4096)]
rows='\n'.join('    {"GLM Q3_K %dx%d", 11, %d, %d, 256, 110, {108, -1}},'%(a,b,a,b) for a,b in shapes)
s=s.replace(anchor,anchor+'\n'+rows)
s=s.replace('const int types[] = {23, 12, 13, 14};','const int types[] = {23, 11, 12, 13, 14};').replace('const char* names[] = {"IQ4_XS", "Q4_K", "Q5_K", "Q6_K"};','const char* names[] = {"IQ4_XS", "Q3_K", "Q4_K", "Q5_K", "Q6_K"};').replace('ti < 4; ++ti','ti < 5; ++ti')
# Require Q3 to remain inactive without explicit tuning; existing tables unchanged.
anchor='    // Volta has its own table';assert s.count(anchor)==1
s=s.replace(anchor,'''    for (int nc = 2; nc <= 4; ++nc)
        for (int n : {7, 1536, 2051, 4096, 8192, 12288})
            if (native_mmvq_il_rows_for(120, 11, nc, n) != 0) {
                ++bad; std::printf("FAIL Q3 must remain default-inactive\\n");
            }
'''+anchor)
(r/'dense-q3-mmvq-il-parity.cpp').write_text(s)
files=['dense-q3-native-mmvq.cu','dense-q3-mmvq-il-parity.cpp']
(r/'q3-il-prepared.json').write_text(json.dumps({'built':False,'default_table_unchanged':True,'activation':'Explicit STRATA_MMVQ_IL_ROWS type11 override plus existing GLM IL opt-in','sha256':{p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
print('Prepared isolated Q3 IL kernel and parity cases; not built or enabled')
