"""Prepare an opt-in Q3 dense repack experiment without mutating any built tree."""
from pathlib import Path
import difflib,hashlib,json
r=Path(__file__).resolve().parent
source=r/'renorm-source/src/program/glm_decode.cpp'
original=source.read_text();s=original
anchor='    bool early_route_sync = std::getenv("STRATA_GLM_EARLY_ROUTE_SYNC") != nullptr;'
assert s.count(anchor)==1
s=s.replace(anchor,'''    // Separate opt-in precision experiment; Q4 and Q3 modes cannot be combined.
    const int dense_q3_repack = [] {
        const char *value = std::getenv("STRATA_GLM_DENSE_Q3");
        if (!value || std::string(value) == "0") return 0;
        if (std::string(value) == "mixers") return 1;
        if (std::string(value) == "all") return 2;
        if (std::string(value) == "q5") return 3;
        if (std::string(value) == "outputs") return 4;
        throw std::invalid_argument("GLM: dense Q3 must be 0, mixers, all, q5 or outputs");
    }();
'''+anchor)
old='''        if (!dense_q4_repack || t.shape.size() != 2'''
new='''        if (dense_q3_repack && dense_q4_repack)
            throw std::invalid_argument("GLM: dense Q3 and Q4 repacks are mutually exclusive");
        const int mode = dense_q3_repack ? dense_q3_repack : dense_q4_repack;
        if (!mode || t.shape.size() != 2'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('(dense_q4_repack == 3 && t.type != GGML_TYPE_Q5_K)', '(mode == 3 && t.type != GGML_TYPE_Q5_K)').replace('(dense_q4_repack == 4 && !name.ends_with("attn_output.weight"))','(mode == 4 && !name.ends_with("attn_output.weight"))')
old='''        return layer >= 3 && (mixer || (dense_q4_repack == 2 && shared)) ? int(GGML_TYPE_Q4_K) : int(t.type);'''
new='''        const int target_type = dense_q3_repack ? GGML_TYPE_Q3_K : GGML_TYPE_Q4_K;
        return layer >= 3 && (mixer || (mode == 2 && shared)) ? target_type : int(t.type);'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('affinity(repack_type == GGML_TYPE_Q4_K);','affinity(repack_type == GGML_TYPE_Q4_K || repack_type == GGML_TYPE_Q3_K);')
s=s.replace('const size_t workers = repack_type == GGML_TYPE_Q4_K ?', 'const size_t workers = (repack_type == GGML_TYPE_Q4_K || repack_type == GGML_TYPE_Q3_K) ?')
s=s.replace('std::cerr << (repack_type == GGML_TYPE_Q8_0 ? "FLOATING_Q8" : "DENSE_Q4")','std::cerr << (repack_type == GGML_TYPE_Q8_0 ? "FLOATING_Q8" :\n                          repack_type == GGML_TYPE_Q3_K ? "DENSE_Q3" : "DENSE_Q4")')
# GPU draft+target cache stays opt-in. Own draft weights/history before cache admission.
helper="""bool cached_gpu_mtp_enabled() {
    const char *value = std::getenv(\"STRATA_GLM_MTP_GPU_CACHE\");
    if (!value || std::string(value) == \"0\") return false;
    if (std::string(value) != \"1\")
        throw std::invalid_argument(\"GLM: GPU MTP target cache must be 0 or 1\");
    return true;
}
"""
assert s.count('bool cached_lookup_enabled() {')==1
s=s.replace('bool cached_lookup_enabled() {',helper+'bool cached_lookup_enabled() {')
old="""        if (!tier_owned_reserve || !capture_hidden || !mtp_cpu_experts ||
            (!decode_cache_budget && !decode_cache_auto)) return false;
        prepare_mtp(prompt);
        prepare_mtp_history(depth);
        return true;"""
new="""        if (!tier_owned_reserve || !capture_hidden ||
            (!mtp_cpu_experts && !cached_gpu_mtp_enabled()) ||
            (!decode_cache_budget && !decode_cache_auto)) return false;
        if (!mtp_cpu_experts && (secondary || remote_tp || decode_cache_auto))
            throw std::invalid_argument(\"GLM: cached GPU MTP requires one GPU and a fixed target cache\");
        prepare_mtp(prompt);
        prepare_mtp_history(depth);
        if (!mtp_cpu_experts) {
            if (!mtp_experts || !mtp_ready || !verify_history)
                throw std::runtime_error(\"GLM: GPU MTP memory was not owned before target cache admission\");
            std::cerr << \"GPU_MTP_CACHE reserved_before_target=1 resident_MiB=\"
                      << mtp_experts->bytes / double(MiB) << '\\n';
        }
        return true;"""
assert s.count(old)==1;s=s.replace(old,new)
old='((use_mtp && !mtp_cpu_experts) || (lookup_depth &&'
new='((use_mtp && !mtp_cpu_experts && !cached_gpu_mtp_enabled()) || (lookup_depth &&'
assert s.count(old)==1;s=s.replace(old,new)
anchor='        if (decode_cache_auto && use_mtp)'
validation="""        if (cached_gpu_mtp_enabled()) {
            const char *owned = std::getenv(\"STRATA_GLM_TIER_OWNED_RESERVE\");
            if (!use_mtp || mtp_cpu_experts || !decode_cache_mib || decode_cache_auto ||
                !owned || std::string(owned) != \"1\")
                throw std::invalid_argument(\"GLM: GPU MTP target cache requires GPU draft, fixed cache and owned reserve\");
        }
"""
assert s.count(anchor)==1;s=s.replace(anchor,validation+anchor)
(r/'dense-q3-decoder.cpp').write_text(s)
patch=''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/src/program/glm_decode.cpp',tofile='b/src/program/glm_decode.cpp'))
(r/'dense-q3-incremental.patch').write_text(patch)
(r/'dense-q3-prepared.json').write_text(json.dumps({'base_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'decoder_sha256':hashlib.sha256(s.encode()).hexdigest(),'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'built':False,'default':'off','quality_qualified':False,'gpu_mtp_cache_optin':True,'purpose':'Measure Q3 dense traffic/cache capacity and opt-in GPU draft reservation before target-cache admission; quality and speed unproven'},indent=2)+'\n')
print('Prepared Q3 decoder and incremental patch; no built source modified')
