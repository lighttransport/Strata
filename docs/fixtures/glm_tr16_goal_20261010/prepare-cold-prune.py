"""Prepare an isolated opt-in experiment; never edit an existing measured build."""
import difflib,hashlib,json,shutil
from pathlib import Path
r=Path(__file__).resolve().parent
old=r/'gpu-cache-v2-source';new=r/'cold-prune-source'
assert not new.exists()
shutil.copytree(old,new)
p=new/'src/kernels/cuda/glm.cu';s=p.read_text()
s=s.replace('__device__ void renormalize_routes(float *weights, int k, float cutoff) {','__device__ void renormalize_routes(float *weights, int k, float cutoff, const int *ids, const float *bonus, bool cold_only) {')
s=s.replace('if (w < cutoff) { weights[j] = 0.f; removed |= w != 0.f; }','if (w < cutoff && (!cold_only || (bonus && bonus[ids[j]] <= 0.f))) { weights[j] = 0.f; removed |= w != 0.f; }')
assert s.count('float min_share, bool renormalize, const float *bonus')==4
s=s.replace('float min_share, bool renormalize, const float *bonus','float min_share, bool renormalize, bool cold_only, const float *bonus')
assert s.count('if (!renormalize && weights[j] < min_share * scale)')==2
s=s.replace('if (!renormalize && weights[j] < min_share * scale)','if (!renormalize && weights[j] < min_share * scale && (!cold_only || (bonus && bonus[ids[j]] <= 0.f)))')
s=s.replace('renormalize_routes(weights, k, min_share * scale);','renormalize_routes(weights, k, min_share * scale, ids, bonus, cold_only);')
s=s.replace('scale, min_share, renormalize, bonus, threadIdx.x','scale, min_share, renormalize, cold_only, bonus, threadIdx.x')
s=s.replace('route_min_share(), route_renormalize(), bonus','route_min_share(), route_renormalize(), route_prune_cold_only(), bonus')
pos=s.index('bool route_renormalize()')
s=s[:pos]+'''// Research option: prune only nonpositive-bonus routes. Positive affinity marks
// resident routes in the fixed-cache experiment. A null bonus disables pruning,
// preserving the draft router and untrained prefill. This is not a general
// residency query for configurations with other kinds of routing bonuses.
bool route_prune_cold_only() {
    static const bool enabled = [] {
        const char *value = std::getenv("STRATA_GLM_ROUTE_PRUNE_COLD_ONLY");
        if (!value || (value[0] == '0' && value[1] == '\\0')) return false;
        if (value[0] == '1' && value[1] == '\\0') return true;
        throw std::invalid_argument("STRATA_GLM_ROUTE_PRUNE_COLD_ONLY must be 0 or 1");
    }();
    return enabled;
}

'''+s[pos:]
p.write_text(s)
p=new/'src/kernels/glm_route_prune_test.cpp';s=p.read_text()
s=s.replace('    constexpr int top=8', '    const bool cold=std::getenv("STRATA_GLM_ROUTE_PRUNE_COLD_ONLY") && std::string(std::getenv("STRATA_GLM_ROUTE_PRUNE_COLD_ONLY"))=="1";\n    constexpr int top=8')
s=s.replace('for(int shape=0;shape<4;++shape) {','for(int shape=0;shape<4;++shape) for(int protection=0;protection<4;++protection) {')
s=s.replace('bonus[e]=(e%3)*0.01f;', 'bonus[e]=(protection==0 || protection==3) ? 0.f : protection==1 ? (e%3)*0.01f : 0.02f;')
s=s.replace('original[j]<double(share)*scale ? 0 : original[j]', 'original[j]<double(share)*scale && (!cold || (protection && bonus[ranking[j]]<=0.f)) ? 0 : original[j]')
s=s.replace('nullptr,dbonus.p)', 'nullptr,protection ? dbonus.p : nullptr)')
s=s.replace('scale,dbonus.p,dv', 'scale,protection ? dbonus.p : nullptr,dv')
s=s.replace('<<" rows="<<checks', '<<" cold_only="<<cold<<" rows="<<checks')
p.write_text(s)
p=new/'CMakeLists.txt';s=p.read_text()
a=s.index('    foreach(_route_share');b=s.index('    add_executable(glm_parity',a)
s=s[:a]+'''    foreach(_route_share 0 0.08 0.125 0.2 0.99)
      foreach(_route_renorm 0 1)
        foreach(_route_cold 0 1)
          add_test(NAME glm_route_prune_${_route_share}_${_route_renorm}_${_route_cold} COMMAND glm_route_prune_test)
          set_tests_properties(glm_route_prune_${_route_share}_${_route_renorm}_${_route_cold} PROPERTIES
            ENVIRONMENT "STRATA_GLM_ROUTE_MIN_SHARE=${_route_share};STRATA_GLM_ROUTE_RENORMALIZE=${_route_renorm};STRATA_GLM_ROUTE_PRUNE_COLD_ONLY=${_route_cold}" LABELS glm)
        endforeach()
      endforeach()
    endforeach()
'''+s[b:];p.write_text(s)
files=['CMakeLists.txt','src/kernels/cuda/glm.cu','src/kernels/glm_route_prune_test.cpp']
patch=''.join(''.join(difflib.unified_diff((old/f).read_text().splitlines(True),(new/f).read_text().splitlines(True),fromfile='a/'+f,tofile='b/'+f)) for f in files)
(r/'cold-prune-incremental.patch').write_text(patch)
(r/'cold-prune-prepared.json').write_text(json.dumps(dict(status='prepared_not_built',base='gpu-cache-v2-source',files=files,sha256={f:hashlib.sha256((new/f).read_bytes()).hexdigest() for f in files},scope='Fixed target cache with positive residency affinity only; preserve positive-bonus routes and all null-bonus routing including draft; default off'),indent=2)+'\n')
print('Prepared cold-prune source; builds and tests pending')
