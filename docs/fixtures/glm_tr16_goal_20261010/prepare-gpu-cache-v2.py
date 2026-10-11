from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'build-dense-q3.py').read_text()
a=s.index('# Repeat the anomalous');b=s.index("root=Path",a);s=s[:a]+s[b:]
a=s.index("state=json.loads");b=s.index("files=json.loads",a)
preamble='''state=json.loads((r/'pipeline-v8-state.json').read_text())
assert state['current'] is None and state['completed'][-1]['name']=='gpu-draft-retry' and state['completed'][-1]['exit_code']==1
assert 'prefix-trained decode cache requires unpacked CPU target experts' in (r/'gpu-draft-retry/aff0-depth1-gpu/000-baseline.log').read_text()
wait_idle(min(900,max(1,end-time.time()-30)))
s=r/'gpu-cache-v2-source'
if s.exists():raise RuntimeError('Refuse overwrite GPU cache v2 source')
shutil.copytree(r/'dense-q3-source',s)
p=s/'src/program/glm_decode.cpp';text=p.read_text()
old='        if (!prepared_experts.empty() || mtp_experts || gpu_decode_experts)'
new='''+repr('''        // The opt-in draft owns its weights and rollback history before cache
        // admission. It is separate from the unpacked CPU target experts.
        const bool reserved_gpu_draft = cached_gpu_mtp_enabled() && tier_owned_reserve &&
            capture_hidden && !mtp_cpu_experts && mtp_ready && verify_history &&
            !secondary && !remote_tp && !decode_cache_auto;
        if (!prepared_experts.empty() || (mtp_experts && !reserved_gpu_draft) || gpu_decode_experts)''')+'''
assert text.count(old)==1;text=text.replace(old,new);p.write_text(text)
'''
s=s[:a]+preamble+s[b:]
s=s.replace("'dense-q3", "'gpu-cache-v2").replace("r/'gpu-cache-v2-source',s", "r/'dense-q3-source',s")
# Preserve old source as copy origin; all outputs use a new namespace.
s=s.replace("(r/'gpu-cache-v2-source')", "(r/'dense-q3-source')")
(r/'build-gpu-cache-v2.py').write_text(s)
compile(s,str(r/'build-gpu-cache-v2.py'),'exec')
