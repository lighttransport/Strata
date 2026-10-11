"""Extract the exact-state experiment independently of lossy experiments."""
import difflib,json,subprocess,tempfile
from pathlib import Path
r=Path(__file__).resolve().parent
rel='src/program/glm_decode.cpp'
original=(r/'source'/rel).read_text()
prior=(r/'pre-sync-state.cpp').read_text()
new=(r/'renorm-source'/rel).read_text()
def member(text,prefix):
 start=text.index(prefix)
 end=text.index('\n    }\n',start)+7
 return text[start:end]
s=original
# This function has no dependence on IL or approximate acceptance changes.
old_mla=member(original,'    void mla_gpu(')
assert old_mla==member(prior,'    void mla_gpu(')
s=s.replace(old_mla,member(new,'    void mla_gpu('),1)
start=s.index('    std::vector<float> mtp_step(');end=s.index('    void rebalance_decode_cache()',start)
start_new=new.index('    const bool mtp_sync_state_only =');end_new=new.index('    void rebalance_decode_cache()',start_new)
s=s[:start]+new[start_new:end_new]+s[end:]
s=s.replace(member(s,'    Checkpoint mtp_checkpoint('),member(new,'    Checkpoint mtp_checkpoint('),1)
anchor='    void check_sequences('
assert s.count(anchor)==1
s=s.replace(anchor,member(new,'    void check_mtp_state(')+'\n'+anchor,1)
anchor='        trace = StepTrace{};\n'
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'        mtp_sync_omitted = mtp_sync_checked = 0;\n',1)
anchor='        if (step_trace && trace.steps)\n'
assert s.count(anchor)==1
s=s.replace(anchor,'        if (mtp_sync_state_only)\n            std::cerr << "MTP_SYNC_STATE_ONLY omitted=" << mtp_sync_omitted\n                      << " checked=" << mtp_sync_checked << \'\\n\';\n'+anchor,1)
assert 'STRATA_GLM_MTP_ACCEPT_MARGIN' not in s
assert 'STRATA_GLM_ROUTE_RENORMALIZE' not in s
assert 'STRATA_GLM_MMVQ_IL' not in s
patch=''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
(r/'state-only-standalone.cpp').write_text(s)
(r/'state-only-standalone.patch').write_text(patch)
with tempfile.TemporaryDirectory(prefix='strata-state-patch-') as d:
 p=Path(d)/rel;p.parent.mkdir(parents=True);p.write_text(original)
 subprocess.run(['patch','-p1','--batch','--input',str(r/'state-only-standalone.patch')],cwd=d,check=True,capture_output=True)
 assert p.read_text()==s
(r/'state-only-standalone-check.json').write_text(json.dumps({'patch_reproduction':True,'standalone_build_tested':False,'source_commit':'e5232ca3f2e926ffb41f9d3ff686efd4c8cc91de','note':'Requires existing portability header fix for build. Extracted functions passed combined-experiment tests; standalone build/runtime remain pending.'},indent=2)+'\n')
print('Standalone state-only patch reproduced exactly; standalone build pending')
