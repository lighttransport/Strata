"""Integrate only the independently verified opt-in MTP state synchronization change."""
import hashlib,json,subprocess,tempfile,time
from pathlib import Path
root=Path('/mnt/nvme02/work/Strata');r=root/'build-tr16-goal-20261010'
proof=json.loads((r/'state-only-build-complete.json').read_text())
assert proof['cuda_parities_passed'] and proof['hip_build_passed']
for path,expected in proof['sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
rows=json.loads((r/'proposal-model-controls/summary.json').read_text())
for name in ('00-standalone-default','01-standalone-state-only'):
 selected=[v for v in rows if v['name']==name];assert len(selected)==1 and selected[0]['tokens_identical']==256
state=json.loads((r/'proposal-model-controls/state-diagnostic-8k.json').read_text());assert state['passed'] and len(state['checked'])==2 and all(v[1]==16 for v in state['checked'])
p=root/'src/program/glm_decode.cpp';before=p.read_bytes()
previous=json.loads((r/'state-only-integration-check.json').read_text())
assert hashlib.sha256(before).hexdigest()==previous['root_decoder_sha256'],'Shared decoder changed; inspect before integrating'
patch=r/'state-only-standalone.patch';assert hashlib.sha256(patch.read_bytes()).hexdigest()==previous['patch_sha256']
marker=r/'state-only-integration.json';assert not marker.exists(),'Already integrated'
with tempfile.TemporaryDirectory(prefix='strata-state-integration-') as tmp:
 t=Path(tmp)/'src/program/glm_decode.cpp';t.parent.mkdir(parents=True);t.write_bytes(before)
 subprocess.run(['patch','-p1','--batch','--fuzz=0','-i',str(patch)],cwd=tmp,check=True,stdout=subprocess.DEVNULL)
 expected=t.read_bytes()
subprocess.run(['git','apply','--check',str(patch)],cwd=root,check=True)
assert p.read_bytes()==before,'Shared decoder changed during preflight'
(r/'state-only-root-before.cpp').write_bytes(before)
subprocess.run(['git','apply',str(patch)],cwd=root,check=True)
after=p.read_bytes();assert after==expected,'Integrated source differs from reviewed application'
subprocess.run(['git','diff','--check'],cwd=root,check=True)
marker.write_text(json.dumps({'time':time.time(),'source_integrated':True,'default':'off','concurrent_deferral_edits_preserved':True,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),'independent_cuda_hip_built':True,'independent_model_tokens_compared':512,'independent_full_state_checks_8k':32,'assembled_root_builds_pending':True,'quality_qualified':False},indent=2)+'\n')
print('Integrated state-only opt-in; existing deferral edits preserved. Assembled builds pending.')
