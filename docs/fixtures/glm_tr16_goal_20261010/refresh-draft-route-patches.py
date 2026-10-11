from pathlib import Path
import difflib,subprocess,tempfile,hashlib,json
r=Path(__file__).resolve().parent
names=('include/strata/kernels/glm.hpp','CMakeLists.txt','src/kernels/cuda/glm.cu','src/kernels/glm_route_prune_test.cpp','src/kernels/mmvq_il_parity.cpp','src/program/glm_decode.cpp','include/strata/spec/bounded_accept.hpp','include/strata/kernels/glm_prefill.hpp')
patch='';incremental='';original={}
for name in names:
 result=subprocess.run(['git','show','e5232ca3:'+name],capture_output=True,text=True)
 original[name]=result.stdout if result.returncode==0 else None
 a=result.stdout.splitlines(True) if result.returncode==0 else [];b=(r/'draft-route-source'/name).read_text().splitlines(True)
 patch+=''.join(difflib.unified_diff(a,b,fromfile='a/'+name if a else '/dev/null',tofile='b/'+name))
 old=r/'renorm-source'/name;a=old.read_text().splitlines(True) if old.exists() else []
 incremental+=''.join(difflib.unified_diff(a,b,fromfile='a/'+name if a else '/dev/null',tofile='b/'+name))
(r/'draft-route.patch').write_text(patch);(r/'draft-route-incremental.patch').write_text(incremental)
with tempfile.TemporaryDirectory(prefix='strata-draft-route-patch-') as tmp:
 for name,value in original.items():
  p=Path(tmp)/name;p.parent.mkdir(parents=True,exist_ok=True)
  if value is not None:p.write_text(value)
 subprocess.run(['patch','-p1','--batch','-i',str(r/'draft-route.patch')],cwd=tmp,check=True,stdout=subprocess.DEVNULL)
 for name in names:
  if (Path(tmp)/name).read_bytes()!=(r/'draft-route-source'/name).read_bytes():raise RuntimeError('Patch does not reproduce '+name)
(r/'draft-route-patch-check.json').write_text(json.dumps({'applies_to':'e5232ca3','reproduces_all_changed_files':True,'patch_sha256':hashlib.sha256((r/'draft-route.patch').read_bytes()).hexdigest(),'files':list(names)},indent=2)+'\n')
print('Patch reproduces all changed source files')
