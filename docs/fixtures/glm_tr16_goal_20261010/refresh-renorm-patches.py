from pathlib import Path
import difflib,subprocess,tempfile,hashlib,json
r=Path(__file__).resolve().parent
names=('CMakeLists.txt','src/kernels/cuda/glm.cu','src/kernels/glm_route_prune_test.cpp','src/kernels/mmvq_il_parity.cpp','src/program/glm_decode.cpp','include/strata/spec/bounded_accept.hpp','include/strata/kernels/glm_prefill.hpp')
patch='';incremental='';original={}
for name in names:
 result=subprocess.run(['git','show','e5232ca3:'+name],capture_output=True,text=True)
 original[name]=result.stdout if result.returncode==0 else None
 a=result.stdout.splitlines(True) if result.returncode==0 else [];b=(r/'renorm-source'/name).read_text().splitlines(True)
 patch+=''.join(difflib.unified_diff(a,b,fromfile='a/'+name if a else '/dev/null',tofile='b/'+name))
 old=r/'il-source'/name;a=old.read_text().splitlines(True) if old.exists() else []
 incremental+=''.join(difflib.unified_diff(a,b,fromfile='a/'+name if a else '/dev/null',tofile='b/'+name))
(r/'route-renorm.patch').write_text(patch);(r/'route-renorm-incremental.patch').write_text(incremental)
a=(r/'pre-sync-state.cpp').read_text().splitlines(True);b=(r/'renorm-source/src/program/glm_decode.cpp').read_text().splitlines(True)
(r/'sync-state-incremental.patch').write_text(''.join(difflib.unified_diff(a,b,fromfile='a/src/program/glm_decode.cpp',tofile='b/src/program/glm_decode.cpp')))
with tempfile.TemporaryDirectory(prefix='strata-renorm-patch-') as tmp:
 for name,value in original.items():
  p=Path(tmp)/name;p.parent.mkdir(parents=True,exist_ok=True)
  if value is not None:p.write_text(value)
 subprocess.run(['patch','-p1','--batch','-i',str(r/'route-renorm.patch')],cwd=tmp,check=True,stdout=subprocess.DEVNULL)
 for name in names:
  if (Path(tmp)/name).read_bytes()!=(r/'renorm-source'/name).read_bytes():raise RuntimeError('Patch does not reproduce '+name)
(r/'renorm-patch-check.json').write_text(json.dumps({'applies_to':'e5232ca3','reproduces_all_changed_files':True,'patch_sha256':hashlib.sha256((r/'route-renorm.patch').read_bytes()).hexdigest(),'files':list(names)},indent=2)+'\n')
print('Patch reproduces all changed source files')
