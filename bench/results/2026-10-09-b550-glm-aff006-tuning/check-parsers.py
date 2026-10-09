"""Compile supplied parser blocks unchanged against the shared contract harness.

Run from the repository root; arguments: quality run directory, output directory.
Capped answers with a complete function remain marked length/unclosed fence.
Manual review of the full answer is still required.
"""
import hashlib,json,re,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); out=Path(sys.argv[2]) if len(sys.argv)>2 else root/'parser-checks';out.mkdir(exist_ok=True)
harness=Path('bench/results/2026-10-09-b550-glm-sweet-spot/parser-harness.cpp.in').read_text()
version=subprocess.run(['g++','--version'],capture_output=True,text=True,check=True).stdout
rows=[]; cached={}
for answer in sorted(root.glob('*/cpp-review-trial*.answer.md')):
 response=json.loads(answer.with_name(answer.name.replace('.answer.md','.response.json')).read_text())
 blocks=re.findall(r'```(?:cpp|c\+\+)\s*(.*?)(?:```|$)',answer.read_text(),re.S)
 blocks=[b for b in blocks if 'bool parse_u64' in b]
 row={'answer':str(answer.relative_to(root)), 'finish_reason':response['choices'][0]['finish_reason'], 'closed_code_fence':answer.read_text().count('```') % 2 == 0}
 if len(blocks)!=1:
  row.update(status='no_unique_complete_parser',blocks=len(blocks));rows.append(row);continue
 code=blocks[0]; digest=hashlib.sha256(code.encode()).hexdigest(); row['function_sha256']=digest
 previous_source=out/(digest[:16]+'.cpp'); previous_result=out/(digest[:16]+'.result.json')
 if previous_source.exists() and previous_result.exists() and previous_source.read_text()==harness.replace('// MODEL_FUNCTION',code):
  previous=json.loads(previous_result.read_text())
  if previous.get('compiler')==version: cached[digest]=previous
 if digest not in cached:
  source=out/(digest[:16]+'.cpp'); binary=Path('/tmp')/('strata-aff006-parser-'+digest[:16])
  source.write_text(harness.replace('// MODEL_FUNCTION',code))
  command=['nice','-n','19','g++','-std=c++17','-O2',str(source),'-o',str(binary)]
  compiled=subprocess.run(command,capture_output=True,text=True,timeout=30)
  result={'compiler':version,'compile_command':command,'compile_returncode':compiled.returncode,'compile_stdout':compiled.stdout,'compile_stderr':compiled.stderr}
  if compiled.returncode==0:
   tested=subprocess.run(['nice','-n','19',str(binary)],capture_output=True,text=True,timeout=10)
   result.update(test_returncode=tested.returncode,test_stdout=tested.stdout,test_stderr=tested.stderr)
  (out/(digest[:16]+'.result.json')).write_text(json.dumps(result,indent=2)+'\n');cached[digest]=result
 result=cached[digest];row.update(status='pass' if result.get('test_returncode')==0 else 'fail',result=digest[:16]+'.result.json')
 rows.append(row)
(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
