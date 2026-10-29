# Sourced by final_screen.sh at an inference-free boundary.
export W_ARGS='--threads=16 --weights-dir=/dev/shm --reserve-gib=3 --gpu-reserve-mib=1024'
for repeat in 1 2; do
 for q8 in 0 1; do
  check_time || exit 0
  bash production_trial.sh "t41_draftq8_${q8}_r$repeat" STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=$mode STRATA_DS41_DSPARK_Q8=$q8
  qualified "t41_draftq8_${q8}_r$repeat"
 done
done
q8=$(python3 - <<'PY'
import pathlib,re,json,statistics
r=pathlib.Path('logs');rates={}
for mode in (0,1):
 rates[mode]=[float(re.search(r'decode_tok_s=([0-9.]+)',(r/f't41_draftq8_{mode}_r{i}.log').read_text())[1]) for i in (1,2)]
best=int(statistics.mean(rates[1])>1.01*statistics.mean(rates[0]) and min(rates[1])>=min(rates[0]))
(r/'t41_draft_selection.json').write_text(json.dumps({'rates':rates,'selected':best,'scope':'draft matrices only; target verification unchanged'},indent=2)+'\n');print(best)
PY
)
export STRATA_DS41_DSPARK_Q8=$q8
if [[ $q8 == 1 ]]; then
 check_time || exit 0
 DS41_GATE_WORKER_ARGS="$W_ARGS" bash cpp_gate/production_run.sh t41_cpp_draftq8 STRATA_DS41_MERGE_DIRECT=1 STRATA_IQ1S_GRID8=$mode STRATA_DS41_DSPARK_Q8=1
 python3 - <<'PY'
import pathlib,json,re
r=pathlib.Path('logs')
assert json.loads(pathlib.Path('cpp_gate/t41_cpp_draftq8.grade.json').read_text())['passed']
assert re.search(r'^tokens: .+$',(r/'t31_cpp_exact.log').read_text(),re.M)[0]==re.search(r'^tokens: .+$',(r/'t41_cpp_draftq8.log').read_text(),re.M)[0]
PY
fi
