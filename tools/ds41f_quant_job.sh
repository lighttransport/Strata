#!/bin/bash
# DeepSeek V4.1 Flash quantization on the xeon+tesla Slurm GPU node (partition gpu1: 36 CPUs, 2x V100, 160-181 GB, 180 min, ~100 MB/s storage).
# Streamable and resumable: every job quantizes whole layers into a chunk directory, stops cleanly before the
# allocation ends (strata-ds41f-pack exits 3 = "more to do"), and requeues itself until the layer range is complete.
#
#   DS41F_CKPT=/path/ckpt DS41F_CHUNKS=/path/experts-iq1s.d DS41F_FORMATS="--gate=iq1_s --up=iq1_s --down=iq1_s --imatrix=/path/imatrix.gguf" \
#       sbatch tools/ds41f_quant_job.sh                    # first job; later jobs are submitted by the script itself
#   tools/ds41f_quant_job.sh --dry-run                      # print the plan without Slurm (uses a fake scontrol if SLURM_JOB_ID is unset)
#
# Environment (all optional except the first three):
#   DS41F_CKPT      checkpoint directory (the shards; fetched by tools/ds41f_fetch.py, which may run first inside the job)
#   DS41F_CHUNKS    chunk directory (plan.json, blk.*.chunk, *.chunk.json)
#   DS41F_FORMATS   the pack tool's format/imatrix arguments, e.g. "--gate=q2_k --up=q2_k --down=q3_k --uncalibrated"
#   DS41F_LAYERS    layer range, default 0-39
#   DS41F_THREADS   default = SLURM_CPUS_PER_TASK or 36
#   DS41F_PACK      the strata-ds41f-pack binary, default build-ds41f/strata-ds41f-pack next to the repo root
#   DS41F_FETCH     "1" to run tools/ds41f_fetch.py --out $DS41F_CKPT first (needs the venv python in DS41F_PYTHON)
#   DS41F_MARGIN_S  seconds kept free before EndTime, default 600
#   DS41F_MAX_JOBS  stop requeueing after this many jobs, default 20
#   DS41F_SBATCH    extra sbatch arguments for the requeue, default "--partition=gpu1 --time=180 --cpus-per-task=36 --mem=160G --gres=gpu:2"
#SBATCH --job-name=ds41f-quant
#SBATCH --partition=gpu1
#SBATCH --time=180
#SBATCH --cpus-per-task=36
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --output=ds41f-quant-%j.log
set -u
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
: "${DS41F_CKPT:?set DS41F_CKPT}"; : "${DS41F_CHUNKS:?set DS41F_CHUNKS}"; : "${DS41F_FORMATS:?set DS41F_FORMATS}"
LAYERS=${DS41F_LAYERS:-0-39}
THREADS=${DS41F_THREADS:-${SLURM_CPUS_PER_TASK:-36}}
PACK=${DS41F_PACK:-$here/build-ds41f/strata-ds41f-pack}
MARGIN=${DS41F_MARGIN_S:-600}
MAXJ=${DS41F_MAX_JOBS:-20}
SBATCH_ARGS=${DS41F_SBATCH:-"--partition=gpu1 --time=180 --cpus-per-task=36 --mem=160G --gres=gpu:2"}
JOBS_DONE=${DS41F_JOBS_DONE:-0}

# Deadline from Slurm's EndTime (ISO local time); without Slurm (dry run, local test) 3 hours from now.
end_epoch() {
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        local end; end=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o 'EndTime=[^ ]*' | head -1 | cut -d= -f2)
        if [ -n "$end" ] && [ "$end" != "Unknown" ]; then date -d "$end" +%s && return; fi
    fi
    echo $(( $(date +%s) + 10800 ))
}
END=$(end_epoch); DEADLINE=$((END - MARGIN)); NOW=$(date +%s)
echo "ds41f-quant job=${SLURM_JOB_ID:-none} host=$(hostname) now=$(date -d @$NOW +%FT%T) end=$(date -d @$END +%FT%T) deadline=$(date -d @$DEADLINE +%FT%T) budget_s=$((DEADLINE - NOW)) jobs_done=$JOBS_DONE"
echo "  ckpt=$DS41F_CKPT chunks=$DS41F_CHUNKS layers=$LAYERS threads=$THREADS formats=$DS41F_FORMATS"
PACK_CMD=("$PACK" experts "$DS41F_CKPT" "--chunk-dir=$DS41F_CHUNKS" $DS41F_FORMATS "--layers=$LAYERS" "--threads=$THREADS" "--deadline=$DEADLINE")
echo "  pack: ${PACK_CMD[*]}"
if [ "$DRY" = 1 ]; then
    echo "  requeue on exit 3: sbatch $SBATCH_ARGS --export=ALL,DS41F_JOBS_DONE=$((JOBS_DONE + 1)) $0"
    echo "  complete when: $PACK assemble $DS41F_CHUNKS --check  exits 0"
    exit 0
fi
if [ "${DS41F_FETCH:-0}" = 1 ]; then
    "${DS41F_PYTHON:-python3}" "$here/tools/ds41f_fetch.py" --out "$DS41F_CKPT" --layers "$LAYERS" || { echo "fetch failed"; exit 1; }
fi
"${PACK_CMD[@]}"; rc=$?
echo "pack exit=$rc at $(date +%FT%T)"
if [ $rc -eq 0 ]; then
    "$PACK" assemble "$DS41F_CHUNKS" --check && echo "DS41F_QUANT_COMPLETE chunks=$DS41F_CHUNKS" && exit 0
    echo "pack returned 0 but the range is incomplete"; exit 1
fi
if [ $rc -eq 3 ]; then
    if [ "$JOBS_DONE" -ge "$MAXJ" ]; then echo "max jobs reached; not requeueing"; exit 3; fi
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "requeueing: more layers remain"
        sbatch $SBATCH_ARGS --dependency=afterany:$SLURM_JOB_ID --export=ALL,DS41F_JOBS_DONE=$((JOBS_DONE + 1)) "$0" || { echo "sbatch failed"; exit 1; }
    fi
    exit 3
fi
exit $rc
