#!/usr/bin/env bash
# Run one DS4.1 prefill screen in a memory-limited systemd scope and stop on
# host pressure, low physical VRAM, OOM accounting, or a new AMD gfxhub fault.
set -euo pipefail

usage() {
  echo "usage: STRATA_DS41_PREFILL_BIN=/path/to/strata-glm-decode $0 RUN_ID MODEL_GGUF PROMPT_IDS [OUTPUT_TOKENS] [-- --check=DIR [--check-force]]" >&2
  exit 2
}

parse_uint() {
  local destination=$1 raw=$2 minimum=$3 maximum=$4 max_digits=$5 value
  [[ "$raw" =~ ^[0-9]+$ && ${#raw} -le $max_digits ]] || return 1
  value=$((10#$raw))
  (( value >= minimum && value <= maximum )) || return 1
  printf -v "$destination" '%d' "$value"
}

[[ $# -ge 3 ]] || usage
run_id=$1; model=$2; prompt=$3; steps=${4:-0}
if [[ $# -ge 4 ]]; then shift 4; else shift 3; fi
if [[ ${1:-} == -- ]]; then shift; fi
extra=(); [[ $# -eq 0 ]] || extra=("$@")
for arg in "${extra[@]}"; do
  case "$arg" in
    --check=*|--check-force) ;;
    *) echo "guard refuses extra decoder option: $arg" >&2; exit 2 ;;
  esac
done

[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid RUN_ID" >&2; exit 2; }
[[ "$prompt" != *';'* ]] || { echo "PROMPT_IDS path cannot contain ';'" >&2; exit 2; }
bin=${STRATA_DS41_PREFILL_BIN:?set STRATA_DS41_PREFILL_BIN}
[[ -x "$bin" && -r "$model" && -r "$prompt" ]] || { echo "binary, model, or prompt is unavailable" >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command -v python3 >/dev/null || { echo "refusing prefill: python3 is required for GGUF prompt preflight" >&2; exit 1; }
[[ ${DS41_PREFILL_MMQ_VALIDATE_BOUNDS:-1} == 1 ]] || {
  echo "guarded prefill requires MMQ bounds and row-ID validation" >&2; exit 2;
}
command -v numactl >/dev/null || { echo "numactl is required for the interleaved-host-memory benchmark" >&2; exit 2; }
parse_uint steps "$steps" 0 4096 4 || { echo "OUTPUT_TOKENS must be an integer in [0, 4096]" >&2; exit 2; }
repeats=${DS41_PREFILL_REPEAT:-1}
parse_uint repeats "$repeats" 1 5 1 || { echo "DS41_PREFILL_REPEAT must be in [1, 5]" >&2; exit 2; }
prompt_arg="@$prompt"
for ((i=1; i<repeats; ++i)); do prompt_arg+=";@$prompt"; done

ctx=${DS41_PREFILL_CONTEXT:-8192}
threads=${DS41_PREFILL_THREADS:-15}
batch=${DS41_PREFILL_BATCH:-4096}
gpu_expert_mib=${STRATA_DS41_GPU_EXPERT_MIB:-3800}
parse_uint ctx "$ctx" 8192 131072 6 || { echo "guarded prefill requires context in [8192, 131072]" >&2; exit 2; }
parse_uint threads "$threads" 1 128 3 || { echo "threads must be in [1, 128]" >&2; exit 2; }
parse_uint batch "$batch" 4096 131072 6 || { echo "guarded prompt batch must be in [4096, 131072]" >&2; exit 2; }
(( batch <= ctx )) || { echo "guarded prompt batch must not exceed context" >&2; exit 2; }
parse_uint gpu_expert_mib "$gpu_expert_mib" 256 8192 4 || { echo "GPU expert tier must be in [256, 8192] MiB" >&2; exit 2; }
prompt_bytes=$(wc -c < "$prompt")
[[ "$prompt_bytes" =~ ^[0-9]+$ ]] && (( prompt_bytes <= 16*1024*1024 )) || {
  echo "prompt ID file must be at most 16 MiB" >&2; exit 2;
}
model_vocab=$(python3 "$script_dir/ds41f_guard_prompt.py" "$model" "$prompt") || exit 2
[[ "$model_vocab" =~ ^[1-9][0-9]*$ ]] || { echo "refusing prefill: invalid model vocabulary size" >&2; exit 2; }
prompt_tokens=$(awk -v vocab="$model_vocab" 'BEGIN { n=0; bad=0 } { gsub(/,/, " "); gsub(/\r/, " "); for (i=1; i<=NF; ++i) {
  if ($i !~ /^[0-9]+$/ || length($i) > 10 || $i+0 > 2147483647 || $i+0 >= vocab) { bad=1; exit 1 }
  ++n; if (n > 131072) { bad=1; exit 1 }
} } END { if (!bad) print n }' "$prompt") || {
  echo "prompt file must contain at most 131072 token IDs inside the model vocabulary" >&2; exit 2;
}
[[ "$prompt_tokens" =~ ^[0-9]+$ ]] && (( prompt_tokens > 0 && prompt_tokens <= ctx && prompt_tokens + (steps > 0 ? steps - 1 : 0) <= ctx )) || {
  echo "prompt and requested output do not fit the guarded context" >&2; exit 2;
}
gpu_sysfs=${DS41_GPU_SYSFS:-}
if [[ -z "$gpu_sysfs" ]]; then
  best_total=0
  for d in /sys/class/drm/card*/device; do
    [[ -r "$d/mem_info_vram_total" && -r "$d/mem_info_vram_used" ]] || continue
    n=$(<"$d/mem_info_vram_total")
    if [[ "$n" =~ ^[0-9]+$ ]] && (( n > best_total )); then best_total=$n; gpu_sysfs=$d; fi
  done
fi
[[ -n "$gpu_sysfs" && -r "$gpu_sysfs/mem_info_vram_total" && -r "$gpu_sysfs/mem_info_vram_used" ]] || {
  echo "cannot identify AMD physical VRAM counters" >&2; exit 2;
}
command -v journalctl >/dev/null && command -v systemd-run >/dev/null && command -v systemctl >/dev/null || {
  echo "journalctl, systemd-run, and systemctl are required for a guarded run" >&2; exit 2;
}
[[ -r /sys/fs/cgroup/cgroup.controllers ]] || { echo "refusing prefill: cgroup v2 is unavailable" >&2; exit 1; }
systemctl --user show-environment >/dev/null 2>&1 || { echo "refusing prefill: the user systemd manager is unavailable" >&2; exit 1; }
if ! command -v rocminfo >/dev/null || ! rocminfo 2>/dev/null | rg 'gfx120[01]' >/dev/null; then
  echo "refusing prefill: this guarded WMMA run requires an RDNA4 gfx1200/1201 device" >&2; exit 1
fi

readonly GiB=$((1024*1024*1024))
readonly PREFLIGHT_VRAM=$((8*GiB))
readonly PREFLIGHT_RAM_KIB=$((96*1024*1024))
readonly LIVE_VRAM=$((2560*1024*1024))
readonly PRELAUNCH_GPU_MARGIN=$((1*GiB))
readonly LIVE_RAM_KIB=$((32*1024*1024))
readonly MEMORY_HIGH=$((56*GiB))
readonly MEMORY_MAX=$((64*GiB))
readonly PREFLIGHT_PSI_FULL_AVG60=10.0
readonly LIVE_PSI_FULL_AVG10=20.0
readonly PREFLIGHT_FAULT_WINDOW='24 hours ago'
readonly KERNEL_BAD_PATTERN='(\[gfxhub\].*page fault|GCVM_L2_PROTECTION_FAULT|VM_L2_PROTECTION_FAULT|GPU fault detected|GPU reset|amdgpu.*(page fault|protection fault|allocation failed|failed to allocate|VRAM.*(full|exhausted))|out of memory|oom-kill|oom_reaper|invoked oom-killer|Killed process [0-9]+|page allocation failure|strata-glm-deco.*segfault|segfault.*strata-glm-decode|general protection fault|memory fault)'
readonly JOURNAL_OOM_PATTERN='codex-ds41-decode.*Failed with result.*oom-kill|codex-ds41-prefill.*Failed with result.*oom-kill|systemd-oomd.*Killed|systemd-oomd.*Killing'
MAX_SECONDS=${DS41_PREFILL_TIMEOUT_SECONDS:-600}
parse_uint MAX_SECONDS "$MAX_SECONDS" 30 1800 4 || { echo "timeout must be between 30 and 1800 seconds" >&2; exit 2; }
readonly MAX_SECONDS

mem_available_kib() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
pressure_value() {
  local key=$1 file=$2
  awk -v key="$key" '$1 == "full" { for (i=2; i<=NF; ++i) if ($i ~ ("^" key "=")) {split($i,a,"="); print a[2]} }' "$file"
}
bad_event_since() {
  local entries next_cursor
  if ! entries=$(journalctl --after-cursor="$event_cursor" --show-cursor --no-pager -o cat 2>/dev/null); then return 2; fi
  next_cursor=$(sed -n 's/^-- cursor: //p' <<<"$entries" | tail -n 1)
  [[ -n "$next_cursor" ]] || return 2
  event_cursor=$next_cursor
  if rg -qi -- "$KERNEL_BAD_PATTERN|$JOURNAL_OOM_PATTERN" <<<"$entries"; then
    bad_event_state=1
  else
    bad_event_state=0
  fi
}

vram_total=$(<"$gpu_sysfs/mem_info_vram_total")
vram_used=$(<"$gpu_sysfs/mem_info_vram_used")
available=$(mem_available_kib)
pressure60=$(pressure_value avg60 /proc/pressure/memory)
[[ "$vram_total" =~ ^[0-9]+$ && "$vram_used" =~ ^[0-9]+$ && "$available" =~ ^[0-9]+$ && "$pressure60" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "cannot read GPU or host memory preflight metrics" >&2; exit 2;
}
required_gpu_bytes=$((gpu_expert_mib*1024*1024 + LIVE_VRAM + PRELAUNCH_GPU_MARGIN))
preflight_vram_min=$PREFLIGHT_VRAM
(( required_gpu_bytes > preflight_vram_min )) && preflight_vram_min=$required_gpu_bytes
(( vram_used <= vram_total && vram_total-vram_used >= preflight_vram_min && available >= PREFLIGHT_RAM_KIB )) || {
  echo "refusing prefill: free_vram=$((vram_total-vram_used)) required_vram=$preflight_vram_min available_ram_kib=$available" >&2; exit 1;
}
awk -v p="$pressure60" -v lim="$PREFLIGHT_PSI_FULL_AVG60" 'BEGIN {exit !(p <= lim)}' || {
  echo "refusing prefill: memory PSI full avg60=$pressure60% exceeds ${PREFLIGHT_PSI_FULL_AVG60}%" >&2; exit 1;
}
journal_base=$(journalctl -n 0 --no-pager --show-cursor 2>/dev/null) || {
  echo "refusing prefill: cannot read the system journal" >&2; exit 1;
}
event_cursor=$(sed -n 's/^-- cursor: //p' <<<"$journal_base" | tail -n 1)
[[ -n "$event_cursor" ]] || { echo "refusing prefill: system journal did not provide a cursor" >&2; exit 1; }
kernel_recent=$(journalctl --boot=all -k --since "$PREFLIGHT_FAULT_WINDOW" --no-pager -o cat 2>/dev/null) || {
  echo "refusing prefill: cannot scan recent kernel faults" >&2; exit 1;
}
journal_recent=$(journalctl --boot=all --since "$PREFLIGHT_FAULT_WINDOW" --no-pager --grep="$JOURNAL_OOM_PATTERN" -o cat 2>/dev/null) || {
  echo "refusing prefill: cannot scan recent systemd scope OOM events" >&2; exit 1;
}
if rg -qi -- "$KERNEL_BAD_PATTERN" <<<"$kernel_recent" || [[ -n "$journal_recent" ]]; then
  echo "refusing prefill: AMD fault, OOM, reset, segfault, or systemd scope OOM kill found in the last 24 hours; inspect the journals before retrying" >&2
  exit 1
fi

unit="codex-ds41-prefill-$run_id"
log="${DS41_PREFILL_LOG:-/tmp/ds41-prefill-$run_id.log}"
samples="${DS41_PREFILL_SAMPLES:-/tmp/ds41-prefill-$run_id-memory.log}"
: > "$samples"
gate_dir=$(mktemp -d "${TMPDIR:-/tmp}/ds41-prefill-${run_id}.gate.XXXXXX") || { echo "cannot create prefill launch gate" >&2; exit 1; }
gate_file="$gate_dir/release"

# Keep 2 GiB for the desktop plus 512 MiB headroom in both the engine allocator
# and the physical VRAM watchdog. Direct expert registration stays disabled
# while the historical TCP faults are unresolved.
run_env=(
  STRATA_DS41_DESKTOP_MIB=2048 STRATA_DS41_GPU_RESERVE_MIB=2560 STRATA_DS41_GPU_HEADROOM_MIB=512
  "STRATA_DS41_GPU_EXPERT_MIB=$gpu_expert_mib"
  STRATA_DS41_PIN_GB=0 STRATA_DS41_POPULATE=0 STRATA_DS41_PREFILL_HYBRID=0
  STRATA_DS41_ATTN_WMMA=2 STRATA_DS41_ATTN_BOUNDS_CHECK=1
  STRATA_MMQ_VALIDATE_BOUNDS=1
  STRATA_DS41_PREFILL_HALVES=1 STRATA_DS41_PREFILL_TILE=4096 "STRATA_DS41_PROMPT_BATCH=$batch"
  STRATA_DS41_MMQ_ROWS=256 STRATA_DS41_ENGRAM_TILE=256 STRATA_DS41_MIX_TILE=1024
  STRATA_DS41_ENGRAM_THREADS=256 STRATA_DS41_SHEXP_TILE=2048 STRATA_DS41_OUTA_TILE=4096 STRATA_DS41_SCORE_MIB=16
  STRATA_DS41_STAGE_MIB=96 STRATA_DS41_STAGE_SLOTS=3 STRATA_DS41_STREAM_MIN=256
  STRATA_DS41_KV=f16 STRATA_DS41_INDEX_GEMM_MIN=64 STRATA_DS41_DECODE_GROUPED=1 STRATA_DS41_GROW=0
  STRATA_DS41_FUSE=1 STRATA_DS41_SHARE_Q8=1 STRATA_DS41_ENGRAM_DIRECT=1
  STRATA_IQ1S_FAST=1 STRATA_Q6_V2=1
)
runtime_env=("PATH=${PATH:-/usr/bin:/bin}" "HOME=${HOME:-/tmp}" "LANG=${LANG:-C}")
for name in ROCM_PATH ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER \
            HSA_OVERRIDE_GFX_VERSION OMP_PROC_BIND OMP_PLACES OMP_WAIT_POLICY TMPDIR; do
  [[ -v $name ]] && runtime_env+=("$name=${!name}")
done
runtime_env+=("OMP_NUM_THREADS=$threads")
if [[ -n ${STRATA_DS41_PREFILL_LIBDIR:-} ]]; then
  runtime_libdir=$STRATA_DS41_PREFILL_LIBDIR
  [[ -z ${LD_LIBRARY_PATH:-} ]] || runtime_libdir+="${runtime_libdir:+:}$LD_LIBRARY_PATH"
  runtime_env+=("LD_LIBRARY_PATH=$runtime_libdir")
elif [[ -n ${LD_LIBRARY_PATH:-} ]]; then
  runtime_env+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
fi
if [[ ${DS41_PREFILL_TIMING:-0} == 1 ]]; then
  run_env+=(STRATA_DS41_DECODE_TIMING=1 STRATA_DS41_PREFILL_TIMING=1)
fi
if [[ ${DS41_MMQ_COMPACT:-0} == 1 ]]; then run_env+=(STRATA_DS41_MMQ_COMPACT=1); fi
[[ -z ${STRATA_DS41_EXPERT_PROFILE:-} ]] || run_env+=("STRATA_DS41_EXPERT_PROFILE=$STRATA_DS41_EXPERT_PROFILE")

systemd-run --user --scope --quiet --unit="$unit" \
  --property=MemoryHigh=56G --property=MemoryMax=64G --property=MemorySwapMax=0 --property=OOMPolicy=kill --property=MemoryOOMGroup=yes \
  /bin/bash -c 'gate=$1; shift; while [[ ! -e "$gate" ]]; do sleep 0.05; done; exec "$@"' \
  ds41-prefill-gate "$gate_file" env -i "${runtime_env[@]}" "${run_env[@]}" \
  numactl --interleave=all "$bin" "$model" "$prompt_arg" "$steps" "--context=$ctx" "--threads=$threads" "${extra[@]}" >"$log" 2>&1 &
runner=$!
finished=0
stop_scope() {
  systemctl --user kill --kill-whom=all -s TERM "$unit.scope" >/dev/null 2>&1 || kill -TERM "$runner" 2>/dev/null || true
  for _ in {1..10}; do
    kill -0 "$runner" 2>/dev/null || return 0
    sleep 0.1
  done
  systemctl --user kill --kill-whom=all -s KILL "$unit.scope" >/dev/null 2>&1 || kill -KILL "$runner" 2>/dev/null || true
}
cleanup() {
  if (( ! finished )) && [[ -n ${runner:-} ]] && kill -0 "$runner" 2>/dev/null; then stop_scope; fi
  rm -rf -- "$gate_dir"
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

scope_cgroup=
for _ in {1..50}; do
  scope_cgroup=$(systemctl --user show "$unit.scope" -p ControlGroup --value 2>/dev/null || true)
  if [[ -n "$scope_cgroup" && -r "/sys/fs/cgroup$scope_cgroup/memory.max" && -r "/sys/fs/cgroup$scope_cgroup/memory.current" ]]; then break; fi
  kill -0 "$runner" 2>/dev/null || break
  sleep 0.1
done
cgdir="/sys/fs/cgroup$scope_cgroup"
if [[ -z "$scope_cgroup" || ! -r "$cgdir/memory.max" || ! -r "$cgdir/memory.high" || ! -r "$cgdir/memory.swap.max" ||
      ! -r "$cgdir/memory.oom.group" || ! -r "$cgdir/memory.events" ]]; then
  stop_scope
  status=0; wait "$runner" || status=$?
  finished=1
  echo "refusing benchmark: systemd did not establish the required memory cgroup" >&2
  tail -n 12 "$log" 2>/dev/null || true
  exit 1
fi
cg_max=$(<"$cgdir/memory.max"); cg_high=$(<"$cgdir/memory.high"); cg_swap=$(<"$cgdir/memory.swap.max")
cg_oom_group=$(<"$cgdir/memory.oom.group")
if [[ ! "$cg_max" =~ ^[0-9]+$ || ! "$cg_high" =~ ^[0-9]+$ || ! "$cg_swap" =~ ^[0-9]+$ ]] ||
   [[ "$cg_oom_group" != 1 ]] || (( cg_max != MEMORY_MAX || cg_high != MEMORY_HIGH || cg_swap != 0 )); then
  stop_scope
  status=0; wait "$runner" || status=$?
  finished=1
  echo "refusing benchmark: unexpected memory cgroup limits high=$cg_high max=$cg_max swap=$cg_swap" >&2
  exit 1
fi

base_high=$(awk '$1 == "high" {print $2}' "$cgdir/memory.events")
base_max=$(awk '$1 == "max" {print $2}' "$cgdir/memory.events")
base_oom=$(awk '$1 == "oom" {print $2}' "$cgdir/memory.events")
base_oom_kill=$(awk '$1 == "oom_kill" {print $2}' "$cgdir/memory.events")
base_oom_group_kill=$(awk '$1 == "oom_group_kill" {print $2}' "$cgdir/memory.events")
[[ "$base_high" =~ ^[0-9]+$ && "$base_max" =~ ^[0-9]+$ && "$base_oom" =~ ^[0-9]+$ &&
   "$base_oom_kill" =~ ^[0-9]+$ && "$base_oom_group_kill" =~ ^[0-9]+$ ]] || {
  stop_scope; wait "$runner" || true; finished=1
  echo "refusing benchmark: cannot read cgroup OOM counters" >&2; exit 1;
}
if (( base_high != 0 || base_max != 0 || base_oom != 0 || base_oom_kill != 0 || base_oom_group_kill != 0 )); then
  stop_scope; wait "$runner" || true; finished=1
  echo "refusing benchmark: the new scope recorded a memory event before monitoring began" >&2
  exit 1
fi

vram_used=$(<"$gpu_sysfs/mem_info_vram_used")
available=$(mem_available_kib)
if ! cgroup_used=$(<"$cgdir/memory.current"); then
  guard_reason=prelaunch_scope_unreadable
  cgroup_used=0
fi
pressure10=$(pressure_value avg10 /proc/pressure/memory)
peak_used=$vram_used; peak_cgroup=$cgroup_used; min_available=$available; peak_psi10=$pressure10
started=$SECONDS; last_kernel_check=$SECONDS; guard_reason=${guard_reason:-none}; status=0
if [[ ! "$vram_used" =~ ^[0-9]+$ || ! "$available" =~ ^[0-9]+$ || ! "$cgroup_used" =~ ^[0-9]+$ ||
      ! "$pressure10" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  guard_reason=prelaunch_metric_unreadable
elif (( vram_used > vram_total || vram_total-vram_used < preflight_vram_min )); then
  guard_reason=prelaunch_planned_gpu_budget
elif (( available < LIVE_RAM_KIB )); then
  guard_reason=prelaunch_available_ram
elif (( cgroup_used >= cg_high-8*GiB )); then
  guard_reason=prelaunch_scope_memory
elif awk -v p="$pressure10" -v lim="$LIVE_PSI_FULL_AVG10" 'BEGIN {exit !(p > lim)}'; then
  guard_reason=prelaunch_memory_pressure
fi
if [[ "$guard_reason" == none ]]; then
  if bad_event_since; then current_kernel_event=$bad_event_state; else current_kernel_event=invalid; fi
  if [[ "$current_kernel_event" == 1 ]]; then guard_reason=prelaunch_unsafe_journal_event
  elif [[ "$current_kernel_event" != 0 ]]; then guard_reason=system_journal_unreadable
  fi
fi
if [[ "$guard_reason" == none ]]; then : > "$gate_file"; else stop_scope; fi
while kill -0 "$runner" 2>/dev/null; do
  [[ "$guard_reason" == none ]] || break
  if [[ ! -r "$cgdir/memory.current" || ! -r "$cgdir/memory.events" ]]; then
    status=0; wait "$runner" || status=$?; finished=1
    (( status == 0 )) || guard_reason=scope_process_exit
    break
  fi
  vram_used=$(<"$gpu_sysfs/mem_info_vram_used") || { guard_reason=physical_vram_unreadable; break; }
  available=$(mem_available_kib) || { guard_reason=available_ram_unreadable; break; }
  cgroup_used=$(<"$cgdir/memory.current") || { guard_reason=scope_memory_unreadable; break; }
  if [[ ! "$vram_used" =~ ^[0-9]+$ || ! "$available" =~ ^[0-9]+$ || ! "$cgroup_used" =~ ^[0-9]+$ ]]; then
    guard_reason=resource_metric_unreadable
    break
  fi
  (( vram_used > peak_used )) && peak_used=$vram_used
  (( cgroup_used > peak_cgroup )) && peak_cgroup=$cgroup_used
  (( available < min_available )) && min_available=$available
  printf '%s %s %s\n' "$vram_used" "$available" "$cgroup_used" >> "$samples"
  pressure10=$(pressure_value avg10 /proc/pressure/memory)
  [[ "$pressure10" =~ ^[0-9]+([.][0-9]+)?$ ]] || { guard_reason=memory_psi_unreadable; }
  if [[ "$pressure10" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
     awk -v p="$pressure10" -v peak="$peak_psi10" 'BEGIN {exit !(p > peak)}'; then peak_psi10=$pressure10; fi
  if (( SECONDS > last_kernel_check )); then
    if bad_event_since; then current_kernel_event=$bad_event_state; else current_kernel_event=invalid; fi
    last_kernel_check=$SECONDS
    if [[ "$current_kernel_event" != 0 ]]; then
      guard_reason=unsafe_journal_event
    fi
  fi
  if [[ "$guard_reason" == none ]]; then
    if (( vram_used > vram_total || vram_total-vram_used < LIVE_VRAM )); then guard_reason=physical_vram
    elif (( available < LIVE_RAM_KIB )); then guard_reason=available_ram
    elif (( cgroup_used >= cg_high-8*GiB )); then guard_reason=scope_memory_high
    elif awk -v p="$pressure10" -v lim="$LIVE_PSI_FULL_AVG10" 'BEGIN {exit !(p > lim)}'; then guard_reason=memory_pressure
    elif (( SECONDS-started > MAX_SECONDS )); then guard_reason=timeout
    fi
  fi
  cg_high_events=$(awk '$1 == "high" {print $2}' "$cgdir/memory.events")
  cg_max_events=$(awk '$1 == "max" {print $2}' "$cgdir/memory.events")
  cg_oom=$(awk '$1 == "oom" {print $2}' "$cgdir/memory.events")
  cg_oom_kill=$(awk '$1 == "oom_kill" {print $2}' "$cgdir/memory.events")
  cg_oom_group_kill=$(awk '$1 == "oom_group_kill" {print $2}' "$cgdir/memory.events")
  if [[ ! "$cg_high_events" =~ ^[0-9]+$ || ! "$cg_max_events" =~ ^[0-9]+$ || ! "$cg_oom" =~ ^[0-9]+$ ||
        ! "$cg_oom_kill" =~ ^[0-9]+$ || ! "$cg_oom_group_kill" =~ ^[0-9]+$ ]]; then
    guard_reason=scope_memory_counters_unreadable
  elif (( cg_high_events > base_high )); then
    guard_reason=scope_memory_high_event
  elif (( cg_max_events > base_max || cg_oom > base_oom || cg_oom_kill > base_oom_kill ||
          cg_oom_group_kill > base_oom_group_kill )); then
    guard_reason=scope_memory_event
  fi
  if [[ "$guard_reason" != none ]]; then
    stop_scope
    break
  fi
  sleep 0.1
done

if [[ "$guard_reason" != none ]] && (( ! finished )); then stop_scope; fi
if (( ! finished )); then wait "$runner" || status=$?; fi
finished=1
if (( status != 0 )) && [[ "$guard_reason" == none ]]; then guard_reason=scope_process_exit; fi
if bad_event_since; then final_kernel_event=$bad_event_state; else final_kernel_event=invalid; fi
if [[ "$final_kernel_event" != 0 ]]; then
  guard_reason=unsafe_journal_event
fi
scope=$(systemctl --user show "$unit.scope" -p ControlGroup --value 2>/dev/null || true)
if [[ -n "$scope" && -r "/sys/fs/cgroup$scope/memory.events" ]]; then
  read -r final_high final_max final_oom final_oom_kill final_oom_group_kill < <(
    awk '$1 == "high" {h=$2} $1 == "max" {m=$2} $1 == "oom" {o=$2} $1 == "oom_kill" {k=$2} $1 == "oom_group_kill" {g=$2} END {print h, m, o, k, g}' \
      "/sys/fs/cgroup$scope/memory.events"
  )
  if [[ ! ${final_high:-} =~ ^[0-9]+$ || ! ${final_max:-} =~ ^[0-9]+$ || ! ${final_oom:-} =~ ^[0-9]+$ ||
        ! ${final_oom_kill:-} =~ ^[0-9]+$ || ! ${final_oom_group_kill:-} =~ ^[0-9]+$ ]]; then
    guard_reason=scope_memory_counters_unreadable
  elif (( final_high != base_high || final_max != base_max || final_oom != base_oom || final_oom_kill != base_oom_kill ||
          final_oom_group_kill != base_oom_group_kill )); then
    guard_reason=scope_memory_event
  fi
fi
printf 'run=%s repeats=%s exit=%s guard=%s peak_vram_used=%s minimum_vram_free=%s peak_scope_memory=%s scope_memory_high=%s minimum_available_ram_kib=%s peak_memory_psi_full_avg10=%s new_unsafe_journal_event=%s\n' \
  "$run_id" "$repeats" "$status" "$guard_reason" "$peak_used" "$((vram_total-peak_used))" "$peak_cgroup" "$cg_high" "$min_available" "$peak_psi10" \
  "$final_kernel_event"
tail -n 25 "$log" 2>/dev/null || true
[[ "$guard_reason" == none && $status -eq 0 ]]
