#!/usr/bin/env bash
# Run one DS4.1 decode screen inside a hard memory-limited scope and stop on
# host pressure, low physical VRAM, cgroup OOM, or new AMD/kernel faults.
set -euo pipefail

usage() {
  echo "usage: STRATA_DS41_DECODE_BIN=/path/to/strata-glm-decode $0 RUN_ID MODEL_GGUF PROMPT_IDS OUTPUT_TOKENS [-- EXTRA_ARGS...]" >&2
  exit 2
}

[[ $# -ge 4 ]] || usage
run_id=$1; model=$2; prompt=$3; steps=$4
shift 4
if [[ ${1:-} == -- ]]; then shift; fi
extra=(); [[ $# -eq 0 ]] || extra=("$@")
for arg in "${extra[@]}"; do
  case "$arg" in
    --check=*|--check-force) ;;
    *) echo "guard refuses extra decoder option: $arg" >&2; exit 2 ;;
  esac
done

parse_uint() {
  local destination=$1 raw=$2 minimum=$3 maximum=$4 max_digits=$5 value
  [[ "$raw" =~ ^[0-9]+$ && ${#raw} -le $max_digits ]] || return 1
  value=$((10#$raw))
  (( value >= minimum && value <= maximum )) || return 1
  printf -v "$destination" '%d' "$value"
}

[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid RUN_ID" >&2; exit 2; }
[[ "$prompt" != *';'* ]] || { echo "PROMPT_IDS path cannot contain ';'" >&2; exit 2; }
bin=${STRATA_DS41_DECODE_BIN:?set STRATA_DS41_DECODE_BIN}
[[ -x "$bin" && -r "$model" && -r "$prompt" ]] || { echo "binary, model, or prompt is unavailable" >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command -v python3 >/dev/null || { echo "refusing decode: python3 is required for GGUF prompt preflight" >&2; exit 1; }
parse_uint steps "$steps" 1 4096 4 || { echo "OUTPUT_TOKENS must be in [1, 4096]" >&2; exit 2; }
[[ ${DS41_DECODE_MMQ_VALIDATE_BOUNDS:-1} == 1 ]] || {
  echo "guarded decode requires MMQ bounds and row-ID validation" >&2; exit 2;
}
if [[ -n ${STRATA_DS41_DSPARK:-} && ! -r ${STRATA_DS41_DSPARK} ]]; then
  echo "STRATA_DS41_DSPARK is not readable" >&2; exit 2
fi

ctx=${DS41_DECODE_CONTEXT:-8192}
threads=${DS41_DECODE_THREADS:-15}
parse_uint ctx "$ctx" 1 131072 6 || { echo "context must be in [1, 131072]" >&2; exit 2; }
parse_uint threads "$threads" 1 512 3 || { echo "threads must be in [1, 512]" >&2; exit 2; }
max_seconds=${DS41_DECODE_TIMEOUT_SECONDS:-600}
parse_uint max_seconds "$max_seconds" 30 1800 4 || { echo "timeout must be in [30, 1800] seconds" >&2; exit 2; }
quality_logits=${DS41_DECODE_SAVE_LOGITS:-0}
[[ "$quality_logits" == 0 || "$quality_logits" == 1 ]] || { echo "DS41_DECODE_SAVE_LOGITS must be 0 or 1" >&2; exit 2; }
prefill_chunk=64
if [[ "$quality_logits" == 1 ]]; then
  prefill_chunk=${DS41_DECODE_QUALITY_CHUNK:-1}
  parse_uint prefill_chunk "$prefill_chunk" 1 8 1 || { echo "quality chunk must be in [1, 8]" >&2; exit 2; }
elif [[ -n ${DS41_DECODE_QUALITY_CHUNK:-} ]]; then
  echo "quality chunk requires teacher-forced logits capture" >&2; exit 2
fi
if [[ "$quality_logits" == 1 && -n ${STRATA_DS41_DSPARK:-} ]]; then
  echo "guarded teacher-forced capture requires target-only decoding" >&2; exit 2
fi
fault_boot_scope=${DS41_DECODE_FAULT_BOOT_SCOPE:-all}
[[ "$fault_boot_scope" == all || "$fault_boot_scope" == current ]] || {
  echo "DS41_DECODE_FAULT_BOOT_SCOPE must be all or current" >&2; exit 2;
}
dsp_batch_stage=${STRATA_DS41_DSPARK_BATCH_STAGE:-0}
dsp_cache_experts=${STRATA_DS41_DSPARK_CACHE_EXPERTS:-0}
dsp_cache_slots=${STRATA_DS41_DSPARK_CACHE_SLOTS:-45}
dsp_cache_sim=${STRATA_DS41_DSPARK_CACHE_SIM:-0}
gpu_route=${STRATA_DS41_GPU_ROUTE:-0}
decode_grouped=${STRATA_DS41_DECODE_GROUPED:-0}
defer_all=${STRATA_DS41_DEFER_ALL:-0}
defer_rank=${STRATA_DS41_DEFER_RANK:-0}
fuse=${STRATA_DS41_FUSE:-0}
share_q8=${STRATA_DS41_SHARE_Q8:-0}
iq1s_pair=${STRATA_IQ1S_PAIR:-0}
iq1s_fast=${STRATA_IQ1S_FAST:-0}
q6_v2=${STRATA_Q6_V2:-0}
numa_interleave=${DS41_DECODE_NUMA_INTERLEAVE:-0}
lane_skip_unused=${STRATA_DS41_LANE_SKIP_UNUSED:-0}
lane_group_exact=${STRATA_DS41_LANE_GROUP_EXACT:-0}
lane_async=${STRATA_DS41_LANE_ASYNC:-0}
stage_cpu_expert=${STRATA_DS41_STAGE_CPU_EXPERT:-0}
stage_cpu_order=${STRATA_DS41_STAGE_CPU_ORDER:-0}
stage_cpu_collect=${STRATA_DS41_STAGE_CPU_COLLECT:-0}
stage_cpu_host_act=${STRATA_DS41_STAGE_CPU_HOST_ACT:-0}
stage_cpu_quota=${STRATA_DS41_STAGE_CPU_QUOTA:-1}
stage_cpu_timing=${STRATA_DS41_STAGE_TIMING:-0}
stage_cache_slots=${STRATA_DS41_STAGE_CACHE_SLOTS:-0}
stage_cache_throttle=${STRATA_DS41_STAGE_CACHE_THROTTLE:-0}
stage_cache_frequency=${STRATA_DS41_STAGE_CACHE_FREQUENCY:-0}
stage_cache_lfu=${STRATA_DS41_STAGE_CACHE_LFU:-0}
stage_pair_dot=${STRATA_DS41_STAGE_PAIR_DOT:-0}
stage_batch_act=${STRATA_DS41_STAGE_BATCH_ACT:-0}
stage_extra_cached=${STRATA_DS41_STAGE_EXTRA_CACHED:-0}
topk_original_weight=${STRATA_DS41_TOPK_ORIGINAL_WEIGHT:-0}
stage_cpu_order_check=${STRATA_DS41_STAGE_CPU_ORDER_CHECK:-0}
gpu_topk_min=${STRATA_DS41_GPU_TOPK_MIN:-16}
topk_check=${DS41_DECODE_TOPK_CHECK:-0}
gpu_expert_mib=${STRATA_DS41_GPU_EXPERT_MIB:-3584}
# the canonical tier (CPU-order GPU experts, a static arena loaded from a route profile)
canon_mib=${STRATA_DS41_CANON_MIB:-0}
canon_check=${STRATA_DS41_CANON_CHECK:-0}
parse_uint canon_mib "$canon_mib" 0 12288 5 || { echo "canonical tier must be in [0, 12288] MiB" >&2; exit 2; }
[[ "$canon_check" =~ ^[01]$ ]] || { echo "STRATA_DS41_CANON_CHECK must be 0 or 1" >&2; exit 2; }
if (( canon_mib > 0 )); then
  [[ -f ${STRATA_DS41_CANON_PROFILE:-} && -r ${STRATA_DS41_CANON_PROFILE} ]] || { echo "STRATA_DS41_CANON_PROFILE must be a readable route trace" >&2; exit 2; }
  canon_profile_bytes=$(stat -c %s -- "$STRATA_DS41_CANON_PROFILE")
  (( canon_profile_bytes > 0 && canon_profile_bytes <= 1073741824 )) || { echo "canonical tier profile size out of bounds" >&2; exit 2; }
  [[ ${STRATA_IQ1S_FAST:-0} == 0 && ${STRATA_DS41_GPU_ROUTE:-0} == 0 && ${STRATA_DS41_STAGE_CPU_EXPERT:-0} == 0 ]] || {
    echo "the canonical tier needs default IQ1_S arithmetic, no GPU_ROUTE and no CPU staging" >&2; exit 2; }
fi
[[ "$dsp_batch_stage" =~ ^[01]$ && "$dsp_cache_experts" =~ ^[01]$ && "$dsp_cache_sim" =~ ^[01]$ &&
   "$gpu_route" =~ ^[01]$ && "$decode_grouped" =~ ^[01]$ && "$defer_all" =~ ^[01]$ &&
   "$fuse" =~ ^[01]$ && "$share_q8" =~ ^[01]$ && "$iq1s_fast" =~ ^[01]$ && "$iq1s_pair" =~ ^[01]$ && "$q6_v2" =~ ^[01]$ &&
   "$numa_interleave" =~ ^[01]$ && "$lane_skip_unused" =~ ^[01]$ && "$lane_group_exact" =~ ^[01]$ && "$lane_async" =~ ^[01]$ && "$stage_cpu_expert" =~ ^[01]$ && "$stage_cpu_order" =~ ^[01]$ && "$stage_cpu_collect" =~ ^[01]$ && "$stage_cpu_host_act" =~ ^[01]$ && "$stage_cpu_timing" =~ ^[01]$ && "$stage_cpu_order_check" =~ ^[01]$ && "$topk_check" =~ ^[01]$ && "$topk_original_weight" =~ ^[01]$ && "$stage_cache_throttle" =~ ^[01]$ && "$stage_cache_frequency" =~ ^[01]$ && "$stage_cache_lfu" =~ ^[01]$ && "$stage_pair_dot" =~ ^[01]$ && "$stage_batch_act" =~ ^[01]$ && "$stage_extra_cached" =~ ^[01]$ ]] || {
  echo "cache, routing, fusion, kernel and NUMA switches must be 0 or 1" >&2; exit 2;
}
parse_uint dsp_cache_slots "$dsp_cache_slots" 1 100 3 || { echo "DSpark cache slots must be in [1, 100]" >&2; exit 2; }
parse_uint defer_rank "$defer_rank" 0 5 1 || { echo "DEFER_RANK must be in [0, 5]" >&2; exit 2; }
parse_uint stage_cpu_quota "$stage_cpu_quota" 1 4 1 || { echo "CPU staging quota must be in [1, 4]" >&2; exit 2; }
parse_uint gpu_topk_min "$gpu_topk_min" 1 64 2 || { echo "GPU top-k minimum must be in [1, 64]" >&2; exit 2; }
parse_uint stage_cache_slots "$stage_cache_slots" 0 192 3 || { echo "CPU staging cache slots must be in [0, 192]" >&2; exit 2; }
if (( topk_original_weight )); then
  parse_uint topk_value "${STRATA_DS41_CPU_TOPK:-}" 0 5 1 || {
    echo "original top-k weights require CPU_TOPK in [0, 5]" >&2; exit 2;
  }
fi
(( dsp_cache_experts == 0 || dsp_batch_stage == 1 )) || { echo "DSpark expert cache requires batch staging" >&2; exit 2; }
(( gpu_route == 0 || decode_grouped == 1 )) || { echo "GPU routing requires grouped decode" >&2; exit 2; }
(( defer_all == 0 || defer_rank == 0 )) || { echo "deferred routing must be DEFER_ALL or rank 1 through 5" >&2; exit 2; }
if (( stage_cpu_order )) && (( !stage_cpu_expert || iq1s_fast )); then
  echo "CPU-order staging requires checked staging and default IQ1_S arithmetic" >&2; exit 2
fi
if (( stage_cpu_order_check && !stage_cpu_order )); then
  echo "CPU-order expert diagnostics require CPU-order staging" >&2; exit 2
fi
if (( stage_pair_dot && !stage_cpu_order )); then
  echo "paired staging dots require CPU-order staging" >&2; exit 2
fi
if (( stage_batch_act && (!stage_cpu_host_act || stage_cpu_quota != 1 || stage_cpu_order_check) )); then
  echo "batched staging activation requires host activation, quota one, and no per-layer diagnostic" >&2; exit 2
fi
if (( stage_cpu_collect && !stage_cpu_order )); then
  echo "CPU-order collection requires CPU-order staging" >&2; exit 2
fi
if (( stage_cpu_host_act && !stage_cpu_collect )); then
  echo "host activation requires CPU-order collection" >&2; exit 2
fi
if (( stage_cpu_quota > 1 && !stage_cpu_collect )); then
  echo "multiple staged experts require CPU-order collection" >&2; exit 2
fi
if (( stage_cache_slots > 0 && (!stage_cpu_collect || stage_cache_slots < stage_cpu_quota) )); then
  echo "staging cache requires ordered collection and at least one slot per staged expert" >&2; exit 2
fi
if (( (stage_cache_throttle || stage_cache_frequency) && stage_cache_slots == 0 )); then
  echo "staging cache policies require a nonempty staging cache" >&2; exit 2
fi
if (( stage_cache_lfu && !stage_cache_frequency )); then
  echo "LFU eviction requires staging frequency tracking" >&2; exit 2
fi
if (( stage_extra_cached && (stage_cache_slots == 0 || stage_cpu_quota < 2) )); then
  echo "cached-only extra staging requires a cache and quota of at least two" >&2; exit 2
fi
if (( stage_cpu_expert )); then
  [[ ${STRATA_DS41_VERIFY_EXACT:-0} == 1 && "$gpu_route" == 0 && "$decode_grouped" == 0 &&
     "$defer_all" == 0 && "$defer_rank" == 0 && -z ${STRATA_DS41_CPU_TOPK:-} && -z ${STRATA_DS41_CPU_TRIM:-} &&
     ${STRATA_DS41_TIER_PROMOTE:-} == 0 && -n ${STRATA_DS41_EXPERT_PROFILE:-} ]] || {
    echo "staged CPU experts require exact verification, CPU routing, a fixed tier profile and no lossy route options" >&2; exit 2;
  }
fi
(( numa_interleave == 0 )) || command -v numactl >/dev/null || { echo "NUMA interleave requested but numactl is unavailable" >&2; exit 2; }
parse_uint gpu_expert_mib "$gpu_expert_mib" 0 16384 5 || { echo "target GPU expert tier must be in [0, 16384] MiB" >&2; exit 2; }

prompt_bytes=$(wc -c < "$prompt")
[[ "$prompt_bytes" =~ ^[0-9]+$ ]] && (( prompt_bytes <= 16*1024*1024 )) || {
  echo "prompt ID file must be at most 16 MiB" >&2; exit 2;
}
model_vocab=$(python3 "$script_dir/ds41f_guard_prompt.py" "$model" "$prompt") || exit 2
[[ "$model_vocab" =~ ^[1-9][0-9]*$ ]] || { echo "refusing decode: invalid model vocabulary size" >&2; exit 2; }
prompt_tokens=$(awk -v vocab="$model_vocab" 'BEGIN { n=0; bad=0 } { gsub(/,/, " "); gsub(/\r/, " "); for (i=1; i<=NF; ++i) {
  if ($i !~ /^[0-9]+$/ || length($i) > 10 || $i+0 > 2147483647 || $i+0 >= vocab) { bad=1; exit 1 }
  ++n; if (n > 131072) { bad=1; exit 1 }
} } END { if (!bad) print n }' "$prompt") || {
  echo "prompt file must contain at most 131072 token IDs inside the model vocabulary" >&2; exit 2;
}
[[ "$prompt_tokens" =~ ^[0-9]+$ ]] && (( prompt_tokens > 0 && prompt_tokens <= ctx && prompt_tokens + steps - 1 <= ctx )) || {
  echo "prompt and requested output do not fit the guarded context" >&2; exit 2;
}
if [[ -n ${STRATA_DS41_ORDER_TRACE:-} ]]; then
  [[ "$STRATA_DS41_ORDER_TRACE" =~ ^/tmp/ds41-cpuorder-trace-[A-Za-z0-9_-]+\.bin$ ]] &&
    (( prompt_tokens == 1 && steps == 1 && fuse == 0 )) && [[ "$quality_logits" == 0 ]] || {
    echo "order tracing requires one token, an unfused path, and a bounded /tmp trace name" >&2; exit 2;
  }
  [[ ! -e "$STRATA_DS41_ORDER_TRACE" && ! -L "$STRATA_DS41_ORDER_TRACE" ]] || {
    echo "order trace destination already exists" >&2; exit 2;
  }
fi
if [[ -n ${STRATA_DS41_ATTN_TRACE:-} ]]; then
  [[ "$STRATA_DS41_ATTN_TRACE" =~ ^/tmp/ds41-attn-trace-[A-Za-z0-9_-]+\.bin$ ]] &&
    (( prompt_tokens == 1 && steps == 1 && fuse == 0 )) && [[ "$quality_logits" == 0 ]] || {
    echo "attention tracing requires one token, an unfused path, and a bounded /tmp trace name" >&2; exit 2;
  }
  [[ ! -e "$STRATA_DS41_ATTN_TRACE" && ! -L "$STRATA_DS41_ATTN_TRACE" ]] || {
    echo "attention trace destination already exists" >&2; exit 2;
  }
fi
if [[ "$quality_logits" == 1 ]]; then
  quality_bytes=$((prompt_tokens * model_vocab * 4))
  (( quality_bytes <= 512*1024*1024 )) || { echo "teacher-forced logits exceed the 512 MiB guard" >&2; exit 2; }
  quality_free_kib=$(df -Pk /tmp | awk 'NR == 2 {print $4}')
  [[ "$quality_free_kib" =~ ^[0-9]+$ ]] && (( quality_free_kib * 1024 >= quality_bytes + 1024*1024*1024 )) || {
    echo "insufficient /tmp space for guarded logits capture" >&2; exit 1;
  }
fi

command -v rocminfo >/dev/null && command -v journalctl >/dev/null && command -v systemd-run >/dev/null && command -v systemctl >/dev/null || {
  echo "rocminfo, journalctl, systemd-run, and systemctl are required" >&2; exit 2;
}
[[ -r /sys/fs/cgroup/cgroup.controllers ]] || { echo "refusing decode: cgroup v2 is unavailable" >&2; exit 1; }
systemctl --user show-environment >/dev/null 2>&1 || { echo "refusing decode: the user systemd manager is unavailable" >&2; exit 1; }
rocminfo 2>/dev/null | rg 'gfx120[01]' >/dev/null || { echo "refusing decode: expected an AMD gfx1200/1201 device" >&2; exit 1; }

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
  echo "refusing decode: cannot identify AMD physical VRAM counters" >&2; exit 1;
}

readonly GiB=$((1024*1024*1024))
readonly PREFLIGHT_VRAM=$((8*GiB))
readonly LIVE_VRAM=$((2560*1024*1024))
readonly PRELAUNCH_GPU_MARGIN=$((1*GiB))
readonly PREFLIGHT_RAM_KIB=$((96*1024*1024))
readonly LIVE_RAM_KIB=$((32*1024*1024))
readonly MEMORY_HIGH=$((48*GiB))
readonly MEMORY_MAX=$((64*GiB))
readonly PREFLIGHT_PSI_FULL_AVG60=3.0
readonly LIVE_PSI_FULL_AVG10=5.0
readonly PREFLIGHT_FAULT_WINDOW='24 hours ago'
readonly KERNEL_BAD_PATTERN='(\[gfxhub\].*page fault|GCVM_L2_PROTECTION_FAULT|VM_L2_PROTECTION_FAULT|GPU fault detected|GPU reset|amdgpu.*(page fault|protection fault|allocation failed|failed to allocate|VRAM.*(full|exhausted))|out of memory|oom-kill|oom_reaper|invoked oom-killer|Killed process [0-9]+|page allocation failure|strata-glm-deco.*segfault|segfault.*strata-glm-decode|general protection fault)'
readonly JOURNAL_OOM_PATTERN='codex-ds41-decode.*Failed with result.*oom-kill|codex-ds41-prefill.*Failed with result.*oom-kill|systemd-oomd.*Killed|systemd-oomd.*Killing'

mem_available_kib() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
pressure_value() {
  local key=$1 file=$2
  awk -v key="$key" '$1 == "full" { for (i=2; i<=NF; ++i) if ($i ~ ("^" key "=")) {split($i,a,"="); print a[2]} }' "$file"
}
event_value() { awk -v key="$1" '$1 == key {print $2}' "$2" 2>/dev/null || true; }

vram_total=$(<"$gpu_sysfs/mem_info_vram_total")
vram_used=$(<"$gpu_sysfs/mem_info_vram_used")
available=$(mem_available_kib)
pressure60=$(pressure_value avg60 /proc/pressure/memory)
[[ "$vram_total" =~ ^[0-9]+$ && "$vram_used" =~ ^[0-9]+$ && "$available" =~ ^[0-9]+$ && "$pressure60" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "refusing decode: cannot read GPU or host memory preflight metrics" >&2; exit 1;
}
required_gpu_bytes=$((gpu_expert_mib*1024*1024 + canon_mib*1024*1024 + stage_cache_slots*16*1024*1024 + LIVE_VRAM + PRELAUNCH_GPU_MARGIN))
preflight_vram_min=$PREFLIGHT_VRAM
(( required_gpu_bytes > preflight_vram_min )) && preflight_vram_min=$required_gpu_bytes
(( vram_used <= vram_total && vram_total-vram_used >= preflight_vram_min && available >= PREFLIGHT_RAM_KIB )) || {
  echo "refusing decode: free_vram_bytes=$((vram_total-vram_used)) required_vram_bytes=$preflight_vram_min available_ram_kib=$available" >&2; exit 1;
}
awk -v p="$pressure60" -v lim="$PREFLIGHT_PSI_FULL_AVG60" 'BEGIN {exit !(p <= lim)}' || {
  echo "refusing decode: memory PSI full avg60=$pressure60% exceeds ${PREFLIGHT_PSI_FULL_AVG60}%" >&2; exit 1;
}

# Refuse recent GPU faults and systemd scope OOM kills. Capture a journal-wide
# cursor first so relevant events arriving during preflight and scope setup are caught.
journal_base=$(journalctl -n 0 --no-pager --show-cursor 2>/dev/null) || {
  echo "refusing decode: cannot read the system journal" >&2; exit 1;
}
event_cursor=$(sed -n 's/^-- cursor: //p' <<<"$journal_base" | tail -n 1)
[[ -n "$event_cursor" ]] || { echo "refusing decode: system journal did not provide a cursor" >&2; exit 1; }
fault_boot_args=(--boot=all)
if [[ "$fault_boot_scope" == current ]]; then
  fault_boot_args=(--boot=0)
  echo "DS41 decode preflight: scanning the current boot only by explicit operator setting; runtime journal monitoring remains enabled" >&2
fi
kernel_recent=$(journalctl "${fault_boot_args[@]}" -k --since "$PREFLIGHT_FAULT_WINDOW" --no-pager -o cat 2>/dev/null) || {
  echo "refusing decode: cannot scan recent kernel faults" >&2; exit 1;
}
journal_recent=$(journalctl "${fault_boot_args[@]}" --since "$PREFLIGHT_FAULT_WINDOW" --no-pager -o cat 2>/dev/null) || {
  echo "refusing decode: cannot scan recent systemd scope OOM events" >&2; exit 1;
}
if rg -qi -- "$KERNEL_BAD_PATTERN" <<<"$kernel_recent" || rg -qi -- "$JOURNAL_OOM_PATTERN" <<<"$journal_recent"; then
  echo "refusing decode: AMD fault, OOM, reset, segfault, or systemd scope OOM kill found in the last 24 hours; inspect the journals before retrying" >&2
  exit 1
fi
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

unit="codex-ds41-decode-$run_id"
log=${DS41_DECODE_LOG:-/tmp/ds41-decode-$run_id.log}
samples=${DS41_DECODE_SAMPLES:-/tmp/ds41-decode-$run_id-memory.log}
: > "$samples"
gate_dir=$(mktemp -d "${TMPDIR:-/tmp}/ds41-decode-${run_id}.gate.XXXXXX") || { echo "cannot create decode launch gate" >&2; exit 1; }
gate_file="$gate_dir/release"
decoder_done="$gate_dir/decoder_done"
monitor_done="$gate_dir/monitor_done"
prompt_arg="@$prompt"
if [[ "$quality_logits" == 1 ]]; then
  quality_file=$(mktemp "/tmp/ds41-logits-${run_id}.XXXXXX") || { echo "cannot create guarded logits file" >&2; exit 1; }
  extra+=("--save-all-logits=$quality_file")
fi

# Restrict the process to bounded staging, disable bulk page population and
# host pinning, and keep 2 GiB for the desktop plus 512 MiB physical headroom.
# WMMA attention's optional row/selection checks are mandatory for this run.
run_env=(
  STRATA_DS41_DESKTOP_MIB=2048 STRATA_DS41_GPU_RESERVE_MIB=2560 STRATA_DS41_GPU_HEADROOM_MIB=512
  "STRATA_DS41_GPU_EXPERT_MIB=$gpu_expert_mib"
  "STRATA_DS41_CANON_MIB=$canon_mib" "STRATA_DS41_CANON_CHECK=$canon_check"
  "STRATA_DS41_FUSE=$fuse" "STRATA_DS41_SHARE_Q8=$share_q8"
  "STRATA_IQ1S_PAIR=$iq1s_pair" "STRATA_IQ1S_FAST=$iq1s_fast" "STRATA_Q6_V2=$q6_v2"
  "STRATA_DS41_LANE_SKIP_UNUSED=$lane_skip_unused"
  "STRATA_DS41_LANE_GROUP_EXACT=$lane_group_exact"
  "STRATA_DS41_LANE_ASYNC=$lane_async"
  "STRATA_DS41_STAGE_CPU_EXPERT=$stage_cpu_expert" "STRATA_DS41_STAGE_CPU_ORDER=$stage_cpu_order" STRATA_DS41_STAGE_CPU_CHECK=1
  "STRATA_DS41_STAGE_CPU_COLLECT=$stage_cpu_collect"
  "STRATA_DS41_STAGE_CPU_HOST_ACT=$stage_cpu_host_act"
  "STRATA_DS41_STAGE_CACHE_THROTTLE=$stage_cache_throttle"
  "STRATA_DS41_STAGE_CACHE_FREQUENCY=$stage_cache_frequency"
  "STRATA_DS41_STAGE_CACHE_LFU=$stage_cache_lfu"
  "STRATA_DS41_STAGE_PAIR_DOT=$stage_pair_dot"
  "STRATA_DS41_STAGE_BATCH_ACT=$stage_batch_act"
  "STRATA_DS41_STAGE_EXTRA_CACHED=$stage_extra_cached"
  "STRATA_DS41_STAGE_CPU_QUOTA=$stage_cpu_quota"
  "STRATA_DS41_STAGE_TIMING=$stage_cpu_timing"
  "STRATA_DS41_STAGE_CACHE_SLOTS=$stage_cache_slots"
  "STRATA_DS41_TOPK_ORIGINAL_WEIGHT=$topk_original_weight"
  "STRATA_DS41_STAGE_CPU_ORDER_CHECK=$stage_cpu_order_check"
  STRATA_DS41_PIN_GB=0 STRATA_DS41_POPULATE=0 STRATA_DS41_RAM_COPY=0
  STRATA_DS41_DSPARK_STREAM_EXPERTS=1 STRATA_DS41_DSPARK_PIN_EXPERTS=0
  "STRATA_DS41_DSPARK_BATCH_STAGE=$dsp_batch_stage" "STRATA_DS41_DSPARK_CACHE_EXPERTS=$dsp_cache_experts"
  "STRATA_DS41_DSPARK_CACHE_SLOTS=$dsp_cache_slots"
  "STRATA_DS41_DSPARK_CACHE_SIM=$dsp_cache_sim"
  "STRATA_DS41_GPU_ROUTE=$gpu_route" "STRATA_DS41_DECODE_GROUPED=$decode_grouped"
  "STRATA_DS41_DEFER_ALL=$defer_all" "STRATA_DS41_DEFER_RANK=$defer_rank"
  "STRATA_DS41_PREFILL_CHUNK=$prefill_chunk" STRATA_DS41_GROW=0 STRATA_DS41_PREFILL_HYBRID=0 STRATA_DS41_KV=f16
  STRATA_DS41_ATTN_WMMA=2 STRATA_DS41_ATTN_BOUNDS_CHECK=1 STRATA_DS41_TOPK_2STAGE=0 "STRATA_DS41_GPU_TOPK_MIN=$gpu_topk_min"
  STRATA_DS41_DENSE_MMQ=0 STRATA_DS41_DECODE_MMQ=0 STRATA_MMQ_VALIDATE_BOUNDS=1
)
(( topk_check == 0 )) || run_env+=(STRATA_DS41_TOPK_CHECK=1)
runtime_env=("PATH=${PATH:-/usr/bin:/bin}" "HOME=${HOME:-/tmp}" "LANG=${LANG:-C}")
for name in LD_LIBRARY_PATH ROCM_PATH ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER \
            HSA_OVERRIDE_GFX_VERSION OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES OMP_WAIT_POLICY TMPDIR; do
  [[ -v $name ]] && runtime_env+=("$name=${!name}")
done
[[ -z ${STRATA_DS41_DSPARK:-} ]] || run_env+=("STRATA_DS41_DSPARK=$STRATA_DS41_DSPARK")
for name in STRATA_DS41_CPU_TRIM STRATA_DS41_CPU_TOPK STRATA_DS41_VERIFY_EXACT \
            STRATA_DS41_ORDER_TRACE STRATA_DS41_ATTN_TRACE \
            STRATA_DS41_DSPARK_EXTEND STRATA_DS41_DSPARK_N STRATA_DS41_DSPARK_PMIN \
            STRATA_DS41_DSPARK_AUTO STRATA_DS41_DSPARK_AUTO_BACKOFF STRATA_DS41_DSPARK_ACCEPT_HIST \
            STRATA_DS41_DSPARK_SCHED STRATA_DS41_DSPARK_Q8 STRATA_DS41_DSPARK_DEVICE_HEAD STRATA_DS41_VERIFY_ARGMAX_GPU \
            STRATA_DS41_DSPARK_PARALLEL_STAGE STRATA_DS41_LOOKUP STRATA_DS41_LOOKUP_N \
            STRATA_DS41_LOOKUP_ADAPT STRATA_DS41_LOOKUP_BONUS STRATA_DS41_LOOKUP_GATE \
            STRATA_DS41_GPU_EXPERT_MIB STRATA_DS41_TIER_PROMOTE STRATA_DS41_TIER_DECAY \
            STRATA_DS41_EXPERT_PROFILE STRATA_DS41_ROUTES STRATA_DS41_DECODE_TIMING STRATA_DS41_ENGRAM_ASYNC \
            STRATA_DS41_CANON_PROFILE STRATA_DS41_CANON_STREAM STRATA_DS41_CANON_STREAM_THREADS STRATA_DS41_VERIFY_FAST_TRUNK STRATA_DS41_VERIFY_FAST_BF16 STRATA_DS41_ATTN_MULTI STRATA_DS41_ROUTE_AFFINITY STRATA_DS41_WINDOW_AFFINITY STRATA_DS41_DENSE_Q4K STRATA_DS41_CANON_COPY_STREAMS STRATA_DS41_MULTI_ROWS STRATA_DS41_CANON_MULTI STRATA_DS41_REMOTE_TP STRATA_DS41_REMOTE_SHARE STRATA_DS41_REMOTE_OWN STRATA_DS41_REMOTE_OWN_MIB STRATA_DS41_REMOTE_OWN_PROFILE STRATA_DS41_REMOTE_OWN_LOAD STRATA_DS41_REMOTE_OWN_SKIP_MIB UCOMM_BACKEND UCOMM_TIMEOUT_MS UCOMM_IB_DEV STRATA_DS41_LANE_SLEEP STRATA_DS41_LANE_SPIN; do
  [[ -v $name ]] && run_env+=("$name=${!name}")
done

decode_cmd=("$bin")
(( numa_interleave == 0 )) || decode_cmd=(numactl --interleave=all "$bin")

# Start only a small waiting shell. The decoder cannot allocate GPU or host
# memory until the scope and its limits have been verified below.
systemd-run --user --scope --quiet --unit="$unit" \
  --property=MemoryHigh=48G --property=MemoryMax=64G --property=MemorySwapMax=0 --property=OOMPolicy=kill \
  /bin/bash -c 'gate=$1; done_file=$2; ack=$3; shift 3; while [[ ! -e "$gate" ]]; do sleep 0.05; done; "$@"; rc=$?; printf "%s\n" "$rc" > "$done_file.tmp"; mv -- "$done_file.tmp" "$done_file"; while [[ ! -e "$ack" ]]; do sleep 0.05; done; exit "$rc"' \
  ds41-decode-gate "$gate_file" "$decoder_done" "$monitor_done" env -i "${runtime_env[@]}" "${run_env[@]}" \
  "${decode_cmd[@]}" "$model" "$prompt_arg" "$steps" "--context=$ctx" "--threads=$threads" "${extra[@]}" >"$log" 2>&1 &
runner=$!
scope_cgroup=
finished=0
stop_scope() {
  systemctl --user kill --kill-whom=all -s TERM "$unit.scope" >/dev/null 2>&1 || kill "$runner" 2>/dev/null || true
  sleep 1
  systemctl --user kill --kill-whom=all -s KILL "$unit.scope" >/dev/null 2>&1 || true
}
cleanup() {
  if (( ! finished )) && [[ -n "$runner" ]] && kill -0 "$runner" 2>/dev/null; then stop_scope; fi
  rm -rf -- "$gate_dir"
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

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

base_high=$(event_value high "$cgdir/memory.events"); base_max=$(event_value max "$cgdir/memory.events")
base_oom=$(event_value oom "$cgdir/memory.events"); base_oom_kill=$(event_value oom_kill "$cgdir/memory.events")
base_oom_group_kill=$(event_value oom_group_kill "$cgdir/memory.events")
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
  stop_scope
  status=0; wait "$runner" || status=$?
  finished=1
  echo "refusing benchmark: memory cgroup disappeared before the first live sample (exit=$status)" >&2
  tail -n 12 "$log" 2>/dev/null || true
  exit 1
fi
pressure10=$(pressure_value avg10 /proc/pressure/memory)
[[ "$vram_used" =~ ^[0-9]+$ && "$available" =~ ^[0-9]+$ && "$cgroup_used" =~ ^[0-9]+$ && "$pressure10" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  stop_scope; wait "$runner" || true; finished=1
  echo "refusing benchmark: cannot read initial live resource metrics" >&2; exit 1;
}
peak_used=$vram_used; peak_cgroup=$cgroup_used; min_available=$available; peak_psi10=$pressure10
started=$SECONDS; guard_reason=none; kernel_state=clear; runner_waited=0; final_events_confirmed=0
if (( vram_used > vram_total || vram_total-vram_used < preflight_vram_min )); then guard_reason=planned_gpu_budget
elif (( available < LIVE_RAM_KIB )); then guard_reason=available_ram
elif (( cgroup_used >= MEMORY_HIGH-4*GiB )); then guard_reason=scope_memory_high
elif awk -v p="$pressure10" -v lim="$LIVE_PSI_FULL_AVG10" 'BEGIN {exit !(p > lim)}'; then guard_reason=memory_pressure
fi
if [[ "$guard_reason" == none ]]; then
  if bad_event_since; then event=$bad_event_state; else event=invalid; fi
  if [[ "$event" == 1 ]]; then guard_reason=prelaunch_unsafe_journal_event; kernel_state=new_event
  elif [[ "$event" != 0 ]]; then guard_reason=system_journal_unreadable; kernel_state=unreadable
  fi
fi
if [[ "$guard_reason" != none ]]; then stop_scope; fi
if [[ "$guard_reason" == none ]]; then
  : > "$gate_file"
fi
while [[ "$guard_reason" == none ]] && kill -0 "$runner" 2>/dev/null; do
  vram_used=$(<"$gpu_sysfs/mem_info_vram_used")
  available=$(mem_available_kib)
  if ! cgroup_used=$(<"$cgdir/memory.current"); then
    guard_reason=scope_cgroup_unreadable
    stop_scope
    break
  fi
  (( vram_used > peak_used )) && peak_used=$vram_used
  (( cgroup_used > peak_cgroup )) && peak_cgroup=$cgroup_used
  (( available < min_available )) && min_available=$available
  printf '%s %s %s\n' "$vram_used" "$available" "$cgroup_used" >> "$samples"
  pressure10=$(pressure_value avg10 /proc/pressure/memory)
  if [[ ! "$vram_used" =~ ^[0-9]+$ || ! "$available" =~ ^[0-9]+$ || ! "$cgroup_used" =~ ^[0-9]+$ || ! "$pressure10" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    guard_reason=resource_metric_unreadable
  elif awk -v p="$pressure10" -v peak="$peak_psi10" 'BEGIN {exit !(p > peak)}'; then
    peak_psi10=$pressure10
  fi

  if [[ "$guard_reason" == none ]] && [[ -r "$log" ]] &&
     rg -q 'HSA_STATUS_ERROR_EXCEPTION|GPU HANG ANALYSIS|GPU fault detected|unspecified launch failure' "$log"; then
    guard_reason=gpu_runtime_exception
  fi
  if [[ "$guard_reason" == none ]]; then
    if bad_event_since; then event=$bad_event_state; else event=invalid; fi
    if [[ "$event" == 1 ]]; then guard_reason=new_unsafe_journal_event; kernel_state=new_event
    elif [[ "$event" != 0 ]]; then guard_reason=system_journal_unreadable; kernel_state=unreadable
    fi
  fi
  if [[ "$guard_reason" == none ]]; then
    if (( vram_used > vram_total || vram_total-vram_used < LIVE_VRAM )); then guard_reason=physical_vram_reserve
    elif (( available < LIVE_RAM_KIB )); then guard_reason=available_ram
    elif (( cgroup_used >= MEMORY_HIGH-4*GiB )); then guard_reason=scope_memory_high
    elif awk -v p="$pressure10" -v lim="$LIVE_PSI_FULL_AVG10" 'BEGIN {exit !(p > lim)}'; then guard_reason=memory_pressure
    elif (( SECONDS-started > max_seconds )); then guard_reason=timeout
    fi
  fi

  if [[ "$guard_reason" != none ]]; then stop_scope; break; fi

  if [[ ! -r "$cgdir/memory.events" ]]; then
    guard_reason=cgroup_metric_unreadable
    stop_scope
    break
  fi
  cur_high=$(event_value high "$cgdir/memory.events"); cur_max=$(event_value max "$cgdir/memory.events")
  cur_oom=$(event_value oom "$cgdir/memory.events"); cur_oom_kill=$(event_value oom_kill "$cgdir/memory.events")
  cur_oom_group_kill=$(event_value oom_group_kill "$cgdir/memory.events")
  if [[ ! "$cur_high" =~ ^[0-9]+$ || ! "$cur_max" =~ ^[0-9]+$ || ! "$cur_oom" =~ ^[0-9]+$ ||
        ! "$cur_oom_kill" =~ ^[0-9]+$ || ! "$cur_oom_group_kill" =~ ^[0-9]+$ ]]; then
    guard_reason=cgroup_metric_unreadable
  elif (( cur_high > base_high || cur_max > base_max || cur_oom > base_oom || cur_oom_kill > base_oom_kill ||
          cur_oom_group_kill > base_oom_group_kill )); then
    guard_reason=scope_memory_event
  fi
  if [[ "$guard_reason" != none ]]; then stop_scope; break; fi
  if [[ -f "$decoder_done" ]]; then
    # The child has exited while the wrapper keeps the scope alive. Read the
    # counters again after that exit before allowing systemd to remove it.
    final_high=$(event_value high "$cgdir/memory.events"); final_max=$(event_value max "$cgdir/memory.events")
    final_oom=$(event_value oom "$cgdir/memory.events"); final_oom_kill=$(event_value oom_kill "$cgdir/memory.events")
    final_oom_group_kill=$(event_value oom_group_kill "$cgdir/memory.events")
    if [[ ! "$final_high" =~ ^[0-9]+$ || ! "$final_max" =~ ^[0-9]+$ || ! "$final_oom" =~ ^[0-9]+$ ||
          ! "$final_oom_kill" =~ ^[0-9]+$ || ! "$final_oom_group_kill" =~ ^[0-9]+$ ]]; then
      guard_reason=cgroup_metric_unreadable
    elif (( final_high > base_high || final_max > base_max || final_oom > base_oom || final_oom_kill > base_oom_kill ||
            final_oom_group_kill > base_oom_group_kill )); then
      guard_reason=scope_memory_event
    fi
    if [[ "$guard_reason" != none ]]; then
      stop_scope
    else
      final_events_confirmed=1
      : > "$monitor_done"
    fi
    break
  fi
  sleep 0.2
done

status=0
if (( ! runner_waited )); then wait "$runner" || status=$?; fi
finished=1
if (( status != 0 )) && [[ "$guard_reason" == none ]]; then guard_reason=scope_process_exit; fi
if (( ! final_events_confirmed )) && [[ -r "$cgdir/memory.events" ]]; then
  final_high=$(event_value high "$cgdir/memory.events"); final_max=$(event_value max "$cgdir/memory.events")
  final_oom=$(event_value oom "$cgdir/memory.events"); final_oom_kill=$(event_value oom_kill "$cgdir/memory.events")
  final_oom_group_kill=$(event_value oom_group_kill "$cgdir/memory.events")
  if [[ ! "$final_high" =~ ^[0-9]+$ || ! "$final_max" =~ ^[0-9]+$ || ! "$final_oom" =~ ^[0-9]+$ ||
        ! "$final_oom_kill" =~ ^[0-9]+$ || ! "$final_oom_group_kill" =~ ^[0-9]+$ ]]; then
    guard_reason=cgroup_metric_unreadable
  elif (( final_high > base_high || final_max > base_max || final_oom > base_oom || final_oom_kill > base_oom_kill ||
          final_oom_group_kill > base_oom_group_kill )); then
    guard_reason=scope_memory_event
  fi
fi
if (( ! final_events_confirmed )) && [[ "$guard_reason" == none ]]; then guard_reason=cgroup_metric_unreadable; fi
if [[ -r "$log" ]] && rg -q 'HSA_STATUS_ERROR_EXCEPTION|GPU HANG ANALYSIS|GPU fault detected|unspecified launch failure' "$log"; then
  guard_reason=gpu_runtime_exception
fi
if bad_event_since; then final_event=$bad_event_state; else final_event=invalid; fi
if [[ "$final_event" == 1 ]]; then guard_reason=new_unsafe_journal_event; kernel_state=new_event
elif [[ "$final_event" != 0 ]]; then guard_reason=system_journal_unreadable; kernel_state=unreadable
fi
if [[ "$quality_logits" == 1 && "$guard_reason" == none && $status -eq 0 ]]; then
  quality_actual=$(stat -c '%s' -- "$quality_file" 2>/dev/null || true)
  [[ "$quality_actual" =~ ^[0-9]+$ ]] && (( quality_actual == quality_bytes )) || guard_reason=logits_capture_incomplete
fi
printf 'run=%s exit=%s guard=%s peak_vram_used_bytes=%s minimum_vram_free_bytes=%s peak_scope_memory_bytes=%s memory_high_bytes=%s memory_max_bytes=%s minimum_available_ram_kib=%s peak_memory_psi_full_avg10=%s unsafe_journal_events=%s\n' \
  "$run_id" "$status" "$guard_reason" "$peak_used" "$((vram_total-peak_used))" "$peak_cgroup" "$cg_high" "$cg_max" "$min_available" "$peak_psi10" "$kernel_state"
if [[ "$quality_logits" == 1 ]]; then printf 'teacher_forced_logits=%s expected_bytes=%s\n' "$quality_file" "$quality_bytes"; fi
tail -n 25 "$log" 2>/dev/null || true
[[ "$guard_reason" == none && $status -eq 0 ]]
