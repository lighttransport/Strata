# Standalone RAM check

Copy this directory to another app or machine. It has no Strata imports and
does not change Strata's default startup. Supports Linux x86-64 with AVX2.
Requires Python 3, g++ with OpenMP, and `stressapptest` for stress/check mode.
On Ubuntu: `sudo apt install g++ stressapptest`. No GPU, model download or
root access is needed to run the checks. Windows and other CPU architectures
are not supported by this version.

Run bandwidth and then a five-minute verified RAM stress test:

```bash
python3 tools/memory_check/memory_check.py --output /tmp/ram-check
```

Run either part separately:

```bash
python3 tools/memory_check/memory_check.py --mode bandwidth --threads 1 4 8 16
python3 tools/memory_check/memory_check.py --mode stress --seconds 300 --memory-mib 49152
```

Pass `--stressapptest /path/to/stressapptest` to use an extracted package
without installing it. The tool does not download dependencies automatically.

Use as an opt-in gate before any app, including the Strata runner:

```bash
python3 tools/memory_check/memory_check.py --output /tmp/strata-preflight \
  --run python3 serve/server.py YOUR_EXISTING_SERVER_ARGUMENTS
```

Replace the command after `--run` with your actual launch command. It starts
only after bandwidth and stress both succeed. No shell expansion is applied.
The wrapper returns the launched application's exit status. Without `--run`,
exit codes are 0 for pass, 1 for a failed check, 2 for setup/tool errors, and
130 for interruption. Do not run simultaneous checks into the same output folder.

Without `--run`, stdout is JSON; progress goes to stderr. With `--run`, the
launched app inherits standard input/output after the JSON result is printed.
The output folder contains
`result.json` and, for stress mode, `stress.log`. Other apps can inspect the
JSON `passed`, `bandwidth`, `peak_read_gbps`, and `stress` fields. A result for
bandwidth-only mode does not certify stability. Missing or inconclusive stress
results fail the gate; detected data errors stop the stress process early.
Stress passes only when its completion summary and wall time both reach the
requested duration. An early `PASS`, missing completion summary, timeout or
forced termination fails the gate and prevents app launch.

The benchmark uses three 1 GiB arrays by default, one warmup and seven timed
runs per kernel. Rates are decimal GB/s of logical bytes, not controller
traffic. Writes are sampled for correctness; the verified stress test is the
stability check. `--array-mib` adjusts each array size; use arrays much larger
than CPU cache. `--pages thp` requests huge pages and reports the actual amount.
CPU affinity is recorded and OpenMP threads are bound to available CPUs.

Stress allocation defaults to 80% of available RAM with at least 4 GiB
reserved. Cgroup v2 limits, including parent limits, are also considered. Explicit allocations exceeding
the available budget are rejected. Other processes can change available RAM
after the check; run on an otherwise idle machine. Cgroup v1 limits are not
detected. Kernel/ECC logs and SMBIOS speed are not checked automatically.
Check the active DIMM speed manually with `sudo dmidecode --type 17`.

There is no universal bandwidth floor. `--min-read-gbps NUMBER` sets an
application-specific floor and skips stress/launch if bandwidth falls below it.
On this B550/Ryzen 9 3950X/64 GiB machine, the 16-thread read median was
29.63 GB/s at the user's confirmed 2166 MT/s and 42.62 GB/s at 3000 MT/s.
The 3000 setting produced memory mismatches starting at 86 seconds; higher
bandwidth did not mean stable memory. Five minutes passing is a short check,
not proof of long-term overclock stability. Rerun after changing BIOS settings.

Run the failure-path tests without stressing RAM:

```bash
python3 -m unittest discover -s tools/memory_check -p 'test_*.py'
```

## CPU and GPU system pre-check

The same standalone directory also includes `system_check.py`. It reuses the
RAM bandwidth tool and adds CPU dependency-chain timing, CPU AVX2/FMA
throughput, pinned PCIe transfers, GPU streaming copy, and library FP32 GEMM.
It needs no model files or Strata build. GPU checks require a CUDA toolkit
(`nvcc`, cuBLAS) or ROCm (`hipcc`, hipBLAS) and access to the GPU devices.
The HIP build uses `/opt/rocm/lib` as its runtime library search path.

CPU/RAM only:

```bash
python3 tools/memory_check/system_check.py --threads 1 4 16 --output /tmp/system-check
```

Include the GPU, and optionally record a RAM setting supplied by the user:

```bash
python3 tools/memory_check/system_check.py --backend hip --ram-mt-s 2666 \
  --output /tmp/system-check
# For NVIDIA, use --backend cuda.
```

Use `--device N` to select the runtime GPU ordinal. The tool records and
monitors the PCI identities of all enumerated AMD/NVIDIA display GPUs rather
than assuming a fixed PCI address. A motherboard change can change addresses.
Run on an otherwise idle system; the tool does not stop other GPU workloads.

`--run COMMAND ...` starts an app only after all requested measurements and
correctness checks succeed. It returns the app's exit code. Without a launch,
exit codes are 0 for passed measurements, 2 for a failed check/setup error,
and 130 for interruption. No universal performance thresholds are imposed.
An app can inspect `result.json`; raw GPU rows are in `gpu.jsonl`, with separate
build, stderr and kernel logs. Partial logs remain after a failed check.

By default the GPU checks allocate two 256 MiB pinned host buffers, two
256 MiB GPU transfer buffers and three 2048-by-2048 FP32 matrices. They reserve
256 MiB of free VRAM and check available host RAM before allocation. Use
`--gpu-buffer-mib 16..1024`, `--gemm-n 256..4096` (multiples of 256), and
`--repeats 3..31` to adjust the workload. GPU execution is limited to 120 seconds
by default (`--timeout`, maximum 600). Compilation time is separate.
The monitor checks PCI configuration and, when readable, new kernel fault
messages; an already logged GPU fault prevents launch. Fault/timeout handling
kills only the benchmark child process group, never resets the GPU or stops
an unrelated service. If kernel logs cannot be read, the result explicitly
reports `kernel_log_checked: false`.

Each GPU measurement uses one warmup and seven timed samples, with four
operations per sample. Event timing includes completion. Transfers count one
buffer's bytes; the GPU copy counts both read and write bytes. Rates use
decimal GB/s and are effective logical bandwidth, not PCIe or DRAM controller
traffic. All transferred/copied bytes are checked; output guards are checked.
Buffers smaller than the GPU cache can measure cache bandwidth instead of
VRAM bandwidth. PCIe directions run separately, not full duplex.

The CPU compute test uses one warmup and seven timed samples. Its scalar FMA
chain measures dependency latency plus loop/OpenMP overhead; compare the
one-thread row for latency. Eight independent AVX2 accumulators measure
register-resident FP32 throughput (two FLOPs per FMA), with exact checks of
all lanes. This is not an in-model quantized expert throughput measurement.

GPU compute is library FP32 SGEMM, with a structured exactly representable
input and full output/guard verification. CUDA uses pedantic math to avoid
silently measuring TF32. These numbers do not measure BF16 tensor/matrix-core
or integer GEMM throughput. Both CUDA and HIP sources are supported; SYCL is
not implemented by this standalone tool.

`passed` means the requested short measurements and their correctness checks
passed. It does not certify system stability or replace the verified RAM
stress test above. The existing `memory_check.py` command and launch gate
keep their behavior; this extension does not change Strata startup.

### Disk read measurement

Add `--disk-file /path/to/existing/large/file` to the system pre-check, or run
just the disk test:

```bash
python3 tools/memory_check/disk_read.py /path/to/model.gguf \
  --mib 1024 --repeats 3 --output /tmp/disk-read.json
```

The test opens an existing regular file read-only with Linux `O_DIRECT`; it
never creates, truncates or modifies the input. Symlinks and raw devices are
rejected. It reads the first 1 GiB three times in 8 MiB blocks by default,
using a page-aligned buffer. A shorter file is read up to its last complete
4 KiB page; files smaller than 1 MiB are rejected. SHA-256 checksums must match
between repeats, and source size/timestamps must stay unchanged. This checks
repeatability, not agreement with a known reference checksum or filesystem
integrity. Use a file containing real allocated data, not a sparse file.

`median_read_GB_s` times read calls only. `median_wall_GB_s` also includes
hashing and loop overhead. Every sample and checksum is recorded. Direct I/O
bypasses the OS page cache; device/controller caching remains possible.
There is no buffered fallback: unsupported direct I/O or short reads fail
the check. The deadline is checked between reads; the system wrapper also
limits the child to 130 seconds. It does not clear global filesystem caches.

For the integrated check, use `--disk-mib` and `--disk-repeats`. Disk checking
is opt-in and contributes to the `--run` launch gate when requested. GPU
checks remain separately opt-in with `--backend`.
