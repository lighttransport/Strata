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
