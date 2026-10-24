# Worker node memory: holding nearly all of the RAM as expert rows

A two-node run keeps a share of every routed expert on the worker node (`strata-ds41-worker`, docs/DS41F_PLAN.md
section 8c). The more of its RAM the worker can hold, the larger the share it takes off the primary node. This page
is the setup that lets a 64 GB worker hold 59.6 GiB of rows (DS4.1-Flash share 5/9) with the rest of the machine
still working, how it was measured, and the same setup for a 128 GB worker.

Measured on the B550 (Ryzen 9 3950X, 4 x 16 GiB DDR4, RTX 5060 Ti 16 GB on PCIe 4 x8, ConnectX-3 on PCIe 3 x8,
Ubuntu 24.04, kernel 7.0.0-38-generic, NVIDIA driver 615.71, CUDA 13.2), 2026-10-23 to 2026-10-25. The 128 GB numbers
are computed from those measurements and have not been run.

## Result

| rows (B550, 62.7 GiB visible) | storage | outcome |
|---|---|---|
| 59.625 GiB (share 5/9) | 2 MiB hugetlb pool, `cudaHostRegister` | mapped 4.1 s, written 4.0 s, registered 1.3 s; IB registration of all 40 layers (1.49 GiB each) 0.006 s per layer; GPU copy 14.4 GB/s; **1.23 GiB of RAM left** outside the pool |
| 59.55 GiB | `cudaHostAlloc`, one buffer per layer | 0.35 GiB left; the worker's own buffers do not fit beside it |
| 57.5 GiB | `cudaHostAlloc` | 2.06 GiB left, no memory pressure |
| 53.7 GiB and up | `cudaHostAlloc`, default kernel settings | the machine stalled for over 5 minutes (see below) |

The worker needs about 0.3 GiB more than the probe (message buffers, CPU lane), which leaves about 0.9 GiB for the OS
at share 5/9. That is tight; with the settings below an overrun kills the worker at once instead of stalling the
machine.

## What stops a large allocation by default

Each of these was hit on the B550 with the stock Ubuntu configuration.

1. **Kexec HandOver scratch memory (5.17 GiB).** Ubuntu's 7.0 kernels are built with
   `CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y`. The kernel marks a scratch area as CMA pageblocks at boot (2,649 blocks
   of 2 MiB on the B550, `/proc/pagetypeinfo`), so only movable pages may use it. No CMA area is registered:
   `CmaTotal` is 0 and the boot log says "0K cma-reserved", but `CmaFree` reads about 5.2 GiB. Page-locked memory is
   never placed there, so the B550 ran out of usable memory at 54.6 GiB of rows with 4.8 GiB still free. Kernel
   7.0.0-34 shows the same. `kho=off` removes it (`CmaFree` 0).
2. **Swap and page-locked memory.** `cudaHostAlloc` on this driver pins shared-memory pages that stay on the
   swap-backed reclaim lists (they show as `Shmem` and `Cached`; `MemAvailable` still counts them correctly). With
   swap on, the kernel kept trying to reclaim them near the limit, drained the file cache to 32 MB and stalled every
   process; the allocation hung inside the driver. With swap off and `min_ttl_ms` set, the same limit ends in an
   instant kill of the largest process instead.
3. **systemd-oomd.** Ubuntu kills a whole user session when memory pressure on `user@.service` stays above 50 % for
   20 s. It killed unrelated tmux sessions twice (2026-10-21 and 2026-10-23), never the process causing the pressure.
4. **Per-page bookkeeping.** With 4 KiB pages the kernel keeps 64 bytes per page (1 GiB on a 64 GiB machine), and
   page-locking 60 GiB through `cudaHostAlloc` costs another 0.8 GiB. 2 MiB hugetlb pages with
   `hugetlb_optimize_vmemmap` return about 0.93 GiB (`MemTotal` grew from 65,725,592 to 66,697,688 kB on the B550),
   and their pages are not on the reclaim lists.

Huge pages do not change read bandwidth here: 29.3 GB/s with 4 KiB and with huge pages at 12 threads
(docs/B550_MEMORY_BANDWIDTH.md). They are used for capacity only. InfiniBand registration of a 1.49 GiB layer drops
from 0.08-0.12 s to 0.006 s as a side effect.

## Setup (persistent, needs sudo)

Run on the worker node. All of it survives reboots.

```bash
# 1. Kernel command line: no KHO scratch area; reserve the hugetlb pool at boot (memory is not fragmented yet)
#    PAGES: see the table below (30600 for share 5/9 on 64 GB)
sudo sed -i 's/^GRUB_CMDLINE_LINUX="/GRUB_CMDLINE_LINUX="kho=off hugepagesz=2M hugepages=PAGES hugetlb_free_vmemmap=on /' /etc/default/grub
sudo update-grub

# 2. Free the struct-page memory of huge pages reserved later at runtime as well
echo 'vm.hugetlb_optimize_vmemmap = 1' | sudo tee /etc/sysctl.d/91-hugetlb.conf

# 3. Out of memory: kill instead of stalling, and kill the process that asked
echo 'vm.oom_kill_allocating_task = 1' | sudo tee /etc/sysctl.d/90-oom.conf
echo 'w /sys/kernel/mm/lru_gen/min_ttl_ms - - - - 1000' | sudo tee /etc/tmpfiles.d/mglru-min-ttl.conf

# 4. No swap
sudo swapoff -a
sudo sed -i 's|^/swap.img|#/swap.img|' /etc/fstab
sudo rm -f /swap.img

# 5. No systemd-oomd
sudo systemctl disable --now systemd-oomd.service systemd-oomd.socket
sudo systemctl mask systemd-oomd.service systemd-oomd.socket

# 6. Never kill the InfiniBand subnet manager (opensm.service only launches the opensm@<port> instances)
sudo mkdir -p /etc/systemd/system/opensm@.service.d
printf '[Service]\nOOMScoreAdjust=-1000\n' | sudo tee /etc/systemd/system/opensm@.service.d/oom.conf

# 7. No desktop (about 0.5 GiB)
sudo systemctl set-default multi-user.target

sudo reboot
```

`systemctl isolate multi-user.target` on a running machine also stops the `opensm@` instances; the InfiniBand port
stays ACTIVE until the next link reset, then stays down. Restart them with `sudo systemctl restart opensm`, or reboot.

Check after the reboot:

```bash
grep -E 'CmaFree|HugePages_Total|MemAvailable|SwapTotal' /proc/meminfo   # 0, PAGES, about 2 GiB (64 GB), 0
cat /sys/kernel/mm/lru_gen/min_ttl_ms /proc/sys/vm/hugetlb_optimize_vmemmap  # 1000, 1
systemctl is-active systemd-oomd; pidof opensm                             # inactive, two PIDs
```

## Pool size

One DS4.1-Flash ninth is 11.93 GiB of rows (6,106 pages of 2 MiB). The pool needs the rows' pages plus a few spare.

| share | rows | pool pages | fits 64 GB (62.7 GiB visible) | fits 128 GB (about 125 GiB visible) |
|---|---|---|---|---|
| 4/9 | 47.70 GiB | 24,500 | yes, about 13 GiB left | yes |
| 5/9 | 59.63 GiB | 30,600 | yes, 1.23 GiB left (measured) | yes |
| 6/9 | 71.55 GiB | 36,800 | no | yes |
| 7/9 | 83.48 GiB | 42,900 | no | yes |
| 8/9 | 95.40 GiB | 49,000 | no | yes, about 27 GiB left (computed) |

8/9 is the largest share the decoder places on a worker. GLM's `--remote-tp-share` takes any fraction, so a GLM
worker can use a pool sized to the machine instead.

**128 GB worker.** Size the pool to leave about 4 GiB outside it: the idle headless system (1.9 GiB measured on the
B550), `cudaHostRegister` (1.07 % of the rows, 0.64 GiB at 59.6 GiB), the worker's buffers (about 0.3 GiB) and
about 1 GiB of headroom. For DS4.1 use 49,000 pages (share 8/9). For a GLM worker, about 61,000 pages (119 GiB)
leaves about 4 GiB; check `MemAvailable` with the pool in place before raising it. On a machine with two NUMA nodes
the boot parameter splits the pool evenly (`hugepages=N` reserves N/2 per node); set it per node with
`hugepages=0:N0,1:N1` if one node holds more of the system's own memory.

## Running the worker

```bash
strata-ds41-worker --root=ADDR:PORT --rows=hugetlb --reserve-gib=0.75 [--threads=N]
```

- `--rows=hugetlb` maps the rows on the pool (populated) before the decoder is told to send, so a pool shortfall
  refuses the setup with a message instead of failing mid-transfer. The fit check uses the pool's free pages, and
  `--reserve-gib` then covers only the RAM outside the pool.
- `--rows=hugetlb` cannot be combined with `--weights-dir` (the rows are not kept between runs; they are sent again,
  about 110 s at 5/9 by extrapolation from 67 s at 3/9).
- `tools/ds41_remote_tp.sh` does not pass `--rows` yet; start the worker by hand or add it to the wrapper.

Optional cap on the worker alone (the memory controller is delegated to user services on Ubuntu, so no sudo):
`systemd-run --user --scope -p MemoryMax=60.2G -p MemorySwapMax=0 strata-ds41-worker ...`. A cap was verified to
kill a page-locking process at the limit instantly (4 GiB cap, 6 GiB request).

## Giving the memory back

- At runtime: `sudo sysctl vm.nr_hugepages=0` returns the pool to the system until the next boot. Reserving it again
  at runtime needs defragmented memory: `sudo sysctl vm.drop_caches=3; sudo sysctl vm.compact_memory=1; sudo sysctl
  vm.nr_hugepages=PAGES`. Without the first two, the B550 got 30,371-30,378 of 30,600 pages. Wait a minute after
  dropping caches before starting the worker: the first probe right after a drop was killed by the `min_ttl_ms`
  check (every page was younger than 1 s; not confirmed).
- For good: remove `hugepagesz=2M hugepages=PAGES hugetlb_free_vmemmap=on` from `GRUB_CMDLINE_LINUX`, run
  `sudo update-grub` and reboot. Keep `kho=off` unless the node needs kexec handover.
