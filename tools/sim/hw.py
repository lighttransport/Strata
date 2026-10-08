"""Hardware descriptions for the GLM simulator.

A HardwareConfig is a plain tree of dataclasses so it can be written as JSON, loaded back, and patched with
`--set cpu.dram_gbps=90` style overrides. Presets carry the measured numbers of the machines in docs/; the
"tr16" preset is the Threadripper 1950X + RTX 5060 Ti box that produced the calibration data.
"""
import dataclasses
import json
import pathlib

GIB = 2 ** 30
MIB = 2 ** 20


@dataclasses.dataclass
class Cpu:
    name: str = "generic"
    cores: int = 8                 # physical cores
    ghz: float = 3.0               # sustained all-core clock
    numa_nodes: int = 1
    gflops: float = 0.0            # FP32 FMA peak over all cores; 0 = derive 16 flop/cycle/core (AVX2)
    simd: str = "avx2"             # avx2 | avx512 (scales the quantized dot efficiency)
    reserve_cores: int = 1         # cores not used as expert workers (host thread / desktop)

    def peak_gflops(self):
        if self.gflops:
            return self.gflops
        return self.cores * self.ghz * (32.0 if self.simd == "avx512" else 16.0)

    def workers(self, threads=None):
        if threads:
            return min(threads, self.cores)
        return max(1, self.cores - self.reserve_cores)


@dataclasses.dataclass
class Memory:
    gib: float = 64.0              # usable host RAM for the engine
    dram_gbps: float = 40.0        # streaming read bandwidth reachable by the expert kernel, all nodes
    per_core_gbps: float = 5.0     # what one core can stream on its own (limits few-thread runs)
    page_cache_gib: float = 0.0    # extra RAM beyond `gib` that holds file pages (cgroup setups: 0)


@dataclasses.dataclass
class Gpu:
    name: str = "generic"
    count: int = 1
    vram_mib: float = 16384        # usable by CUDA/HIP after the driver's reservation
    bandwidth_gbps: float = 400.0  # memory bandwidth peak
    tflops: float = 20.0           # FP16/INT8 tensor throughput for prefill GEMMs
    gemv_efficiency: float = 0.85  # measured dense GEMV share of peak bandwidth (5060 Ti: 380/448)
    tier_gbps: float = 235.0       # routed-expert kernel on 8..24 experts per launch
    launch_us: float = 3.0         # exposed cost per kernel launch in the decode chain
    desktop_mib: float = 0.0       # VRAM held by the desktop session / other processes


@dataclasses.dataclass
class Pcie:
    gen: int = 3
    lanes: int = 8
    h2d_gbps: float = 0.0          # measured host-to-device; 0 = 0.9 x link peak
    d2h_gbps: float = 0.0
    latency_us: float = 16.0       # one mailbox round trip (16 KiB each way + 3 kernels), measured 15.6-17.1

    def link_peak_gbps(self):
        per_lane = {3: 0.985, 4: 1.969, 5: 3.938}[self.gen]
        return per_lane * self.lanes

    def h2d(self):
        return self.h2d_gbps or 0.9 * self.link_peak_gbps()

    def d2h(self):
        return self.d2h_gbps or self.h2d()


@dataclasses.dataclass
class Disk:
    read_gbps: float = 1.0         # one-stream page-fault reads (lazy prefill staging; tr16: ~1 GB/s)
    fetch_gbps: float = 1.0        # decode expert fetches by the RAM tier's reader threads (misses)


@dataclasses.dataclass
class Link:
    """Network link to a second node for remote expert tensor parallelism."""
    name: str = "none"
    rtt_us: float = 0.0            # decode-shaped round trip (4.8 KB request, 8 KB reply)
    gbps: float = 0.0              # bulk bandwidth (weight shipping)
    exposed_us: float = 0.0        # exposed wait per layer after overlap, measured where available


@dataclasses.dataclass
class Node:
    """A remote worker node: only its expert-row bandwidth matters for decode."""
    name: str = "none"
    expert_gbps: float = 0.0
    ram_gib: float = 0.0


@dataclasses.dataclass
class HardwareConfig:
    name: str = "generic"
    cpu: Cpu = dataclasses.field(default_factory=Cpu)
    memory: Memory = dataclasses.field(default_factory=Memory)
    gpu: Gpu = dataclasses.field(default_factory=Gpu)
    pcie: Pcie = dataclasses.field(default_factory=Pcie)
    disk: Disk = dataclasses.field(default_factory=Disk)
    link: Link = dataclasses.field(default_factory=Link)
    remote: Node = dataclasses.field(default_factory=Node)
    notes: str = ""

    def to_dict(self):
        return dataclasses.asdict(self)

    def copy(self):
        return from_dict(self.to_dict())

    def set(self, dotted, value):
        """Patch `cpu.dram_gbps=90`: the value is parsed with the field's type."""
        obj = self
        parts = dotted.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        field = {f.name: f for f in dataclasses.fields(obj)}[parts[-1]]
        current = getattr(obj, parts[-1])
        if isinstance(current, bool):
            value = str(value).lower() in ("1", "true", "yes")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(float(value))
        elif isinstance(current, float):
            value = float(value)
        setattr(obj, field.name, value)
        return self


def from_dict(data):
    def build(cls, part):
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in part:
                continue
            if f.name in _SUB:
                kwargs[f.name] = build(_SUB[f.name], part[f.name])
            else:
                kwargs[f.name] = part[f.name]
        return cls(**kwargs)
    return build(HardwareConfig, data)


_SUB = dict(cpu=Cpu, memory=Memory, gpu=Gpu, pcie=Pcie, disk=Disk, link=Link, remote=Node)


def tr16():
    """This PC: Threadripper 1950X, 2 NUMA nodes, 125 GiB DDR4, RTX 5060 Ti 16 GB on PCIe Gen3 x8.

    DRAM: 72 GB/s is the plain 15-thread stream with the q23 kernel (docs/GLM_Q2_DECODE_REDESIGN.md); the generic
    read test gave 50.7 GB/s on 16 threads (docs/glm53_flash_ddr4_bandwidth_measurement.json). PCIe: 7.18 GB/s is
    the nsys-measured H2D rate of prefill staging (docs/GLM_SINGLE_OPTIMIZATION.md). GPU: 380-393 GB/s dense
    GEMVs of a 448 GB/s peak; the desktop holds about 2.1 GiB of VRAM.
    """
    return HardwareConfig(
        name="tr16",
        cpu=Cpu(name="AMD Ryzen Threadripper 1950X", cores=16, ghz=3.4, numa_nodes=2, simd="avx2"),
        memory=Memory(gib=124.0, dram_gbps=72.0, per_core_gbps=4.8),
        gpu=Gpu(name="NVIDIA GeForce RTX 5060 Ti 16 GB", vram_mib=15907, bandwidth_gbps=448.0, tflops=47.0,
                gemv_efficiency=0.85, tier_gbps=235.0, launch_us=3.0, desktop_mib=2150),
        pcie=Pcie(gen=3, lanes=8, h2d_gbps=7.18, latency_us=16.0),
        disk=Disk(read_gbps=1.0, fetch_gbps=1.0),
        notes="calibration machine; model drive nvme0n1 is x1 (~1 GB/s), pack drive nvme1n1 is x4",
    )


def b550():
    """Ryzen 9 3950X (1 NUMA node) + RX 9070 XT 16 GB; expert rows measured at ~26 GB/s under load."""
    return HardwareConfig(
        name="b550",
        cpu=Cpu(name="AMD Ryzen 9 3950X", cores=16, ghz=3.5, numa_nodes=1, simd="avx2"),
        memory=Memory(gib=60.0, dram_gbps=40.0, per_core_gbps=5.0),
        gpu=Gpu(name="AMD Radeon RX 9070 XT", vram_mib=16304, bandwidth_gbps=640.0, tflops=97.0,
                gemv_efficiency=0.7, tier_gbps=200.0, launch_us=5.0, desktop_mib=500),
        pcie=Pcie(gen=4, lanes=16, h2d_gbps=20.0, latency_us=25.0),
        disk=Disk(read_gbps=2.0),
        notes="docs/GLM_B550_REAP50.md: the GPU memory clock was stuck at 96 MHz during measurement",
    )


def xeon_v100():
    """2 x Xeon Gold 6240 (36 cores, interleaved memory) + 1-2 x V100 32 GB on PCIe Gen3 x16 (docs/GLM53_V100.md).

    The measured runs (28.9 tok/s with two cards, 0.82-0.84 tier hits) used the pre-canonical tier kernels at
    about 50-100 GB/s and the original Q2 pack; `tier_gbps` reflects that. Use `--set gpu.tier_gbps=400` for the
    current kernel. `tflops` is the throughput the MMQ prefill GEMMs reach on SM70 (dp4a, no int8 tensor cores),
    fitted to the measured 501 / 304 tok/s prefill, not the card's FP16 tensor peak.
    """
    return HardwareConfig(
        name="xeon_v100",
        cpu=Cpu(name="2x Intel Xeon Gold 6240", cores=36, ghz=2.6, numa_nodes=2, simd="avx512"),
        memory=Memory(gib=160.0, dram_gbps=150.0, per_core_gbps=8.0),
        gpu=Gpu(name="NVIDIA V100 PCIe 32 GB", count=2, vram_mib=32000, bandwidth_gbps=900.0, tflops=50.0,
                gemv_efficiency=0.8, tier_gbps=100.0, launch_us=4.0),
        pcie=Pcie(gen=3, lanes=16, h2d_gbps=12.0, latency_us=20.0),
        disk=Disk(read_gbps=2.0, fetch_gbps=2.0),
    )


def b550_32g_3070():
    """Unmeasured: the B550's Ryzen 9 3950X (16 cores Zen 2) with 2 x 16 GB DDR4-2666 (42.7 GB/s peak, about
    34 GB/s for the expert stream) and an RTX 3070 8 GB (448 GB/s, PCIe Gen4 x16), one NVMe. Estimated from
    specs; the 3950X's expert rows measured 26 GB/s while the box ran other jobs (GLM_REMOTE_TP_COMM.md)."""
    return HardwareConfig(
        name="b550_32g_3070",
        cpu=Cpu(name="AMD Ryzen 9 3950X", cores=16, ghz=3.5, numa_nodes=1, simd="avx2"),
        memory=Memory(gib=30.0, dram_gbps=34.0, per_core_gbps=5.0),
        gpu=Gpu(name="NVIDIA GeForce RTX 3070 8 GB", vram_mib=7900, bandwidth_gbps=448.0, tflops=40.0,
                gemv_efficiency=0.85, tier_gbps=235.0, launch_us=3.5, desktop_mib=600),
        pcie=Pcie(gen=4, lanes=16, h2d_gbps=22.0, latency_us=15.0),
        disk=Disk(read_gbps=2.5, fetch_gbps=2.5),
        notes="unmeasured preset: B550 board with 32 GB and an RTX 3070 instead of the 9070 XT",
    )


def tr16_plus_b550_ib():
    """tr16 decoder with the B550 as a remote expert worker over ConnectX-3 InfiniBand."""
    hw = tr16()
    hw.name = "tr16+b550-ib"
    hw.link = Link(name="ib-connectx3", rtt_us=20.9, gbps=1.65, exposed_us=25.0)
    hw.remote = Node(name="b550", expert_gbps=26.0, ram_gib=60.0)
    return hw


def tr16_plus_b550_gbe():
    hw = tr16()
    hw.name = "tr16+b550-1gbe"
    hw.link = Link(name="1gbe", rtt_us=401.0, gbps=0.117, exposed_us=80.0)
    hw.remote = Node(name="b550", expert_gbps=26.0, ram_gib=60.0)
    return hw


PRESETS = {
    "tr16": tr16,
    "b550": b550,
    "xeon_v100": xeon_v100,
    "b550_32g_3070": b550_32g_3070,
    "tr16+b550-ib": tr16_plus_b550_ib,
    "tr16+b550-1gbe": tr16_plus_b550_gbe,
}


def load(spec):
    """A preset name, or a JSON file written by HardwareConfig.to_dict()."""
    if spec in PRESETS:
        return PRESETS[spec]()
    path = pathlib.Path(spec)
    if path.exists():
        return from_dict(json.loads(path.read_text()))
    raise ValueError(f"unknown hardware preset or file: {spec} (presets: {', '.join(PRESETS)})")
