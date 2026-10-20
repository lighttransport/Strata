#!/usr/bin/env python3
"""Python wrapper for ucfile (tools/ucomm/tools/ucfile.c): discovery, remote file management and transfer jobs.

    from ucfile import UcFile
    uc = UcFile()                                   # finds ../build/ucfile, or $UCFILE_BIN
    uc.discover()                                   # [{'name': 'node-b', 'reply_from': ..., 'ib': True, ...}]
    uc.df('node-b'); uc.ls('node-b', '/data'); uc.rm('node-b', '/data/old.bin')
    job = uc.job(self_name='node-a', nodes={'node-a': {}, 'node-b': {}}, verify='crc32c',
                 transfers=[uc.copy('node-a:/models/a.gguf', 'node-b:/data/a.gguf'),
                            uc.shard('node-a:/models/b.gguf', to=['node-b', 'node-c'], weights=[1, 1],
                                     dst='{node}:/data/b.{index}', manifest='/models/b.manifest.json')])
    result = uc.run(job, progress=print)            # dict: per node and item: bytes, seconds, GBps, status

Command line: python3 ucfile.py discover | df NODE | ls NODE PATH | run JOB.json [--dry-run]
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent


def find_binary():
    for candidate in (os.environ.get("UCFILE_BIN"), HERE.parent / "build" / "ucfile", shutil.which("ucfile")):
        if candidate and pathlib.Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError("ucfile binary not found: build tools/ucomm (make) or set UCFILE_BIN")


class UcFileError(RuntimeError):
    pass


class UcFile:
    def __init__(self, binary=None, port=None):
        self.binary = binary or find_binary()
        self.port = port

    def _args(self, *args):
        cmd = [self.binary, *map(str, args)]
        if self.port:
            cmd += ["--port", str(self.port)]
        return cmd

    def _json(self, *args, timeout=60):
        p = subprocess.run(self._args(*args), capture_output=True, text=True, timeout=timeout)
        out = p.stdout.strip()
        if not out:
            raise UcFileError(p.stderr.strip() or f"ucfile {' '.join(map(str, args))} failed ({p.returncode})")
        lines = [json.loads(line) for line in out.splitlines() if line.strip()]
        return lines[0] if len(lines) == 1 else lines

    # ---- discovery and management
    def discover(self, timeout_ms=1000, to=()):
        """Agents answering a broadcast on every interface (Ethernet and IPoIB), plus unicast `to` addresses."""
        args = ["discover", "--timeout", timeout_ms]
        for addr in to:
            args += ["--to", addr]
        p = subprocess.run(self._args(*args), capture_output=True, text=True, timeout=timeout_ms / 1000 + 10)
        peers = json.loads(p.stdout or "[]")
        return [dict(p["info"], reply_from=p["reply_from"], via_ipoib=p["via_ipoib"]) for p in peers]

    def df(self, node):
        return self._json("df", node)

    def ls(self, node, path):
        return self._json("ls", node, path)

    def stat(self, node, path):
        return self._json("stat", node, path)

    def rm(self, node, *paths):
        return self._json("rm", node, *paths)

    def mkdir(self, node, path):
        return self._json("mkdir", node, path)

    def agent(self, roots, name=None, verbose=False):
        """Starts an agent on this machine; returns the Popen (terminate() stops it)."""
        args = ["agent"] + [x for r in roots for x in ("--root", r)]
        if name:
            args += ["--name", name]
        if verbose:
            args.append("-v")
        return subprocess.Popen(self._args(*args), stderr=subprocess.PIPE, text=True)

    # ---- jobs
    @staticmethod
    def copy(src, dst, offset=None, length=None):
        """A whole file, or [offset, offset + length) of it: 'node:/path' -> 'node:/path'."""
        t = {"src": src, "dst": dst}
        if offset is not None:
            t["offset"] = int(offset)
        if length is not None:
            t["length"] = int(length)
        return t

    @staticmethod
    def shard(src, to, dst, weights=None, manifest=None):
        """Split src into contiguous 1 MiB-aligned ranges by weight, one per node in `to`. dst uses {node}, {index}."""
        s = {"to": list(to), "dst": dst}
        if weights:
            s["weights"] = list(weights)
        if manifest:
            s["manifest"] = manifest
        return {"src": src, "shard": s}

    @staticmethod
    def job(self_name, nodes, transfers, **options):
        """options: backend ('auto'|'ib'|'udp'), verify ('none'|'crc32c'), chunk_mib, depth, overwrite,
        reserve_gib, keep_mtime, root_addr, data_port, port, timeout_ms, discover (extra addresses)."""
        return dict(options, self=self_name, nodes=nodes, transfers=transfers)

    def run(self, job, dry_run=False, progress=None, timeout=None):
        """Runs a job (dict or path to JSON). progress(node_or_none, dict) gets progress lines when given."""
        with tempfile.TemporaryDirectory() as tmp:
            path = job if isinstance(job, (str, pathlib.Path)) else pathlib.Path(tmp) / "job.json"
            if not isinstance(job, (str, pathlib.Path)):
                path.write_text(json.dumps(job))
            args = ["run", path] + (["--dry-run"] if dry_run else []) + (["-v"] if progress else [])
            p = subprocess.Popen(self._args(*args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if progress:
                for line in p.stderr:
                    node, _, rest = line.partition(": ")
                    try:
                        progress(node, json.loads(rest))
                    except ValueError:
                        progress(None, {"log": line.rstrip()})
            out, err = p.communicate(timeout=timeout)
            if not out.strip():
                raise UcFileError((err or "").strip() or f"ucfile run failed ({p.returncode})")
            result = json.loads(out.strip().splitlines()[-1])
            if p.returncode and not dry_run:
                raise UcFileError(json.dumps(result))
            return result

    @staticmethod
    def reassemble(manifest_path, out_path, shard_dir=None):
        """Concatenates shards that are on this machine (by the manifest's paths, or the same names in shard_dir)
        into out_path, checking every shard's length and the total size."""
        m = json.loads(pathlib.Path(manifest_path).read_text())
        with open(out_path, "wb") as out:
            for s in sorted(m["shards"], key=lambda s: s["offset"]):
                src = pathlib.Path(shard_dir) / pathlib.Path(s["path"]).name if shard_dir else pathlib.Path(s["path"])
                if src.stat().st_size != s["length"] or out.tell() != s["offset"]:
                    raise UcFileError(f"shard {s['index']} ({src}) does not match the manifest")
                with open(src, "rb") as f:
                    shutil.copyfileobj(f, out, 64 << 20)
            if out.tell() != m["size"]:
                raise UcFileError("reassembled size differs from the source")
        return out_path


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    uc, cmd, rest = UcFile(), argv[0], argv[1:]
    if cmd == "discover":
        result = uc.discover()
    elif cmd in ("df", "ls", "stat", "rm", "mkdir"):
        result = getattr(uc, cmd)(*rest)
    elif cmd == "run":
        result = uc.run(rest[0], dry_run="--dry-run" in rest,
                        progress=lambda node, msg: print(node, msg, file=sys.stderr))
    elif cmd == "reassemble":
        result = uc.reassemble(*rest)
    else:
        print(__doc__)
        return 2
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
