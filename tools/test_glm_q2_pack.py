"""Integration checks for a locally generated expert sidecar; source files stay read-only."""
import argparse
import pathlib
import struct
import subprocess
import tempfile
from gguf_reader import GGUFFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("pack", type=pathlib.Path)
    parser.add_argument("--converter", default="build-glm/strata-glm-q2-pack")
    args = parser.parse_args()
    info = GGUFFile(args.pack)
    with args.pack.open("rb") as source:
        header = source.read(info.data_start)
    def verify(path, error=None):
        result = subprocess.run([args.converter, "--verify", str(args.model), str(path)], capture_output=True, text=True, timeout=300)
        if error is None:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0 and error in result.stderr, result.stderr
    verify(args.pack)
    with tempfile.TemporaryDirectory(prefix="strata-pack-test-") as directory:
        path = pathlib.Path(directory) / "bad.gguf"
        def write(value, size=None):
            with path.open("wb") as out:
                out.write(value)
                if size is not None:
                    out.truncate(size)
        def change_metadata(key, data):
            value = bytearray(header)
            marker = struct.pack("<Q", len(key)) + key.encode()
            offset = value.index(marker) + len(marker) + 4
            value[offset:offset + len(data)] = data
            return value
        write(change_metadata("strata.expert_pack.version", struct.pack("<I", 999)))
        verify(path, "version or source fingerprint mismatch")
        source_id = info.metadata["strata.expert_pack.source"]
        write(change_metadata("strata.expert_pack.source", struct.pack("<Q", source_id ^ 1)))
        verify(path, "version or source fingerprint mismatch")
        write(header)
        verify(path, "invalid payload bounds")
        write(header, args.pack.stat().st_size)
        verify(path, "payload checksum mismatch")
    print("expert pack PASS: valid source, version/source mismatch, truncation, payload corruption")


if __name__ == "__main__":
    main()
