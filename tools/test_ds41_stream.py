#!/usr/bin/env python3
"""Worker staging-ring regression tests with an injected transport (no model/GPU/network).

python3 tools/test_ds41_stream.py --ggml-include <llama.cpp>/ggml/include
"""
import argparse
import os
import pathlib
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ggml-include", required=True, type=pathlib.Path)
    parser.add_argument("--cuda-object", type=pathlib.Path, help="optional compiled CUDA ds41.cu object for mocked dispatch tests")
    parser.add_argument("--cuda-root", type=pathlib.Path, default=pathlib.Path("/usr/local/cuda"))
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="ds41-stream-test-") as tmp:
        env_object = pathlib.Path(tmp) / "glm_env.o"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-I" + str(root / "include"),
                        "-c", str(root / "src/host/glm_env.c"), "-o", str(env_object)], check=True)
        config_test = pathlib.Path(tmp) / "config_test"
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++20", "-I" + str(root / "include"),
                        str(root / "tests/ds41/stream_config_test.cpp"), str(env_object), "-o", str(config_test)], check=True)
        subprocess.run([str(config_test)], check=True, timeout=10)
        binary = pathlib.Path(tmp) / "stream_test"
        subprocess.run([
            os.environ.get("CXX", "c++"), "-std=c++20", "-O1", "-pthread",
            "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
            "-I" + str(root / "include"), "-I" + str(root / "tools/ucomm/include"),
            "-I" + str(args.ggml_include.resolve()),
            str(root / "tests/ds41/stream_failure_test.cpp"), "-o", str(binary),
        ], check=True)
        for argv, expected in (([], 0), (["fail-send"], 1)):
            result = subprocess.run([str(binary), *argv], capture_output=True, text=True, timeout=10)
            if result.returncode != expected:
                raise AssertionError(f"{argv}: expected exit {expected}, got {result.returncode}\n{result.stderr}")
            if argv and "injected send failure" not in result.stderr:
                raise AssertionError(result.stderr)
        print("PASS: sparse-shard configuration, ordered ring reuse and failed-send cancellation")
        if args.cuda_object:
            dispatch = pathlib.Path(tmp) / "attention_dispatch"
            lib = args.cuda_root / "lib64"
            subprocess.run([
                os.environ.get("CXX", "c++"), "-std=c++20", "-I" + str(root / "include"),
                "-I" + str(args.cuda_root / "include"),
                str(root / "tests/ds41/attention_capability_test.cpp"), str(args.cuda_object.resolve()),
                "-L" + str(lib), "-Wl,-rpath," + str(lib), "-lcudart", "-o", str(dispatch),
            ], check=True)
            subprocess.run([str(dispatch)], check=True, timeout=10)
            subprocess.run([str(dispatch), "default"], check=True, timeout=10)
            print("PASS: mocked CUDA capability rejection, checked setup and default dispatch")


if __name__ == "__main__":
    main()
