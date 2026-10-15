import csv
from pathlib import Path
import tempfile
import unittest
from glm_runner_profile_report import analyze, nsys_lanes, read_nsys


class MultiGpuProfileTest(unittest.TestCase):
    def test_devices_and_streams_are_separated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            with path.open("w") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Start (ns)", "Duration (ns)", "Device", "Stream", "Name"]
                )
                for i in range(12):
                    writer.writerow([i * 100000, 20000, 0, 7, "wait_add"])
                    writer.writerow([i * 100000 + 20000, 20000, 0, 7, "glm_kda_kernel"])
                    writer.writerow([i * 100000 + 50000, 30000, 1, 7, "wait_add"])
                    writer.writerow(
                        [i * 100000 + 10000, 60000, 0, 8, "resident_gu_kernel"]
                    )
            self.assertEqual(nsys_lanes(path), {("0", "7"), ("1", "7"), ("0", "8")})
            a = analyze(list(read_nsys(path, "0", "7")))
            b = analyze(list(read_nsys(path, "1", "7")))
            self.assertAlmostEqual(a["period_ms"]["mean"], 0.1)
            self.assertAlmostEqual(a["kernel_ms_mean"], 0.04)
            self.assertAlmostEqual(b["kernel_ms_mean"], 0.03)
            self.assertEqual(len(list(read_nsys(path, "0", "8"))), 12)

    def test_missing_identity_cannot_satisfy_explicit_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text("Start,Duration,Name\n0,1,kernel\n")
            with self.assertRaises(ValueError):
                list(read_nsys(path, device="0"))


if __name__ == "__main__":
    unittest.main()
