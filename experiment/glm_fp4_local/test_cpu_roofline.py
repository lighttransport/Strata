import importlib.util
import math
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('cpu_roofline',Path(__file__).with_name('cpu_roofline.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RooflineTest(unittest.TestCase):
    def test_missing_spec_remains_unknown(self):
        r=m.assess(72,80)
        self.assertIsNone(r['theoretical_memory_gb_s'])
        self.assertAlmostEqual(r['payload_over_read_control'],.9)
        self.assertEqual(r['nt1_useful_gops_s'],256)
    def test_ddr_transfer_units_and_scale_bytes(self):
        r=m.assess(72,80,channels=4,memory_mts=2500)
        self.assertEqual(r['theoretical_memory_gb_s'],80)
        self.assertAlmostEqual(r['payload_over_theoretical_memory'],.9)
        self.assertEqual(r['maximum_theoretical_memory_gb_s_for_90pct_payload'],80)
    def test_invalid_denominators(self):
        for value in [0,-1,math.inf,math.nan]:
            with self.assertRaises(ValueError):m.assess(72,value)
        for kw in [dict(channels=4),dict(memory_mts=2500),dict(channels=1.5,memory_mts=2500),dict(channels=4,memory_mts=0)]:
            with self.assertRaises(ValueError):m.assess(72,80,**kw)

if __name__=='__main__':unittest.main()
