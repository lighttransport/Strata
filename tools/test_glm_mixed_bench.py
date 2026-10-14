"""CPU-only checks for the experimental timing harness."""
import unittest
from glm_mixed_bench import stream_health
class HealthTest(unittest.TestCase):
    def test_unique_stream(self):
        self.assertIsNone(stream_health(list(range(512)))['consecutive_cycle'])
    def test_cycle_and_eos(self):
        self.assertEqual(stream_health(list(range(32))*4)['consecutive_cycle']['period'],32)
        self.assertEqual(stream_health([154829]*80)['consecutive_cycle']['period'],1)
    def test_repeated_fragment_not_consecutive(self):
        data=list(range(32))+list(range(100,200))+list(range(32))+list(range(300,400))+list(range(32))
        report=stream_health(data)
        self.assertIsNone(report['consecutive_cycle'])
        self.assertEqual(report['max_32gram_occurrences'],3)
if __name__=='__main__': unittest.main()
