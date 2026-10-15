import unittest
from glm_mixed_fraction_calibrate import climb, fraction_for, initial_groups


class FractionCalibrateTest(unittest.TestCase):
    def test_fraction_rounds_back_to_groups(self):
        for groups in (14, 18):
            for g in range(1, groups):
                self.assertEqual(round(fraction_for(g, groups) * groups), g)

    def test_initial_guess(self):
        self.assertEqual(initial_groups(14), 7)
        self.assertEqual(initial_groups(14, 611, 406), 8)
        self.assertEqual(initial_groups(14, 1000, 1), 13)

    def test_climb_finds_peak_and_measures_once(self):
        calls = []
        def evaluate(g):
            calls.append(g)
            return 900 - (g - 11) ** 2
        best, scores = climb(evaluate, 14, 7)
        self.assertEqual(best, 11)
        self.assertEqual(len(calls), len(set(calls)))

    def test_failed_neighbour_is_skipped(self):
        best, _ = climb(lambda g: None if g == 8 else 100 - abs(g - 5), 14, 7)
        self.assertEqual(best, 5)


if __name__ == '__main__':
    unittest.main()
