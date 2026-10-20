"""Candidate-path oracle and unchanged sequential selector checks."""
from types import SimpleNamespace
import itertools
import unittest

try:
    import torch
    from glm_dflash2_worker import select_path
    from glm_dflash2_selector_lookahead import (
        path_indices, transition_tables, select_candidates, install_lookahead,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed')
class LookaheadTest(unittest.TestCase):
    @staticmethod
    def oracle(tables, discount):
        def score(path):
            previous, total = 0, 0.
            for row, index in enumerate(path):
                total += discount**row * tables[row][previous, index].item()
                previous = index
            return total
        paths = itertools.product(*(range(table.shape[1]) for table in tables))
        return max(paths, key=lambda path: (score(path), tuple(-i for i in path)))

    @staticmethod
    def model(rows=7):
        hidden, projection = torch.randn(rows, 8), torch.randn(5, 8)
        predecessor, successor = torch.randn(19, 5), torch.randn(19, 5)
        model = SimpleNamespace(
            gemm_f32=True, last_hidden=hidden,
            linear=lambda x, name: x @ projection.T,
            rounded=lambda x: x, selector_scale=1.,
            w={'candidate_selector.predecessor_codebook': predecessor,
               'candidate_selector.successor_codebook': successor})
        return model, projection

    def test_discounted_paths_match_exhaustive_oracle_and_stable_ties(self):
        torch.manual_seed(253)
        for _ in range(20):
            tables = [torch.randint(-4, 5, (1, 3)).float()]
            tables += [torch.randint(-4, 5, (3, 3)).float() for _ in range(3)]
            for discount in (.25, .5, 1.):
                expected = self.oracle(tables, discount)
                self.assertEqual(path_indices(tables, discount).tolist(), list(expected))
        self.assertEqual(path_indices([torch.zeros(1, 3), torch.zeros(3, 3)], 1.).tolist(), [0, 0])

    def test_zero_discount_matches_existing_sequential_selector(self):
        for seed in range(20):
            torch.manual_seed(seed)
            model, projection = self.model()
            ids = torch.stack([torch.randperm(19)[:4] for _ in range(7)])
            unary = torch.randn(7, 4)
            actual = select_candidates(model, ids, unary, 3, 0.)
            expected = select_path(model.last_hidden, ids, unary, 3, projection,
                                   model.w['candidate_selector.predecessor_codebook'],
                                   model.w['candidate_selector.successor_codebook'])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_installed_selector_matches_oracle_without_changing_inputs(self):
        torch.manual_seed(254)
        model, _ = self.model(rows=3)
        ids = torch.stack([torch.randperm(19)[:3] for _ in range(3)])
        unary = torch.randn(3, 3)
        weights = {key: value.clone() for key, value in model.w.items()}
        hidden, original_ids, original_unary = model.last_hidden.clone(), ids.clone(), unary.clone()
        tables = transition_tables(model, ids, unary, 3)
        indices = self.oracle(tables, .5)
        expected = torch.tensor([ids[row, index].item() for row, index in enumerate(indices)])
        install_lookahead(model, .5)
        torch.testing.assert_close(model.select(ids, unary, 3), expected, rtol=0, atol=0)
        for key, value in model.w.items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        for value, original in ((model.last_hidden, hidden), (ids, original_ids), (unary, original_unary)):
            torch.testing.assert_close(value, original, rtol=0, atol=0)

    def test_bad_discount_and_incompatible_paths_rejected(self):
        for value in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                path_indices([torch.zeros(1, 3)], value)
        with self.assertRaises(ValueError):
            path_indices([torch.zeros(1, 3), torch.zeros(2, 2)], .5)

    def test_zero_discount_keeps_selector_and_positive_requires_fp32(self):
        original = lambda *args: None
        model = SimpleNamespace(gemm_f32=False, select=original)
        install_lookahead(model, 0.)
        self.assertIs(model.select, original)
        with self.assertRaisesRegex(ValueError, 'FP32-output'):
            install_lookahead(model, .25)
        for discount in (-1., 2., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                install_lookahead(model, discount)


if __name__ == '__main__':
    if torch is not None:
        torch.set_num_threads(1)
    unittest.main()
