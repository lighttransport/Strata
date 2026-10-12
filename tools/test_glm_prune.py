"""Expert pruning tests: strata-glm-prune-experts (byte slicing, checksums, loader acceptance, plan refusals) and
tools/glm_reap_rank.py (saliency order). Small synthetic GGUFs, no GPU or model download."""
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

from gguf_reader import GGUFFile
from test_glm_assemble import fnv, model_metadata, write_gguf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OLD, NEW = 32, 16
LAYERS = range(3, 45)  # the routed layers the loader's assembled-pack rule checks


def tensor(model, name):
    return next(t for t in model.tensors if t.name == name)


def tensor_data(model, name):
    t = tensor(model, name)
    with open(model.path, 'rb') as fh:
        fh.seek(model.data_start + t.offset)
        return fh.read(t.expected_bytes())


def expert_bytes(layer, part, expert, size):
    return bytes([(layer * 7 + {'gate': 1, 'up': 2, 'down': 3}[part] * 31 + expert) % 251]) * size


class PruneToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        directory = pathlib.Path(cls.work.name)
        host = [str(ROOT / 'src/host/glm_expert_pack.c'), str(ROOT / 'src/host/glm_names.c')]
        cls.tool = directory / 'prune'
        subprocess.run(['c++', '-O2', '-std=c++20', '-I' + str(ROOT / 'include'),
                        str(ROOT / 'tools/glm_prune_experts.cpp')] + host + ['-o', str(cls.tool)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        d = pathlib.Path(self.temp.name)
        self.input, self.plan, self.output = d / 'in.gguf', d / 'plan.txt', d / 'out.gguf'
        meta = model_metadata()
        meta['glm5next.expert_count'] = ('u32', OLD)
        tensors = []
        for layer in LAYERS:
            for part in ('gate', 'up', 'down'):
                size = 110 if part == 'down' else 84  # one Q3_K / Q2_K block per expert
                data = b''.join(expert_bytes(layer, part, e, size) for e in range(OLD))
                tensors.append((f'blk.{layer}.ffn_{part}_exps.weight', 11 if part == 'down' else 10, [256, 1, OLD], data))
            tensors.append((f'blk.{layer}.ffn_gate_inp.weight', 0, [4, OLD],
                            struct.pack(f'<{4 * OLD}f', *[float(layer * 1000 + e * 10 + k) for e in range(OLD) for k in range(4)])))
            tensors.append((f'blk.{layer}.exp_probs_b.bias', 0, [OLD], struct.pack(f'<{OLD}f', *[float(e) for e in range(OLD)])))
        tensors.append(('test.fixed', 0, [4], struct.pack('<4f', 1., -2., 3., 4.)))
        meta.update({'strata.expert_pack.version': ('u32', 1), 'strata.expert_pack.source': ('u64', 12345),
                     'strata.expert_pack.calibrated': ('bool', True), 'strata.expert_pack.assembled': ('bool', True),
                     'strata.expert_pack.retained': ('u32', 0)})
        meta.update({'strata.expert_pack.hash.' + t[0]: ('u64', fnv(t[3])) for t in tensors if t[0].endswith('_exps.weight')})
        write_gguf(self.input, meta, tensors)
        self.keep = {layer: [e for e in range(OLD) if (e + layer) % 2 == 0] for layer in LAYERS}

    def write_plan(self, keep=None, header=f'experts {OLD} {NEW}'):
        keep = self.keep if keep is None else keep
        self.plan.write_text(header + '\n' + ''.join(f'{l} ' + ' '.join(map(str, ids)) + '\n' for l, ids in keep.items()))

    def run_tool(self):
        return subprocess.run([self.tool, self.input, self.plan, self.output], capture_output=True, text=True)

    def test_slices_rehashes_and_reloads(self):
        self.write_plan()
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        out = GGUFFile(self.output)
        self.assertEqual(out.metadata['glm5next.expert_count'], NEW)
        self.assertEqual(out.metadata['strata.prune.source_experts'], OLD)
        self.assertEqual(out.metadata['strata.expert_pack.retained'], 0)
        self.assertTrue(out.metadata['strata.expert_pack.assembled'])
        self.assertEqual(out.metadata['strata.prune.plan_hash'], fnv(self.plan.read_bytes()))
        for layer in (3, 20, 44):
            keep = self.keep[layer]
            for part in ('gate', 'up', 'down'):
                name = f'blk.{layer}.ffn_{part}_exps.weight'
                size = 110 if part == 'down' else 84
                data = tensor_data(out, name)
                self.assertEqual(data, b''.join(expert_bytes(layer, part, e, size) for e in keep), name)
                self.assertEqual(out.metadata['strata.expert_pack.hash.' + name], fnv(data))
                self.assertEqual(tensor(out, name).shape[-1], NEW)
            router = struct.unpack(f'<{4 * NEW}f', tensor_data(out, f'blk.{layer}.ffn_gate_inp.weight'))
            self.assertEqual(router[:4], tuple(float(layer * 1000 + keep[0] * 10 + k) for k in range(4)))
            bias = struct.unpack(f'<{NEW}f', tensor_data(out, f'blk.{layer}.exp_probs_b.bias'))
            self.assertEqual(bias, tuple(float(e) for e in keep))
        self.assertEqual(tensor_data(out, 'test.fixed'), struct.pack('<4f', 1., -2., 3., 4.))

    def test_refuses_bad_plans_and_existing_output(self):
        bad = dict(self.keep)
        del bad[20]
        self.write_plan(bad)
        self.assertIn('missing from the plan', self.run_tool().stderr)
        self.write_plan({**self.keep, 20: list(reversed(self.keep[20]))})
        self.assertIn('ascending', self.run_tool().stderr)
        self.write_plan(header=f'experts 64 {NEW}')
        self.assertIn('does not match', self.run_tool().stderr)
        self.write_plan()
        self.output.write_bytes(b'x')
        self.assertIn('refusing to overwrite', self.run_tool().stderr)
        self.assertFalse(pathlib.Path(str(self.output) + '.partial').exists())


GGUF_PY = os.environ.get('STRATA_GGUF_PY', '/mnt/nvme02/work/llama.cpp/gguf-py')


@unittest.skipUnless(pathlib.Path(GGUF_PY, 'gguf').is_dir(), 'llama.cpp gguf-py not available (STRATA_GGUF_PY)')
class RankTest(unittest.TestCase):
    def test_saliency_order(self):
        """saliency = mean gate * sqrt(sum_k ||W[:,k]||^2 E[h_k^2]); expert e is built so its saliency grows with e,
        except expert 5, which is made the strongest; expert 7 is never visited (saliency 0)."""
        E, cols, rows = 32, 4, 2
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            calib = d / 'calib'
            calib.mkdir()
            meta = model_metadata()
            meta['glm5next.expert_count'] = ('u32', E)
            tensors = []
            for layer in (3, 4):
                w = []
                for e in range(E):
                    scale = 50.0 if e == 5 else 1.0 + e * 0.1
                    w += [scale] * (rows * cols)
                tensors.append((f'blk.{layer}.ffn_down_exps.weight', 0, [cols, rows, E], struct.pack(f'<{len(w)}f', *w)))
                tensors.append((f'blk.{layer}.ffn_gate_exps.weight', 0, [cols, rows, E], bytes(4 * cols * rows * E)))
                (calib / f'blk.{layer}.down.f32').write_bytes(struct.pack(f'<{E * cols}f', *([1.0] * (E * cols))))
            write_gguf(d / 'm.gguf', meta, tensors)
            counts = [0 if e == 7 else 10 for e in range(E)]
            (calib / 'coverage.json').write_text(json.dumps({'projections': [
                {'name': f'blk.{l}.gu', 'counts': counts} for l in (3, 4)]}))
            (calib / 'gates.json').write_text(json.dumps({'layers': [
                {'layer': l, 'sum': [c * 0.1 for c in counts], 'sum_sq': [c * 0.01 for c in counts]} for l in (3, 4)]}))
            subprocess.run([sys.executable, str(ROOT / 'tools/glm_reap_rank.py'), str(d / 'm.gguf'), str(calib),
                            '--keep', '16', '--out', str(d / 'plan.txt'), '--jobs', '2'], check=True, capture_output=True,
                           env={**os.environ, 'STRATA_GGUF_PY': GGUF_PY})
            lines = (d / 'plan.txt').read_text().split('\n')
            self.assertEqual(lines[0], f'experts {E} 16')
            for line in lines[1:3]:
                ids = [int(x) for x in line.split()[1:]]
                self.assertEqual(ids, sorted([5] + list(range(17, 32))))  # strongest + the 15 largest
                self.assertNotIn(7, ids)


if __name__ == '__main__':
    unittest.main()
