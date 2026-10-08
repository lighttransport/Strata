"""Byte-preserving assembly and rejection tests; small synthetic GGUFs, no GPU/model download."""
import pathlib
import struct
import subprocess
import tempfile
import unittest
from gguf_reader import GGUFFile
from gguf_writer import GGUFWriter

ROOT = pathlib.Path(__file__).resolve().parents[1]


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def write_gguf(path, metadata, tensors):
    writer = GGUFWriter()
    header = struct.pack('<IIQQ', 0x46554747, 3, len(tensors), len(metadata))
    for key, (kind, value) in metadata.items():
        header += writer._kv_bytes(key, kind, value)
    payload = b''
    for name, kind, shape, data in tensors:
        encoded = name.encode()
        header += struct.pack('<Q', len(encoded)) + encoded + struct.pack('<I', len(shape))
        header += struct.pack(f'<{len(shape)}Q', *shape) + struct.pack('<IQ', kind, len(payload))
        payload += data + bytes(-len(data) % 32)
    path.write_bytes(header + bytes(-len(header) % 32) + payload)


def model_metadata():
    integers = dict(block_count=45, embedding_length=256, context_length=128, expert_count=1,
                    expert_used_count=1, vocab_size=2, nextn_predict_layers=0, leading_dense_block_count=3,
                    expert_feed_forward_length=256, feed_forward_length=256, expert_group_count=1,
                    expert_group_used_count=1, expert_gating_func=0, expert_shared_count=1)
    integers.update({'attention.head_count': 1, 'hyper_connection.count': 4,
                     'hyper_connection.sinkhorn_iterations': 1, 'attention.kv_lora_rank': 1,
                     'attention.q_lora_rank': 1, 'kda.head_dim': 1, 'ssm.conv_kernel': 1,
                     'attention.indexer.head_count': 1, 'attention.indexer.key_length': 1,
                     'attention.indexer.top_k': 4, 'rope.dimension_count': 1})
    result = {'general.architecture': ('string', 'glm5next')}
    result.update({'glm5next.' + k: ('u32', v) for k, v in integers.items()})
    result.update({'glm5next.' + k: ('f32', v) for k, v in {
        'hyper_connection.epsilon': 1e-6, 'attention.layer_norm_rms_epsilon': 1e-6,
        'kda.gate_lower_bound': -5., 'expert_weights_scale': 1., 'swiglu_limit': 7.}.items()})
    result['glm5next.attention.head_count_kv'] = ('array:u32', [0] * 45)
    result['glm5next.expert_weights_norm'] = ('bool', True)
    result['test.signed'] = ('i64', -17)
    result['test.strings'] = ('array:string', ['hello', '世界'])
    return result


class AssemblyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        directory = pathlib.Path(cls.work.name)
        cls.assembler = directory / 'assemble'
        subprocess.run(['c++', '-O2', '-std=c++20', '-I' + str(ROOT / 'include'),
                        str(ROOT / 'tools/glm_assemble.cpp'), '-o', str(cls.assembler)], check=True)
        helper = directory / 'fingerprint.cpp'
        helper.write_text('#include "strata/core/model.hpp"\n#include <iostream>\n'
                          'int main(int argc,char** argv){std::cout<<strata::core::ModelArtifact(argv[1]).source_fingerprint();}')
        cls.fingerprint = directory / 'fingerprint'
        subprocess.run(['c++', '-O2', '-std=c++20', '-I' + str(ROOT / 'include'), str(helper),
                        '-o', str(cls.fingerprint)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = pathlib.Path(self.temp.name)
        self.base, self.pack, self.output = [self.directory / n for n in ('base.gguf', 'pack.gguf', 'out.gguf')]
        self.original, self.converted = [], []
        for layer in range(3, 45):
            for part in ('gate', 'up', 'down'):
                name = f'blk.{layer}.ffn_{part}_exps.weight'
                self.original.append((name, 12, [256, 1, 1], bytes([layer]) * 144))
                self.converted.append((name, 11 if part == 'down' else 10, [256, 1, 1],
                                       bytes([layer + 1]) * (110 if part == 'down' else 84)))
        self.fixed = ('test.fixed', 0, [4], struct.pack('<4f', 1., -2., 3., 4.))
        self.metadata = model_metadata()
        write_gguf(self.base, self.metadata, self.original + [self.fixed])
        source = int(subprocess.check_output([self.fingerprint, self.base]))
        self.pack_meta = {'strata.expert_pack.version': ('u32', 1), 'strata.expert_pack.source': ('u64', source)}
        self.pack_meta.update({'strata.expert_pack.hash.' + t[0]: ('u64', fnv(t[3])) for t in self.converted})
        write_gguf(self.pack, self.pack_meta, self.converted)

    def run_assembly(self):
        return subprocess.run([self.assembler, self.base, self.pack, self.output], capture_output=True, text=True)

    def test_exact_bytes_and_metadata(self):
        result = self.run_assembly()
        self.assertEqual(result.returncode, 0, result.stderr)
        model = GGUFFile(self.output)
        self.assertTrue(model.metadata['strata.expert_pack.assembled'])
        self.assertEqual(model.metadata['test.signed'], -17)
        self.assertEqual(model.metadata['test.strings'], ['hello', '世界'])
        expected = {t[0]: t for t in self.converted + [self.fixed]}
        with self.output.open('rb') as stream:
            for tensor in model.tensors:
                reference = expected.pop(tensor.name)
                self.assertEqual((tensor.type_id, tensor.shape), (reference[1], reference[2]))
                stream.seek(model.data_start + tensor.offset)
                self.assertEqual(stream.read(tensor.expected_bytes()), reference[3])
        self.assertFalse(expected)
        self.assertIn('refusing to overwrite', self.run_assembly().stderr)

    def test_wrong_source(self):
        self.pack_meta['strata.expert_pack.source'] = ('u64', 0)
        write_gguf(self.pack, self.pack_meta, self.converted)
        self.assertIn('source fingerprint mismatch', self.run_assembly().stderr)
        self.assertFalse(self.output.exists())

    def test_corrupt_payload(self):
        model = GGUFFile(self.pack)
        with self.pack.open('r+b') as stream:
            stream.seek(model.data_start)
            stream.write(b'\xff')
        self.assertIn('payload checksum mismatch', self.run_assembly().stderr)
        self.assertFalse(self.output.exists())

    def test_incomplete_pack(self):
        write_gguf(self.pack, self.pack_meta, self.converted[3:])
        self.assertIn('incomplete assembled Q23 experts', self.run_assembly().stderr)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
