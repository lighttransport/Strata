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


def write_gguf(path, metadata, tensors, pad_header=True):
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
    path.write_bytes(header + (bytes(-len(header) % 32) if pad_header else b'') + payload)


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
        # The loader and the assembler call the host checks written in C11 (src/host), so a build that does not
        # go through CMake compiles that unit next to them.
        host = [str(ROOT / 'src/host/glm_expert_pack.c'), str(ROOT / 'src/host/glm_names.c')]
        cls.assembler = directory / 'assemble'
        subprocess.run(['c++', '-O2', '-std=c++20', '-I' + str(ROOT / 'include'),
                        str(ROOT / 'tools/glm_assemble.cpp')] + host + ['-o', str(cls.assembler)], check=True)
        helper = directory / 'fingerprint.cpp'
        helper.write_text('#include "strata/core/model.hpp"\n#include <iostream>\n'
                          'int main(int argc,char** argv){std::cout<<strata::core::ModelArtifact(argv[1]).source_fingerprint();}')
        cls.fingerprint = directory / 'fingerprint'
        subprocess.run(['c++', '-O2', '-std=c++20', '-I' + str(ROOT / 'include'), str(helper),
                        '-o', str(cls.fingerprint)] + host, check=True)
        # Loads an assembled file through the loader's validation and prints the retained-projection count it accepted.
        census = directory / 'census.cpp'
        census.write_text('#include "strata/core/model.hpp"\n#include <iostream>\n'
                          'int main(int argc,char** argv){try{strata::core::ModelArtifact m(argv[1]);'
                          'const auto* r=m.tensors().begin()->second.file->get("strata.expert_pack.retained");'
                          'std::cout<<(r?r->u:0);return 0;}catch(const std::exception& e){std::cerr<<e.what();return 1;}}')
        cls.census = directory / 'census'
        subprocess.run(['c++', '-O2', '-std=c++20', '-I' + str(ROOT / 'include'), str(census)] + host +
                       ['-o', str(cls.census)], check=True)
        helper = directory / 'overlay.cpp'
        helper.write_text('''#include "strata/core/model.hpp"
#include <iostream>
int main(int argc,char** argv){try {
    strata::core::ModelArtifact m(argv[1]);
    const auto gate=m.at("blk.3.ffn_gate_exps.weight"), up=m.at("blk.3.ffn_up_exps.weight"), fixed=m.at("test.fixed");
    m.overlay_experts(argv[2]);
    if(argc==4)m.overlay_experts(argv[3]);
    if(m.at("blk.3.ffn_gate_exps.weight").data()!=gate.data() ||
       m.at("blk.3.ffn_up_exps.weight").data()!=up.data() || m.at("test.fixed").data()!=fixed.data())
        throw std::runtime_error("untouched tensor changed");
    for(int l=3;l<=44;++l){const auto& d=m.at("blk."+std::to_string(l)+".ffn_down_exps.weight");
        if(d.tensor->type!=10 || d.bytes!=84 || d.data()[0]!=77)throw std::runtime_error("down overlay differs");}
    std::cout<<"Q22 down overlay preserves gate/up/fixed tensors";
}catch(const std::exception& e){std::cerr<<e.what();return 1;}}
''')
        cls.overlay = directory / 'overlay'
        subprocess.run(['c++', '-O2', '-std=c++20', '-I' + str(ROOT / 'include'), str(helper),
                        '-o', str(cls.overlay)] + host, check=True)

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

    def partial_pack_meta(self, skip):
        """Pack metadata for a pack that omits the first `skip` converted projections (a real pack hashes only its tensors)."""
        omitted = {'strata.expert_pack.hash.' + t[0] for t in self.converted[:skip]}
        return {k: v for k, v in self.pack_meta.items() if k not in omitted}

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

    def test_retained_projections(self):
        # A pack that leaves one layer's projections in their original (K-quant) format: the assembly copies them as
        # they are and records the count, and the assembled file loads with retained == 3.
        write_gguf(self.pack, self.partial_pack_meta(3), self.converted[3:])
        result = self.run_assembly()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('retained_original_projections=3', result.stdout)
        model = GGUFFile(self.output)
        self.assertEqual(model.metadata['strata.expert_pack.retained'], 3)
        kept = {t[0]: t for t in self.original[:3]}
        with self.output.open('rb') as stream:
            for tensor in model.tensors:
                if tensor.name in kept:
                    self.assertEqual(tensor.type_id, 12)
                    stream.seek(model.data_start + tensor.offset)
                    self.assertEqual(stream.read(tensor.expected_bytes()), kept[tensor.name][3])
        self.assertEqual(int(subprocess.check_output([self.census, self.output])), 3)

    def test_retained_count_is_enforced(self):
        # Changing the recorded retained count of an assembly with original-format projections is rejected by the loader.
        write_gguf(self.pack, self.partial_pack_meta(3), self.converted[3:])
        self.assertEqual(self.run_assembly().returncode, 0)
        data = bytearray(self.output.read_bytes())
        key = b'strata.expert_pack.retained'
        at = data.index(key) + len(key)
        self.assertEqual(struct.unpack_from('<II', data, at), (4, 3))   # u32 type, value 3
        struct.pack_into('<I', data, at + 4, 2)
        patched = self.directory / 'patched.gguf'
        patched.write_bytes(data)
        result = subprocess.run([self.census, patched], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('retained-projection count mismatch', result.stderr)

    def test_metadata_only_first_shard(self):
        # Unsloth's first shard has KV metadata only and no tensors (its file ends before the aligned data offset).
        shards = [self.directory / f'meta-0000{i}-of-00003.gguf' for i in (1, 2, 3)]
        half = len(self.original) // 2
        total = len(self.original) + 1
        for i, (path, tensors) in enumerate(zip(shards, ([], self.original[:half], self.original[half:] + [self.fixed]))):
            meta = dict(self.metadata)
            meta.update({'split.no': ('u16', i), 'split.count': ('u16', 3), 'split.tensors.count': ('i32', total)})
            write_gguf(path, meta, tensors, pad_header=bool(tensors))   # the metadata shard ends unpadded
        source = int(subprocess.check_output([self.fingerprint, shards[0]]))
        self.pack_meta['strata.expert_pack.source'] = ('u64', source)
        write_gguf(self.pack, self.pack_meta, self.converted)
        result = subprocess.run([self.assembler, shards[0], self.pack, self.output], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(GGUFFile(self.output).tensors), total)

    def test_split_base(self):
        # The base may be a split GGUF: the shards are read through the shard-aware artifact and split.* keys are dropped.
        half = len(self.original) // 2
        shards = [self.directory / f'split-0000{i}-of-00002.gguf' for i in (1, 2)]
        total = len(self.original) + 1
        for i, (path, tensors) in enumerate(zip(shards, (self.original[:half], self.original[half:] + [self.fixed]))):
            meta = dict(self.metadata)
            meta.update({'split.no': ('u16', i), 'split.count': ('u16', 2), 'split.tensors.count': ('i32', total)})
            write_gguf(path, meta, tensors)
        source = int(subprocess.check_output([self.fingerprint, shards[0]]))
        self.pack_meta['strata.expert_pack.source'] = ('u64', source)
        write_gguf(self.pack, self.pack_meta, self.converted)
        result = subprocess.run([self.assembler, shards[0], self.pack, self.output], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        model = GGUFFile(self.output)
        self.assertFalse([k for k in model.metadata if k.startswith('split.')])
        self.assertEqual(len(model.tensors), total)
        self.assertEqual(int(subprocess.check_output([self.census, self.output])), 0)

    def make_q22_sidecar(self):
        result = self.run_assembly()
        self.assertEqual(result.returncode, 0, result.stderr)
        path = self.directory / 'q22.gguf'
        source = int(subprocess.check_output([self.fingerprint, self.output]))
        tensors = [(f'blk.{l}.ffn_down_exps.weight', 10, [256, 1, 1], bytes([77]) * 84)
                   for l in range(3, 45)]
        metadata = {'strata.expert_pack.version': ('u32', 1), 'strata.expert_pack.source': ('u64', source)}
        metadata.update({'strata.expert_pack.hash.' + t[0]: ('u64', fnv(t[3])) for t in tensors})
        write_gguf(path, metadata, tensors)
        return path, metadata, tensors

    def test_assembled_q22_down_overlay(self):
        path, _, _ = self.make_q22_sidecar()
        before = self.output.read_bytes()
        result = subprocess.run([self.overlay, self.output, path], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_bytes(), before)
        result = subprocess.run([self.overlay, self.output, path, path], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('expert pack requires an original GGUF model', result.stderr)

    def test_assembled_q22_wrong_source(self):
        path, metadata, tensors = self.make_q22_sidecar()
        metadata['strata.expert_pack.source'] = ('u64', 0)
        write_gguf(path, metadata, tensors)
        result = subprocess.run([self.overlay, self.output, path], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('source fingerprint mismatch', result.stderr)

    def test_assembled_q22_rejects_gate_replacement(self):
        path, metadata, tensors = self.make_q22_sidecar()
        for part in ('gate', 'up'):
            tensor = (f'blk.3.ffn_{part}_exps.weight', 10, [256, 1, 1], bytes([77]) * 84)
            tensors.append(tensor)
            metadata['strata.expert_pack.hash.' + tensor[0]] = ('u64', fnv(tensor[3]))
        write_gguf(path, metadata, tensors)
        result = subprocess.run([self.overlay, self.output, path], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid type transition', result.stderr)


if __name__ == '__main__':
    unittest.main()
