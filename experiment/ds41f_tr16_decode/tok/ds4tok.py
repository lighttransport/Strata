import json, sys
from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders
D = '/mnt/nvme01/models/ds41f-gsq/IQ2_XXS/maya-ds4/tokenizer/'
def load():
    vocab = json.load(open(D + 'vocab.json'))
    merges = [tuple(l.rstrip('\n').split(' ', 1)) for l in open(D + 'merges.txt', encoding='utf-8') if l.strip() and not l.startswith('#version')]
    t = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False))
    pats = [r"\p{N}{1,3}", r"[一-龥぀-ゟ゠-ヿ]+",
            r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"]
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(p), behavior='isolated') for p in pats] +
                                              [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    t.decoder = decoders.ByteLevel()
    return t
if __name__ == '__main__':
    t = load()
    ids = [int(x) for x in open(sys.argv[1]).read().split()]
    txt = t.decode(ids)
    re_ids = t.encode(txt).ids
    print(len(vocab := t.get_vocab()), len(ids), 'roundtrip_equal=', re_ids == ids, 'bos?', ids[:1])
    print(repr(txt[:300]))
