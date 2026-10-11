import unittest
from glm_chat_stops import chat_stop_ids


class ChatStopTest(unittest.TestCase):
    def test_metadata_eos_and_serving_role_stops(self):
        metadata={'tokenizer.ggml.eos_token_id':100,
                  'tokenizer.ggml.tokens':['ordinary','<|user|>','<|observation|>','<|assistant|>']}
        self.assertEqual(chat_stop_ids(metadata),[1,2,100])

    def test_explicit_stops_are_deduplicated_without_vocabulary(self):
        self.assertEqual(chat_stop_ids({'tokenizer.ggml.eos_token_id':2,
                                       'tokenizer.ggml.eot_token_id':3,
                                       'tokenizer.ggml.eom_token_id':3}),[2,3])


if __name__=='__main__':unittest.main()
