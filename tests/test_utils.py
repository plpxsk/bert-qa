import unittest
import numpy as np

from utils import find_context_start_end
from extra.train_pt import load_model_tokenizer_hf


class TestUtils(unittest.TestCase):

    def test_find_context_start_end(self):
        sequence_ids = [None, 0, 0, None, 1, 1, 1, None]
        estart, eend = 4, 6

        start, end = find_context_start_end(sequence_ids)
        self.assertEqual(start, estart)
        self.assertEqual(end, eend)

    def test_load_model_tokenizer_hf(self):
        batch = ["This is an example of BERT working in MLX."]
        model_str = "bert-base-uncased"

        model, tokenizer = load_model_tokenizer_hf(model_str=model_str, hf_auto_class="AutoModel")
        model2, tokenizer2 = load_model_tokenizer_hf(model_str=model_str, hf_auto_class="Bert")

        tok = tokenizer(batch)
        tok2 = tokenizer2(batch)

        np.testing.assert_equal(tok["input_ids"], tok2["input_ids"])
        np.testing.assert_equal(tok["token_type_ids"], tok2["token_type_ids"])
        np.testing.assert_equal(tok["attention_mask"], tok2["attention_mask"])
        np.testing.assert_equal(tok.sequence_ids(), tok2.sequence_ids())
