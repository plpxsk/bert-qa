import unittest
import numpy as np

from utils import get_answers, find_context_start_end
from model import load_model_tokenizer



class TestUtils(unittest.TestCase):

    def setUp(self):
        model_str = "bert-base-uncased"
        weights_pretrain_path = "weights/bert-base-uncased.npz"

        model, tokenizer = load_model_tokenizer(hf_model=model_str,
                                                weights_pretrain_path=weights_pretrain_path)
        self.model = model
        self.tokenizer = tokenizer
        self.question = "How many programming languages does BLOOM support?"
        self.context = "BLOOM has 176 billion parameters and can generate text in 46 natural languages and 13 programming languages."  # noqa

    def test_bert_qa(self):
        """For this trained model, answer quality will be poor, so check only logic
        """

        expected_tokenized_inputs = self.tokenizer(self.question, self.context, return_tensors=None)
        exp_len = len(expected_tokenized_inputs["input_ids"])

        exp_context_start, exp_context_end = find_context_start_end(
            expected_tokenized_inputs.sequence_ids())

        tokenized_inputs = self.tokenizer(self.question, self.context, return_tensors="mlx")
        start_logits, end_logits = self.model(**tokenized_inputs)

        self.assertEqual(exp_len, tokenized_inputs["input_ids"].shape[1])
        self.assertEqual(exp_len, start_logits.shape[1])
        self.assertEqual(exp_len, end_logits.shape[1])

        answers = get_answers(start_logits, end_logits, tokenized_inputs.sequence_ids(),
                              top_k=3)

        # top answer would be answers[0]
        for answer in answers:

            start = answer["start"]
            end = answer["end"]
            score = answer["score"]
            print(score)

            # answers should be within context
            self.assertTrue(exp_context_start <= start)
            self.assertTrue(exp_context_start <= end)
            self.assertTrue(start <= exp_context_end)
            self.assertTrue(end <= exp_context_end)

            # scores should be real numbers
            self.assertTrue(score > -float("inf"))
            self.assertTrue(score < float("inf"))

            answer_tokens = tokenized_inputs["input_ids"][0, start: end + 1]
            # tokenizer can't use MLX array as input
            answer_str = self.tokenizer.decode(np.array(answer_tokens))

            # answer should be verbatim in context (except for case for -uncased models)
            self.assertTrue(answer_str.lower() in self.context.lower())
