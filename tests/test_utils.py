import unittest
import mlx.core as mx

from utils import find_context_start_end, filter_logits_to_context
from utils import find_valid_answers, get_answers


class TestUtils(unittest.TestCase):

    def test_find_context_start_end(self):
        # expect sequence_ids to have at least 3 None
        sequence_ids = [None, 0, None, 1, None]
        estart, eend = 3, 3
        start, end = find_context_start_end(sequence_ids)

        self.assertEqual(estart, start)
        self.assertEqual(eend, end)

        sequence_ids = [None, 0, 0, None, 1, 1, 1, None]
        estart, eend = 4, 6
        start, end = find_context_start_end(sequence_ids)

        self.assertEqual(estart, start)
        self.assertEqual(eend, end)

    def test_filter_logits_to_context(self):
        start_logits = [0.1, 0.9, 1.9, 2.9, 1.9, 0.9, 0.1]
        context_start = 3
        context_end = 6
        expect_len = context_end-context_start+1

        obs_start_logits = filter_logits_to_context(start_logits, context_start_index=context_start,
                                                    context_end_index=context_end,
                                                    flatten_to_list=False)
        obs_len = len(obs_start_logits)

        self.assertEqual(expect_len, obs_len, "not equalz")

    def test_filter_logits_to_context_mlx(self):
        start_logits = [0.1, 0.9, 1.9, 2.9, 1.9, 0.9, 0.1]
        start_logits = mx.array(start_logits)
        context_start = 3
        context_end = 6
        expect_len = context_end-context_start+1

        obs_start_logits = filter_logits_to_context(start_logits, context_start_index=context_start,
                                                    context_end_index=context_end,
                                                    flatten_to_list=True)
        obs_len = len(obs_start_logits)

        self.assertEqual(expect_len, obs_len)

    def test_find_valid_answers(self):
        start_logits = [0.1, 0.9, 1.9, 2.9, 1.9, 0.9, 0.1]
        end_logits = start_logits
        n_best_sizes = [2, len(start_logits), len(start_logits) + 10]

        # score is sum of start and end logits
        exp_top_score = 2*max(start_logits)

        for n_best_size in n_best_sizes:
            answers = find_valid_answers(start_logits, end_logits, context_start_index=3,
                                         n_best_size=n_best_size, sort=True)
            obs_len = len(answers)
            obs_top_score = answers[0]["score"]

            self.assertTrue(obs_len > 0)
            self.assertEqual(exp_top_score, obs_top_score)

    def test_get_answers(self):
        sequence_ids = [None, 0, 0, None, 1, 1, 1, None]
        start_logits = [0.1, 0.9, 1.9, 2.9, 1.9, 0.9, 0.1, 0.1]
        end_logits = start_logits

        # score is sum of start and end logits
        # and should fall within context (where sequence_ids == 1)
        max_score_within_context = max(start_logits[4:])
        exp_top_score = 2*max_score_within_context

        answers = get_answers(start_logits=start_logits, end_logits=end_logits,
                              tokenized_input_sequence_ids=sequence_ids, flatten_to_list=False,
                              top_k=3, n_best_size=20)

        obs_len = len(answers)
        obs_top_score = answers[0]["score"]

        self.assertTrue(obs_len > 0)
        self.assertEqual(exp_top_score, obs_top_score)
