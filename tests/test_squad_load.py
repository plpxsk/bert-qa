import unittest

from utils import load_squad_raw, split_dataset, preproc_squad, load_squad
from utils import preprocess_tokenize_function
from model import load_model_tokenizer_hf


class TestSquadLoad(unittest.TestCase):

    def setUp(self):
        from datasets import load_dataset
        from transformers.utils import logging
        logging.set_verbosity_error()

        rows = 70
        squad_split = "train[:" + str(rows) + "]"
        squad = load_dataset("squad", split=squad_split)

        _, tokenizer = load_model_tokenizer_hf(hf_model="bert-base-uncased")

        self.squad = squad
        self.tokenizer = tokenizer
        self.expected_rows = rows

    def test_load_squad_raw(self):
        squad = load_squad_raw(load_split="train[:70]", torch=False)
        obs_rows = len(squad)

        assert self.expected_rows == obs_rows, \
            f"load_squad_raw() should return {self.expected_rows} rows, but got: {obs_rows}"

    def test_split_dataset(self):
        squad = split_dataset(self.squad)
        n_splits = len(squad)
        observed_rows = len(squad["train"]) + len(squad["valid"]) + len(squad["test"])

        assert 3 == n_splits, f"split_dataset() should return 3 datasets, but got: {n_splits}"
        assert self.expected_rows == observed_rows, \
            f"Expected row size is {self.expected_rows} but got: {observed_rows}"

    def test_preprocess_tokenize_function(self):
        squad = preprocess_tokenize_function(self.squad, self.tokenizer)
        exp_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions',
                    'end_positions']
        obs_keys = squad.keys()
        len_input_ids = len(squad['input_ids'])
        len_starts = len(squad['start_positions'])
        len_ends = len(squad['end_positions'])

        assert set(exp_keys) == set(obs_keys), \
            f"Expected keys and observed keys dont match. Expected: {exp_keys} Observed: {obs_keys}"
        assert self.expected_rows == len_input_ids, \
            f"preprocess_tokenize_function() returns unexpected lengths in {len_input_ids}"
        assert self.expected_rows == len_starts, \
            f"preprocess_tokenize_function() returns unexpected lengths in {len_starts}"
        assert self.expected_rows == len_ends, \
            f"preprocess_tokenize_function() returns unexpected lengths in {len_ends}"

    def test_preproc_squad(self):
        columns = self.squad.column_names
        squad = preproc_squad(self.squad, tokenizer=self.tokenizer,
                              preproc_function=preprocess_tokenize_function, remove_columns=columns)

        exp_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions',
                    'end_positions']
        obs_keys = squad.features.keys()

        assert set(exp_keys) == set(obs_keys), \
            f"Expected keys and observed keys dont match. Expected: {exp_keys} Observed: {obs_keys}"

    def test_load_squad(self):
        train, valid, test = load_squad(load_split="train[:70]", tokenizer=self.tokenizer,
                                        preproc_function=preprocess_tokenize_function,
                                        test_valid_frac=0.25, test_frac=0.5,
                                        return_tuples=True, torch=False)
        obs_rows = len(train) + len(valid) + len(test)

        assert self.expected_rows == obs_rows, \
            f"Expect {self.expected_rows} but got {obs_rows}"
