from typing import Tuple

import mlx.nn as nn
import mlx.core as mx

from deps.model import Bert


class BertQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Bert(config, add_pooler=False)
        self.num_labels = config.num_labels
        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)

    def load_weights_pretrain(self, path: str, strict=True):
        # use strict=False to omit loading pooler.bias, pooler.weight
        self.model.load_weights(path, strict=strict)

    def __call__(
            self,
            input_ids: mx.array,
            token_type_ids: mx.array,
            attention_mask: mx.array
    ) -> Tuple[mx.array, mx.array]:

        # if batch_size = 16 then shape of input_ids is like: (16, 512, 768)
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        logits = self.qa_output(outputs)

        # split shape (b, x, 2) into two shapes (b, x, 1)
        # then, remove last dim so shape is (b, x)
        start_logits, end_logits = mx.split(logits, indices_or_sections=2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


def load_model_tokenizer_hf(hf_model: str = "bert-base-uncased"):
    from transformers import BertForQuestionAnswering, AutoTokenizer
    # https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/bert#transformers.BertForQuestionAnswering
    model = BertForQuestionAnswering.from_pretrained(hf_model)
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    return model, tokenizer


def load_model_tokenizer(hf_model: str,
                         weights_pretrain_path: str = None,
                         weights_finetuned_path: str = None
                         ):
    assert weights_pretrain_path is not None or weights_finetuned_path is not None, (
        "Must pass one weights_* parameter")

    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    config = AutoConfig.from_pretrained(hf_model)

    model = BertQA(config)
    if weights_pretrain_path is not None:
        # strict=False to omit loading pooler.bias, pooler.weight
        model.load_weights_pretrain(weights_pretrain_path, strict=False)
    else:
        model.load_weights(weights_finetuned_path)

    return model, tokenizer
