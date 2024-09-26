import mlx.nn as nn
import mlx.core as mx

from deps import Bert


class BertQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Bert(config, add_pooler=False)
        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        # TODO: factor this out? no strict?
    def load_weights2(self, path: str):
        # strict=False to omit loading pooler.bias, pooler.weight
        self.model.load_weights(path, strict=False)

    def __call__(
            self,
            input_ids: mx.array,
            token_type_ids: mx.array,
            attention_mask: mx.array
        # TODO return type?
    ) -> Tuple[mx.array, mx.array]:

        # if batch_size = 16 then shape of input_ids is like: (16, 512, 768)
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        # # TODO check argument 0
        # sequence_output = outputs[0]
        # # ... so take the only batch

        logits = self.qa_output(outputs)

        # start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = mx.split(logits, 2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # do I need outputs??
        # return outputs, start_logits, end_logits
        return start_logits, end_logits
