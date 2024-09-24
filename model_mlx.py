import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase


class TransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.ln1 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)
        self.gelu = nn.GELU()

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(dims, num_heads, mlp_dims)
            for i in range(num_layers)
        ]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self, input_ids: mx.array, token_type_ids: mx.array = None
    ) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )

        if token_type_ids is None:
            # If token_type_ids is not provided, default to zeros
            token_type_ids = mx.zeros_like(input_ids)

        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        return self.norm(embeddings)


class Bert(nn.Module):
    def __init__(self, config, add_pooler):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.hidden_size,
            num_heads=config.num_attention_heads,
            mlp_dims=config.intermediate_size,
        )
        self.pooler = nn.Linear(
            config.hidden_size, config.hidden_size) if add_pooler else None

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array = None,
        attention_mask: mx.array = None
    ) -> Tuple[mx.array, mx.array]:
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            # convert 0's to -infs, 1's to 0's, and make it broadcastable
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        y = self.encoder(x, attention_mask)
        if self.pooler is not None:
            return y, mx.tanh(self.pooler(y[:, 0]))
        else:
            return y


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
            attention_mask: mx.array,
            start_positions: mx.array,
            end_positions: mx.array
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


def tmp():
    def tmp():
        # NEXT: take this into qa.py with loss_fn()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # # If we are on multi-GPU, split add a dimension
            # if len(start_positions.size()) > 1:
            #     start_positions = start_positions.squeeze(-1)
            # if len(end_positions.size()) > 1:
            #     end_positions = end_positions.squeeze(-1)

            # # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # ignored_index = start_logits.size(1)
            # start_positions = start_positions.clamp(0, ignored_index)
            # end_positions = end_positions.clamp(0, ignored_index)

            loss_fn = nn.losses.cross_entropy()
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # # TODO check argument 2:
        # output = (start_logits, end_logits) + outputs[2:]
        # return ((total_loss,) + output)


def load_model(
    bert_model: str, weights_path: str
) -> Tuple[Bert, PreTrainedTokenizerBase]:
    if not Path(weights_path).exists():
        raise ValueError(f"No model weights found in {weights_path}")

    config = AutoConfig.from_pretrained(bert_model)

    # create and update the model
    model = Bert(config)
    model.load_weights(weights_path)

    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    return model, tokenizer


def run(bert_model: str, mlx_model: str, batch: List[str]):
    model, tokenizer = load_model(bert_model, mlx_model)

    tokens = tokenizer(batch, return_tensors="mlx", padding=True)

    return model(**tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the BERT model using MLX.")
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-uncased",
        help="The huggingface name of the BERT model to save.",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="weights/bert-base-uncased.npz",
        help="The path of the stored MLX BERT weights (npz file).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is an example of BERT working in MLX",
        help="The text to generate embeddings for.",
    )
    args = parser.parse_args()
    run(args.bert_model, args.mlx_model, args.text)