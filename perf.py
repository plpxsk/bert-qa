"""Note: High memory usage in HF `evaluate` library

Memory usage seems to spike between 8 and 20+ GB memory for batches of 100+
examples

This script crashes at 2000+ validation records on 32GB system

Based on:

https://huggingface.co/learn/nlp-course/en/chapter7/7
"""

import collections

import numpy as np
import mlx.core as mx
from tqdm.auto import tqdm
import evaluate
from datasets import load_dataset

from model import load_model_tokenizer


def main(args):
    model, tokenizer = load_model_tokenizer(hf_model=args.model_str,
                                            weights_finetuned_path=args.weights_finetuned)
    raw_squad_validation = load_dataset("squad", split="validation")

    args_dict = dict(tokenizer=tokenizer)
    processed_squad_validation = raw_squad_validation.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_squad_validation.column_names,
        fn_kwargs=args_dict
    )
    # (10570, 10626)
    # len(raw_squad_validation, len(processed_squad_validation)

    for s in range(0, len(raw_squad_validation), args.batch_size):
        ids = range(s, s + args.batch_size)
        batch = processed_squad_validation.select(ids)

        batch_model = batch.remove_columns(["example_id", "offset_mapping"])
        batch_model = batch_model.to_dict()
        batch_model = {key: mx.array(batch_model[key]) for key in batch_model.keys()}

        start_logits, end_logits = model(**batch_model)

        print(compute_metrics(start_logits, end_logits, batch, raw_squad_validation.select(ids)))


def compute_metrics(start_logits, end_logits, features, examples, n_best=20):
    """Lower n_best and max_answer_length speeds up processing.

    n_best = 20 and max_answer_length = 30 seem robust, with little performance
    improvement for n_best=50 or max_answer_length=100

    Source:
    https://huggingface.co/learn/nlp-course/en/chapter7/7
    """
    metric = evaluate.load("squad")
    max_answer_length = 30

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


def preprocess_validation_examples(examples, tokenizer):
    """For squad _validation_ set

    More straightforwad than preprocessing of the squad training set, for which
    see utils.preprocess_tokenize_function()

    Source:
    https://huggingface.co/learn/nlp-course/en/chapter7/7
    """

    max_length = tokenizer.model_max_length
    stride = max_length // 3

    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


# from qa.py infer()
def get_answer_from_tokenized_inputs(tokenized_inputs, start, end, tokenizer):
    tokens = tokenized_inputs["input_ids"][0, start: end + 1]
    # tokenizer can't use MLX array as input
    answer = tokenizer.decode(np.array(tokens))
    return answer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Fine tune BERT for Q&A")
    parser.add_argument(
        "--model_str",
        default="bert-base-uncased",
        help="Name of pre-trained BERT model for tokenizer and parameters"
    )
    parser.add_argument(
        "--weights_finetuned",
        default="weights/tmp-fine-tuned.npz",
        help="Check performance for model with these trained weights"
    )
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Size of validation set batch. Memory failures may occur for large batches. Default is 1000.")  # noqa
    args = parser.parse_args()

    main(args)
