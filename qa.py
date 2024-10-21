import numpy as np
import time
import argparse
import math
import logging
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from utils import load_squad, init_logger, preprocess_tokenize_function, get_answers
from model import load_model_tokenizer


def main(args):
    model, tokenizer = load_model_tokenizer(
        hf_model=args.model_str, weights_pretrain_path=args.weights_pretrain)

    if not args.infer:
        print("Loading datasets...")
        train_ds, valid_ds, test_ds = load_processed_datasets(tokenizer, args.dataset_size)

    # set logger after loading squad
    init_logger(args.log)

    if args.train:
        print(f"Training for {args.num_iters} iters...")
        train(model, train_ds, valid_ds, loss_fn, args)
        print(f"Saving fine-tuned weights to {args.weights_finetuned}")
        mx.savez(args.weights_finetuned, **dict(tree_flatten(model.trainable_parameters())))

    # Weights should exist after training
    if not Path(args.weights_finetuned).is_file():
        raise ValueError(
            f"Fine-tuned weights file {args.weights_finetuned} is missing. "
            "Use --train to learn and save fine-tuned weights."
        )
    model.load_weights(args.weights_finetuned, strict=True)

    if args.test:
        print("Checking test loss...")
        test(model, test_ds, args.batch_size)

    if args.infer:
        assert args.question is not None and args.context is not None, (
            "With --infer, must pass both --question and --context")
        print("Running inference...")
        infer(model, tokenizer, args.question, args.context, top_k=args.top_k)


def load_processed_datasets(tokenizer, dataset_size=None):
    load_split = ("train" if dataset_size is None
                  else "train[:" + str(dataset_size) + "]")
    train_ds, valid_ds, test_ds = load_squad(
        load_split=load_split, tokenizer=tokenizer,
        preproc_function=preprocess_tokenize_function,
        test_valid_frac=0.2, test_frac=0.5, return_tuples=True, torch=False)
    return train_ds, valid_ds, test_ds


def train(model, train_ds, valid_ds, loss_fn, args):
    optimizer = optim.AdamW(learning_rate=1e-5)

    mx.eval(model.parameters())
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(input_ids, token_type_ids, attention_mask, start_positions,
             end_positions):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, input_ids, token_type_ids, attention_mask,
                                       start_positions, end_positions)
        optimizer.update(model, grads)
        return loss

    train_iterator = batch_iterate(train_ds, batch_size=args.batch_size)
    losses = []
    tic = time.perf_counter()

    for it, batch in zip(range(args.num_iters), train_iterator):
        logging.info(f"Iteration {it}...")
        input_ids, token_type_ids, attention_mask, start_positions, end_positions = map(
            mx.array,
            (batch['input_ids'], batch['token_type_ids'], batch['attention_mask'],
             batch['start_positions'], batch['end_positions'])
        )

        loss = step(input_ids, token_type_ids, attention_mask, start_positions, end_positions)

        mx.eval(state)
        losses.append(loss.item())
        if (it + 1) % args.steps_per_report == 0:
            logging.info("Running report...")
            train_loss = np.mean(losses)
            toc = time.perf_counter()
            print(
                f"Iter {it + 1}: "
                f"Train loss {train_loss:.3f}, "
                f"Train ppl {math.exp(train_loss):.3f}, "
                f"It/sec {args.steps_per_report / (toc - tic):.3f}"
            )
            losses = []
            tic = time.perf_counter()
        if (it + 1) % args.steps_per_eval == 0:
            logging.info("Checking validation loss...")
            val_loss = eval_fn(valid_ds, model, batch_size=args.batch_size)
            toc = time.perf_counter()
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val ppl {math.exp(val_loss):.3f}, "
                f"Val took {(toc - tic):.3f}s, "
            )
            tic = time.perf_counter()


def test(model, test_ds, batch_size):
    tic = time.perf_counter()
    model.eval()
    test_loss = eval_fn(test_ds, model, batch_size=batch_size)
    toc = time.perf_counter()
    print(
        f"Test loss {test_loss:.3f}, "
        f"Test ppl {math.exp(test_loss):.3f}, "
        f"Test eval took {(toc - tic):.3f}s"
    )


def infer(model, tokenizer, question, context, top_k=1):
    tokenized_inputs = tokenizer(question, context, return_tensors="mlx")
    start_logits, end_logits = model(**tokenized_inputs)
    answers = get_answers(start_logits, end_logits, tokenized_inputs.sequence_ids(), top_k=top_k)

    def get_answer_from_tokenized_inputs(tokenized_inputs, start, end):
        tokens = tokenized_inputs["input_ids"][0, start: end + 1]
        # tokenizer can't use MLX array as input
        answer = tokenizer.decode(np.array(tokens))
        return answer

    print("### Context, context:")
    print(context)
    print(question, "\n")

    for answer in answers:
        start = answer["start"]
        end = answer["end"]
        score = answer["score"]
        answer = get_answer_from_tokenized_inputs(tokenized_inputs, start, end)
        print("A: ", answer)
        print("Score: ", score, "\n")


def batch_iterate(dataset, batch_size):
    perm = np.random.default_rng(12345).permutation(len(dataset))
    for s in range(0, len(dataset), batch_size):
        ids = perm[s: s + batch_size]
        yield dataset[ids]


def loss_fn(model, input_ids, token_type_ids, attention_mask, start_positions,
            end_positions, reduction="mean"):
    start_logits, end_logits = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask)
    slosses = nn.losses.cross_entropy(start_logits, start_positions, reduction=reduction)
    elosses = nn.losses.cross_entropy(end_logits, end_positions, reduction=reduction)
    loss = (slosses + elosses) / 2
    return loss


def eval_fn(dataset, model, batch_size=8):
    loss = 0
    for s in range(0, len(dataset), batch_size):
        batch = dataset[s: s + batch_size]
        input_ids, token_type_ids, attention_mask, start_positions, end_positions = map(
            mx.array,
            (batch['input_ids'], batch['token_type_ids'], batch['attention_mask'],
             batch['start_positions'], batch['end_positions'])
        )
        losses = loss_fn(model, input_ids, token_type_ids, attention_mask,
                         start_positions, end_positions, reduction="none")
        losses_have_nans = mx.isnan(losses).any()
        if losses_have_nans:
            logging.debug(f"eval_fn() found NANs in losses: {losses}")
        loss += mx.sum(losses).item()
    logging.debug(f"eval_fn() final loss: {loss}")
    logging.debug(f"eval_fn() len(dataset): {len(dataset)}")
    return loss / len(dataset)


def build_parser():
    def none_or_int(value):
        if value == 'None':
            return None
        return int(value)

    parser = argparse.ArgumentParser(description="Fine tune BERT for Q&A")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run fine-tune training and save weights to --weights_finetuned",
    )
    parser.add_argument(
        "--model_str",
        default="bert-base-uncased",
        help="Name of pre-trained BERT model for tokenizer and parameters",
    )
    parser.add_argument(
        "--weights_pretrain",
        default="weights/bert-base-uncased.npz",
        help="The path to the local pre-trained MLX model weights",
    )
    parser.add_argument(
        "--weights_finetuned",
        default="weights/tmp-fine-tuned.npz",
        help="Path to save fine-tuned model weights, or to load weights for testing or inference",
    )
    parser.add_argument("--dataset_size", type=none_or_int, default=None,
                        help="Number of records to load for entire dataset. Default is None (full data)")  # noqa
    parser.add_argument("--batch_size", type=int, default=10, help="Minibatch size. Default is 10")
    parser.add_argument(
        "--num_iters", type=int, default=100, help="Iterations to train for. Default is 100"
    )
    parser.add_argument(
        "--steps_per_report",
        type=int,
        default=5,
        help="Number of training steps between loss reporting. Default is 5",
    )
    parser.add_argument(
        "--steps_per_eval",
        type=int,
        default=10,
        help="Number of training steps between validations. Default is 10",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training, or after loading fine-tuned weights",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run inference on --question and --context"
    )
    parser.add_argument(
        "--question",
        help="Run inference with this question. Must also pass --context"
    )
    parser.add_argument(
        "--context",
        help="Run inference on this context. Must also pass --question"
    )
    parser.add_argument(
        "--top_k",
        default=1,
        type=int,
        help="Number of top answers to return for --infer"
    )
    parser.add_argument(
        "--log",
        default="warning",
        help="Set python logging level from: warn (default)"
    )
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    main(args)
