from functools import partial
import numpy as np
import time
import argparse
import math
import logging

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from utils import load_processed_datasets


def main(args):
    model, tokenizer = load_model_tokenizer(
        hf_model=args.model_str, weights_pretrain_path=args.load_weights)

    train_ds, valid_ds, test_ds = load_processed_datasets(
        filter_size=args.dataset_size, model_max_length=tokenizer.model_max_length,
        tokenizer=tokenizer)

    # set logger after loading squad
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=numeric_level)

    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=1e-5)
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
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
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

    # if args.train:
    #     opt = opt
    #     train()
    #     mx.savez(save_file)

    tic = time.perf_counter()
    if args.test:
        logging.info("Checking test loss...")
        test_loss = eval_fn(test_ds, model, batch_size=args.batch_size)
        toc = time.perf_counter()
        print(
            f"Test loss {test_loss:.3f}, "
            f"Test ppl {math.exp(test_loss):.3f}, "
            f"Test eval took {(toc - tic):.3f}s, "
        )

    print(f"Saving fine-tuned weights to {args.save_weights}")
    mx.savez(args.save_weights, **dict(tree_flatten(model.trainable_parameters())))

    # if args.inference:
    #     context, question = context, question
    #     run()


def batch_iterate(dataset, batch_size):
    perm = np.random.default_rng(12345).permutation(len(dataset))
    # # do not use this??
    # # it won't work at least in juypyter because need ids[iter].item()
    # perm = mx.array(perm)
    for s in range(0, len(dataset), batch_size):
        ids = perm[s: s + batch_size]
        yield dataset[ids]


def loss_fn(model, input_ids, token_type_ids, attention_mask, start_positions,
            end_positions, reduce=True):
    start_logits, end_logits = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions)
    slosses = nn.losses.cross_entropy(start_logits, start_positions)
    elosses = nn.losses.cross_entropy(end_logits, end_positions)
    if reduce:
        slosses = mx.mean(slosses)
        elosses = mx.mean(elosses)
    loss = (slosses + elosses) / 2
    return loss


# TODO review this
def eval_fn(dataset, model, batch_size=8):
    loss = 0
    # refactor with batch_iterate?
    for s in range(0, len(dataset), batch_size):
        batch = dataset[s: s + batch_size]
        input_ids, token_type_ids, attention_mask, start_positions, end_positions = map(
            mx.array,
            (batch['input_ids'], batch['token_type_ids'], batch['attention_mask'],
             batch['start_positions'], batch['end_positions'])
        )
        losses = loss_fn(model, input_ids, token_type_ids, attention_mask,
                         start_positions, end_positions, reduce=False)
        losses_have_nans = mx.isnan(losses).any()
        if losses_have_nans:
            logging.debug(f"eval_fn() found losses with nans: {losses}")
        loss += mx.sum(losses).item()
    logging.debug(f"eval_fn() final loss: {loss}")
    logging.debug(f"eval_fn() len(dataset): {len(dataset)}")
    return loss / len(dataset)


def load_model_tokenizer(hf_model: str,
                         weights_pretrain_path: str = None,
                         weights_finetuned_path: str = None,
                         ):
    assert weights_pretrain_path is not None or weights_finetuned_path is not None, "Must pass one weights_* parameter"

    from transformers import AutoConfig, AutoTokenizer
    from model_mlx import BertQA

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    config = AutoConfig.from_pretrained(hf_model)

    model = BertQA(config)
    if weights_pretrain_path is not None:
        # use load_weights2()
        model.load_weights2(weights_pretrain_path)
    else:
        # uses mx standard load_weights()
        model.load_weights(weights_finetuned_path)

    return model, tokenizer


def build_parser():
    parser = argparse.ArgumentParser(description="TBD")
    parser.add_argument(
        "--model_str",
        default="bert-base-uncased",
        help="Name of BERT model for tokenizer and parameters",
    )
    parser.add_argument(
        "--load_weights",
        default="weights/bert-base-uncased.npz",
        help="The path to the local pre-trained MLX model weights",
    )
    parser.add_argument(
        "--save_weights",
        default="weights/tmp-fine-tuned.npz",
        help="Path to save fine-tuned model weights",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument("--dataset_size", type=int, default=1000,
                        help="Number of records to load for entire dataset. Default is 1,000")
    parser.add_argument("--batch_size", type=int, default=10, help="Minibatch size. Default is 10")
    parser.add_argument(
        "--num_iters", type=int, default=4, help="Iterations to train for. Default is 4"
    )
    parser.add_argument(
        "--steps_per_report",
        type=int,
        default=2,
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
        help="Evaluate on the test set after training",
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
