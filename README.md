# Fine tune BERT model for Q&A with Apple MLX

Using `squad` Question and Answer dataset and Mac with Apple Silicon.

After fine-tune training the base BERT model, you can find answers to questions
in your document.

As typical, the answer is an exact extract from the provided document ("context").

# Set up

1. Install `requirements.txt`
    * `torch` is required only for next step conversion
1. Convert pre-trained BERT model weights to MLX with `make convert`

# Run

_The main script is `qa.py` which imports `model.py`, `utils.py` and depends on
`/deps`._

Run a demo fine-tune training script with evaluation on a held-out set, via
`make demo`.

Then, run a demo inference with `make infer`.

Both demos take only a few seconds. See "Demo" below.

For more training, testing, inference, and performance code recipes, see
[Makefile](Makefile).

# Performance

With 1000 iters (~1 hour) of training as per `make train`, the model seems to
get quite good performance.

Results on the `squad` validation set, via `make perf`, in batches of 1000:

```shell
(.venv) bert-qa# make perf
python perf.py \
		--model_str bert-base-uncased \
		--weights_finetuned weights/final_fine_tuned_full_data_1000_iter.npz
100%|██████████████████████████████████████████████████████████████| 1000/1000 [01:24<00:00, 11.90it/s]
{'exact_match': 73.7, 'f1': 80.74097983020103}
100%|██████████████████████████████████████████████████████████████| 1000/1000 [01:29<00:00, 11.15it/s]
{'exact_match': 64.1, 'f1': 74.99522715078461}
100%|██████████████████████████████████████████████████████████████| 1000/1000 [01:28<00:00, 11.25it/s]
{'exact_match': 66.7, 'f1': 76.48260749828081}
100%|██████████████████████████████████████████████████████████████| 1000/1000 [01:26<00:00, 11.52it/s]
{'exact_match': 70.0, 'f1': 79.48454740980043}
100%|██████████████████████████████████████████████████████████████| 1000/1000 [01:25<00:00, 11.65it/s]
{'exact_match': 64.2, 'f1': 76.3785847219809}
100%|██████████████████████████████████████████████████████████████| 1000/1000 [01:25<00:00, 11.70it/s]
{'exact_match': 72.7, 'f1': 81.2466704812215}
100%|██████████████████████████████████████████████████████████████| 1000/1000 [01:23<00:00, 12.00it/s]
{'exact_match': 63.2, 'f1': 72.27569857845408}
100%|██████████████████████████████████████████████████████████████| 1000/1000 [01:24<00:00, 11.79it/s]
{'exact_match': 64.3, 'f1': 73.15070496821069}
100%|██████████████████████████████████████████████████████████████| 1000/1000 [01:25<00:00, 11.75it/s]
{'exact_match': 62.6, 'f1': 73.709639718001}
100%|██████████████████████████████████████████████████████████████| 1000/1000 [01:26<00:00, 11.58it/s]
{'exact_match': 64.7, 'f1': 74.43623684426922}
```

For reference, `DistilBERT` fine-tuned on SQuAD obtains 79.1 and 86.9 for those
scores on the whole dataset. See [ref](https://arxiv.org/abs/1910.01108v2).


# Demo

_On MacBook with M1 Pro and 32 GB RAM_

### Training

This should take only a few seconds, unless HuggingFace needs to download the
`squad` dataset, which should also be fast.

```
(.venv) bert-qa# make demo
python qa.py \
                --train \
                --test \
                --model_str bert-base-uncased \
                --weights_pretrain weights/bert-base-uncased.npz \
                --weights_finetuned weights/demo_fine_tuned.npz \
                --dataset_size 1000 \
                --num_iters 10
Loading datasets...
Map: 100%|█████████| 800/800 [00:00<00:00, 4922.23 examples/s]
Map: 100%|█████████| 100/100 [00:00<00:00, 1593.32 examples/s]
Map: 100%|█████████| 100/100 [00:00<00:00, 4938.48 examples/s]
Training for 10 iters...
Iter 5: Train loss 5.474, Train ppl 238.462, It/sec 0.939
Iter 10: Train loss 5.115, Train ppl 166.544, It/sec 0.949
Iter 10: Val loss 4.958, Val ppl 142.341, Val took 3.414s, 
Saving fine-tuned weights to weights/demo_fine_tuned.npz
Checking test loss...
Test loss 4.918, Test ppl 136.747, Test eval took 3.465s
```

### Inference

Inference is very quick (but can be quite wrong with short demo fine-tuning):

```
(.venv) bert-qa# make infer
python qa.py \
                --infer \
                --weights_finetuned weights/demo_fine_tuned.npz \
                --question "How many programming languages does BLOOM support?" \
                --context "BLOOM has 176 billion parameters and can generate text in 46 natural languages and 13 programming languages."
Running inference...
# Context, Question:
BLOOM has 176 billion parameters and can generate text in 46 natural languages and 13 programming languages.

How many programming languages does BLOOM support? 

Answer:  languages and 13 programming
Score:  4.9233527183532715 
```

### Performance

This light demo model does poorly, as expected. Results on the first 1000
validation records are representative:

```
(.venv) bert-qa# python perf.py \
		--model_str bert-base-uncased \
		--weights_finetuned weights/demo_fine_tuned.npz
100%|██████████████████████████████████████████████████████████████| 1000/1000 [01:18<00:00, 12.80it/s]
{'exact_match': 0.2, 'f1': 4.331335339379247}
```

# Tests

_Install per `requirements_extra.txt`_

Run some tests with `make test` per [Makefile](Makefile).

Tests include:

  * Confirm model architecture against Huggingface model
  * Confirm loss against official BERT implementation loss
  * Unit tests

# Alternative Implementations

See [README](alt/README.md) in `/alt` for alternative fine-tune training and
inference with PyTorch and/or higher-level HuggingFace pipelines.

Run as in `make alts` per [Makefile](Makefile).

# Dependencies

Dependent BERT base model architecture and conversion script in `/deps` is from
[ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/bert).

`MLX` is a NumPy-like array framework designed for efficient and flexible
machine learning on Apple silicon. See
[github.com/ml-explore/mlx](https://github.com/ml-explore/mlx).

For original `BERT` model, see
[github.com/google-research/bert](https://github.com/google-research/bert).

_For more, see [resources.md](resources.md)_
