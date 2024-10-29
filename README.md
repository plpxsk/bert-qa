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

Run a demo fine-tuning script with evaluation on test set, via `make demo`.

Then, run a demo inference with `make infer`. See "Demo" below.

For full training and inference code recipes, see [Makefile](Makefile).

# Demo

_On MacBook with M1 Pro and 32 GB RAM_

### Training

This should take only a few seconds, unless HuggingFace needs to download the
`squad` dataset.

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


# Tests

_Install per `requirements_extra.txt`_

Run some tests with `make test` per Makefile.

Tests includes:

  * confirm model architecture against Huggingface model
  * confirm loss against official BERT implementation loss

# Alternative Implementations

See [README](alt/README.md) in `/alt` for alternative fine-tune training and
inference with PyTorch and/or higher-level HuggingFace pipelines.

# Dependencies

Dependent BERT base model architecture and conversion script in `/deps` is from
[ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/bert).

`MLX` is a NumPy-like array framework designed for efficient and flexible
machine learning on Apple silicon. See
[github.com/ml-explore/mlx](https://github.com/ml-explore/mlx).

For original `BERT` model, see
[github.com/google-research/bert](https://github.com/google-research/bert).
