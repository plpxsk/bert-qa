# Fine tune BERT model for Q&A on MacBook

Using `squad` Question and Answer dataset and Apple MLX (ML-Explore) for Mac
with Apple Silicon.

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

Both demos take only a few seconds. **See "Demo" below**.

For more training, testing, inference, and performance code recipes, see
[Makefile](Makefile).

# Performance

_All results on MacBook with M1 Pro and 32 GB RAM_

Fine-tuning took ~3 hours, for a single pass through the `squad 1.1` training
dataset, with pre-trained `bert-base-cased`, batch size of 32, and learning
rate of 5e-5 as in `make train`.

Results on the `squad` validation set, as per `make perf`, in batches of 1000:

```shell
(.venv) bert-qa# python perf.py --model_str bert-base-cased --weights_finetuned weights/final_fine_tuned_1_epoch.npz          
100%|█████████████████████████████████████████| 1000/1000 [01:30<00:00, 11.06it/s]
{'exact_match': 83.1, 'f1': 87.99243539054682}
100%|█████████████████████████████████████████| 1000/1000 [01:32<00:00, 10.80it/s]
{'exact_match': 78.0, 'f1': 86.65255557501183}
100%|█████████████████████████████████████████| 1000/1000 [01:32<00:00, 10.79it/s]
{'exact_match': 76.5, 'f1': 85.35768350029493}
100%|█████████████████████████████████████████| 1000/1000 [01:31<00:00, 10.91it/s]
{'exact_match': 81.5, 'f1': 87.90210248581334}
100%|█████████████████████████████████████████| 1000/1000 [01:29<00:00, 11.15it/s]
{'exact_match': 70.4, 'f1': 81.141941325805}
100%|█████████████████████████████████████████| 1000/1000 [01:30<00:00, 11.00it/s]
{'exact_match': 79.3, 'f1': 85.1800862625974}
100%|█████████████████████████████████████████| 1000/1000 [01:27<00:00, 11.44it/s]
{'exact_match': 69.5, 'f1': 77.68816921507332}
100%|█████████████████████████████████████████| 1000/1000 [01:28<00:00, 11.35it/s]
{'exact_match': 73.6, 'f1': 80.23080772684483}
100%|█████████████████████████████████████████| 1000/1000 [01:28<00:00, 11.33it/s]
{'exact_match': 68.5, 'f1': 78.53797196528662}
100%|█████████████████████████████████████████| 1000/1000 [01:29<00:00, 11.23it/s]
{'exact_match': 74.2, 'f1': 81.61097223228957}
```

Performance is in line with:

  * original BERT: 80.8 EM and 88.5 F1 (base) and 84-87 EM and 91-93 F1 (large) [1]
  * `DistilBERT`: 79.1 EM and 86.9 F1 [2]

[1] See Table 2 for squad 1.1 results, for which "We fine-tune for 3 epochs
with a learning rate of 5e-5 and a batch size of 32"
[https://arxiv.org/abs/1810.04805](arxiv.org/abs/1810.04805)

[2] [https://arxiv.org/abs/1910.01108v2](arxiv.org/abs/1910.01108v2)


# Demo

### Training

This should take only a few seconds, unless HuggingFace needs to download the
`squad` dataset, which should also be fast.

```
(.venv) bert-qa# make demo
python qa.py \
                --train \
                --test \
                --model_str bert-base-cased \
                --weights_pretrain weights/bert-base-cased.npz \
                --weights_finetuned weights/demo_fine_tuned.npz \
                --dataset_size 1000 \
                --n_iters 30
Loading datasets...
Map: 100%|██████████████████████████████████| 800/800 [00:00<00:00, 5059.07 examples/s]
Map: 100%|██████████████████████████████████| 100/100 [00:00<00:00, 1601.85 examples/s]
Map: 100%|██████████████████████████████████| 100/100 [00:00<00:00, 5035.48 examples/s]
Training for 1 epochs and 30 iters...
Iter (batch) 5: Train loss 5.942, Train ppl 380.855, It/sec 0.881
Iter (batch) 10: Train loss 4.851, Train ppl 127.869, It/sec 0.915
Iter (batch) 10: Val loss 4.227, Val ppl 68.516, Val took 3.525s, 
Iter (batch) 15: Train loss 4.137, Train ppl 62.625, It/sec 0.918
Iter (batch) 20: Train loss 3.977, Train ppl 53.348, It/sec 0.908
Iter (batch) 20: Val loss 3.551, Val ppl 34.850, Val took 3.536s, 
Iter (batch) 25: Train loss 3.807, Train ppl 45.007, It/sec 0.916
Iter (batch) 30: Train loss 3.380, Train ppl 29.374, It/sec 0.917
Iter (batch) 30: Val loss 2.733, Val ppl 15.377, Val took 3.548s, 
Saving fine-tuned weights to weights/demo_fine_tuned.npz
Checking test loss...
Test loss 2.799, Test ppl 16.423, Test eval took 3.608s
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

Answer:  176 billion
Score:  10.352056980133057 
```

### Performance

This light demo model does poorly, as expected. Results on the first 1000
validation records are representative:

```
(.venv) bert-qa# python perf.py \ 
>       --model_str bert-base-cased \
>       --weights_finetuned weights/demo_fine_tuned.npz 
100%|██████████████████████████████████████████| 1000/1000 [01:22<00:00, 12.15it/s]
{'exact_match': 21.7, 'f1': 30.74530582015016}
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
