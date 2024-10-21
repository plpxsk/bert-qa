# Fine tune BERT for Q&A on MacBook using Apple MLX

Work in Progress

Bert fine-tuning for Q&A with squad dataset:

- [x] Implement fine-tune training and inference with HuggingFace AutoClasses (/alt)
- [x] Implement fine-tune training and inference with PyTorch (/alt)
- [ ] WIP: port to Apple MLX

Uses Apple MLX for training using Apple Silicon. See [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx).

# Run

For code recipes, see [Makefile](Makefile).

# Test

Run some tests with `make test` per Makefile

# Alternative Implementations

See `/alt` for POCs of training and inference with HuggingFace pipelines and/or
PyTorch
