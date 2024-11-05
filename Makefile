convert:
	mkdir -p weights
	python deps/convert.py --bert-model bert-base-cased --mlx-model weights/bert-base-cased.npz

demo:
	python qa.py \
		--train \
		--test \
		--model_str bert-base-cased \
		--weights_pretrain weights/bert-base-cased.npz \
		--weights_finetuned weights/demo_fine_tuned.npz \
		--dataset_size 1000 \
		--n_iters 30

train:
	python qa.py \
		--train \
		--test \
		--model_str bert-base-cased \
		--weights_pretrain weights/bert-base-cased.npz \
		--weights_finetuned weights/final_fine_tuned_1_epoch.npz \
		--batch_size 32 \
		--n_epoch 1 \
		--steps_per_report 100 \
		--steps_per_eval 900

testit:
	python qa.py \
		--test \
		--weights_finetuned weights/demo_fine_tuned.npz \
		--dataset_size 1000

infer:
	python qa.py \
		--infer \
		--weights_finetuned weights/demo_fine_tuned.npz \
		--question "How many programming languages does BLOOM support?" \
		--context "BLOOM has 176 billion parameters and can generate text in 46 natural languages and 13 programming languages."

perf:
	python perf.py \
		--model_str bert-base-cased \
		--weights_finetuned weights/demo_fine_tuned.npz \
		--batch_size 100

alts:
	PYTHONPATH=. python alt/hf/train.py
	PYTHONPATH=. python alt/hf/infer.py

	PYTHONPATH=. python alt/pt/train.py
	PYTHONPATH=. python alt/pt/infer.py

test:
	PYTHONPATH=. python deps/test_base_bert.py
	PYTHONPATH=. pytest

