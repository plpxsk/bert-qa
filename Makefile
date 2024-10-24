convert:
	mkdir -p weights
	python deps/convert.py --bert-model bert-base-uncased --mlx-model weights/bert-base-uncased.npz

demo:
	python qa.py \
		--train \
		--test \
		--model_str bert-base-uncased \
		--weights_pretrain weights/bert-base-uncased.npz \
		--weights_finetuned weights/demo_fine_tuned.npz \
		--dataset_size 1000 \
		--num_iters 10

train:
	python qa.py \
		--train \
		--test \
		--model_str bert-base-uncased \
		--weights_pretrain weights/bert-base-uncased.npz \
		--weights_finetuned weights/fine_tuned_full_data_1000_iter.npz \
		--batch_size 10 \
		--num_iters 1000 \
		--steps_per_report 100 \
		--steps_per_eval 500 \

testit:
	python qa.py \
		--test \
		--weights_finetuned weights/final_fine_tuned_full_data_1000_iter.npz \
		--dataset_size 1000

infer:
	python qa.py \
		--infer \
		--weights_finetuned weights/demo_fine_tuned.npz \
		--question "How many programming languages does BLOOM support?" \
		--context "BLOOM has 176 billion parameters and can generate text in 46 natural languages and 13 programming languages."

alts:
	PYTHONPATH=. python alt/hf/train.py
	PYTHONPATH=. python alt/hf/infer.py

	PYTHONPATH=. python alt/pt/train.py
	PYTHONPATH=. python alt/pt/infer.py

test:
	PYTHONPATH=. python deps/test_base_bert.py
	PYTHONPATH=. pytest

