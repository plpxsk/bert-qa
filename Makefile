run:
	python qa.py \
		--model_str bert-base-uncased \
		--load_weights weights/bert-base-uncased.npz \
		--save_weights weights/fine_tuned_full_data_1000_iter.npz \
		--batch_size 10 \
		--num_iters 1000 \
		--steps_per_report 100 \
		--steps_per_eval 500 \
		--test


alternatives:
	PYTHONPATH=. python alt/hf/train.py
	PYTHONPATH=. python alt/hf/infer.py

	PYTHONPATH=. python alt/pt/train.py
	PYTHONPATH=. python alt/pt/infer.py

test:
	PYTHONPATH=. pytest
	PYTHONPATH=. python tests/test_compare_hf.py
