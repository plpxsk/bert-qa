run:
	python qa.py --train
	python qa.py --infer


alternatives:
	PYTHONPATH=. python alt/hf/train.py
	PYTHONPATH=. python alt/hf/infer.py

	PYTHONPATH=. python alt/pt/train.py
	PYTHONPATH=. python alt/pt/infer.py

test:
	PYTHONPATH=. pytest
	PYTHONPATH=. python tests/test_compare_hf.py
