extras:
	PYTHONPATH=. python extra/train_hf.py
	PYTHONPATH=. python extra/train_pt.py

	PYTHONPATH=. python extra/inference_hf.py
	PYTHONPATH=. python extra/inference_pt.py

test:
	PYTHONPATH=. pytest
	PYTHONPATH=. python tests/test_compare_hf.py
