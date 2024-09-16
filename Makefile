extras:
	PYTHONPATH=. python extra/train_hf.py
	PYTHONPATH=. python extra/train_pt.py

	PYTHONPATH=. python extra/inference_hf
	PYTHONPATH=. python extra/inference_pt.py

