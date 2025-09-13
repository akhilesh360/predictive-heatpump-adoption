.PHONY: install train compare explain score shap calibrate fairness ci
install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt || true
train:
	python run_pipeline.py
compare:
	python run_compare.py
explain:
	python -m src.explain_xgb
score:
	python -m src.score_dataset
calibrate:
	python -m src.calibrate
fairness:
	python -m src.fairness
ci: install train compare explain score calibrate fairness
