from __future__ import annotations
from pathlib import Path

from src.model_logistic import train_model
from src.evaluate import evaluate_and_save
from src.preprocess import load_data, get_feature_lists

DATA_PATH = "data/synthetic_households.csv"
MODEL_PATH = "outputs/model.pkl"

def main():
    pipe, (X_test, y_test) = train_model(DATA_PATH, model_out=MODEL_PATH)
    # Evaluate and save plots/metrics
    evaluate_and_save(MODEL_PATH, X_test, y_test)

if __name__ == "__main__":
    main()
