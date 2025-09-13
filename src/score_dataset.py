from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import setup_logger, ensure_dir, save_json
from .preprocess import get_feature_lists, load_data
from .model_xgb import train_model as train_xgb  # We use XGBoost as the main scoring model

DATA_PATH = "data/synthetic_households.csv"
MODEL_PATH = "outputs/model_xgb.pkl"

def main():
    logger = setup_logger()
    # Make sure the XGBoost model is trained and ready
    if not Path(MODEL_PATH).exists():
        logger.info("Training XGBoost model for scoring (artifact not found)...")
        pipe, _ = train_xgb(DATA_PATH, model_out=MODEL_PATH)
    else:
        artifact = joblib.load(MODEL_PATH)
        pipe = artifact["pipeline"]

    df = load_data(DATA_PATH).copy()
    num_cols, cat_cols, target_col = get_feature_lists()
    X = df[num_cols + cat_cols]

    # Predict probabilities for each household
    proba = pipe.predict_proba(X)[:,1]
    df["adoption_probability"] = proba

    # Split scores into deciles, with 10 being the highest chance to adopt
    df["prob_decile"] = pd.qcut(df["adoption_probability"], 10, labels=False) + 1
    # Put decile 10 at the top for easier viewing
    df["prob_decile"] = 11 - df["prob_decile"]

    # Build a table showing gains and lift by decile
    gains = df.groupby("prob_decile")["adopted_heat_pump"].agg(["count","sum"]).reset_index()
    gains = gains.sort_values("prob_decile", ascending=True)  # By default, decile 10 is at the bottom
    gains["adoption_rate"] = gains["sum"] / gains["count"]
    gains.to_csv("outputs/gains_table.csv", index=False)

    # Plot adoption rate for each decile
    plt.figure()
    plt.plot(gains["prob_decile"], gains["adoption_rate"], marker="o")
    plt.xlabel("Propensity Decile (10 = Highest)")
    plt.ylabel("Observed Adoption Rate")
    plt.title("Lift by Propensity Decile")
    ensure_dir("outputs")
    plt.savefig("outputs/lift_by_decile.png", bbox_inches="tight")
    plt.close()

    # Write the scored results to a CSV file
    Path("data").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/scored_households.csv", index=False)
    logger.info("Wrote data/scored_households.csv and outputs/gains_table.csv + lift_by_decile.png.")

if __name__ == "__main__":
    main()
