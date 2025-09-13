from __future__ import annotations
from pathlib import Path
import json, joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import precision_recall_curve

from .utils import setup_logger, ensure_dir

MODEL_ART = "outputs/model_xgb.pkl"  # default: calibrate XGB artifact
TEST_SPLIT = "outputs/test_split.pkl"

def main():
    logger = setup_logger()
    if not Path(MODEL_ART).exists():
        logger.error(f"Missing model artifact: {MODEL_ART}. Run `python run_compare.py` or train XGB first.")
        return
    if not Path(TEST_SPLIT).exists():
        logger.error(f"Missing test split: {TEST_SPLIT}. Run `python run_compare.py` or `python run_pipeline.py` first.")
        return

    art = joblib.load(MODEL_ART)
    pipe = art["pipeline"]
    X_test, y_test = joblib.load(TEST_SPLIT)

    logger.info("Scoring probabilities for calibration and threshold analysis...")
    y_score = pipe.predict_proba(X_test)[:, 1]

    # Reliability curve
    disp = CalibrationDisplay.from_predictions(y_test, y_score, n_bins=10)
    ensure_dir("outputs")
    plt.savefig("outputs/calibration_curve.png", bbox_inches="tight")
    plt.close()

    # Threshold selection examples
    p, r, th = precision_recall_curve(y_test, y_score)
    # Example policy: choose threshold where precision >= 0.6 (if available), otherwise F1-like tradeoff
    idx = None
    for i in range(len(r)):
        if i < len(th) and p[i] >= 0.60:
            idx = i
            break
    if idx is None:
        # fallback: maximize 0.5*precision + 0.5*recall
        scores = 0.5 * p[:-1] + 0.5 * r[:-1]
        idx = int(np.argmax(scores))

    chosen = float(th[idx]) if idx < len(th) else 0.5
    report = {"chosen_threshold": chosen, "policy_note": "Target precision >= 0.60 if achievable; else trade-off."}
    with open("outputs/threshold_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved outputs/calibration_curve.png and outputs/threshold_report.json")

if __name__ == "__main__":
    main()
