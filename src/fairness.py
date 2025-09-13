from __future__ import annotations
from pathlib import Path
import json, joblib
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

from .utils import setup_logger, ensure_dir

MODEL_ART = "outputs/model_xgb.pkl"
TEST_SPLIT = "outputs/test_split.pkl"
TEST_META = "outputs/test_meta.pkl"

def _slice_report(y_true, y_score, mask):
    return {
        "n": int(mask.sum()),
        "AUC": float(roc_auc_score(y_true[mask], y_score[mask])) if mask.sum() > 1 else None,
        "Brier": float(brier_score_loss(y_true[mask], y_score[mask])) if mask.sum() > 1 else None,
    }

def main():
    logger = setup_logger()
    if not (Path(MODEL_ART).exists() and Path(TEST_SPLIT).exists() and Path(TEST_META).exists()):
        logger.error("Missing artifacts. Run training first.")
        return

    art = joblib.load(MODEL_ART)
    pipe = art["pipeline"]
    X_test, y_test = joblib.load(TEST_SPLIT)
    idx_list, meta = joblib.load(TEST_META)

    y_score = pipe.predict_proba(X_test)[:, 1]

    # Analyze results for DAC (Disadvantaged Community) groups
    dac = meta["dac_flag"].values.astype(int)
    dac_mask = dac == 1
    non_mask = dac == 0

    report = {
        "overall": {"AUC": float(roc_auc_score(y_test, y_score)), "Brier": float(brier_score_loss(y_test, y_score)), "n": int(len(y_test))},
        "DAC": _slice_report(y_test.values, y_score, dac_mask),
        "non_DAC": _slice_report(y_test.values, y_score, non_mask),
        "by_region": {}
    }

    # Analyze results by region
    regions = meta["region"].astype(str).values
    for r in np.unique(regions):
        mask = regions == r
        report["by_region"][r] = _slice_report(y_test.values, y_score, mask)

    ensure_dir("outputs")
    with open("outputs/fairness_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved outputs/fairness_report.json")

if __name__ == "__main__":
    main()
