from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from src.model_logistic import train_model as train_logit
from src.model_rf import train_model as train_rf
from src.model_xgb import train_model as train_xgb
from src.utils import ensure_dir, save_json

DATA_PATH = "data/synthetic_households.csv"

def _auc_and_curve(pipe, X_test, y_test):
    y_score = pipe.predict_proba(X_test)[:,1]
    auc = float(roc_auc_score(y_test, y_score))
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return auc, fpr, tpr

def main():
    # Train three models on the same split for fair comparison
    logit, (X_test, y_test) = train_logit(DATA_PATH, model_out="outputs/model.pkl")
    rf, _ = train_rf(DATA_PATH, model_out="outputs/model_rf.pkl")
    xgb, _ = train_xgb(DATA_PATH, model_out="outputs/model_xgb.pkl")

    # Compute AUCs and ROC curves
    auc_logit, fpr_l, tpr_l = _auc_and_curve(logit, X_test, y_test)
    auc_rf, fpr_r, tpr_r = _auc_and_curve(rf, X_test, y_test)
    auc_xgb, fpr_x, tpr_x = _auc_and_curve(xgb, X_test, y_test)

    # Save metrics
    metrics = {
        "LogisticRegression": {"AUC": auc_logit},
        "RandomForest": {"AUC": auc_rf},
        "XGBoost": {"AUC": auc_xgb}
    }
    save_json(metrics, "outputs/metrics_compare.json")

    # Plot ROC curves on one chart (single figure, multiple lines)
    plt.figure()
    plt.plot(fpr_l, tpr_l, label=f"Logit (AUC={auc_logit:.3f})")
    plt.plot(fpr_r, tpr_r, label=f"RandomForest (AUC={auc_rf:.3f})")
    plt.plot(fpr_x, tpr_x, label=f"XGBoost (AUC={auc_xgb:.3f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    ensure_dir("outputs")
    plt.savefig("outputs/roc_compare.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
