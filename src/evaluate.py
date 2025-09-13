from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

from .utils import setup_logger, ensure_dir, save_json

def _get_feature_names(preprocessor, num_cols, cat_cols):
    # Get names from ColumnTransformer pieces
    num_features = num_cols
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    return list(num_features) + list(cat_feature_names)

def evaluate_and_save(model_artifact: str, X_test, y_test,
                      roc_path: str = "outputs/roc_curve.png",
                      feat_path: str = "outputs/feature_importance.png",
                      metrics_path: str = "outputs/metrics.json"):
    logger = setup_logger()
    logger.info("Loading trained model artifact...")
    artifact = joblib.load(model_artifact)
    pipe = artifact["pipeline"]
    num_cols = artifact["num_cols"]
    cat_cols = artifact["cat_cols"]

    logger.info("Scoring test set...")
    y_scores = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, y_scores))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {"AUC": auc, "confusion_matrix": cm, "classification_report": report}
    save_json(metrics, metrics_path)
    logger.info(f"Saved metrics to {metrics_path}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    ensure_dir(Path(roc_path).parent)
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    # Feature importance (coefficients)
    # Extract coefficients from final estimator
    clf = pipe.named_steps["clf"]
    pre = pipe.named_steps["pre"]
    feature_names = _get_feature_names(pre, num_cols, cat_cols)

    coefs = clf.coef_.flatten()
    # Get top absolute coefficients for readability
    top_idx = np.argsort(np.abs(coefs))[::-1][:20]
    top_features = [feature_names[i] for i in top_idx]
    top_values = coefs[top_idx]

    # Bar plot
    plt.figure(figsize=(8,6))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_values)
    plt.yticks(y_pos, top_features)
    plt.xlabel("Logistic Coefficient")
    plt.title("Top Feature Importances (absolute)")
    plt.gca().invert_yaxis()
    ensure_dir(Path(feat_path).parent)
    plt.savefig(feat_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved ROC curve to {roc_path} and feature importances to {feat_path}")
