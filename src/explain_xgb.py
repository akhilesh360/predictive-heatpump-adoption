from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import setup_logger, ensure_dir
from .preprocess import get_feature_lists, load_data
from .model_xgb import train_model as train_xgb

DATA_PATH = "data/synthetic_households.csv"
MODEL_PATH = "outputs/model_xgb.pkl"

def main():
    logger = setup_logger()
    # Make sure the XGBoost model is trained or loaded
    if not Path(MODEL_PATH).exists():
        logger.info("Training XGBoost model (artifact not found)...")
        pipe, (X_test, y_test) = train_xgb(DATA_PATH, model_out=MODEL_PATH)
    else:
        logger.info("Loading existing XGBoost artifact...")
        artifact = joblib.load(MODEL_PATH)
        pipe = artifact["pipeline"]

    # Load the raw data and create a small sample for SHAP analysis
    df = load_data(DATA_PATH)
    num_cols, cat_cols, target_col = get_feature_lists()
    X = df[num_cols + cat_cols]
    y = df[target_col]

    # Use a small sample for SHAP to keep things fast
    sample = X.sample(n=min(1000, len(X)), random_state=42)

    # Get the fitted model components
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["clf"]

    # Transform features so they match what the model expects
    X_trans = pre.transform(sample)

    # Build a SHAP explainer using the XGBoost model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # Get feature names after preprocessing (numeric and one-hot categorical)
    from .evaluate import _get_feature_names
    feature_names = _get_feature_names(pre, num_cols, cat_cols)

    # Make a SHAP summary plot (beeswarm)
    plt.figure(figsize=(9,6))
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
    ensure_dir("outputs")
    plt.savefig("outputs/shap_summary_xgb.png", bbox_inches="tight")
    plt.close()

    # 2) Mean absolute SHAP values (bar) plot
    shap_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(shap_abs)[::-1][:20]
    top_feats = [feature_names[i] for i in order]
    top_vals = shap_abs[order]

    plt.figure(figsize=(8,6))
    y_pos = np.arange(len(top_feats))
    plt.barh(y_pos, top_vals)
    plt.yticks(y_pos, top_feats)
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top XGBoost Feature Importance (SHAP)")
    plt.gca().invert_yaxis()
    plt.savefig("outputs/shap_bar_xgb.png", bbox_inches="tight")
    plt.close()

    # Save table of SHAP importances
    out_df = pd.DataFrame({"feature": top_feats, "mean_abs_shap": top_vals})
    out_df.to_csv("outputs/shap_top_features_xgb.csv", index=False)
    logger.info("Saved SHAP plots and table to outputs/.")

if __name__ == "__main__":
    main()
