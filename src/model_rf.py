from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocess import build_preprocessor, get_feature_lists, load_data
from .utils import setup_logger, ensure_dir

def train_model(csv_path: str, model_out: str = "outputs/model_rf.pkl", test_size: float = 0.2, random_state: int = 42):
    logger = setup_logger()
    logger.info("Loading data...")
    df = load_data(csv_path)
    num_cols, cat_cols, target_col = get_feature_lists()

    X = df[num_cols + cat_cols]
    y = df[target_col]

    logger.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info("Building pipeline (preprocessor + RandomForestClassifier)...")
    preprocessor = build_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

    logger.info("Training model...")
    pipe.fit(X_train, y_train)

    ensure_dir(Path(model_out).parent)
    joblib.dump({"pipeline": pipe, "num_cols": num_cols, "cat_cols": cat_cols}, model_out)
    logger.info(f"Saved trained pipeline to {model_out}")

    return pipe, (X_test, y_test)
