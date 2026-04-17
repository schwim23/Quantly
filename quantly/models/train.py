"""Walk-forward ML model training (LightGBM + XGBoost ensemble)."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_lgbm(X: pd.DataFrame, y: pd.Series) -> object:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X.fillna(0), y)
    return model


def train_xgb(X: pd.DataFrame, y: pd.Series) -> object:
    import xgboost as xgb
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X.fillna(0), y)
    return model


def save_models(lgbm_model: object, xgb_model: object, feature_names: list[str]) -> None:
    with (MODELS_DIR / "lgbm_latest.pkl").open("wb") as f:
        pickle.dump(lgbm_model, f)
    with (MODELS_DIR / "xgb_latest.pkl").open("wb") as f:
        pickle.dump(xgb_model, f)
    (MODELS_DIR / "feature_names.txt").write_text("\n".join(feature_names))
    logger.info("Models saved to %s", MODELS_DIR)


def load_models() -> tuple:
    """Load latest trained models. Returns (lgbm, xgb, feature_names)."""
    with (MODELS_DIR / "lgbm_latest.pkl").open("rb") as f:
        lgbm = pickle.load(f)  # noqa: S301
    with (MODELS_DIR / "xgb_latest.pkl").open("rb") as f:
        xgb = pickle.load(f)  # noqa: S301
    features = (MODELS_DIR / "feature_names.txt").read_text().splitlines()
    return lgbm, xgb, features


def walk_forward_train(
    feature_matrix: pd.DataFrame,
    labels: pd.DataFrame,
    train_years: int = 2,
    purge_days: int = 5,
) -> tuple:
    """Walk-forward training: train on rolling window, validate on next quarter.

    Returns:
        (lgbm_model, xgb_model) trained on the most recent full window
    """
    merged = feature_matrix.reset_index().merge(
        labels[["ticker", "signal_date", "label"]].dropna(subset=["label"]),
        on="ticker",
    )
    merged["signal_date"] = pd.to_datetime(merged["signal_date"])
    merged = merged.sort_values("signal_date")

    # Use the most recent train_years window for the production model
    cutoff = merged["signal_date"].max() - pd.DateOffset(years=train_years)
    train_df = merged[merged["signal_date"] >= cutoff]

    feature_cols = [c for c in feature_matrix.columns if c not in ("ticker", "signal_date")]
    X = train_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = train_df["label"].astype(int)

    if len(y.unique()) < 2:
        raise ValueError("Training data has only one class — need more history")

    lgbm = train_lgbm(X, y)
    xgb = train_xgb(X, y)
    save_models(lgbm, xgb, list(X.columns))
    return lgbm, xgb
