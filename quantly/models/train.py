"""Walk-forward ML model training.

Trains LightGBM and XGBoost models using walk-forward cross-validation
to prevent lookahead bias and test generalization across time periods.
"""

from __future__ import annotations

import logging
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def walk_forward_train(
    feature_matrix: pd.DataFrame,
    labels: pd.DataFrame,
    train_years: int = 2,
    val_months: int = 3,
    purge_days: int = 5,
) -> dict:
    """Walk-forward training loop.

    Args:
        feature_matrix: Full feature DataFrame indexed by (ticker, signal_date)
        labels: Labels DataFrame with columns ticker, signal_date, label
        train_years: Rolling training window size in years
        val_months: Validation window size in months
        purge_days: Gap between train and val sets to prevent leakage

    Returns:
        Dict with keys: lgbm_model, xgb_model, val_metrics, feature_importances
    """
    import lightgbm as lgb
    import xgboost as xgb

    # TODO: implement full walk-forward loop with date-based splits
    raise NotImplementedError


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series, params: dict | None = None) -> object:
    """Train a single LightGBM classifier.

    Args:
        X_train: Feature matrix
        y_train: Binary labels (0/1)
        params: LightGBM hyperparameters (uses defaults if None)

    Returns:
        Trained LGBMClassifier
    """
    import lightgbm as lgb

    default_params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**(params or default_params))
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, params: dict | None = None) -> object:
    """Train a single XGBoost classifier."""
    import xgboost as xgb

    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "random_state": 42,
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**(params or default_params))
    model.fit(X_train, y_train)
    return model


def save_models(lgbm_model: object, xgb_model: object) -> None:
    """Save trained models to disk."""
    with (MODELS_DIR / "lgbm_latest.pkl").open("wb") as f:
        pickle.dump(lgbm_model, f)
    with (MODELS_DIR / "xgb_latest.pkl").open("wb") as f:
        pickle.dump(xgb_model, f)
    logger.info("Models saved to %s", MODELS_DIR)


def load_models() -> tuple:
    """Load the latest trained models from disk.

    Returns:
        Tuple of (lgbm_model, xgb_model)
    """
    with (MODELS_DIR / "lgbm_latest.pkl").open("rb") as f:
        lgbm = pickle.load(f)  # noqa: S301
    with (MODELS_DIR / "xgb_latest.pkl").open("rb") as f:
        xgb = pickle.load(f)  # noqa: S301
    return lgbm, xgb
