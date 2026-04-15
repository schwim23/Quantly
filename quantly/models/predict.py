"""Model inference + SHAP explainability.

For each candidate ticker, produces:
  - conviction score (0–100): probability of outperforming SPY × 100
  - top_drivers: top 3 feature names by absolute SHAP value
  - shap_values: dict of feature → SHAP contribution
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from quantly.features.pipeline import FEATURE_COLUMNS
from quantly.models.train import load_model

logger = logging.getLogger(__name__)


def _safe_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def score_candidates(
    features_df: pd.DataFrame,
    model: Optional[Any] = None,
) -> pd.DataFrame:
    """Score each ticker in *features_df* and return ranked results.

    Args:
        features_df: DataFrame indexed by ticker (output of build_feature_matrix).
        model: Fitted LGBMClassifier. If None, loads from disk.

    Returns:
        DataFrame indexed by ticker with columns:
            conviction      - 0–100 score (probability × 100)
            top_drivers     - list of top 3 feature names
            shap_values     - dict of feature → SHAP float
        Sorted by conviction descending.
    """
    if model is None:
        model = load_model()

    if model is None:
        logger.warning("No model available — returning empty scores")
        return pd.DataFrame(columns=["conviction", "top_drivers", "shap_values"])

    if features_df.empty:
        return pd.DataFrame(columns=["conviction", "top_drivers", "shap_values"])

    feat_cols = _safe_feature_columns(features_df)
    X = features_df[feat_cols].fillna(0.0).astype(float)

    # Predict probability of outperformance
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return pd.DataFrame(columns=["conviction", "top_drivers", "shap_values"])

    # SHAP values for explainability
    shap_results: list[dict] = []
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_matrix = explainer.shap_values(X)
        if isinstance(shap_matrix, list):
            shap_matrix = shap_matrix[1]  # class 1 (positive)

        for i, ticker in enumerate(features_df.index):
            row_shap = dict(zip(feat_cols, shap_matrix[i]))
            top_3 = sorted(row_shap, key=lambda k: abs(row_shap[k]), reverse=True)[:3]
            shap_results.append({
                "ticker": ticker,
                "shap_values": row_shap,
                "top_drivers": top_3,
            })
    except Exception as exc:
        logger.warning("SHAP computation failed: %s — using feature importance fallback", exc)
        try:
            importances = dict(zip(feat_cols, model.feature_importances_))
            top_3_global = sorted(importances, key=importances.get, reverse=True)[:3]
        except Exception:
            top_3_global = feat_cols[:3]
        for ticker in features_df.index:
            shap_results.append({
                "ticker": ticker,
                "shap_values": {},
                "top_drivers": top_3_global,
            })

    results = []
    for i, row in enumerate(shap_results):
        results.append({
            "ticker": row["ticker"],
            "conviction": round(float(proba[i]) * 100, 1),
            "top_drivers": row["top_drivers"],
            "shap_values": row["shap_values"],
        })

    df = pd.DataFrame(results).set_index("ticker")
    return df.sort_values("conviction", ascending=False)
