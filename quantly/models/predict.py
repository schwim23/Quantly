"""Model inference and SHAP explainability."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def score_candidates(feature_matrix: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Score a feature matrix and return top-N candidates by ensemble probability.

    Returns:
        DataFrame with columns: ticker, lgbm_prob, xgb_prob, ensemble_prob
        Sorted descending by ensemble_prob
    """
    from quantly.models.train import load_models

    lgbm, xgb, feature_names = load_models()

    # Align to trained feature set
    X = feature_matrix.reindex(columns=feature_names, fill_value=0).fillna(0)

    lgbm_probs = lgbm.predict_proba(X)[:, 1]
    xgb_probs = xgb.predict_proba(X)[:, 1]
    ensemble = (lgbm_probs + xgb_probs) / 2

    results = pd.DataFrame({
        "ticker": feature_matrix.index,
        "lgbm_prob": lgbm_probs,
        "xgb_prob": xgb_probs,
        "ensemble_prob": ensemble,
    }).sort_values("ensemble_prob", ascending=False)

    return results.head(top_n).reset_index(drop=True)


def compute_shap_values(feature_matrix: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Compute SHAP values for a subset of tickers using the LightGBM model."""
    import shap
    from quantly.models.train import load_models

    lgbm, _, feature_names = load_models()
    X = feature_matrix.loc[tickers].reindex(columns=feature_names, fill_value=0).fillna(0)
    explainer = shap.TreeExplainer(lgbm)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    return pd.DataFrame(shap_vals, index=tickers, columns=X.columns)


def get_top_shap_features(shap_df: pd.DataFrame, ticker: str, top_n: int = 5) -> list[dict]:
    """Top N SHAP features for a single ticker, sorted by absolute importance."""
    row = shap_df.loc[ticker]
    top = row.abs().sort_values(ascending=False).head(top_n)
    return [
        {
            "feature": feat,
            "shap_value": float(row[feat]),
            "direction": "bullish" if row[feat] > 0 else "bearish",
        }
        for feat in top.index
    ]
