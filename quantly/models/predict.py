"""Model inference and SHAP value generation.

Scores a feature matrix and returns probability estimates
plus SHAP feature attributions for explainability.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quantly.models.train import load_models

logger = logging.getLogger(__name__)


def score_candidates(
    feature_matrix: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Score a feature matrix and return top-N candidates.

    Args:
        feature_matrix: Feature DataFrame indexed by ticker
        top_n: Number of top candidates to return

    Returns:
        DataFrame with columns: ticker, lgbm_prob, xgb_prob, ensemble_prob,
        ranked by ensemble_prob descending
    """
    lgbm, xgb = load_models()
    X = feature_matrix.select_dtypes(include=[np.number]).fillna(0)

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


def compute_shap_values(
    feature_matrix: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    """Compute SHAP values for the top candidates.

    Args:
        feature_matrix: Feature matrix subset for tickers of interest
        tickers: Which tickers to explain

    Returns:
        DataFrame with SHAP values per feature per ticker
    """
    import shap

    lgbm, _ = load_models()
    X = feature_matrix.loc[tickers].select_dtypes(include=[np.number]).fillna(0)
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    return pd.DataFrame(shap_values, index=tickers, columns=X.columns)


def get_top_shap_features(
    shap_df: pd.DataFrame,
    ticker: str,
    top_n: int = 5,
) -> list[dict]:
    """Get the top N most important features for a single ticker's prediction.

    Returns:
        List of dicts with keys: feature, shap_value, direction (positive/negative)
    """
    row = shap_df.loc[ticker].abs().sort_values(ascending=False).head(top_n)
    return [
        {
            "feature": feat,
            "shap_value": shap_df.loc[ticker, feat],
            "direction": "bullish" if shap_df.loc[ticker, feat] > 0 else "bearish",
        }
        for feat in row.index
    ]
