from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans


def _to_categorical(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) >= 2:
        # Bin numeric values to evaluate categorical-style dependence.
        return pd.qcut(series, q=min(10, series.nunique()), duplicates="drop").astype(str)
    return series.fillna("missing").astype(str)


def _cramers_v(feature: pd.Series, sensitive: pd.Series) -> Optional[float]:
    contingency = pd.crosstab(_to_categorical(feature), _to_categorical(sensitive))
    if contingency.empty or min(contingency.shape) < 2:
        return None

    n = contingency.to_numpy().sum()
    if n <= 0:
        return None

    chi2 = chi2_contingency(contingency)[0]
    phi2 = chi2 / n
    rows, cols = contingency.shape
    denominator = max(min(rows - 1, cols - 1), 1)
    return float(np.sqrt(phi2 / denominator))


def _kmeans_purity(feature: pd.Series, sensitive: pd.Series) -> Optional[float]:
    work = pd.DataFrame({"feature": feature, "sensitive": sensitive}).dropna()
    if work.empty:
        return None

    if not pd.api.types.is_numeric_dtype(work["feature"]):
        work["feature"] = pd.Categorical(work["feature"].astype(str)).codes.astype(float)
    else:
        work["feature"] = pd.to_numeric(work["feature"], errors="coerce")

    work = work.dropna()
    if len(work) < 10:
        return None

    sensitive_series = work["sensitive"].astype(str)
    unique_groups = sensitive_series.nunique()
    if unique_groups < 2:
        return None

    n_clusters = int(min(max(2, unique_groups), 6))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(work[["feature"]].to_numpy())

    purity_count = 0
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        cluster_labels = sensitive_series[cluster_mask]
        if cluster_labels.empty:
            continue
        purity_count += int(cluster_labels.value_counts().iloc[0])

    return float(purity_count / len(work)) if len(work) else None


def detect_proxy_features(df: pd.DataFrame, sensitive_cols: List[str]) -> Dict[str, Any]:
    """Detect likely proxy features via correlation and cluster purity signals."""
    if df.empty:
        raise ValueError("Proxy detection requires a non-empty DataFrame.")

    missing_sensitive = [col for col in sensitive_cols if col not in df.columns]
    if missing_sensitive:
        raise ValueError(f"Sensitive columns not found: {missing_sensitive}")

    excluded_names = {"target", "label", "outcome", "approved", "income_encoded"}
    candidate_features = [
        col
        for col in df.columns
        if col not in set(sensitive_cols)
        and col.lower() not in excluded_names
    ]

    proxy_candidates: List[Dict[str, Any]] = []
    proxy_feature_names = set()

    for feature_col in candidate_features:
        for sensitive_col in sensitive_cols:
            cramers_score = _cramers_v(df[feature_col], df[sensitive_col])
            purity_score = _kmeans_purity(df[feature_col], df[sensitive_col])

            methods: List[str] = []
            score_map: Dict[str, float] = {}
            if cramers_score is not None and cramers_score >= 0.60:
                methods.append("cramers_v")
                score_map["cramers_v"] = round(float(cramers_score), 4)
            if purity_score is not None and purity_score >= 0.80:
                methods.append("kmeans_purity")
                score_map["kmeans_purity"] = round(float(purity_score), 4)

            if not methods:
                continue

            proxy_feature_names.add(feature_col)
            proxy_candidates.append(
                {
                    "feature": feature_col,
                    "sensitive_attribute": sensitive_col,
                    "methods": methods,
                    "scores": score_map,
                    "combined_score": round(max(score_map.values()), 4),
                }
            )

    proxy_candidates.sort(key=lambda item: item["combined_score"], reverse=True)
    safe_features = sorted([col for col in candidate_features if col not in proxy_feature_names])

    return {
        "proxy_candidates": proxy_candidates,
        "safe_features": safe_features,
    }
