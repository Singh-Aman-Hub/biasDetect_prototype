from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _encode_with_feature_cols(
    df: pd.DataFrame,
    target_col: str,
    sensitive_cols: List[str],
    feature_cols: List[str],
) -> pd.DataFrame:
    X = df.drop(columns=[target_col], errors="ignore").copy()
    sensitive_set = set(sensitive_cols)

    encoded = X.copy()
    for col in encoded.select_dtypes(include=["object", "string"]).columns:
        if col not in sensitive_set:
            encoded = pd.get_dummies(encoded, columns=[col], drop_first=True, dtype=float)

    encoded = encoded.drop(columns=[col for col in encoded.columns if col in sensitive_set], errors="ignore")

    for col in encoded.columns:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
        if encoded[col].isna().all():
            encoded[col] = 0.0
        else:
            encoded[col] = encoded[col].fillna(encoded[col].median())

    return encoded.reindex(columns=feature_cols, fill_value=0.0)


def _flip_series(series: pd.Series) -> pd.Series:
    values = [item for item in series.fillna("missing").astype(str).unique().tolist()]
    if len(values) < 2:
        return series.copy()

    values_sorted = sorted(values)
    if len(values_sorted) == 2:
        mapping = {
            values_sorted[0]: values_sorted[1],
            values_sorted[1]: values_sorted[0],
        }
    else:
        rotated = values_sorted[1:] + values_sorted[:1]
        mapping = dict(zip(values_sorted, rotated))

    flipped = series.fillna("missing").astype(str).map(mapping)
    return flipped


def run_counterfactual_test(
    df: pd.DataFrame,
    shared_model: Any,
    primary_sensitive_col: str,
    target_col: str,
) -> Dict[str, Any]:
    """Flip one sensitive attribute and measure decision instability."""
    if df.empty:
        raise ValueError("Counterfactual test requires a non-empty DataFrame.")
    if primary_sensitive_col not in df.columns:
        raise ValueError(f"Primary sensitive column '{primary_sensitive_col}' not found.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if not isinstance(shared_model, dict) or "model" not in shared_model:
        raise ValueError("shared_model must be a model bundle dict with keys: model, feature_cols.")

    model = shared_model["model"]
    feature_cols = list(shared_model.get("feature_cols", []))
    sensitive_cols = list(shared_model.get("sensitive_cols", [primary_sensitive_col]))
    if primary_sensitive_col not in sensitive_cols:
        sensitive_cols.append(primary_sensitive_col)

    X_original = _encode_with_feature_cols(
        df=df,
        target_col=target_col,
        sensitive_cols=sensitive_cols,
        feature_cols=feature_cols,
    )
    # Safely convert predictions to int, handling NaN and float values
    original_pred_raw = pd.Series(model.predict(X_original), index=df.index)
    original_pred = original_pred_raw.fillna(0).astype(float).round().astype(int)

    flipped_df = df.copy()
    flipped_df[primary_sensitive_col] = _flip_series(flipped_df[primary_sensitive_col])
    X_flipped = _encode_with_feature_cols(
        df=flipped_df,
        target_col=target_col,
        sensitive_cols=sensitive_cols,
        feature_cols=feature_cols,
    )
    # Safely convert predictions to int, handling NaN and float values
    flipped_pred_raw = pd.Series(model.predict(X_flipped), index=df.index)
    flipped_pred = flipped_pred_raw.fillna(0).astype(float).round().astype(int)

    changed_mask = original_pred != flipped_pred
    total = len(df)
    changed = int(changed_mask.sum())
    flip_rate = float(changed / total) if total else 0.0

    neg_to_pos = int(((original_pred == 0) & (flipped_pred == 1)).sum())
    pos_to_neg = int(((original_pred == 1) & (flipped_pred == 0)).sum())

    original_groups = df[primary_sensitive_col].fillna("missing").astype(str)
    by_group: Dict[str, float] = {}
    for group in sorted(original_groups.unique().tolist()):
        mask = original_groups == group
        rate = float((original_pred[mask] != flipped_pred[mask]).mean()) if int(mask.sum()) else 0.0
        by_group[str(group)] = round(rate, 4)

    return {
        "flip_rate": round(flip_rate, 4),
        "flip_direction_breakdown": {
            "negative_to_positive": round(float(neg_to_pos / total) if total else 0.0, 4),
            "positive_to_negative": round(float(pos_to_neg / total) if total else 0.0, 4),
            "by_original_group": by_group,
        },
        "changed_decisions": changed,
        "total_records": total,
    }
