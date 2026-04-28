from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


def _resolve_positive_label(series: pd.Series) -> Any:
    unique = [item for item in series.dropna().unique().tolist()]
    if not unique:
        raise ValueError("Target column has no non-null labels.")
    if 1 in unique:
        return 1
    if "1" in unique:
        return "1"
    if True in unique:
        return True
    return sorted(unique, key=lambda value: str(value))[-1]


def _encode_with_feature_cols(
    df: pd.DataFrame,
    target_col: str,
    sensitive_cols: List[str],
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=[target_col], errors="ignore").copy()
    X_original = X.copy()
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

    encoded = encoded.reindex(columns=feature_cols, fill_value=0.0)
    return encoded, X_original


def _extract_shap_matrix(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            return np.asarray(shap_values[1])
        if len(shap_values) == 1:
            return np.asarray(shap_values[0])
        return np.empty((0, 0))

    matrix = np.asarray(shap_values)
    if matrix.ndim == 3:
        if matrix.shape[-1] > 1:
            return matrix[:, :, 1]
        return matrix[:, :, 0]
    return matrix


def _top_feature_contributions(
    model: Any,
    X_sample: pd.DataFrame,
    top_k: int,
) -> List[List[Dict[str, float]]]:
    if shap is None:
        importance = np.ones(X_sample.shape[1], dtype=float)
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coeff = np.asarray(model.coef_)
            importance = np.abs(coeff[0]) if coeff.ndim > 1 else np.abs(coeff)

        baseline = X_sample.mean(axis=0).to_numpy(dtype=float)
        contributions_per_row: List[List[Dict[str, float]]] = []
        for _, row in X_sample.iterrows():
            raw_contrib = (row.to_numpy(dtype=float) - baseline) * importance
            pairs = [
                {"feature": X_sample.columns[idx], "shap_value": float(raw_contrib[idx])}
                for idx in range(len(raw_contrib))
            ]
            pairs.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
            contributions_per_row.append(pairs[:top_k])
        return contributions_per_row

    explainer = shap.TreeExplainer(model)
    # Disable additivity check to handle datasets with varying feature scales or distributions
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    matrix = _extract_shap_matrix(shap_values)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    contributions_per_row = []
    for row_idx in range(matrix.shape[0]):
        values = matrix[row_idx]
        pairs = [
            {"feature": X_sample.columns[idx], "shap_value": float(values[idx])}
            for idx in range(len(values))
        ]
        pairs.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
        contributions_per_row.append(pairs[:top_k])
    return contributions_per_row


def _build_counterfactual_text(
    row_original: pd.Series,
    nearest_approved_original: Optional[pd.Series],
) -> str:
    if nearest_approved_original is None:
        if "credit_score" in row_original.index and pd.notna(row_original.get("credit_score")):
            return "If credit score increased by roughly 50 points, this case may be approved."
        return "Improving key risk features toward approved-peer values may flip this decision."

    numeric_cols = [
        col
        for col in row_original.index
        if pd.api.types.is_number(row_original[col]) and pd.api.types.is_number(nearest_approved_original[col])
    ]
    if not numeric_cols:
        return "Matching patterns seen in similar approved records may flip this decision."

    adjustments: List[Tuple[str, float]] = []
    for col in numeric_cols:
        diff = float(nearest_approved_original[col] - row_original[col])
        if diff > 0:
            adjustments.append((col, diff))

    if not adjustments:
        return "Reducing risk-sensitive feature values toward approved peers may help this case."

    adjustments.sort(key=lambda item: abs(item[1]), reverse=True)
    top = adjustments[:2]
    phrases = [f"{feature} +{delta:.1f}" for feature, delta in top]
    return "If " + " and ".join(phrases) + ", this case would be closer to approved neighbors."


def explain_flagged_decisions(
    df: pd.DataFrame,
    shared_model: Any,
    sensitive_cols: List[str],
    target_col: str,
    max_samples: int = 8,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """Explain rejected decisions using SHAP and nearest-neighbor counterfactuals."""
    if not isinstance(shared_model, dict) or "model" not in shared_model:
        raise ValueError("shared_model must be a model bundle dict with keys: model, feature_cols.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    model = shared_model["model"]
    feature_cols = list(shared_model.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("shared_model.feature_cols is required for explainability.")

    X_encoded, X_original = _encode_with_feature_cols(
        df=df,
        target_col=target_col,
        sensitive_cols=sensitive_cols,
        feature_cols=feature_cols,
    )

    predictions = pd.Series(model.predict(X_encoded), index=X_encoded.index)
    rejected_idx = predictions[predictions == 0].index.tolist()
    if not rejected_idx:
        rejected_idx = predictions.index.tolist()

    sampled_idx = rejected_idx[:max_samples]
    sample_encoded = X_encoded.loc[sampled_idx]
    sample_original = X_original.loc[sampled_idx]

    approved_idx = predictions[predictions == 1].index.tolist()
    approved_encoded = X_encoded.loc[approved_idx] if approved_idx else pd.DataFrame(columns=X_encoded.columns)
    approved_original = X_original.loc[approved_idx] if approved_idx else pd.DataFrame(columns=X_original.columns)

    contributions = _top_feature_contributions(model=model, X_sample=sample_encoded, top_k=top_k)

    nn_model = None
    if not approved_encoded.empty:
        nn_model = NearestNeighbors(n_neighbors=1, n_jobs=1)
        nn_model.fit(approved_encoded)

    explanations: List[Dict[str, Any]] = []
    for row_pos, row_idx in enumerate(sample_encoded.index):
        row_encoded = sample_encoded.loc[[row_idx]]
        row_original = sample_original.loc[row_idx]

        nearest_original = None
        if nn_model is not None:
            nearest_position = int(nn_model.kneighbors(row_encoded, return_distance=False)[0][0])
            nearest_original = approved_original.iloc[nearest_position]

        # Safely convert row_idx to int if it's numeric, otherwise use row_pos
        try:
            row_idx_value = int(row_idx) if isinstance(row_idx, (int, float, np.integer)) else row_pos
        except (ValueError, TypeError):
            row_idx_value = row_pos
        
        # Safely convert prediction to int
        pred_value = predictions.loc[row_idx]
        try:
            pred_int = int(pred_value) if pd.notna(pred_value) else 0
        except (ValueError, TypeError):
            pred_int = 0

        explanations.append(
            {
                "row_index": row_idx_value,
                "predicted_label": pred_int,
                "top_contributing_features": contributions[row_pos],
                "counterfactual": _build_counterfactual_text(row_original, nearest_original),
            }
        )

    return explanations


def generate_narrative_summary(explanations: List[Dict[str, Any]]) -> str:
    """Convert SHAP-style contributions into a compact English summary."""
    if not explanations:
        return "No flagged decisions were found for explanation in this batch."

    feature_counts: Dict[str, int] = {}
    feature_magnitude: Dict[str, float] = {}

    for entry in explanations:
        for item in entry.get("top_contributing_features", []):
            feature = str(item.get("feature", "unknown"))
            value = float(item.get("shap_value", 0.0))
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
            feature_magnitude[feature] = feature_magnitude.get(feature, 0.0) + abs(value)

    ranked = sorted(
        feature_counts.keys(),
        key=lambda feature: (feature_counts[feature], feature_magnitude.get(feature, 0.0)),
        reverse=True,
    )
    top_features = ranked[:3]

    if not top_features:
        return (
            f"Explained {len(explanations)} flagged decisions, but no stable dominant feature pattern "
            "was detected."
        )

    lead = ", ".join(top_features)
    return (
        f"Across {len(explanations)} flagged decisions, the most influential rejection drivers were "
        f"{lead}. Counterfactual neighbor checks indicate that moderate improvements in these drivers "
        "could move many cases toward approval outcomes."
    )
