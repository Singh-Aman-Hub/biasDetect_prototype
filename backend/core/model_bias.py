from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DEFAULT_METRIC_WEIGHTS = {
    "demographic_parity": 0.40,
    "equal_opportunity": 0.35,
    "fpr_gap": 0.25,
}


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


def _encode_features(
    df: pd.DataFrame,
    target_col: str,
    sensitive_cols: List[str],
    positive_label: Any,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
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

    if feature_cols is not None:
        encoded = encoded.reindex(columns=feature_cols, fill_value=0.0)

    y = (df[target_col] == positive_label).astype(int)
    return encoded, y, X_original


def _extract_model_bundle(
    df: pd.DataFrame,
    target_col: str,
    sensitive_cols: List[str],
    shared_model: Any,
) -> Tuple[Any, pd.DataFrame, pd.Series, pd.DataFrame, Any]:
    positive_label = _resolve_positive_label(df[target_col])

    if isinstance(shared_model, dict) and "model" in shared_model:
        model = shared_model["model"]
        if all(key in shared_model for key in ["X_test", "y_test", "X_test_original"]):
            return (
                model,
                shared_model["X_test"].copy(),
                pd.Series(shared_model["y_test"]).copy(),
                shared_model["X_test_original"].copy(),
                shared_model.get("positive_label", positive_label),
            )

        feature_cols = shared_model.get("feature_cols")
        X, y, X_original = _encode_features(
            df=df,
            target_col=target_col,
            sensitive_cols=sensitive_cols,
            positive_label=shared_model.get("positive_label", positive_label),
            feature_cols=feature_cols,
        )
        stratify_target = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_target,
        )
        X_test_original = X_original.loc[X_test.index]
        return model, X_test, y_test, X_test_original, shared_model.get("positive_label", positive_label)

    if shared_model is not None and hasattr(shared_model, "predict"):
        model = shared_model
    else:
        model = None

    X, y, X_original = _encode_features(
        df=df,
        target_col=target_col,
        sensitive_cols=sensitive_cols,
        positive_label=positive_label,
    )
    stratify_target = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )
    X_test_original = X_original.loc[X_test.index]

    if model is None:
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
        model.fit(X_train, y_train)

    return model, X_test, y_test, X_test_original, positive_label


def run_model_bias_analysis(
    df: pd.DataFrame,
    sensitive_cols: List[str],
    target_col: str,
    shared_model: Any,
    metric_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run model-level fairness diagnostics with fairlearn."""
    if df.empty:
        raise ValueError("Model bias analysis requires a non-empty DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    weights = DEFAULT_METRIC_WEIGHTS.copy()
    if metric_weights:
        for key, value in metric_weights.items():
            if key in weights:
                weights[key] = float(value)

    model, X_test, y_test, X_test_original, _ = _extract_model_bundle(
        df=df,
        target_col=target_col,
        sensitive_cols=sensitive_cols,
        shared_model=shared_model,
    )

    y_true = pd.Series(y_test).reset_index(drop=True)
    y_pred = pd.Series(model.predict(X_test)).reset_index(drop=True)

    metrics_by_group: Dict[str, Any] = {}
    fairness_gaps = {
        "max_demographic_parity_difference": 0.0,
        "max_equal_opportunity_difference": 0.0,
        "max_fpr_gap": 0.0,
        "max_equalized_odds_difference": 0.0,
    }

    per_attribute_scores: List[float] = []

    for sensitive_col in sensitive_cols:
        if sensitive_col not in X_test_original.columns:
            continue

        sensitive_values = X_test_original[sensitive_col].reset_index(drop=True).astype(str)

        frame = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "selection_rate": selection_rate,
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_values,
        )

        by_group_df = frame.by_group
        if isinstance(by_group_df, pd.Series):
            by_group_df = by_group_df.to_frame().T

        group_metrics: Dict[str, Dict[str, float]] = {}
        tpr_values: List[float] = []
        fpr_values: List[float] = []

        for group_name, row in by_group_df.iterrows():
            tpr = float(row.get("tpr", 0.0))
            fpr = float(row.get("fpr", 0.0))
            tpr_values.append(tpr)
            fpr_values.append(fpr)
            group_metrics[str(group_name)] = {
                "accuracy": float(row.get("accuracy", 0.0)),
                "selection_rate": float(row.get("selection_rate", 0.0)),
                "tpr": tpr,
                "fpr": fpr,
            }

        dpd = float(
            abs(
                demographic_parity_difference(
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive_values,
                )
            )
        )
        eod = float(
            abs(
                equalized_odds_difference(
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive_values,
                )
            )
        )
        tpr_gap = float(max(tpr_values) - min(tpr_values)) if tpr_values else 0.0
        fpr_gap = float(max(fpr_values) - min(fpr_values)) if fpr_values else 0.0

        fairness_gaps["max_demographic_parity_difference"] = max(
            fairness_gaps["max_demographic_parity_difference"],
            dpd,
        )
        fairness_gaps["max_equal_opportunity_difference"] = max(
            fairness_gaps["max_equal_opportunity_difference"],
            tpr_gap,
        )
        fairness_gaps["max_fpr_gap"] = max(fairness_gaps["max_fpr_gap"], fpr_gap)
        fairness_gaps["max_equalized_odds_difference"] = max(
            fairness_gaps["max_equalized_odds_difference"],
            eod,
        )

        weighted_penalty = (
            weights["demographic_parity"] * dpd
            + weights["equal_opportunity"] * tpr_gap
            + weights["fpr_gap"] * fpr_gap
        )
        per_attribute_scores.append(max(0.0, 100.0 * (1.0 - weighted_penalty)))

        metrics_by_group[sensitive_col] = {
            "groups": group_metrics,
            "demographic_parity_difference": dpd,
            "equal_opportunity_difference": tpr_gap,
            "equalized_odds_difference": eod,
            "fpr_gap": fpr_gap,
        }

    fairness_score = float(np.mean(per_attribute_scores)) if per_attribute_scores else 0.0

    return {
        "fairness_score": round(fairness_score, 2),
        "metrics_by_group": metrics_by_group,
        "fairness_gaps": {
            key: round(float(value), 4) for key, value in fairness_gaps.items()
        },
        "metric_weights": {k: float(v) for k, v in weights.items()},
    }
