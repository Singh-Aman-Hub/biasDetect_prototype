from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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


def _encode_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    sensitive_cols: List[str],
    positive_label: Any,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    X_train_raw = train_df.drop(columns=[target_col], errors="ignore").copy()
    X_test_raw = test_df.drop(columns=[target_col], errors="ignore").copy()

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    sensitive_set = set(sensitive_cols)

    combined = pd.concat([X_train, X_test], axis=0)
    for col in combined.select_dtypes(include=["object", "string"]).columns:
        if col not in sensitive_set:
            combined = pd.get_dummies(combined, columns=[col], drop_first=True, dtype=float)

    combined = combined.drop(columns=[col for col in combined.columns if col in sensitive_set], errors="ignore")
    for col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        if combined[col].isna().all():
            combined[col] = 0.0
        else:
            combined[col] = combined[col].fillna(combined[col].median())

    X_train_enc = combined.iloc[: len(X_train)].copy()
    X_test_enc = combined.iloc[len(X_train) :].copy()

    # Safely convert boolean to int, handling NaN values
    y_train = (train_df[target_col] == positive_label).fillna(False).astype(int)
    y_test = (test_df[target_col] == positive_label).fillna(False).astype(int)

    return X_train_enc, y_train, X_test_enc, y_test, X_test_raw


def _evaluate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_test_original: pd.DataFrame,
    sensitive_cols: List[str],
) -> Dict[str, float]:
    # Safely convert predictions to int, handling NaN and float values
    y_pred_raw = pd.Series(model.predict(X_test), index=y_test.index)
    y_pred = y_pred_raw.fillna(0).astype(float).round().astype(int)
    accuracy = float(accuracy_score(y_test, y_pred))

    dpd_values: List[float] = []
    eod_values: List[float] = []
    for sensitive_col in sensitive_cols:
        if sensitive_col not in X_test_original.columns:
            continue
        sf = X_test_original[sensitive_col].fillna("missing").astype(str)
        dpd_values.append(
            float(
                abs(
                    demographic_parity_difference(
                        y_true=y_test,
                        y_pred=y_pred,
                        sensitive_features=sf,
                    )
                )
            )
        )
        eod_values.append(
            float(
                abs(
                    equalized_odds_difference(
                        y_true=y_test,
                        y_pred=y_pred,
                        sensitive_features=sf,
                    )
                )
            )
        )

    avg_gap = float(np.mean(dpd_values + eod_values)) if (dpd_values or eod_values) else 0.0
    fairness_score = float(max(0.0, 100.0 * (1.0 - avg_gap)))

    return {
        "accuracy": round(accuracy, 4),
        "fairness_score": round(fairness_score, 2),
        "demographic_parity_gap": round(float(np.mean(dpd_values)) if dpd_values else 0.0, 4),
        "equalized_odds_gap": round(float(np.mean(eod_values)) if eod_values else 0.0, 4),
    }


def _minority_group(series: pd.Series) -> str:
    counts = series.fillna("missing").astype(str).value_counts()
    return str(counts.idxmin()) if not counts.empty else "missing"


def _fragility_label(max_acc_drop: float, max_fairness_drop: float) -> str:
    if max_acc_drop >= 0.15 or max_fairness_drop >= 15:
        return "HIGH"
    if max_acc_drop >= 0.08 or max_fairness_drop >= 8:
        return "MEDIUM"
    return "LOW"


def run_stress_tests(
    df: pd.DataFrame,
    shared_model: Any,
    sensitive_cols: List[str],
    target_col: str,
) -> Dict[str, Any]:
    """Stress test model robustness under adverse fairness scenarios."""
    if df.empty:
        raise ValueError("Stress tests require a non-empty DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    missing_sensitive = [col for col in sensitive_cols if col not in df.columns]
    if missing_sensitive:
        raise ValueError(f"Sensitive columns not found: {missing_sensitive}")

    positive_label = _resolve_positive_label(df[target_col])
    primary_sensitive = sensitive_cols[0]

    y_binary = (df[target_col] == positive_label).astype(int)
    stratify_target = y_binary if y_binary.nunique() > 1 else None
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    X_train, y_train, X_test, y_test, X_test_original = _encode_split(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        sensitive_cols=sensitive_cols,
        positive_label=positive_label,
    )

    if isinstance(shared_model, dict) and "model" in shared_model:
        baseline_model = shared_model["model"]
    else:
        baseline_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
        baseline_model.fit(X_train, y_train)

    baseline = _evaluate(
        model=baseline_model,
        X_test=X_test,
        y_test=y_test,
        X_test_original=X_test_original,
        sensitive_cols=sensitive_cols,
    )

    scenarios: Dict[str, Dict[str, float]] = {}

    minority_value = _minority_group(train_df[primary_sensitive])

    # Scenario 1: Minority under-sampling.
    train_undersampled = train_df.copy()
    minority_mask = train_undersampled[primary_sensitive].fillna("missing").astype(str) == minority_value
    minority_subset = train_undersampled[minority_mask]
    keep_count = int(np.ceil(len(minority_subset) * 0.5))
    minority_kept = minority_subset.sample(keep_count, random_state=42) if len(minority_subset) > 0 else minority_subset
    train_undersampled = pd.concat([train_undersampled[~minority_mask], minority_kept], axis=0)

    Xu_train, yu_train, Xu_test, yu_test, Xu_test_original = _encode_split(
        train_df=train_undersampled,
        test_df=test_df,
        target_col=target_col,
        sensitive_cols=sensitive_cols,
        positive_label=positive_label,
    )
    model_under = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    model_under.fit(Xu_train, yu_train)
    scenarios["minority_undersampling_50pct"] = _evaluate(
        model=model_under,
        X_test=Xu_test,
        y_test=yu_test,
        X_test_original=Xu_test_original,
        sensitive_cols=sensitive_cols,
    )

    # Scenario 2: Label noise on minority group.
    Xn_train, yn_train, Xn_test, yn_test, Xn_test_original = _encode_split(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        sensitive_cols=sensitive_cols,
        positive_label=positive_label,
    )
    yn_train_noisy = yn_train.copy()
    noise_mask = train_df[primary_sensitive].fillna("missing").astype(str) == minority_value
    noise_indices = train_df[noise_mask].index.tolist()
    noise_count = int(np.ceil(len(noise_indices) * 0.10))
    # Flip labels by direct boolean negation instead of using index assignment
    if noise_count > 0 and len(noise_indices) > 0:
        noisy_idx_list = noise_indices[:noise_count]
        for idx in noisy_idx_list:
            if idx in yn_train_noisy.index:
                yn_train_noisy.loc[idx] = 1 - yn_train_noisy.loc[idx]

    model_noise = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    model_noise.fit(Xn_train, yn_train_noisy)
    scenarios["label_noise_10pct_minority"] = _evaluate(
        model=model_noise,
        X_test=Xn_test,
        y_test=yn_test,
        X_test_original=Xn_test_original,
        sensitive_cols=sensitive_cols,
    )

    # Scenario 3: Distribution shift on minority group numeric values.
    shifted_test_df = test_df.copy()
    shift_mask = shifted_test_df[primary_sensitive].fillna("missing").astype(str) == minority_value
    numeric_cols = [
        col
        for col in shifted_test_df.columns
        if col != target_col and col not in sensitive_cols and pd.api.types.is_numeric_dtype(shifted_test_df[col])
    ]
    for col in numeric_cols:
        # Convert to float to allow fractional values from multiplication
        shifted_test_df[col] = shifted_test_df[col].astype(float)
        shifted_test_df.loc[shift_mask, col] = shifted_test_df.loc[shift_mask, col] * 0.85

    Xs_train, ys_train, Xs_test, ys_test, Xs_test_original = _encode_split(
        train_df=train_df,
        test_df=shifted_test_df,
        target_col=target_col,
        sensitive_cols=sensitive_cols,
        positive_label=positive_label,
    )
    model_shift = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    model_shift.fit(Xs_train, ys_train)
    scenarios["distribution_shift_minority_numeric_down"] = _evaluate(
        model=model_shift,
        X_test=Xs_test,
        y_test=ys_test,
        X_test_original=Xs_test_original,
        sensitive_cols=sensitive_cols,
    )

    for name, result in scenarios.items():
        result["delta_accuracy"] = round(float(result["accuracy"] - baseline["accuracy"]), 4)
        result["delta_fairness_score"] = round(
            float(result["fairness_score"] - baseline["fairness_score"]),
            2,
        )
        scenarios[name] = result

    max_acc_drop = max(max(0.0, baseline["accuracy"] - result["accuracy"]) for result in scenarios.values())
    max_fairness_drop = max(
        max(0.0, baseline["fairness_score"] - result["fairness_score"])
        for result in scenarios.values()
    )

    return {
        "baseline": baseline,
        "scenarios": scenarios,
        "overall_fragility": _fragility_label(max_acc_drop=max_acc_drop, max_fairness_drop=max_fairness_drop),
    }
