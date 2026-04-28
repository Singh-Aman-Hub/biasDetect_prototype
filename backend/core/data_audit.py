from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


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


def run_data_audit(df: pd.DataFrame, sensitive_cols: List[str], target_col: str) -> Dict[str, Any]:
    """Compute dataset-level quality and representation diagnostics."""
    if df.empty:
        raise ValueError("Data audit requires a non-empty DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    missing_sensitive = [col for col in sensitive_cols if col not in df.columns]
    if missing_sensitive:
        raise ValueError(f"Sensitive columns not found: {missing_sensitive}")

    total_rows = len(df)
    positive_label = _resolve_positive_label(df[target_col])
    # Safely convert boolean to int, handling NaN values
    binary_target = (df[target_col] == positive_label).fillna(False).astype(int)

    missing_data: Dict[str, Dict[str, float]] = {}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_data[col] = {
            "missing_count": missing_count,
            "missing_rate": float(missing_count / total_rows) if total_rows else 0.0,
        }

    overall_class_distribution = (
        df[target_col].astype(str).value_counts(normalize=True).sort_values(ascending=False)
    )

    group_stats: Dict[str, Any] = {}
    under_represented_groups: List[Dict[str, Any]] = []
    max_approval_gap = 0.0

    for sensitive_col in sensitive_cols:
        series = df[sensitive_col].fillna("missing").astype(str)
        counts = series.value_counts(dropna=False)

        groups: List[Dict[str, Any]] = []
        approval_rates: List[float] = []
        for group_name, count in counts.items():
            mask = series == group_name
            positive_rate = float(binary_target[mask].mean()) if int(mask.sum()) > 0 else 0.0
            group_share = float(count / total_rows) if total_rows else 0.0
            approval_rates.append(positive_rate)

            group_record = {
                "group": str(group_name),
                "count": int(count),
                "population_share": group_share,
                "positive_rate": positive_rate,
            }
            groups.append(group_record)

            if group_share < 0.20:
                under_represented_groups.append(
                    {
                        "sensitive_attribute": sensitive_col,
                        "group": str(group_name),
                        "population_share": group_share,
                    }
                )

        group_stats[sensitive_col] = {
            "groups": groups,
            "approval_rate_gap": float(max(approval_rates) - min(approval_rates))
            if approval_rates
            else 0.0,
        }
        if approval_rates:
            max_approval_gap = max(max_approval_gap, max(approval_rates) - min(approval_rates))

    risk_score = 0
    if any(item["missing_rate"] > 0.10 for item in missing_data.values()):
        risk_score += 1
    if under_represented_groups:
        risk_score += 1
    if max_approval_gap > 0.15:
        risk_score += 1

    if risk_score >= 3:
        risk_level = "HIGH"
    elif risk_score == 2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "group_stats": group_stats,
        "missing_data": missing_data,
        "risk_level": risk_level,
        "target_distribution": {
            str(k): float(v) for k, v in overall_class_distribution.to_dict().items()
        },
        "under_represented_groups": under_represented_groups,
    }
