from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from imblearn.over_sampling import RandomOverSampler


class MitigationEngine:
    """Generates and applies mitigation strategies based on fairness diagnostics."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_cols: List[str],
        bias_summary: Dict[str, Any],
    ) -> None:
        self.df = df.copy()
        self.target_col = target_col
        self.sensitive_cols = sensitive_cols
        self.bias_summary = bias_summary

    def _max_equalized_odds_difference(self) -> float:
        model_metrics = self.bias_summary.get("model_metrics", {})
        values: List[float] = []

        for _, details in model_metrics.items():
            value = details.get("equalized_odds_difference")
            if value is not None:
                values.append(abs(float(value)))

        return max(values) if values else 0.0

    def generate_recommendations(self) -> List[Dict[str, str]]:
        recommendations: List[Dict[str, str]] = []

        dataset_metrics = self.bias_summary.get("dataset_metrics", {})
        disparate_impact = dataset_metrics.get("disparate_impact", {})
        proxy_variables = self.bias_summary.get("proxy_variables", [])

        low_disparate_impact = [
            f"{attr}: {value:.3f}"
            for attr, value in disparate_impact.items()
            if value is not None and float(value) < 0.8
        ]

        if low_disparate_impact:
            recommendations.append(
                {
                    "level": "data",
                    "strategy": "Resample underrepresented sensitive groups",
                    "reason": (
                        "Disparate impact is below the 0.8 fairness threshold for "
                        + ", ".join(low_disparate_impact)
                    ),
                    "expected_impact": "Improves parity of favorable outcomes across protected groups.",
                    "code_hint": "from imblearn.over_sampling import RandomOverSampler",
                }
            )

        if proxy_variables:
            proxy_names = sorted({item.get("feature", "") for item in proxy_variables if item.get("feature")})
            recommendations.append(
                {
                    "level": "feature",
                    "strategy": "Remove or transform proxy variables",
                    "reason": (
                        "Potential proxy features were detected: " + ", ".join(proxy_names)
                    ),
                    "expected_impact": "Reduces hidden indirect discrimination from correlated features.",
                    "code_hint": "X = X.drop(columns=proxy_feature_list)",
                }
            )

        max_eq_odds = self._max_equalized_odds_difference()
        if max_eq_odds > 0.1:
            recommendations.append(
                {
                    "level": "model",
                    "strategy": "Apply fairness constraints during training",
                    "reason": (
                        f"Equalized odds difference is {max_eq_odds:.3f}, above the 0.1 alert threshold."
                    ),
                    "expected_impact": "Lowers error-rate disparities between sensitive groups.",
                    "code_hint": (
                        "Use fairlearn.reductions.ExponentiatedGradient with "
                        "EqualizedOdds() constraint."
                    ),
                }
            )

        recommendations.append(
            {
                "level": "threshold",
                "strategy": "Tune decision thresholds per subgroup",
                "reason": "Probability calibration can reduce parity and error-rate gaps.",
                "expected_impact": "Improves fairness while preserving acceptable model utility.",
                "code_hint": "Optimize group-aware thresholds on validation predictions.",
            }
        )

        return recommendations

    def apply_resampling(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not self.sensitive_cols:
            return self.df.copy(), {
                "method": "none",
                "message": "No sensitive columns were provided; skipping resampling.",
                "rows_before": int(len(self.df)),
                "rows_after": int(len(self.df)),
            }

        group_cols = self.df[self.sensitive_cols].fillna("missing")
        for col in group_cols.columns:
            group_cols[col] = group_cols[col].astype(str)
        
        # Use explicit per-row string conversion to avoid pandas join issues
        group_key = group_cols.apply(lambda row: "|".join(row.values.astype(str)), axis=1)

        sampler = RandomOverSampler(random_state=42)
        resampled_df, resampled_groups = sampler.fit_resample(self.df, group_key)

        if not isinstance(resampled_df, pd.DataFrame):
            resampled_df = pd.DataFrame(resampled_df, columns=self.df.columns)

        before_counts = group_key.value_counts().to_dict()
        after_counts = pd.Series(resampled_groups).value_counts().to_dict()

        summary = {
            "method": "RandomOverSampler",
            "rows_before": int(len(self.df)),
            "rows_after": int(len(resampled_df)),
            "group_counts_before": {str(k): int(v) for k, v in before_counts.items()},
            "group_counts_after": {str(k): int(v) for k, v in after_counts.items()},
        }

        return resampled_df.reset_index(drop=True), summary

    def remove_proxy_features(self, proxy_list: Sequence[Any]) -> pd.DataFrame:
        proxy_feature_names = set()
        for item in proxy_list:
            if isinstance(item, str):
                proxy_feature_names.add(item)
            elif isinstance(item, dict) and item.get("feature"):
                proxy_feature_names.add(str(item["feature"]))

        drop_cols = [
            col for col in proxy_feature_names if col in self.df.columns and col != self.target_col
        ]
        return self.df.drop(columns=drop_cols, errors="ignore").copy()
