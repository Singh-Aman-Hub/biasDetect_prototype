from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class CounterfactualEngine:
    """Measures prediction instability when sensitive attributes are perturbed."""

    def __init__(
        self,
        model: Any,
        df: pd.DataFrame,
        sensitive_cols: List[str],
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.df = df.copy()
        self.sensitive_cols = sensitive_cols
        # If feature_cols not provided, use all non-sensitive columns
        if feature_cols is None:
            self.feature_cols = [col for col in df.columns if col not in sensitive_cols]
        else:
            self.feature_cols = feature_cols
        self._results: Dict[str, Any] = {}

    def _flip_value(self, col: str, value: Any) -> Any:
        unique_values = [v for v in self.df[col].dropna().unique().tolist()]
        if not unique_values:
            return value

        if pd.api.types.is_numeric_dtype(self.df[col]):
            unique_values = sorted(unique_values)

        if len(unique_values) == 1:
            return unique_values[0]

        for candidate in unique_values:
            if candidate != value:
                return candidate

        return value

    def run_counterfactual_test(self, n_samples: int = 100) -> Dict[str, Any]:
        if self.df.empty:
            raise ValueError("Counterfactual test requires a non-empty dataset.")

        missing_sensitive = [col for col in self.sensitive_cols if col not in self.df.columns]
        if missing_sensitive:
            raise ValueError(f"Sensitive columns not present in counterfactual data: {missing_sensitive}")

        sampled = self.df.sample(min(n_samples, len(self.df)), random_state=42).copy()

        total_checks = 0
        changed_predictions = 0
        changed_by_attribute = {col: 0 for col in self.sensitive_cols}
        total_by_attribute = {col: 0 for col in self.sensitive_cols}
        examples: List[Dict[str, Any]] = []

        for idx, row in sampled.iterrows():
            # Only use feature columns for prediction (exclude sensitive columns)
            row_features = row[self.feature_cols] if self.feature_cols else row
            row_df = pd.DataFrame([row_features])
            original_prediction = self.model.predict(row_df)[0]

            for sensitive_col in self.sensitive_cols:
                total_checks += 1
                total_by_attribute[sensitive_col] += 1

                modified_row = row.copy()
                original_value = modified_row[sensitive_col]
                modified_value = self._flip_value(sensitive_col, original_value)
                modified_row[sensitive_col] = modified_value

                # Only use feature columns for prediction
                modified_features = modified_row[self.feature_cols] if self.feature_cols else modified_row
                modified_df = pd.DataFrame([modified_features])
                modified_prediction = self.model.predict(modified_df)[0]
                prediction_changed = bool(modified_prediction != original_prediction)

                if prediction_changed:
                    changed_predictions += 1
                    changed_by_attribute[sensitive_col] += 1
                    if len(examples) < 5:
                        examples.append(
                            {
                                "row_index": int(idx),
                                "attribute": sensitive_col,
                                "original_value": original_value,
                                "modified_value": modified_value,
                                "original_prediction": float(original_prediction),
                                "modified_prediction": float(modified_prediction),
                            }
                        )

        if total_checks == 0:
            fairness_score = 100.0
            instability_rate = 0.0
        else:
            instability_rate = (changed_predictions / total_checks) * 100.0
            fairness_score = 100.0 - instability_rate

        attribute_instability = {
            col: (changed_by_attribute[col] / total_by_attribute[col]) * 100.0
            if total_by_attribute[col] > 0
            else 0.0
            for col in self.sensitive_cols
        }

        most_sensitive_attribute: Optional[str] = None
        if attribute_instability:
            most_sensitive_attribute = max(
                attribute_instability.items(), key=lambda item: item[1]
            )[0]

        self._results = {
            "counterfactual_fairness_score": round(float(fairness_score), 2),
            "instability_rate": round(float(instability_rate), 2),
            "most_sensitive_attribute": most_sensitive_attribute,
            "attribute_instability": {
                k: round(float(v), 2) for k, v in attribute_instability.items()
            },
            "example_flip_cases": examples,
            "total_tests": int(total_checks),
            "prediction_flips": int(changed_predictions),
        }
        return self._results

    def explain_instability(self) -> Dict[str, Any]:
        if not self._results:
            self.run_counterfactual_test()

        attribute_instability = self._results.get("attribute_instability", {})
        ordered = sorted(attribute_instability.items(), key=lambda item: item[1], reverse=True)
        explanation = [
            {
                "attribute": attr,
                "instability_percent": value,
                "interpretation": (
                    "High sensitivity impact"
                    if value >= 20
                    else "Moderate sensitivity impact"
                    if value >= 10
                    else "Low sensitivity impact"
                ),
            }
            for attr, value in ordered
        ]

        return {
            "counterfactual_fairness_score": self._results.get("counterfactual_fairness_score", 0.0),
            "instability_rate": self._results.get("instability_rate", 0.0),
            "most_sensitive_attribute": self._results.get("most_sensitive_attribute"),
            "attribute_breakdown": explanation,
            "example_flip_cases": self._results.get("example_flip_cases", []),
        }
