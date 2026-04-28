from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class SimulationEngine:
    """Compares model quality and fairness before/after mitigation changes."""

    def __init__(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        target_col: str,
        sensitive_cols: List[str],
    ) -> None:
        self.original_df = original_df.copy()
        self.cleaned_df = cleaned_df.copy()
        self.target_col = target_col
        self.sensitive_cols = sensitive_cols

        self._comparison: Dict[str, Any] = {}

    @staticmethod
    def _safe_float(value: Any) -> float:
        if isinstance(value, np.generic):
            value = value.item()
        return float(value)

    def _prepare_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found for simulation.")

        working = df.copy()
        target = working[self.target_col]

        unique_labels = [label for label in target.dropna().unique().tolist()]
        if len(unique_labels) < 2:
            raise ValueError("Simulation requires at least two target classes.")

        if len(unique_labels) > 2:
            # For compatibility with binary fairness metrics, use one-vs-rest with the majority class.
            majority_class = target.value_counts().idxmax()
            y = (target == majority_class).astype(int)
        else:
            favorable = sorted(unique_labels, key=lambda item: str(item))[-1]
            y = (target == favorable).astype(int)

        X = working.drop(columns=[self.target_col], errors="ignore").copy()
        X_original = X.copy()  # Keep original for sensitive column values
        
        # Problem 4: Encode categorical features using get_dummies, excluding sensitive columns
        sensitive_set = set(self.sensitive_cols)
        for col in X.select_dtypes(include=["object", "string"]).columns:
            if col not in sensitive_set:
                X = pd.get_dummies(X, columns=[col], drop_first=True, dtype=float)
        
        # Handle remaining columns - remove sensitive columns from X_encoded (for model)
        # but keep them in X_original (for fairness metrics)
        cols_to_drop = []
        for col in X.columns:
            if col in sensitive_set:
                # Mark sensitive columns for removal from X (used for model training)
                cols_to_drop.append(col)
            elif pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")
                # Fill NaN with median if there are any numeric values
                if X[col].notna().any():
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0.0)
            else:
                # Try to convert to numeric
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        
        # Remove sensitive columns from X (for model training)
        X = X.drop(columns=cols_to_drop, errors="ignore")

        return X, y, X_original

    def _evaluate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        X, y, X_original = self._prepare_dataset(df)

        stratify_target = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_target,
        )
        
        # Get corresponding original features for sensitive columns
        X_original_test = X_original.iloc[X_test.index]

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1, min_samples_leaf=4)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        dpd_values: List[float] = []
        eod_values: List[float] = []
        f1_by_sensitive_group: Dict[str, Dict[str, float]] = {}
        sensitive_details: Dict[str, Any] = {}

        for sensitive_col in self.sensitive_cols:
            # Use original values for sensitive column grouping (Problem 7)
            if sensitive_col not in X_original_test.columns:
                continue

            sf = X_original_test[sensitive_col]
            dpd = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sf)
            eod = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sf)
            dpd_values.append(abs(self._safe_float(dpd)))
            eod_values.append(abs(self._safe_float(eod)))

            group_f1: Dict[str, float] = {}
            for group_value in sorted(pd.Series(sf).dropna().unique().tolist()):
                mask = sf == group_value
                if int(mask.sum()) == 0:
                    continue
                group_f1[str(group_value)] = self._safe_float(
                    f1_score(y_test[mask], y_pred[mask], zero_division=0)
                )

            f1_by_sensitive_group[sensitive_col] = group_f1
            sensitive_details[sensitive_col] = {
                "demographic_parity_difference": self._safe_float(abs(dpd)),
                "equalized_odds_difference": self._safe_float(abs(eod)),
                "f1_by_group": group_f1,
            }

        return {
            "accuracy": self._safe_float(accuracy),
            "demographic_parity_difference": self._safe_float(np.mean(dpd_values))
            if dpd_values
            else 0.0,
            "equalized_odds_difference": self._safe_float(np.mean(eod_values))
            if eod_values
            else 0.0,
            "f1_by_sensitive_group": f1_by_sensitive_group,
            "sensitive_details": sensitive_details,
        }

    @staticmethod
    def _compute_delta(original: Dict[str, Any], cleaned: Dict[str, Any]) -> Dict[str, Any]:
        delta = {
            "accuracy": cleaned.get("accuracy", 0.0) - original.get("accuracy", 0.0),
            "demographic_parity_difference": cleaned.get(
                "demographic_parity_difference", 0.0
            )
            - original.get("demographic_parity_difference", 0.0),
            "equalized_odds_difference": cleaned.get("equalized_odds_difference", 0.0)
            - original.get("equalized_odds_difference", 0.0),
            "f1_by_sensitive_group": {},
        }

        original_f1 = original.get("f1_by_sensitive_group", {})
        cleaned_f1 = cleaned.get("f1_by_sensitive_group", {})

        all_sensitive = sorted(set(original_f1.keys()) | set(cleaned_f1.keys()))
        for sensitive_col in all_sensitive:
            original_groups = original_f1.get(sensitive_col, {})
            cleaned_groups = cleaned_f1.get(sensitive_col, {})
            all_groups = sorted(set(original_groups.keys()) | set(cleaned_groups.keys()))

            delta["f1_by_sensitive_group"][sensitive_col] = {
                group: cleaned_groups.get(group, 0.0) - original_groups.get(group, 0.0)
                for group in all_groups
            }

        return delta

    def run_simulation(self) -> Dict[str, Any]:
        original_metrics = self._evaluate_dataset(self.original_df)
        cleaned_metrics = self._evaluate_dataset(self.cleaned_df)

        self._comparison = {
            "original": original_metrics,
            "cleaned": cleaned_metrics,
            "delta": self._compute_delta(original_metrics, cleaned_metrics),
        }
        return self._comparison

    def generate_comparison_chart(self, include_plotlyjs: Any = False) -> str:
        if not self._comparison:
            self.run_simulation()

        original = self._comparison.get("original", {})
        cleaned = self._comparison.get("cleaned", {})

        metrics = [
            "accuracy",
            "demographic_parity_difference",
            "equalized_odds_difference",
        ]
        original_values = [self._safe_float(original.get(metric, 0.0)) for metric in metrics]
        cleaned_values = [self._safe_float(cleaned.get(metric, 0.0)) for metric in metrics]

        figure = go.Figure()
        figure.add_bar(name="Original Model", x=metrics, y=original_values, marker_color="#E24B4A")
        figure.add_bar(name="Cleaned Model", x=metrics, y=cleaned_values, marker_color="#1D9E75")

        figure.update_layout(
            barmode="group",
            title="Original vs Cleaned Fairness Comparison",
            yaxis_title="Metric Value",
            xaxis_title="Metrics",
            template="plotly_white",
            margin={"l": 50, "r": 30, "t": 60, "b": 60},
            height=460,
        )

        return pio.to_html(figure, full_html=False, include_plotlyjs=include_plotlyjs)
