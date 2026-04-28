from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from scipy.stats import chi2_contingency, pearsonr
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric
except Exception:  # pragma: no cover - graceful fallback when optional deps are missing
    BinaryLabelDataset = None
    BinaryLabelDatasetMetric = None


LOGGER = logging.getLogger(__name__)


class BiasEngine:
    """Computes dataset and model fairness diagnostics."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_cols: List[str],
        favorable_label: Any,
    ) -> None:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' was not found in the dataset.")

        missing_sensitive = [col for col in sensitive_cols if col not in df.columns]
        if missing_sensitive:
            raise ValueError(f"Sensitive columns not found: {missing_sensitive}")

        self.df = df.copy()
        self.target_col = target_col
        self.sensitive_cols = sensitive_cols
        self.favorable_label = favorable_label

        self._dataset_metrics: Dict[str, Any] = {}
        self._model_metrics: Dict[str, Any] = {}
        self._proxy_variables: List[Dict[str, Any]] = []
        self._sensitive_code_maps: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _clean_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (np.generic,)):
            value = value.item()
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return None
            return float(value)
        return None

    def _prepare_numeric_dataframe(self) -> pd.DataFrame:
        prepared = self.df.copy()
        prepared[self.target_col] = (prepared[self.target_col] == self.favorable_label).astype(int)

        for col in prepared.columns:
            if col == self.target_col:
                continue

            series = prepared[col]
            if pd.api.types.is_bool_dtype(series):
                prepared[col] = series.astype(int)
            elif pd.api.types.is_numeric_dtype(series):
                prepared[col] = pd.to_numeric(series, errors="coerce")
                if col in self.sensitive_cols:
                    unique_vals = sorted(pd.Series(prepared[col]).dropna().unique().tolist())
                    self._sensitive_code_maps[col] = {str(v): float(v) for v in unique_vals}
            else:
                categorical = pd.Categorical(series.fillna("missing").astype(str))
                prepared[col] = pd.Series(categorical.codes, index=series.index, dtype="float64")
                if col in self.sensitive_cols:
                    self._sensitive_code_maps[col] = {
                        str(category): float(idx)
                        for idx, category in enumerate(categorical.categories)
                    }

        prepared = prepared.replace([np.inf, -np.inf], np.nan)
        for col in prepared.columns:
            if col == self.target_col:
                continue
            if prepared[col].isna().all():
                prepared[col] = 0.0
            else:
                prepared[col] = prepared[col].fillna(prepared[col].median())

        prepared[self.target_col] = prepared[self.target_col].astype(int)
        return prepared

    def _compute_group_rates(self, sensitive_col: str) -> pd.Series:
        working = pd.DataFrame(
            {
                "sensitive": self.df[sensitive_col].fillna("missing").astype(str),
                "target": (self.df[self.target_col] == self.favorable_label).astype(int),
            }
        )
        return working.groupby("sensitive")["target"].mean().sort_values()

    def _group_values_for_aif360(self, sensitive_col: str) -> Tuple[float, float]:
        rates = self._compute_group_rates(sensitive_col)
        if rates.empty:
            return 0.0, 0.0

        unpriv_label = rates.index[0]
        priv_label = rates.index[-1]

        mapping = self._sensitive_code_maps.get(sensitive_col, {})
        unpriv_value = mapping.get(str(unpriv_label))
        priv_value = mapping.get(str(priv_label))

        if unpriv_value is None:
            try:
                unpriv_value = float(unpriv_label)
            except ValueError:
                unpriv_value = 0.0

        if priv_value is None:
            try:
                priv_value = float(priv_label)
            except ValueError:
                priv_value = 1.0

        if priv_value == unpriv_value:
            priv_value = unpriv_value + 1.0

        return float(unpriv_value), float(priv_value)

    def _compute_consistency_fallback(self, prepared_df: pd.DataFrame) -> Optional[float]:
        if len(prepared_df) < 6:
            return None

        # Cap sample size to keep NearestNeighbors tractable on large datasets
        CONSISTENCY_SAMPLE_CAP = 2000
        if len(prepared_df) > CONSISTENCY_SAMPLE_CAP:
            prepared_df = prepared_df.sample(CONSISTENCY_SAMPLE_CAP, random_state=42)

        features = prepared_df.drop(columns=[self.target_col], errors="ignore")
        labels = prepared_df[self.target_col].astype(float).to_numpy()
        if features.empty:
            return None

        n_neighbors = min(6, len(features))
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1)
        neighbors.fit(features)
        neighbor_indices = neighbors.kneighbors(return_distance=False)

        scores: List[float] = []
        for idx, near_idx in enumerate(neighbor_indices):
            comparison = [n for n in near_idx if n != idx]
            if not comparison:
                continue
            peer_mean = labels[comparison].mean()
            scores.append(1.0 - abs(labels[idx] - peer_mean))

        if not scores:
            return None

        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _compute_class_imbalance(self, sensitive_col: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        grouped = self.df.groupby(self.df[sensitive_col].fillna("missing").astype(str), dropna=False)

        for group_name, group_df in grouped:
            target_series = group_df[self.target_col]
            target_distribution = (
                target_series.astype(str).value_counts(normalize=True).sort_values(ascending=False)
            )
            rows.append(
                {
                    "group": str(group_name),
                    "group_size": int(len(group_df)),
                    "positive_rate": self._clean_float(
                        (group_df[self.target_col] == self.favorable_label).mean()
                    ),
                    "class_distribution": {
                        str(k): self._clean_float(v)
                        for k, v in target_distribution.to_dict().items()
                    },
                }
            )

        return rows

    def compute_dataset_metrics(self) -> Dict[str, Any]:
        prepared = self._prepare_numeric_dataframe()

        disparate_impact: Dict[str, Optional[float]] = {}
        statistical_parity: Dict[str, Optional[float]] = {}
        consistency_values: List[float] = []
        class_imbalance: Dict[str, List[Dict[str, Any]]] = {}

        for sensitive_col in self.sensitive_cols:
            class_imbalance[sensitive_col] = self._compute_class_imbalance(sensitive_col)

            rates = self._compute_group_rates(sensitive_col)
            if rates.empty:
                disparate_impact[sensitive_col] = None
                statistical_parity[sensitive_col] = None
                continue

            unpriv_rate = float(rates.iloc[0])
            priv_rate = float(rates.iloc[-1])
            fallback_di = None if priv_rate == 0 else unpriv_rate / priv_rate
            fallback_spd = unpriv_rate - priv_rate

            if BinaryLabelDataset is None or BinaryLabelDatasetMetric is None:
                disparate_impact[sensitive_col] = self._clean_float(fallback_di)
                statistical_parity[sensitive_col] = self._clean_float(fallback_spd)
                continue

            try:
                unpriv_value, priv_value = self._group_values_for_aif360(sensitive_col)
                bld = BinaryLabelDataset(
                    df=prepared,
                    label_names=[self.target_col],
                    protected_attribute_names=[sensitive_col],
                    favorable_label=1.0,
                    unfavorable_label=0.0,
                )

                metric = BinaryLabelDatasetMetric(
                    bld,
                    unprivileged_groups=[{sensitive_col: unpriv_value}],
                    privileged_groups=[{sensitive_col: priv_value}],
                )

                disparate_impact[sensitive_col] = self._clean_float(metric.disparate_impact())
                statistical_parity[sensitive_col] = self._clean_float(
                    metric.statistical_parity_difference()
                )
                consistency = self._clean_float(metric.consistency())
                if consistency is not None:
                    consistency_values.append(consistency)
            except Exception as exc:  # pragma: no cover - defensive path for unsupported datasets
                LOGGER.warning(
                    "AIF360 metric computation failed for %s: %s",
                    sensitive_col,
                    exc,
                )
                disparate_impact[sensitive_col] = self._clean_float(fallback_di)
                statistical_parity[sensitive_col] = self._clean_float(fallback_spd)

        consistency_score = (
            float(np.mean(consistency_values))
            if consistency_values
            else self._compute_consistency_fallback(prepared)
        )

        self._dataset_metrics = {
            "disparate_impact": disparate_impact,
            "statistical_parity_difference": statistical_parity,
            "consistency_score": self._clean_float(consistency_score),
            "class_imbalance_by_sensitive_group": class_imbalance,
        }
        return self._dataset_metrics

    def compute_model_metrics(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_test_original: Optional[pd.DataFrame] = None,
        sensitive_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        y_true = pd.Series(y_test).reset_index(drop=True)
        y_pred = pd.Series(model.predict(X_test)).reset_index(drop=True)

        metrics_by_sensitive: Dict[str, Any] = {}
        
        # Use provided sensitive_cols if given, otherwise use self.sensitive_cols
        cols_to_evaluate = sensitive_cols if sensitive_cols is not None else self.sensitive_cols

        for sensitive_col in cols_to_evaluate:
            if sensitive_col not in X_test.columns and (X_test_original is None or sensitive_col not in X_test_original.columns):
                LOGGER.warning("Sensitive column '%s' not in X_test; skipping.", sensitive_col)
                continue

            # Problem 7: Use original string values for group labels in MetricFrame
            # If X_test_original is provided, get the original (unencoded) values
            if X_test_original is not None and sensitive_col in X_test_original.columns:
                sensitive_values = X_test_original[sensitive_col].reset_index(drop=True)
            elif sensitive_col in X_test.columns:
                sensitive_values = X_test[sensitive_col].reset_index(drop=True)
            else:
                LOGGER.warning("Sensitive column '%s' not found; skipping.", sensitive_col)
                continue
            
            metric_frame = MetricFrame(
                metrics={
                    "accuracy_score": accuracy_score,
                    "selection_rate": selection_rate,
                    "false_positive_rate": false_positive_rate,
                    "false_negative_rate": false_negative_rate,
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_values,
            )

            by_group_df = metric_frame.by_group
            if isinstance(by_group_df, pd.Series):
                by_group_df = by_group_df.to_frame().T

            group_metrics: Dict[str, Dict[str, Optional[float]]] = {}
            for group, row in by_group_df.iterrows():
                group_metrics[str(group)] = {
                    "accuracy_score": self._clean_float(row.get("accuracy_score")),
                    "selection_rate": self._clean_float(row.get("selection_rate")),
                    "false_positive_rate": self._clean_float(row.get("false_positive_rate")),
                    "false_negative_rate": self._clean_float(row.get("false_negative_rate")),
                }

            eq_odds_diff = equalized_odds_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_values,
            )
            demo_parity_diff = demographic_parity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_values,
            )

            metrics_by_sensitive[sensitive_col] = {
                "by_group": group_metrics,
                "equalized_odds_difference": self._clean_float(eq_odds_diff),
                "demographic_parity_difference": self._clean_float(demo_parity_diff),
            }

        self._model_metrics = metrics_by_sensitive
        return metrics_by_sensitive

    def _compute_proxy_correlation(
        self,
        feature_col: str,
        sensitive_col: str,
    ) -> Tuple[Optional[float], Optional[str]]:
        feature_series = self.df[feature_col]
        sensitive_series = self.df[sensitive_col]

        if pd.api.types.is_numeric_dtype(feature_series) and pd.api.types.is_numeric_dtype(
            sensitive_series
        ):
            clean = pd.DataFrame({"f": feature_series, "s": sensitive_series}).dropna()
            if len(clean) < 3 or clean["f"].nunique() < 2 or clean["s"].nunique() < 2:
                return None, None
            corr, _ = pearsonr(clean["f"], clean["s"])
            return abs(float(corr)), "pearson"

        contingency = pd.crosstab(feature_series.fillna("missing"), sensitive_series.fillna("missing"))
        if contingency.empty or min(contingency.shape) < 2:
            return None, None

        chi2 = chi2_contingency(contingency)[0]
        n = contingency.to_numpy().sum()
        if n == 0:
            return None, None

        phi2 = chi2 / n
        rows, cols = contingency.shape
        denominator = max(min(rows - 1, cols - 1), 1)
        cramers_v = float(np.sqrt(phi2 / denominator))
        return cramers_v, "cramers_v"

    def detect_proxy_variables(self) -> List[Dict[str, Any]]:
        proxy_flags: List[Dict[str, Any]] = []
        candidate_features = [
            col
            for col in self.df.columns
            if col not in set(self.sensitive_cols + [self.target_col])
        ]

        for feature_col in candidate_features:
            for sensitive_col in self.sensitive_cols:
                score, metric_name = self._compute_proxy_correlation(feature_col, sensitive_col)
                if score is None or metric_name is None:
                    continue

                if score > 0.6:
                    proxy_flags.append(
                        {
                            "feature": feature_col,
                            "correlated_with": sensitive_col,
                            "correlation_score": round(float(score), 4),
                            "risk_level": "high" if score > 0.8 else "medium",
                            "metric": metric_name,
                        }
                    )

        proxy_flags.sort(key=lambda item: item["correlation_score"], reverse=True)
        self._proxy_variables = proxy_flags
        return proxy_flags

    def generate_bias_summary(self) -> Dict[str, Any]:
        if not self._dataset_metrics:
            self.compute_dataset_metrics()
        if not self._proxy_variables:
            self.detect_proxy_variables()

        summary = {
            "target_column": self.target_col,
            "sensitive_columns": self.sensitive_cols,
            "favorable_label": self.favorable_label,
            "dataset_metrics": self._dataset_metrics,
            "model_metrics": self._model_metrics,
            "proxy_variables": self._proxy_variables,
        }
        return summary
