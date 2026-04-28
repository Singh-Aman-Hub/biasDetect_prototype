from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

try:
    import shap
except Exception:  # pragma: no cover - graceful fallback when SHAP is unavailable
    shap = None


class ExplainabilityEngine:
    """Computes SHAP-based feature importance and single prediction explanations."""

    def __init__(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        self.model = model
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()

        self._explainer: Any = None
        self._shap_matrix: Optional[np.ndarray] = None
        self._global_importance: Dict[str, float] = {}
        self._mean_signed_shap: Dict[str, float] = {}

    def _is_tree_based_model(self) -> bool:
        class_name = self.model.__class__.__name__.lower()
        tree_keywords = ["forest", "tree", "boost", "xgb", "gbm"]
        return any(keyword in class_name for keyword in tree_keywords)

    @staticmethod
    def _extract_class_shap_values(shap_values: Any) -> np.ndarray:
        if isinstance(shap_values, list):
            if not shap_values:
                return np.empty((0, 0))
            if len(shap_values) > 1:
                return np.asarray(shap_values[1])
            return np.asarray(shap_values[0])

        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            if shap_array.shape[-1] > 1:
                return shap_array[:, :, 1]
            return shap_array[:, :, 0]
        return shap_array

    def _ensure_explainer(self) -> None:
        if self._explainer is not None:
            return

        if shap is None:
            raise RuntimeError("SHAP is not installed. Please install the 'shap' package.")

        if self._is_tree_based_model():
            self._explainer = shap.TreeExplainer(self.model)
        else:
            self._explainer = shap.LinearExplainer(self.model, self.X_train)

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, np.generic):
            return float(value.item())
        return float(value)

    def compute_shap_values(self) -> Dict[str, Any]:
        self._ensure_explainer()
        shap_values = self._explainer.shap_values(self.X_test)
        shap_matrix = self._extract_class_shap_values(shap_values)

        if shap_matrix.size == 0:
            raise RuntimeError("Unable to compute SHAP values for the given model and data.")

        self._shap_matrix = shap_matrix
        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        mean_signed = np.mean(shap_matrix, axis=0)

        feature_names = list(self.X_test.columns)
        self._global_importance = {
            feature: self._to_float(mean_abs[idx]) for idx, feature in enumerate(feature_names)
        }
        self._mean_signed_shap = {
            feature: self._to_float(mean_signed[idx]) for idx, feature in enumerate(feature_names)
        }

        sorted_features = sorted(
            self._global_importance.items(), key=lambda item: item[1], reverse=True
        )
        top_10 = [
            {"feature": feature, "importance": self._to_float(score)}
            for feature, score in sorted_features[:10]
        ]

        return {
            "global_feature_importance": self._global_importance,
            "top_10_features": top_10,
        }

    def _extract_base_value(self) -> Optional[float]:
        if self._explainer is None or not hasattr(self._explainer, "expected_value"):
            return None

        expected = self._explainer.expected_value
        if isinstance(expected, (list, tuple, np.ndarray)):
            if len(expected) > 1:
                return self._to_float(expected[1])
            if len(expected) == 1:
                return self._to_float(expected[0])
            return None
        return self._to_float(expected)

    def explain_single_prediction(self, row: Any) -> Dict[str, Any]:
        self._ensure_explainer()

        if isinstance(row, pd.Series):
            row_df = pd.DataFrame([row.to_dict()])
        elif isinstance(row, dict):
            row_df = pd.DataFrame([row])
        elif isinstance(row, pd.DataFrame):
            row_df = row.copy().head(1)
        else:
            raise ValueError("Row must be a pandas Series, dict, or pandas DataFrame.")

        row_df = row_df.reindex(columns=self.X_test.columns, fill_value=0)
        shap_values = self._explainer.shap_values(row_df)
        row_shap = self._extract_class_shap_values(shap_values)

        if row_shap.ndim == 1:
            row_shap = row_shap.reshape(1, -1)

        contributions = {
            feature: self._to_float(row_shap[0, idx])
            for idx, feature in enumerate(self.X_test.columns)
        }

        prediction_probability: Optional[float]
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(row_df)
            if probabilities.shape[1] > 1:
                prediction_probability = self._to_float(probabilities[0, 1])
            else:
                prediction_probability = self._to_float(probabilities[0, 0])
        else:
            prediction_probability = self._to_float(self.model.predict(row_df)[0])

        return {
            "base_value": self._extract_base_value(),
            "prediction_probability": prediction_probability,
            "feature_contributions": contributions,
        }

    def generate_plotly_importance_chart(
        self,
        include_plotlyjs: Any = False,
        top_n: int = 10,
    ) -> str:
        if not self._global_importance:
            self.compute_shap_values()

        sorted_items = sorted(
            self._global_importance.items(), key=lambda item: item[1], reverse=True
        )[:top_n]
        features = [item[0] for item in sorted_items][::-1]
        importances = [item[1] for item in sorted_items][::-1]

        colors: List[str] = []
        for feature in features:
            signed_value = self._mean_signed_shap.get(feature, 0.0)
            colors.append("#1D9E75" if signed_value >= 0 else "#E24B4A")

        figure = go.Figure(
            data=[
                go.Bar(
                    x=importances,
                    y=features,
                    orientation="h",
                    marker_color=colors,
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                )
            ]
        )
        figure.update_layout(
            title="Top SHAP Feature Importance",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            template="plotly_white",
            margin={"l": 150, "r": 30, "t": 60, "b": 50},
            height=420,
        )

        return pio.to_html(
            figure,
            full_html=False,
            include_plotlyjs=include_plotlyjs,
        )
