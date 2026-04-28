"""Core modules for the unified backend pipeline."""

from .auto_fix import generate_fix_recommendations
from .counterfactual import run_counterfactual_test
from .data_audit import run_data_audit
from .explainability import explain_flagged_decisions, generate_narrative_summary
from .feature_intelligence import detect_proxy_features
from .model_bias import run_model_bias_analysis
from .stress_test import run_stress_tests

__all__ = [
    "run_data_audit",
    "detect_proxy_features",
    "run_model_bias_analysis",
    "explain_flagged_decisions",
    "generate_narrative_summary",
    "run_counterfactual_test",
    "run_stress_tests",
    "generate_fix_recommendations",
]
