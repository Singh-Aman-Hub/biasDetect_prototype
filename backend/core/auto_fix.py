from __future__ import annotations

from typing import Any, Dict, List


def _max_fpr_gap(model_bias: Dict[str, Any]) -> float:
    gaps = model_bias.get("fairness_gaps", {})
    try:
        return float(gaps.get("max_fpr_gap", 0.0))
    except Exception:
        return 0.0


def _has_under_representation(data_audit: Dict[str, Any]) -> bool:
    if data_audit.get("under_represented_groups"):
        return True

    for details in data_audit.get("group_stats", {}).values():
        for group in details.get("groups", []):
            if float(group.get("population_share", 0.0)) < 0.20:
                return True
    return False


def generate_fix_recommendations(
    data_audit: Dict[str, Any],
    proxy: Dict[str, Any],
    model_bias: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Rule-based remediation recommendations for UI/action workflows."""
    recommendations: List[Dict[str, Any]] = []

    fpr_gap = _max_fpr_gap(model_bias)
    fairness_score = float(model_bias.get("fairness_score", 0.0))
    proxy_candidates = proxy.get("proxy_candidates", [])

    if fpr_gap >= 0.10:
        recommendations.append(
            {
                "id": "threshold_tuning",
                "title": "Threshold Tuning",
                "priority": "high",
                "reason": (
                    f"Observed FPR gap is {fpr_gap:.3f}, indicating uneven false-positive burden "
                    "across groups."
                ),
                "action": "Tune decision thresholds per sensitive subgroup on a validation set.",
                "code_hint": "Use group-specific probability thresholds after calibration.",
            }
        )

    if proxy_candidates:
        proxy_names = sorted({item.get("feature", "") for item in proxy_candidates if item.get("feature")})
        recommendations.append(
            {
                "id": "proxy_feature_mitigation",
                "title": "Proxy Feature Mitigation",
                "priority": "high",
                "reason": (
                    "Detected likely proxy features: " + ", ".join(proxy_names[:8])
                    + ("..." if len(proxy_names) > 8 else "")
                ),
                "action": "Drop, bucket, or monotonic-transform proxy features before training.",
                "code_hint": "X = X.drop(columns=proxy_candidates)",
            }
        )

    if _has_under_representation(data_audit):
        recommendations.append(
            {
                "id": "smote_resampling",
                "title": "SMOTE / Group-Aware Resampling",
                "priority": "medium",
                "reason": "At least one sensitive subgroup is under-represented (<20% population share).",
                "action": "Apply SMOTE or stratified oversampling for minority sensitive groups.",
                "code_hint": "from imblearn.over_sampling import SMOTE",
            }
        )

    if fairness_score < 70:
        recommendations.append(
            {
                "id": "fairness_constrained_training",
                "title": "Fairness-Constrained Training",
                "priority": "medium",
                "reason": f"Fairness score is {fairness_score:.1f}, below target operational threshold.",
                "action": "Train with equalized-odds or demographic-parity constraints.",
                "code_hint": (
                    "Use fairlearn.reductions.ExponentiatedGradient with EqualizedOdds or "
                    "DemographicParity constraints."
                ),
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "id": "monitor_only",
                "title": "Continue Monitoring",
                "priority": "low",
                "reason": "No severe audit trigger was detected in current thresholds.",
                "action": "Keep periodic fairness monitoring in CI and production drift checks.",
                "code_hint": "Schedule recurring bias audits and alerting.",
            }
        )

    return recommendations
