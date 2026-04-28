from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - allows local execution without Gemini dependency installed
    genai = None


class LLMReasoner:
    """Handles Gemini calls and robustly parses reasoning output."""

    def __init__(self) -> None:
        load_dotenv()
        self.model_name = "gemini-1.5-pro"
        self.api_key = (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )
        self.client = None

        if self.api_key and self.api_key != "your_key_here" and genai is not None:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)

    def _call_gemini(
        self,
        prompt: str,
        max_tokens: int = 1200,
        temperature: float = 0.1,
    ) -> str:
        if self.client is None:
            raise RuntimeError("Gemini client is not configured.")

        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json"
                if "Return strict JSON object only" in prompt
                else "text/plain",
            },
        )
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        chunks: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not parts:
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)

        if not chunks:
            raise RuntimeError("Gemini response did not contain text output.")

        return "\n".join(chunks).strip()

    @staticmethod
    def _extract_json(raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^```", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _fallback_interpretation(
        self,
        bias_summary: Dict[str, Any],
        dataset_context: str,
    ) -> Dict[str, Any]:
        dataset_metrics = bias_summary.get("dataset_metrics", {})
        model_metrics = bias_summary.get("model_metrics", {})
        proxies = bias_summary.get("proxy_variables", [])

        score = 100.0
        findings: List[str] = []
        recommendations: List[str] = []

        for attr, di in dataset_metrics.get("disparate_impact", {}).items():
            if di is not None and float(di) < 0.8:
                score -= 18
                findings.append(
                    f"Disparate impact for {attr} is {float(di):.3f}, below the 0.8 fairness threshold."
                )
                recommendations.append(
                    f"Prioritize resampling or reweighting for the {attr} groups to improve outcome parity."
                )

        for attr, metrics in model_metrics.items():
            eod = metrics.get("equalized_odds_difference")
            dpd = metrics.get("demographic_parity_difference")
            if eod is not None and abs(float(eod)) > 0.1:
                score -= 14
                findings.append(
                    f"Equalized odds difference for {attr} is {float(eod):.3f}, indicating uneven error rates."
                )
                recommendations.append(
                    f"Apply constrained optimization (ExponentiatedGradient + EqualizedOdds) for {attr}."
                )
            if dpd is not None and abs(float(dpd)) > 0.1:
                score -= 10
                findings.append(
                    f"Demographic parity difference for {attr} is {float(dpd):.3f}, signaling selection disparity."
                )

        if proxies:
            score -= min(24, len(proxies) * 6)
            proxy_names = sorted({item.get("feature", "") for item in proxies if item.get("feature")})
            findings.append(
                "Potential proxy variables detected: " + ", ".join(proxy_names[:5])
            )
            recommendations.append(
                "Audit and drop or transform proxy features that correlate strongly with protected attributes."
            )

        score = self._clamp(score, 0.0, 100.0)
        score_int = int(round(score))

        if score_int >= 80:
            risk_level = "low"
        elif score_int >= 60:
            risk_level = "medium"
        elif score_int >= 35:
            risk_level = "high"
        else:
            risk_level = "critical"

        if not findings:
            findings = [
                "No severe fairness violations were detected in the computed metrics.",
                "Continue monitoring fairness metrics over time to catch drift.",
                "Validate fairness under real production traffic and retraining cycles.",
            ]

        if not recommendations:
            recommendations = [
                "Establish fairness guardrails in CI and model validation checks.",
                "Add subgroup performance monitoring and periodic bias audits.",
                "Tune thresholds using a validation set to balance utility and parity.",
            ]

        proxy_explanation = (
            "Proxy variables can encode protected attribute information indirectly. "
            "Even when sensitive columns are removed, correlated proxies may let the model "
            "reconstruct protected-group membership and reproduce discriminatory behavior."
        )

        executive_summary = (
            f"For {dataset_context or 'this dataset'}, the fairness audit produced an overall score of "
            f"{score_int}/100 with a {risk_level} risk classification. "
            "The strongest concerns are parity gaps and/or proxy-driven signals. "
            "Addressing data balance, feature controls, and threshold governance should reduce risk."
        )

        return {
            "overall_risk_level": risk_level,
            "overall_fairness_score": score_int,
            "key_findings": findings[:5],
            "proxy_variable_explanation": proxy_explanation,
            "priority_recommendations": recommendations[:3],
            "executive_summary": executive_summary,
        }

    def interpret_bias_report(self, bias_summary: Dict[str, Any], dataset_context: str) -> Dict[str, Any]:
        prompt = (
            "You are a fairness auditor. Analyze the provided bias report JSON and return ONLY valid JSON.\n"
            "Return exactly these keys and no others:\n"
            "- overall_risk_level: one of low, medium, high, critical\n"
            "- overall_fairness_score: integer 0-100\n"
            "- key_findings: list of 3-5 plain-English strings\n"
            "- proxy_variable_explanation: plain-English paragraph\n"
            "- priority_recommendations: ordered list of top 3 actionable strings\n"
            "- executive_summary: 2-3 sentence plain-English paragraph\n\n"
            "Dataset context:\n"
            f"{dataset_context}\n\n"
            "Bias summary JSON:\n"
            f"{json.dumps(bias_summary, indent=2, default=str)}\n\n"
            "Output constraints:\n"
            "- Return strict JSON object only\n"
            "- Do not use markdown code fences\n"
            "- Keep statements specific to the provided metrics\n"
        )

        try:
            raw_response = self._call_gemini(prompt=prompt, max_tokens=1300, temperature=0.1)
            parsed = self._extract_json(raw_response)

            required_keys = {
                "overall_risk_level",
                "overall_fairness_score",
                "key_findings",
                "proxy_variable_explanation",
                "priority_recommendations",
                "executive_summary",
            }
            missing = required_keys - set(parsed.keys())
            if missing:
                raise ValueError(f"Gemini response is missing keys: {sorted(missing)}")

            parsed["overall_fairness_score"] = int(parsed["overall_fairness_score"])
            return parsed
        except Exception:
            return self._fallback_interpretation(bias_summary=bias_summary, dataset_context=dataset_context)

    def explain_single_record(
        self,
        record: Dict[str, Any],
        shap_values: Dict[str, float],
        prediction: str,
        dataset_context: str,
    ) -> str:
        prompt = (
            "Explain this model decision in 2-3 concise sentences for a non-technical stakeholder.\n"
            "Use SHAP values as evidence and mention only the strongest contributing features.\n"
            "Avoid jargon and avoid bullet points.\n\n"
            f"Dataset context: {dataset_context}\n"
            f"Prediction: {prediction}\n"
            f"Record: {json.dumps(record, default=str)}\n"
            f"SHAP contributions: {json.dumps(shap_values, default=str)}\n"
        )

        try:
            response = self._call_gemini(prompt=prompt, max_tokens=220, temperature=0.2)
            return response.strip()
        except Exception:
            sorted_features = sorted(
                shap_values.items(), key=lambda item: abs(float(item[1])), reverse=True
            )[:3]
            feature_bits = [f"{name} ({value:+.3f})" for name, value in sorted_features]
            joined = ", ".join(feature_bits) if feature_bits else "no dominant feature signals"
            return (
                f"The model predicted {prediction} for this record based on the strongest feature effects: "
                f"{joined}. Positive SHAP values pushed the prediction upward, while negative values reduced it. "
                "This explanation should be reviewed alongside group-level fairness metrics before operational use."
            )

    def generate_full_report_narrative(self, full_audit_results: Dict[str, Any]) -> str:
        prompt = (
            "Create a markdown report with these exact sections:\n"
            "1. Executive Summary\n"
            "2. Dataset Bias Findings\n"
            "3. Model Fairness Analysis\n"
            "4. Proxy Variable Risks\n"
            "5. Recommended Actions\n"
            "6. Monitoring Guidance\n\n"
            "Use concise paragraphs and practical recommendations.\n"
            "Do not include any section outside these six.\n\n"
            f"Full audit payload:\n{json.dumps(full_audit_results, indent=2, default=str)}"
        )

        try:
            return self._call_gemini(prompt=prompt, max_tokens=1400, temperature=0.2)
        except Exception:
            llm_report = full_audit_results.get("llm_interpretation", {})
            recommendations = full_audit_results.get("recommendations", [])

            recommendation_lines = "\n".join(
                f"- {item.get('strategy', 'Mitigation strategy')}: {item.get('reason', '')}"
                for item in recommendations[:5]
            )

            if not recommendation_lines:
                recommendation_lines = "- Maintain periodic fairness audits and enforce parity guardrails."

            return (
                "## Executive Summary\n"
                f"{llm_report.get('executive_summary', 'Fairness analysis completed with fallback narrative generation.')}\n\n"
                "## Dataset Bias Findings\n"
                "Dataset-level parity and representation metrics were computed across sensitive groups, "
                "including disparate impact and statistical parity difference.\n\n"
                "## Model Fairness Analysis\n"
                "Model-level fairness diagnostics included group-wise accuracy, selection rate, false-positive "
                "rate, false-negative rate, demographic parity difference, and equalized odds difference.\n\n"
                "## Proxy Variable Risks\n"
                f"{llm_report.get('proxy_variable_explanation', 'Proxy-risk analysis identified correlation-driven concerns.')}\n\n"
                "## Recommended Actions\n"
                f"{recommendation_lines}\n\n"
                "## Monitoring Guidance\n"
                "Track fairness metrics in production by subgroup, monitor drift, and rerun mitigation + simulation "
                "after each retraining cycle."
            )
