from __future__ import annotations

import asyncio
import json
from html import escape
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import File, Form, FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.routers.pipeline import (
    router as pipeline_router,
    run_unified_pipeline_from_csv_bytes,
)
from core import (
    ExplainabilityEngine,
    LLMReasoner,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Unbiased AI Decision Platform")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "templates" / "static")), name="static")

# New modular backend pipeline endpoint.
app.include_router(pipeline_router)

app.state.latest_report_context = None
app.state.latest_compare_context = None
app.state.latest_reasoner = LLMReasoner()
app.state.latest_simulation_engine = None
app.state.latest_explainability_engine = None
app.state.latest_model = None
app.state.latest_dataset_context = ""
app.state.latest_feature_columns = []


class ExplainRecordPayload(BaseModel):
    record: Dict[str, Any]
    index: Optional[int] = None
    dataset_context: Optional[str] = None


def parse_sensitive_columns(raw_sensitive_cols: str) -> List[str]:
    return [item.strip() for item in raw_sensitive_cols.split(",") if item.strip()]


async def _run_unified_analysis(
    *,
    contents: bytes,
    target_col: str,
    sensitive_list: List[str],
    dataset_context: str,
    metric_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    unified_result = await loop.run_in_executor(
        None,
        partial(
            run_unified_pipeline_from_csv_bytes,
            contents=contents,
            target_col=target_col.strip(),
            sensitive_cols=[s.strip() for s in sensitive_list],
            metric_weights=metric_weights,
            include_internals=True,
        ),
    )

    shared_model = unified_result.pop("_shared_model", {})
    result = _adapt_unified_result_for_report(
        unified_result=unified_result,
        dataset_context=dataset_context,
        target_col=target_col.strip(),
        sensitive_list=[s.strip() for s in sensitive_list],
        shared_model=shared_model,
    )

    app.state.latest_reasoner = result.pop("_reasoner")
    app.state.latest_simulation_engine = None
    app.state.latest_model = shared_model.get("model")
    app.state.latest_feature_columns = list(shared_model.get("feature_cols", []))
    app.state.latest_dataset_context = dataset_context

    if shared_model.get("model") is not None and shared_model.get("X_train") is not None and shared_model.get("X_test") is not None:
        app.state.latest_explainability_engine = ExplainabilityEngine(
            model=shared_model["model"],
            X_train=shared_model["X_train"],
            X_test=shared_model["X_test"],
        )
    else:
        app.state.latest_explainability_engine = None

    app.state.latest_report_context = result
    app.state.latest_compare_context = {
        "comparison": result["simulation_comparison"],
        "full_report_narrative": result["full_report_narrative"],
        "dataset_context": dataset_context,
    }

    return result


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "error_message": None,
            "form_data": {
                "target_col": "",
                "sensitive_cols": "",
                "dataset_context": "",
                "favorable_label": "",
            },
        },
    )


@app.get("/report", response_class=HTMLResponse)
async def report(request: Request) -> HTMLResponse:
    if app.state.latest_report_context is None:
        return RedirectResponse(url="/", status_code=302)

    context = {"request": request, **app.state.latest_report_context}
    return templates.TemplateResponse(request=request, name="report.html", context=context)


def _map_proxy_variables_for_report(proxy: Dict[str, Any]) -> List[Dict[str, Any]]:
    mapped: List[Dict[str, Any]] = []
    for item in proxy.get("proxy_candidates", []):
        methods = item.get("methods", [])
        scores = item.get("scores", {})
        metric = methods[0] if methods else "unknown"
        score = float(scores.get(metric, item.get("combined_score", 0.0)))
        mapped.append(
            {
                "feature": item.get("feature", "unknown"),
                "correlated_with": item.get("sensitive_attribute", "unknown"),
                "correlation_score": round(score, 4),
                "risk_level": "high" if score >= 0.8 else "medium" if score >= 0.6 else "low",
                "metric": metric,
            }
        )
    return mapped


def _derive_dataset_metrics_from_unified(unified_result: Dict[str, Any]) -> Dict[str, Any]:
    data_audit = unified_result.get("data_audit", {})
    model_bias = unified_result.get("model_bias", {})
    group_stats = data_audit.get("group_stats", {})

    disparate_impact: Dict[str, Optional[float]] = {}
    statistical_parity_difference: Dict[str, Optional[float]] = {}

    for sensitive_col, details in group_stats.items():
        groups = details.get("groups", [])
        rates = [float(group.get("positive_rate", 0.0)) for group in groups]
        if not rates:
            disparate_impact[sensitive_col] = None
            statistical_parity_difference[sensitive_col] = None
            continue

        priv_rate = max(rates)
        unpriv_rate = min(rates)
        disparate_impact[sensitive_col] = None if priv_rate == 0 else float(unpriv_rate / priv_rate)
        statistical_parity_difference[sensitive_col] = float(unpriv_rate - priv_rate)

    fairness_gaps = model_bias.get("fairness_gaps", {})
    avg_gap = (
        float(fairness_gaps.get("max_demographic_parity_difference", 0.0))
        + float(fairness_gaps.get("max_equalized_odds_difference", 0.0))
    ) / 2.0
    consistency_score = max(0.0, min(1.0, 1.0 - avg_gap))

    return {
        "disparate_impact": disparate_impact,
        "statistical_parity_difference": statistical_parity_difference,
        "consistency_score": consistency_score,
    }


def _map_model_metrics_for_report(model_bias: Dict[str, Any]) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {}
    for sensitive_col, details in model_bias.get("metrics_by_group", {}).items():
        groups = details.get("groups", {})
        by_group: Dict[str, Dict[str, float]] = {}

        for group_name, metrics in groups.items():
            tpr = float(metrics.get("tpr", 0.0))
            by_group[str(group_name)] = {
                "accuracy_score": float(metrics.get("accuracy", 0.0)),
                "selection_rate": float(metrics.get("selection_rate", 0.0)),
                "false_positive_rate": float(metrics.get("fpr", 0.0)),
                "false_negative_rate": float(max(0.0, min(1.0, 1.0 - tpr))),
            }

        mapped[sensitive_col] = {
            "by_group": by_group,
            "equalized_odds_difference": float(details.get("equalized_odds_difference", 0.0)),
            "demographic_parity_difference": float(details.get("demographic_parity_difference", 0.0)),
        }

    return mapped


def _map_counterfactual_for_report(
    counterfactual: Dict[str, Any],
    sensitive_list: List[str],
) -> Dict[str, Any]:
    flip_rate = float(counterfactual.get("flip_rate", 0.0))
    instability_rate = max(0.0, min(100.0, flip_rate * 100.0))
    return {
        "counterfactual_fairness_score": round(100.0 - instability_rate, 2),
        "instability_rate": round(instability_rate, 2),
        "most_sensitive_attribute": sensitive_list[0] if sensitive_list else None,
        "attribute_instability": {
            sensitive_list[0] if sensitive_list else "sensitive": round(instability_rate, 2)
        },
        "example_flip_cases": [],
        "total_tests": int(counterfactual.get("total_records", 0)),
        "prediction_flips": int(counterfactual.get("changed_decisions", 0)),
    }


def _build_shap_summary_from_explanations(explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
    feature_scores: Dict[str, float] = {}
    for explanation in explanations:
        for item in explanation.get("top_contributing_features", []):
            feature = str(item.get("feature", "unknown"))
            value = abs(float(item.get("shap_value", 0.0)))
            feature_scores[feature] = feature_scores.get(feature, 0.0) + value

    sorted_items = sorted(feature_scores.items(), key=lambda pair: pair[1], reverse=True)
    top_10 = [{"feature": feature, "importance": float(score)} for feature, score in sorted_items[:10]]
    return {
        "global_feature_importance": {feature: float(score) for feature, score in sorted_items},
        "top_10_features": top_10,
    }


def _build_shap_chart_html_from_summary(shap_summary: Dict[str, Any]) -> str:
    top_features = shap_summary.get("top_10_features", [])
    if not top_features:
        return "<p>No feature-importance data was generated for this run.</p>"

    rows = []
    max_score = max(float(item.get("importance", 0.0)) for item in top_features) or 1.0
    for item in top_features:
        feature = escape(str(item.get("feature", "unknown")))
        score = float(item.get("importance", 0.0))
        width = int((score / max_score) * 100)
        rows.append(
            "<tr>"
            f"<td>{feature}</td>"
            f"<td>{score:.4f}</td>"
            "<td>"
            f"<div style='background:#edf2f7;border-radius:8px;height:10px;'><div style='width:{width}%;height:10px;background:#3b82f6;border-radius:8px;'></div></div>"
            "</td>"
            "</tr>"
        )

    return (
        "<div class='card'><h3 style='margin-bottom:10px;'>Feature Importance (From Flagged Decisions)</h3>"
        "<table><thead><tr><th>Feature</th><th>Importance</th><th>Relative Impact</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _build_simulation_comparison_from_stress(stress: Dict[str, Any]) -> Dict[str, Any]:
    baseline = stress.get("baseline", {})
    scenarios = stress.get("scenarios", {})

    if scenarios:
        best_name = max(
            scenarios.keys(),
            key=lambda key: (
                float(scenarios[key].get("fairness_score", 0.0)),
                float(scenarios[key].get("accuracy", 0.0)),
            ),
        )
        best = scenarios[best_name]
    else:
        best = baseline

    original = {
        "accuracy": float(baseline.get("accuracy", 0.0)),
        "demographic_parity_difference": float(baseline.get("demographic_parity_gap", 0.0)),
        "equalized_odds_difference": float(baseline.get("equalized_odds_gap", 0.0)),
    }
    cleaned = {
        "accuracy": float(best.get("accuracy", 0.0)),
        "demographic_parity_difference": float(best.get("demographic_parity_gap", 0.0)),
        "equalized_odds_difference": float(best.get("equalized_odds_gap", 0.0)),
    }
    delta = {
        "accuracy": cleaned["accuracy"] - original["accuracy"],
        "demographic_parity_difference": cleaned["demographic_parity_difference"]
        - original["demographic_parity_difference"],
        "equalized_odds_difference": cleaned["equalized_odds_difference"]
        - original["equalized_odds_difference"],
    }

    return {
        "original": original,
        "cleaned": cleaned,
        "delta": delta,
    }


def _adapt_unified_result_for_report(
    unified_result: Dict[str, Any],
    dataset_context: str,
    target_col: str,
    sensitive_list: List[str],
    shared_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dataset_metrics = _derive_dataset_metrics_from_unified(unified_result)
    proxy_variables = _map_proxy_variables_for_report(unified_result.get("proxy", {}))
    model_metrics = _map_model_metrics_for_report(unified_result.get("model_bias", {}))
    counterfactual_results = _map_counterfactual_for_report(
        unified_result.get("counterfactual", {}),
        sensitive_list=sensitive_list,
    )

    recommendations = []
    for item in unified_result.get("recommendations", []):
        recommendations.append(
            {
                "level": item.get("priority", "medium"),
                "strategy": item.get("title", "Recommendation"),
                "reason": item.get("reason", ""),
                "expected_impact": item.get("action", ""),
                "code_hint": item.get("code_hint", ""),
            }
        )

    shap_summary = _build_shap_summary_from_explanations(unified_result.get("explanations", []))
    shap_chart_html = _build_shap_chart_html_from_summary(shap_summary)

    simulation_comparison = _build_simulation_comparison_from_stress(unified_result.get("stress", {}))

    reasoner = LLMReasoner()
    bias_summary = {
        "target_column": target_col,
        "sensitive_columns": sensitive_list,
        "dataset_metrics": dataset_metrics,
        "model_metrics": model_metrics,
        "proxy_variables": proxy_variables,
    }
    llm_interpretation = reasoner.interpret_bias_report(
        bias_summary=bias_summary,
        dataset_context=dataset_context,
    )

    full_report_narrative = reasoner.generate_full_report_narrative(
        {
            "bias_summary": bias_summary,
            "dataset_metrics": dataset_metrics,
            "model_metrics": model_metrics,
            "proxy_variables": proxy_variables,
            "counterfactual_results": counterfactual_results,
            "recommendations": recommendations,
            "simulation_comparison": simulation_comparison,
            "llm_interpretation": llm_interpretation,
            "dataset_context": dataset_context,
            "stress": unified_result.get("stress", {}),
            "explain_summary": unified_result.get("explain_summary", ""),
        }
    )

    # Generate sample predictions for interactive explanations
    sample_predictions = []
    if shared_model and shared_model.get("model") and shared_model.get("X_test") is not None and shared_model.get("y_test") is not None:
        try:
            X_test = shared_model["X_test"]
            y_test = shared_model["y_test"]
            model = shared_model["model"]
            feature_cols = shared_model.get("feature_cols", [])
            
            # Get predictions for test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # Sample up to 10 records for display
            sample_indices = list(range(min(10, len(X_test))))
            
            for idx in sample_indices:
                record_dict = {}
                for col_name in feature_cols:
                    if col_name in X_test.columns:
                        record_dict[col_name] = float(X_test.iloc[idx][col_name])
                
                sample_predictions.append({
                    "index": int(X_test.index[idx]) if hasattr(X_test.index, '__getitem__') else idx,
                    "prediction": int(y_pred[idx]),
                    "confidence": float(y_pred_proba[idx]),
                    "actual": int(y_test.iloc[idx]),
                    "record_json": json.dumps(record_dict),
                    "record_dict": record_dict,
                })
        except Exception as e:
            LOGGER.warning(f"Failed to generate sample predictions: {e}")
            sample_predictions = []

    return {
        "dataset_context": dataset_context,
        "bias_summary": bias_summary,
        "dataset_metrics": dataset_metrics,
        "model_metrics": model_metrics,
        "proxy_variables": proxy_variables,
        "shap_summary": shap_summary,
        "shap_chart_html": shap_chart_html,
        "counterfactual_results": counterfactual_results,
        "counterfactual_explanation": {
            "summary": unified_result.get("counterfactual", {}).get("flip_direction_breakdown", {})
        },
        "recommendations": recommendations,
        "resampling_summary": {
            "method": "stress-scenario-proxy",
            "risk_level": unified_result.get("data_audit", {}).get("risk_level", "unknown"),
        },
        "simulation_comparison": simulation_comparison,
        "llm_interpretation": llm_interpretation,
        "full_report_narrative": full_report_narrative,
        "target_col": target_col,
        "sensitive_cols": sensitive_list,
        "favorable_label": "1",
        "sample_predictions": sample_predictions,
        "_reasoner": reasoner,
    }


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    target_col: str = Form(...),
    sensitive_cols: str = Form(...),
    dataset_context: str = Form(""),
    favorable_label: str = Form(""),
) -> HTMLResponse:
    form_data = {
        "target_col": target_col,
        "sensitive_cols": sensitive_cols,
        "dataset_context": dataset_context,
        "favorable_label": favorable_label,
    }

    try:
        if not file.filename:
            raise ValueError("Please upload a CSV file.")

        contents = await file.read()
        if not contents:
            raise ValueError("Uploaded file is empty.")

        safe_name = Path(file.filename).name
        upload_path = UPLOAD_DIR / safe_name
        upload_path.write_bytes(contents)

        sensitive_list = parse_sensitive_columns(sensitive_cols)
        if not sensitive_list:
            raise ValueError("Please provide at least one sensitive column.")

        result = await _run_unified_analysis(
            contents=contents,
            target_col=target_col,
            sensitive_list=sensitive_list,
            dataset_context=dataset_context,
            metric_weights=None,
        )

        context = {"request": request, **result}
        return templates.TemplateResponse(request=request, name="report.html", context=context)

    except Exception as exc:
        LOGGER.exception("Analysis pipeline failed")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "error_message": f"Analysis failed: {exc}",
                "form_data": form_data,
            },
            status_code=400,
        )


@app.post("/pipeline/run-all", response_class=HTMLResponse)
async def run_all_pipeline_view(
    request: Request,
    file: UploadFile = File(...),
    target_col: str = Form(...),
    sensitive_cols: str = Form(...),
    dataset_context: str = Form(""),
    metric_weights: str = Form("{}"),
) -> HTMLResponse:
    form_data = {
        "target_col": target_col,
        "sensitive_cols": sensitive_cols,
        "dataset_context": dataset_context,
        "metric_weights": metric_weights,
    }

    try:
        if not file.filename:
            raise ValueError("Please upload a CSV file.")

        contents = await file.read()
        if not contents:
            raise ValueError("Uploaded file is empty.")

        safe_name = Path(file.filename).name
        upload_path = UPLOAD_DIR / safe_name
        upload_path.write_bytes(contents)

        sensitive_list = parse_sensitive_columns(sensitive_cols)
        if not sensitive_list:
            raise ValueError("Please provide at least one sensitive column.")

        try:
            parsed_metric_weights = json.loads(metric_weights) if metric_weights else {}
            if not isinstance(parsed_metric_weights, dict):
                parsed_metric_weights = {}
        except json.JSONDecodeError:
            parsed_metric_weights = {}

        result = await _run_unified_analysis(
            contents=contents,
            target_col=target_col,
            sensitive_list=sensitive_list,
            dataset_context=dataset_context,
            metric_weights=parsed_metric_weights,
        )

        context = {"request": request, **result}
        return templates.TemplateResponse(request=request, name="report.html", context=context)

    except Exception as exc:
        LOGGER.exception("Pipeline view failed")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "error_message": f"Pipeline failed: {exc}",
                "form_data": form_data,
            },
            status_code=400,
        )


@app.post("/explain-record")
async def explain_record(payload: ExplainRecordPayload) -> JSONResponse:
    try:
        explainability_engine = app.state.latest_explainability_engine
        reasoner = app.state.latest_reasoner

        if explainability_engine is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No active analysis context. Run /analyze first."},
            )

        record_df = pd.DataFrame([payload.record])
        record_df = record_df.reindex(columns=app.state.latest_feature_columns, fill_value=0)
        for col in record_df.columns:
            record_df[col] = pd.to_numeric(record_df[col], errors="coerce").fillna(0)

        explanation = explainability_engine.explain_single_prediction(record_df.iloc[0])

        llm_text = reasoner.explain_single_record(
            record=payload.record,
            shap_values=explanation.get("feature_contributions", {}),
            prediction=str(explanation.get("prediction_probability")),
            dataset_context=payload.dataset_context or app.state.latest_dataset_context,
        )

        return JSONResponse(
            {
                "index": payload.index,
                "base_value": explanation.get("base_value"),
                "prediction_probability": explanation.get("prediction_probability"),
                "feature_contributions": explanation.get("feature_contributions"),
                "plain_english_explanation": llm_text,
            }
        )
    except Exception as exc:
        LOGGER.exception("Single-record explanation failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/compare", response_class=HTMLResponse)
async def compare(request: Request) -> HTMLResponse:
    if app.state.latest_compare_context is None:
        return templates.TemplateResponse(
            request=request,
            name="compare.html",
            context={
                "request": request,
                "comparison": None,
                "comparison_chart_html": "",
                "full_report_narrative": "Run an analysis to generate comparison output.",
            },
        )

    simulation_engine = app.state.latest_simulation_engine
    comparison_chart_html = ""
    if simulation_engine is not None:
        comparison_chart_html = simulation_engine.generate_comparison_chart(include_plotlyjs="cdn")

    context = {
        "request": request,
        "comparison": app.state.latest_compare_context.get("comparison"),
        "comparison_chart_html": comparison_chart_html,
        "full_report_narrative": app.state.latest_compare_context.get("full_report_narrative", ""),
        "dataset_context": app.state.latest_compare_context.get("dataset_context", ""),
    }
    return templates.TemplateResponse(request=request, name="compare.html", context=context)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
