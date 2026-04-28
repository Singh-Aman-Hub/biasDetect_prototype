from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from backend.core.auto_fix import generate_fix_recommendations
from backend.core.counterfactual import run_counterfactual_test
from backend.core.data_audit import run_data_audit
from backend.core.explainability import explain_flagged_decisions, generate_narrative_summary
from backend.core.feature_intelligence import detect_proxy_features
from backend.core.model_bias import run_model_bias_analysis
from backend.core.stress_test import run_stress_tests


router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _parse_sensitive_cols(raw_sensitive_cols: str) -> List[str]:
    return [item.strip() for item in raw_sensitive_cols.split(",") if item.strip()]


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


def _encode_for_model(
    df: pd.DataFrame,
    target_col: str,
    sensitive_cols: List[str],
    positive_label: Any,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = df.drop(columns=[target_col], errors="ignore").copy()
    X_original = X.copy()

    encoded = X.copy()
    sensitive_set = set(sensitive_cols)

    for col in encoded.select_dtypes(include=["object", "string"]).columns:
        if col not in sensitive_set:
            encoded = pd.get_dummies(encoded, columns=[col], drop_first=True, dtype=float)

    encoded = encoded.drop(columns=[col for col in encoded.columns if col in sensitive_set], errors="ignore")

    for col in encoded.columns:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
        if encoded[col].isna().all():
            encoded[col] = 0.0
        else:
            encoded[col] = encoded[col].fillna(encoded[col].median())

    # Safely convert boolean comparison to int, handling NaN and float values
    y = (df[target_col] == positive_label).fillna(False).astype(int)
    return encoded, y, X_original


def _build_shared_model(df: pd.DataFrame, target_col: str, sensitive_cols: List[str]) -> Dict[str, Any]:
    positive_label = _resolve_positive_label(df[target_col])
    X, y, X_original = _encode_for_model(
        df=df,
        target_col=target_col,
        sensitive_cols=sensitive_cols,
        positive_label=positive_label,
    )

    stratify_target = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )
    X_test_original = X_original.loc[X_test.index]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    return {
        "model": model,
        "feature_cols": list(X.columns),
        "sensitive_cols": list(sensitive_cols),
        "positive_label": positive_label,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_original": X_test_original,
    }


def _clean_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return cleaned


def run_unified_pipeline(
    df: pd.DataFrame,
    target_col: str,
    sensitive_cols: List[str],
    metric_weights: Optional[Dict[str, float]] = None,
    include_internals: bool = False,
) -> Dict[str, Any]:
    """Execute all seven components in required sequence on an in-memory DataFrame."""
    if df.empty:
        raise ValueError("CSV contains no rows.")

    working_df = _clean_input_dataframe(df)
    target_name = target_col.strip()
    sensitive_list = [item.strip() for item in sensitive_cols if item and item.strip()]

    if not sensitive_list:
        raise ValueError("Provide at least one sensitive column.")
    if target_name not in working_df.columns:
        raise ValueError(f"Target column '{target_name}' not found.")

    missing_sensitive = [col for col in sensitive_list if col not in working_df.columns]
    if missing_sensitive:
        raise ValueError(f"Sensitive columns not found: {missing_sensitive}")

    parsed_metric_weights: Dict[str, float] = metric_weights or {}
    shared_model = _build_shared_model(df=working_df, target_col=target_name, sensitive_cols=sensitive_list)

    # 1) Data audit
    data_audit = run_data_audit(df=working_df, sensitive_cols=sensitive_list, target_col=target_name)

    # 2) Feature intelligence / proxy detection
    proxy = detect_proxy_features(df=working_df, sensitive_cols=sensitive_list)

    # 3) Model bias analysis
    model_bias = run_model_bias_analysis(
        df=working_df,
        sensitive_cols=sensitive_list,
        target_col=target_name,
        shared_model=shared_model,
        metric_weights=parsed_metric_weights,
    )

    # 4) Explainability and narrative summary
    explanations = explain_flagged_decisions(
        df=working_df,
        shared_model=shared_model,
        sensitive_cols=sensitive_list,
        target_col=target_name,
    )
    explain_summary = generate_narrative_summary(explanations)

    # 5) Counterfactual sensitivity test (primary sensitive column)
    counterfactual = run_counterfactual_test(
        df=working_df,
        shared_model=shared_model,
        primary_sensitive_col=sensitive_list[0],
        target_col=target_name,
    )

    # 6) Stress tests
    stress = run_stress_tests(
        df=working_df,
        shared_model=shared_model,
        sensitive_cols=sensitive_list,
        target_col=target_name,
    )

    # 7) Auto-fix recommendations
    recommendations = generate_fix_recommendations(
        data_audit=data_audit,
        proxy=proxy,
        model_bias=model_bias,
    )

    result = {
        "data_audit": data_audit,
        "proxy": proxy,
        "model_bias": model_bias,
        "explanations": explanations,
        "explain_summary": explain_summary,
        "counterfactual": counterfactual,
        "stress": stress,
        "recommendations": recommendations,
    }
    if include_internals:
        result["_shared_model"] = shared_model
    return result


def run_unified_pipeline_from_csv_bytes(
    contents: bytes,
    target_col: str,
    sensitive_cols: List[str],
    metric_weights: Optional[Dict[str, float]] = None,
    include_internals: bool = False,
) -> Dict[str, Any]:
    """Execute the unified pipeline from raw CSV bytes."""
    if not contents:
        raise ValueError("Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    return run_unified_pipeline(
        df=df,
        target_col=target_col,
        sensitive_cols=sensitive_cols,
        metric_weights=metric_weights,
        include_internals=include_internals,
    )


@router.post("/run-all-api")
async def run_all_pipeline_api(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    sensitive_cols: str = Form(...),
    metric_weights: str = Form("{}"),
) -> Dict[str, Any]:
    """Unified backend pipeline that executes all seven components in sequence."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    contents = await file.read()
    sensitive_list = _parse_sensitive_cols(sensitive_cols)

    try:
        parsed_metric_weights = json.loads(metric_weights) if metric_weights else {}
        if not isinstance(parsed_metric_weights, dict):
            parsed_metric_weights = {}
    except json.JSONDecodeError:
        parsed_metric_weights = {}

    try:
        return run_unified_pipeline_from_csv_bytes(
            contents=contents,
            target_col=target_col.strip(),
            sensitive_cols=sensitive_list,
            metric_weights=parsed_metric_weights,
            include_internals=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
