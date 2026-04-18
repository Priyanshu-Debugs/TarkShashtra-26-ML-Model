from typing import List, Optional
import os

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .model import FEATURE_COLUMNS, model_store

app = FastAPI(title="TarkShashtra ML API", version="1.0.0")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[os.getenv("RATE_LIMIT", "60/minute")],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class PredictionRequest(BaseModel):
    student_id: Optional[str] = None
    class_id: Optional[str] = None
    subject: Optional[str] = None
    assignment: float = Field(..., ge=0, le=100)
    attendance: float = Field(..., ge=0, le=100)
    lms: float = Field(..., ge=0, le=100)
    marks: float = Field(..., ge=0, le=100)


class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]


class PredictionResponse(BaseModel):
    student_id: Optional[str]
    class_id: Optional[str]
    subject: Optional[str]
    risk_label: str
    risk_label_id: int
    probabilities: Optional[dict]
    risk_score_predicted: Optional[float]
    risk_score_calculated: Optional[float]
    reasons: List[dict]
    suggestions: List[str]


class BatchPredictionResponse(BaseModel):
    items: List[PredictionResponse]


def to_feature_map(payload: PredictionRequest) -> dict:
    return {
        "assignment": payload.assignment,
        "attendance": payload.attendance,
        "lms": payload.lms,
        "marks": payload.marks,
    }


def to_feature_vector(feature_map: dict) -> list[float]:
    return [feature_map[name] for name in FEATURE_COLUMNS]


def build_suggestions(
    risk_label: str, reasons: List[dict], risk_score_value: Optional[float]
) -> List[str]:
    label = (risk_label or "").strip().lower()
    suggestions: List[str] = []

    if risk_score_value is not None:
        if risk_score_value >= 70:
            suggestions.extend(
                [
                    "Schedule weekly mentor check-ins and academic support.",
                    "Create a short-term improvement plan with clear weekly goals.",
                    "Engage guardians and monitor progress every 2 weeks.",
                ]
            )
        elif risk_score_value >= 40:
            suggestions.extend(
                [
                    "Set bi-weekly progress reviews and study targets.",
                    "Focus on consistent assignment completion.",
                    "Provide optional tutoring or peer support.",
                ]
            )
        else:
            suggestions.extend(
                [
                    "Maintain current study routine and attendance.",
                    "Continue monthly performance monitoring.",
                ]
            )
    elif label:
        if label == "high":
            suggestions.extend(
                [
                    "Schedule weekly mentor check-ins and academic support.",
                    "Create a short-term improvement plan with clear weekly goals.",
                    "Engage guardians and monitor progress every 2 weeks.",
                ]
            )
        elif label == "medium":
            suggestions.extend(
                [
                    "Set bi-weekly progress reviews and study targets.",
                    "Focus on consistent assignment completion.",
                    "Provide optional tutoring or peer support.",
                ]
            )
        elif label == "low":
            suggestions.extend(
                [
                    "Maintain current study routine and attendance.",
                    "Continue monthly performance monitoring.",
                ]
            )

    reason_map = {
        "attendance": "Improve attendance with reminders and follow-up calls.",
        "marks": "Offer subject-specific revision sessions and practice quizzes.",
        "assignment": "Set a submission schedule and track weekly completion.",
        "lms": "Increase LMS engagement with weekly activity goals.",
        "risk_score": "Prioritize immediate intervention and closer monitoring.",
    }

    for reason in reasons:
        feature = str(reason.get("feature", "")).strip().lower()
        suggestion = reason_map.get(feature)
        if suggestion:
            suggestions.append(suggestion)

    deduped: List[str] = []
    seen = set()
    for item in suggestions:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


@app.on_event("startup")
def load_model() -> None:
    model_store.load()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model_store.model is not None}


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(os.getenv("PREDICT_RATE_LIMIT", "30/minute"))
def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    try:
        feature_map = to_feature_map(payload)
        predicted_score = model_store.predict_risk_score(to_feature_vector(feature_map))
        calculated_score = model_store.calculate_risk_score(feature_map)
        risk_score_value = (
            predicted_score if predicted_score is not None else calculated_score
        )
        label, label_id, probs = model_store.predict(to_feature_vector(feature_map))
        explain_map = dict(feature_map)
        explain_map["risk_score"] = risk_score_value
        reasons = model_store.explain(explain_map, max_items=3)
        suggestions = build_suggestions(str(label), reasons, risk_score_value)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return PredictionResponse(
        student_id=payload.student_id,
        class_id=payload.class_id,
        subject=payload.subject,
        risk_label=str(label),
        risk_label_id=int(label_id),
        probabilities=probs,
        risk_score_predicted=predicted_score,
        risk_score_calculated=calculated_score,
        reasons=reasons,
        suggestions=suggestions,
    )


@app.post("/predict_batch", response_model=BatchPredictionResponse)
@limiter.limit(os.getenv("PREDICT_RATE_LIMIT", "30/minute"))
def predict_batch(
    request: Request, payload: BatchPredictionRequest
) -> BatchPredictionResponse:
    items: List[PredictionResponse] = []
    for item in payload.items:
        feature_map = to_feature_map(item)
        predicted_score = model_store.predict_risk_score(to_feature_vector(feature_map))
        calculated_score = model_store.calculate_risk_score(feature_map)
        risk_score_value = (
            predicted_score if predicted_score is not None else calculated_score
        )
        label, label_id, probs = model_store.predict(to_feature_vector(feature_map))
        explain_map = dict(feature_map)
        explain_map["risk_score"] = risk_score_value
        reasons = model_store.explain(explain_map, max_items=3)
        suggestions = build_suggestions(str(label), reasons, risk_score_value)
        items.append(
            PredictionResponse(
                student_id=item.student_id,
                class_id=item.class_id,
                subject=item.subject,
                risk_label=str(label),
                risk_label_id=int(label_id),
                probabilities=probs,
                risk_score_predicted=predicted_score,
                risk_score_calculated=calculated_score,
                reasons=reasons,
                suggestions=suggestions,
            )
        )
    return BatchPredictionResponse(items=items)
