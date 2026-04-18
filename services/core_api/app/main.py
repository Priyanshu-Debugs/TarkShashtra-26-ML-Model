from typing import List, Optional
from datetime import datetime
from pathlib import Path
import json
import os
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

app = FastAPI(title="TarkShashtra Core API", version="1.0.0")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[os.getenv("RATE_LIMIT", "120/minute")],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ML_API_BASE_URL = os.getenv("ML_API_BASE_URL", "https://ml-api-ay0v.onrender.com")
ML_API_FALLBACK_URL = os.getenv("ML_API_FALLBACK_URL", "http://localhost:8001")
ML_API_URL = os.getenv("ML_API_URL", f"{ML_API_BASE_URL}/predict")
ML_API_BATCH_URL = os.getenv(
    "ML_API_BATCH_URL", f"{ML_API_BASE_URL}/predict_batch"
)
ML_API_TIMEOUT = float(os.getenv("ML_API_TIMEOUT", "10"))
ALERT_SCORE_THRESHOLD = float(os.getenv("ALERT_SCORE_THRESHOLD", "70"))

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
INTERVENTIONS_FILE = DATA_DIR / "interventions.json"
PERFORMANCE_FILE = DATA_DIR / "performance.json"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"


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


class InterventionRequest(BaseModel):
    student_id: str
    action_type: str
    mentor: Optional[str] = None
    notes: Optional[str] = None
    class_id: Optional[str] = None
    subject: Optional[str] = None


class PerformanceMetrics(BaseModel):
    assignment: float = Field(..., ge=0, le=100)
    attendance: float = Field(..., ge=0, le=100)
    lms: float = Field(..., ge=0, le=100)
    marks: float = Field(..., ge=0, le=100)
    risk_score: Optional[float] = Field(None, ge=0)


class PerformanceRecordRequest(BaseModel):
    student_id: str
    before: PerformanceMetrics
    after: PerformanceMetrics
    class_id: Optional[str] = None
    subject: Optional[str] = None
    notes: Optional[str] = None


def _read_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: list) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


async def call_ml_api(payload: dict, url: str) -> dict:
    if hasattr(payload, "model_dump"):
        payload_data = payload.model_dump()
    elif hasattr(payload, "dict"):
        payload_data = payload.dict()
    else:
        payload_data = payload
    async with httpx.AsyncClient(timeout=ML_API_TIMEOUT) as client:
        try:
            resp = await client.post(url, json=payload_data)
        except httpx.RequestError:
            fallback_url = url.replace(ML_API_BASE_URL, ML_API_FALLBACK_URL)
            resp = await client.post(fallback_url, json=payload_data)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


def intervention_rules(risk_label: str) -> List[str]:
    label = (risk_label or "").strip().lower()
    if label == "high":
        return [
            "Immediate advisor outreach",
            "Weekly progress check-ins",
            "Targeted academic support sessions",
        ]
    if label == "medium":
        return [
            "Bi-weekly progress review",
            "Study plan and attendance nudges",
            "Optional tutoring resources",
        ]
    if label == "low":
        return [
            "Maintain current progress",
            "Monthly performance monitoring",
        ]
    return ["Review manually - label not recognized"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
@limiter.limit(os.getenv("PREDICT_RATE_LIMIT", "60/minute"))
async def predict(request: Request, payload: PredictionRequest) -> dict:
    return await call_ml_api(payload, ML_API_URL)


@app.post("/predict_batch")
@limiter.limit(os.getenv("PREDICT_RATE_LIMIT", "60/minute"))
async def predict_batch(request: Request, payload: BatchPredictionRequest) -> dict:
    response = await call_ml_api(payload, ML_API_BATCH_URL)
    items = response.get("items", [])
    _write_json(PREDICTIONS_FILE, items)
    return response


@app.post("/intervention")
@limiter.limit(os.getenv("INTERVENTION_RATE_LIMIT", "30/minute"))
async def intervention(request: Request, payload: PredictionRequest) -> dict:
    result = await call_ml_api(payload, ML_API_URL)
    label = str(result.get("risk_label", ""))
    return {
        "risk_label": label,
        "recommendations": intervention_rules(label),
        "suggestions": result.get("suggestions", []),
        "model": result,
    }


@app.post("/interventions")
@limiter.limit(os.getenv("INTERVENTION_RATE_LIMIT", "30/minute"))
async def log_intervention(request: Request, payload: InterventionRequest) -> dict:
    record = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    record.update(
        {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )
    data = _read_json(INTERVENTIONS_FILE)
    data.append(record)
    _write_json(INTERVENTIONS_FILE, data)
    return {"status": "logged", "record": record}


@app.get("/interventions")
def list_interventions(
    student_id: Optional[str] = None,
    class_id: Optional[str] = None,
    subject: Optional[str] = None,
) -> dict:
    data = _read_json(INTERVENTIONS_FILE)
    filtered = [
        item
        for item in data
        if (student_id is None or item.get("student_id") == student_id)
        and (class_id is None or item.get("class_id") == class_id)
        and (subject is None or item.get("subject") == subject)
    ]
    return {"items": filtered}


@app.post("/performance")
@limiter.limit(os.getenv("INTERVENTION_RATE_LIMIT", "30/minute"))
async def log_performance(request: Request, payload: PerformanceRecordRequest) -> dict:
    record = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    before = record["before"]
    after = record["after"]
    delta = {
        key: round(float(after.get(key, 0)) - float(before.get(key, 0)), 2)
        for key in ["assignment", "attendance", "lms", "marks", "risk_score"]
        if key in before or key in after
    }
    record.update(
        {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "delta": delta,
        }
    )
    data = _read_json(PERFORMANCE_FILE)
    data.append(record)
    _write_json(PERFORMANCE_FILE, data)
    return {"status": "logged", "record": record}


@app.get("/performance")
def list_performance(student_id: Optional[str] = None) -> dict:
    data = _read_json(PERFORMANCE_FILE)
    filtered = [
        item for item in data if student_id is None or item.get("student_id") == student_id
    ]
    return {"items": filtered}


@app.post("/alerts/high-risk")
@limiter.limit(os.getenv("INTERVENTION_RATE_LIMIT", "30/minute"))
async def high_risk_alerts(request: Request, payload: BatchPredictionRequest) -> dict:
    response = await call_ml_api(payload, ML_API_BATCH_URL)
    items = response.get("items", [])
    alerts = [
        item
        for item in items
        if str(item.get("risk_label", "")).lower() == "high"
        or float(
            item.get("risk_score_predicted")
            or item.get("risk_score_calculated")
            or 0
        )
        >= ALERT_SCORE_THRESHOLD
    ]
    return {"count": len(alerts), "items": alerts}


@app.get("/dashboard/at_risk")
def dashboard_at_risk(
    class_id: Optional[str] = None,
    subject: Optional[str] = None,
    severity: Optional[str] = None,
) -> dict:
    data = _read_json(PREDICTIONS_FILE)
    filtered = [
        item
        for item in data
        if (class_id is None or item.get("class_id") == class_id)
        and (subject is None or item.get("subject") == subject)
        and (
            severity is None
            or str(item.get("risk_label", "")).lower() == severity.lower()
        )
    ]
    return {"items": filtered, "count": len(filtered)}
