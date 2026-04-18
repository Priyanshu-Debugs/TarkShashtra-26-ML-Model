from pathlib import Path
import json
import os

import joblib
import numpy as np

FEATURE_COLUMNS = ["assignment", "attendance", "lms", "marks"]
DEFAULT_RISK_SCORE_WEIGHTS = {
    "attendance": 0.35,
    "marks": 0.30,
    "assignment": 0.20,
    "lms": 0.15,
}


class ModelStore:
    def __init__(
        self,
        model_path: Path,
        regression_path: Path,
        encoder_path: Path,
        metadata_path: Path,
    ) -> None:
        self.model_path = Path(model_path)
        self.regression_path = Path(regression_path)
        self.encoder_path = Path(encoder_path)
        self.metadata_path = Path(metadata_path)
        self.model = None
        self.regressor = None
        self.encoder = None
        self.feature_stats = {}
        self.feature_importance = {}
        self.risk_score_weights = DEFAULT_RISK_SCORE_WEIGHTS

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. Run training/train.py first."
            )
        self.model = joblib.load(self.model_path)
        if self.regression_path.exists():
            self.regressor = joblib.load(self.regression_path)
        if self.encoder_path.exists():
            self.encoder = joblib.load(self.encoder_path)
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.feature_stats = metadata.get("feature_stats", {})
            self.feature_importance = metadata.get("feature_importance", {})
            self.risk_score_weights = metadata.get(
                "risk_score_weights", DEFAULT_RISK_SCORE_WEIGHTS
            )

    def predict(self, features: list[float]) -> tuple[str, int, dict | None]:
        if self.model is None:
            self.load()
        arr = np.array([features], dtype=float)
        pred_id = int(self.model.predict(arr)[0])
        label = str(pred_id)
        if self.encoder is not None:
            try:
                label = self.encoder.inverse_transform([pred_id])[0]
            except Exception:
                label = str(pred_id)

        probs = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(arr)[0]
            if self.encoder is not None and hasattr(self.encoder, "classes_"):
                probs = {str(cls): float(p) for cls, p in zip(self.encoder.classes_, proba)}
            else:
                probs = {str(i): float(p) for i, p in enumerate(proba)}
        return label, pred_id, probs

    def predict_risk_score(self, features: list[float]) -> float | None:
        if self.regressor is None:
            return None
        arr = np.array([features], dtype=float)
        return float(self.regressor.predict(arr)[0])

    def calculate_risk_score(self, feature_map: dict) -> float | None:
        weights = self.risk_score_weights or DEFAULT_RISK_SCORE_WEIGHTS
        total_weight = 0.0
        score = 0.0
        for key, weight in weights.items():
            value = feature_map.get(key)
            if value is None:
                continue
            score += weight * (float(value) / 100.0)
            total_weight += weight
        if total_weight == 0:
            return None
        return round((1.0 - (score / total_weight)) * 100.0, 2)

    def explain(self, feature_map: dict, max_items: int = 3) -> list[dict]:
        if not self.feature_stats:
            return []

        reasons = []

        def add_reason(feature: str, value: float, threshold: float, message: str):
            reasons.append(
                {
                    "feature": feature,
                    "value": round(float(value), 2),
                    "threshold": round(float(threshold), 2),
                    "message": message,
                    "gap": abs(float(value) - float(threshold)),
                }
            )

        for feature in ["attendance", "marks", "assignment", "lms"]:
            value = feature_map.get(feature)
            stats = self.feature_stats.get(feature)
            if value is None or not stats:
                continue
            if float(value) < float(stats.get("p25", 0)):
                add_reason(
                    feature,
                    value,
                    stats.get("p25", 0),
                    f"Low {feature} (below 25th percentile)",
                )

        risk_value = feature_map.get("risk_score")
        risk_stats = self.feature_stats.get("risk_score")
        if risk_value is not None and risk_stats:
            if float(risk_value) > float(risk_stats.get("p75", 0)):
                add_reason(
                    "risk_score",
                    risk_value,
                    risk_stats.get("p75", 0),
                    "High risk score (above 75th percentile)",
                )

        reasons = sorted(reasons, key=lambda r: r["gap"], reverse=True)[:max_items]
        for reason in reasons:
            reason.pop("gap", None)
        return reasons


def _default_models_dir() -> Path:
    core_root = Path(__file__).resolve().parent.parent
    local_models_dir = core_root / "models"
    if local_models_dir.exists():
        return local_models_dir

    sibling_ml_models_dir = core_root.parent / "ml_api" / "models"
    return sibling_ml_models_dir


def get_model_paths() -> tuple[Path, Path, Path, Path]:
    models_dir = _default_models_dir()
    model_path = Path(os.getenv("MODEL_PATH", models_dir / "model.pkl"))
    regression_path = Path(
        os.getenv("REGRESSION_MODEL_PATH", models_dir / "model_regression.pkl")
    )
    encoder_path = Path(os.getenv("LABEL_ENCODER_PATH", models_dir / "label_encoder.pkl"))
    metadata_path = Path(
        os.getenv("MODEL_METADATA_PATH", models_dir / "model_metadata.json")
    )
    return model_path, regression_path, encoder_path, metadata_path


model_path, regression_path, encoder_path, metadata_path = get_model_paths()
model_store = ModelStore(model_path, regression_path, encoder_path, metadata_path)
