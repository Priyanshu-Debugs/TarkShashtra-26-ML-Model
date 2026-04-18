from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_FEATURE_COLUMNS = ["assignment", "attendance", "lms", "marks"]
RISK_SCORE_WEIGHTS = {
    "attendance": 0.35,
    "marks": 0.30,
    "assignment": 0.20,
    "lms": 0.15,
}

DATASET_PATH = Path(os.getenv("DATASET_PATH", "datasets/TS-PS12.csv"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "services/ml_api/models"))


def select_model():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",
        )
    except Exception as exc:
        print(f"XGBoost not available, using RandomForest: {exc}")
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )


def select_regressor():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception as exc:
        print(f"XGBoost not available for regression, using RandomForest: {exc}")
        return RandomForestRegressor(
            n_estimators=300,
            random_state=42,
        )


def compute_feature_stats(dataframe: pd.DataFrame) -> dict:
    stats = {}
    for col in BASE_FEATURE_COLUMNS + ["risk_score"]:
        series = dataframe[col].astype(float)
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
        }
    return stats


def get_feature_importance(model) -> dict:
    if hasattr(model, "feature_importances_"):
        return {
            col: float(val)
            for col, val in zip(BASE_FEATURE_COLUMNS, model.feature_importances_)
        }
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        if coef.ndim == 2:
            coef = coef.mean(axis=0)
        return {col: float(val) for col, val in zip(BASE_FEATURE_COLUMNS, coef)}
    return {}


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    le = LabelEncoder()
    y_cls = le.fit_transform(df["risk_label"])
    X_cls = df[BASE_FEATURE_COLUMNS]

    y_reg = df["risk_score"]
    X_reg = df[BASE_FEATURE_COLUMNS]

    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
        X_cls,
        y_cls,
        test_size=0.3,
        random_state=42,
        stratify=y_cls,
    )

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg,
        y_reg,
        test_size=0.3,
        random_state=42,
    )

    clf_model = select_model()
    clf_model.fit(X_cls_train, y_cls_train)

    reg_model = select_regressor()
    reg_model.fit(X_reg_train, y_reg_train)

    preds_cls = clf_model.predict(X_cls_test)
    print("Accuracy:", round(accuracy_score(y_cls_test, preds_cls), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(y_cls_test, preds_cls), 4))
    print("F1 macro:", round(f1_score(y_cls_test, preds_cls, average="macro"), 4))
    print("F1 weighted:", round(f1_score(y_cls_test, preds_cls, average="weighted"), 4))

    preds_reg = reg_model.predict(X_reg_test)
    rmse = float(np.sqrt(mean_squared_error(y_reg_test, preds_reg)))
    print("MAE:", round(mean_absolute_error(y_reg_test, preds_reg), 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2_score(y_reg_test, preds_reg), 4))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_model, MODEL_DIR / "model.pkl")
    joblib.dump(reg_model, MODEL_DIR / "model_regression.pkl")
    joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
    metadata = {
        "feature_stats": compute_feature_stats(df),
        "classification_feature_importance": get_feature_importance(clf_model),
        "regression_feature_importance": get_feature_importance(reg_model),
        "risk_score_weights": RISK_SCORE_WEIGHTS,
        "feature_columns": BASE_FEATURE_COLUMNS,
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
