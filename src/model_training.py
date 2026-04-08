import os
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_heart_disease_data


MODEL_OUTPUT = Path(__file__).resolve().parent.parent / "models" / "heart_model.pkl"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)),
        ]
    )


def train_model(df: pd.DataFrame) -> Any:
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def save_model(model: Any, output_path: Path = MODEL_OUTPUT) -> None:
    os.makedirs(output_path.parent, exist_ok=True)
    joblib.dump(model, output_path)


def run_training() -> dict:
    df = load_heart_disease_data()
    model, X_test, y_test = train_model(df)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)
    print("Training complete. Model saved to models/heart_model.pkl")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    metrics = run_training()
    print("Classification report:\n", metrics["classification_report"])
    print("Confusion matrix:\n", metrics["confusion_matrix"])
