from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split

from src.data_loader import load_heart_disease_data
from src.model_training import build_pipeline, evaluate_model, save_model, train_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "heart_model.pkl"
DATA_PATH = BASE_DIR.parent / "data" / "heart.csv"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


def initialize_app():
    df = load_heart_disease_data(csv_path=str(DATA_PATH))
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        _, X_test, _, y_test = train_test_split(
            df.drop(columns=["target"]),
            df["target"],
            test_size=0.25,
            random_state=42,
            stratify=df["target"],
        )
        metrics = evaluate_model(model, X_test, y_test)
    else:
        model, X_test, y_test = train_model(df)
        save_model(model, MODEL_PATH)
        metrics = evaluate_model(model, X_test, y_test)
    return df, model, metrics


DATAFRAME, MODEL, METRICS = initialize_app()
FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def parse_feature_value(value: str, feature_name: str):
    if feature_name in {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}:
        return int(value)
    return float(value)


def predict_risk(form_data):
    values = [parse_feature_value(form_data.get(name, "0"), name) for name in FEATURE_ORDER]
    prediction = MODEL.predict([values])[0]
    probability = float(MODEL.predict_proba([values])[0][1])
    return int(prediction), round(probability, 3)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    prediction_probability = None
    form_values = {name: "" for name in FEATURE_ORDER}
    if request.method == "POST":
        form_values = {name: request.form.get(name, "0") for name in FEATURE_ORDER}
        prediction_result, prediction_probability = predict_risk(form_values)

    dataset_preview = DATAFRAME.head(10).to_dict(orient="records")
    columns = DATAFRAME.columns.tolist()
    return render_template(
        "index.html",
        dataset_preview=dataset_preview,
        columns=columns,
        dataset_count=DATAFRAME.shape[0],
        metrics=METRICS,
        prediction_result=prediction_result,
        prediction_probability=prediction_probability,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
