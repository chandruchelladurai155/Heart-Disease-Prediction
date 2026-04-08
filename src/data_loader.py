import os
from pathlib import Path
import pandas as pd
import requests

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)

COLUMNS = [
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
    "target",
]


def download_uci_dataset(destination: str) -> str:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    response = requests.get(UCI_URL, timeout=15)
    response.raise_for_status()
    with open(destination, "w", encoding="utf-8") as fp:
        fp.write(response.text)
    return destination


def load_heart_disease_data(csv_path: str = "data/heart.csv", download_if_missing: bool = True) -> pd.DataFrame:
    if not Path(csv_path).exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        print(f"Downloading dataset from UCI to {csv_path}...")
        download_uci_dataset(csv_path)

    df = pd.read_csv(csv_path, header=None, names=COLUMNS, na_values="?")
    df = df.dropna().reset_index(drop=True)
    df["target"] = (df["target"] > 0).astype(int)
    numeric_columns = [c for c in COLUMNS if c != "target"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df
