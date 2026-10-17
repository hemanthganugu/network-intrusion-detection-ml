"""Shared utilities for the network intrusion detection demo."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_network_traffic.csv"
MODEL_PATH = ROOT / "models" / "intrusion_detector.joblib"
TARGET = "label"
NUMERIC_FEATURES = [
    "duration", "dst_port", "packets", "bytes", "syn_count",
    "failed_logins", "connection_rate", "inbound_packets", "outbound_packets"
]
CATEGORICAL_FEATURES = ["protocol"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def build_pipeline():
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES)
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, class_weight="balanced",
            random_state=42, n_jobs=-1
        ))
    ])

def load_dataset(path=DATA_PATH):
    data = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(sorted(missing)))
    data = data[FEATURES + [TARGET]].copy()
    data[TARGET] = data[TARGET].astype(str).str.upper().str.strip()
    return data

def train_and_save(path=DATA_PATH):
    data = load_dataset(path)
    X, y = data[FEATURES], data[TARGET]
    model = build_pipeline()
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model, X, y

def load_or_train():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    model, _, _ = train_and_save()
    return model
