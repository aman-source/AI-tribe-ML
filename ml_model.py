import json
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# The line `# from dotenv import load_dotenv` is a commented-out import statement in Python. It
# suggests that the `dotenv` library is being used to load environment variables from a `.env` file
# into the Python script. However, since it is commented out with a `#` at the beginning, it is not
# currently being used in the script.

from dotenv import load_dotenv

load_dotenv()


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = os.getenv("CSV_PATH", "data/team_productivity.csv")

# ---------- Load CSV ----------
def load_csv():
    df = pd.read_csv(CSV_PATH)
    for col in ["created_at", "closed_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# ---------- 1. Regression: Predict cycle_time ----------
def train_regression(df):
    df = df.dropna(subset=["cycle_time"])
    if df.empty:
        return None, 0
    df["weekday"] = df["created_at"].dt.weekday
    features = ["task_type", "priority", "team_name", "weekday"]
    X = df[features].copy()
    y = df["cycle_time"]

    encoders = {}
    for c in ["task_type", "priority", "team_name"]:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encoders[c] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = round(model.score(X_test, y_test), 3)
    dump((model, encoders), os.path.join(MODEL_DIR, "closure_time.joblib"))
    return "closure_time", score

# ---------- 2. Classification: Will task close in 24h? ----------
def train_classifier(df):
    df = df.copy()
    df = df[df["status"].isin(["open", "in_progress", "closed"])]
    df["will_close_24h"] = np.where(
        (df["closed_at"].notna()) & ((df["closed_at"] - df["created_at"]).dt.total_seconds() <= 86400),
        1, 0
    )
    if df["will_close_24h"].sum() == 0:
        return None, 0
    df["weekday"] = df["created_at"].dt.weekday
    X = df[["task_type", "priority", "team_name", "weekday"]]
    y = df["will_close_24h"]

    encoders = {}
    for c in ["task_type", "priority", "team_name"]:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encoders[c] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = round(model.score(X_test, y_test), 3)
    dump((model, encoders), os.path.join(MODEL_DIR, "completion_prob.joblib"))
    return "completion_prob", acc

# ---------- 3. Forecast: Task completions trend ----------
def train_forecaster(df):
    if "closed_at" not in df.columns or df["closed_at"].isna().all():
        return None, 0
    df["ds"] = df["closed_at"].dt.date
    daily = df.groupby("ds").size().reset_index(name="y")
    if len(daily) < 5:
        return None, 0
    model = Prophet(daily_seasonality=False, weekly_seasonality=True)
    model.fit(daily)
    with open(os.path.join(MODEL_DIR, "team_forecast.json"), "w") as f:
        json.dump(model_to_json(model), f)
    return "team_forecast", len(daily)

# ---------- Prediction ----------
def predict_cycle_time(task):
    model, encoders = load(os.path.join(MODEL_DIR, "closure_time.joblib"))
    df = pd.DataFrame([task])
    df["weekday"] = pd.to_datetime(df["created_at"]).dt.weekday
    for c in ["task_type", "priority", "team_name"]:
        df[c] = encoders[c].transform(df[c])
    pred = model.predict(df[["task_type","priority","team_name","weekday"]])[0]
    return round(float(pred),2)

def predict_completion_prob(task):
    model, encoders = load(os.path.join(MODEL_DIR, "completion_prob.joblib"))
    df = pd.DataFrame([task])
    df["weekday"] = pd.to_datetime(df["created_at"]).dt.weekday
    for c in ["task_type", "priority", "team_name"]:
        df[c] = encoders[c].transform(df[c])
    prob = model.predict_proba(df[["task_type","priority","team_name","weekday"]])[0][1]
    return round(float(prob),2)

def forecast_next_week():
    with open(os.path.join(MODEL_DIR, "team_forecast.json"), "r") as f:
        model = model_from_json(json.load(f))
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    window = forecast[["ds", "yhat"]].tail(7).copy()
    window["ds"] = window["ds"].dt.strftime("%Y-%m-%d")
    return window.to_dict(orient="records")
