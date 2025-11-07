from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime
from ml_model import (
    load_csv, train_regression, train_classifier, train_forecaster,
    predict_cycle_time, predict_completion_prob, forecast_next_week
)

app = FastAPI(title="Team Productivity CSV ML Backend")

# ---- Schemas ----
class TaskInput(BaseModel):
    task_type: str = Field(..., description="Type of task, e.g. 'feature', 'bug', 'improvement'")
    priority: str = Field(..., description="Priority level, e.g. 'low', 'medium', 'high'")
    team_name: str = Field(..., description="Name of the team")
    created_at: datetime = Field(..., description="Timestamp of creation")

class ForecastItem(BaseModel):
    ds: str
    yhat: float

class ForecastResponse(BaseModel):
    forecast: list[ForecastItem]

# ---- Endpoints ----
@app.post("/train/all")
def train_all():
    df = load_csv()
    results = {}
    for func in [train_regression, train_classifier, train_forecaster]:
        name, score = func(df)
        results[name] = score
    return {"trained_models": results}

@app.post("/predict/closure_time")
def predict_ct(payload: TaskInput):
    pred = predict_cycle_time(payload.dict())
    return {"predicted_cycle_time": pred}

@app.post("/predict/completion_prob")
def predict_cp(payload: TaskInput):
    prob = predict_completion_prob(payload.dict())
    return {"probability_close_24h": prob}

@app.get("/forecast/next_week", response_model=ForecastResponse)
def forecast_week():
    preds = forecast_next_week()
    return {"forecast": preds}
