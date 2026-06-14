import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from model import load
from features import FEATURE_COLS

app = FastAPI(title="Churn Predictor API", version="1.0.0")
model = load()


class UserFeatures(BaseModel):
    avg_rating_given: float
    rating_std: float
    watch_rate_per_week: float
    days_since_last_watch: float
    genre_diversity: float
    preferred_genre_ratio: float
    avg_imdb_rating_watched: float
    avg_runtime_watched: float
    recent_watch_count: float
    account_age_days: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(user: UserFeatures):
    X = np.array([[getattr(user, f) for f in FEATURE_COLS]])
    prob = model.predict_proba(X)[0][1]
    churned = bool(prob >= 0.5)
    return {
        "churned": churned,
        "churn_probability": round(float(prob), 4),
    }