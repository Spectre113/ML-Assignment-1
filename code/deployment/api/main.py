from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

import os
import pickle

model_path = os.path.join(os.path.dirname(__file__), '../models/director_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = FastAPI(title="Director Funding Predictor")

class DirectorFeatures(BaseModel):
    avg_budget: float
    avg_revenue: float
    avg_vote: float
    num_movies: int
    avg_popularity: float
    avg_vote_count: float
    age: int

class_descriptions = {
    0: "Режиссер пытается выбраться из забытых мест, но денег скорее всего не получит.",
    1: "Режиссер может рассчитывать на пачку дешевых сухариков, либо студия - камикадзе.",
    2: "Режиссер может надеяться на пачку дорогих сухариков, есть вероятность, что фильм не закидают помидорами.",
    3: "Режиссер скорее всего получит деньги на свой фильм.",
    4: "У дома режиссера стоят важные дяди и кидают в него мешки с деньгами."
}

@app.post("/predict")
def predict(features: DirectorFeatures):
    X = np.array([[features.avg_budget,
                   features.avg_revenue,
                   features.avg_vote,
                   features.num_movies,
                   features.avg_popularity,
                   features.avg_vote_count]])
    
    pred_class = int(model.predict(X)[0])
    
    if features.age > 60:
        pred_class = max(0, pred_class - 1)
    
    return {
        "predicted_class": pred_class,
        "description": class_descriptions[pred_class]
    }
