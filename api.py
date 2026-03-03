from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel


# Load the trained model
with open('pr_merge_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()


# Define input data structure
class PRFeatures(BaseModel):
    additions: int
    deletions: int
    changed_files: int
    commits: int
    comments: int
    total_lines: int
    lines_per_commit: float
    description_length: int
    time_open_hours: float

@app.get("/")
def home():
    return {"message": "PR Merge Predictor API", "status": "running"}

@app.post("/predict")
def predict(features: PRFeatures):
    # Convert input to DataFrame
    data = pd.DataFrame([features.dict()])
    
    # Make prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    return {
        "prediction": "merged" if prediction == 1 else "not_merged",
        "merge_probability": round(float(probability), 3)
    }
