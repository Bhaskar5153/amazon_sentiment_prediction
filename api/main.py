import os, sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import Predictor
from typing import Dict, List, Any



MODEL_DIR = Path(__file__).parent.parent / "models"

# print(MODEL_DIR)
try:
    predictor = Predictor(model_dir=MODEL_DIR)
except FileNotFoundError as e:
    raise RuntimeError(f"Error occured during the model prediction: {e}")


app = FastAPI(title="Amazon sentiment reviews system API")



class ReviewRequest(BaseModel):
    review_text: str
    review_score: float = 3.0

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "review_text": "Great product, I absolutely loved it",
                "review_score": 5.0
            }]
        }
    }




class SentimentResponse(BaseModel):
    review_text: str
    review_score: float
    sentiment: Dict[str, str]




@app.get('/')
def root():
    return {
        "status": "ok", "message": "Amazon review system API is running..."
    }


@app.post('/predict', response_model=SentimentResponse)
def predict_sentiment(request: ReviewRequest):
    if not request.review_text.strip():
        raise HTTPException(status_code=422, detail="Review text cannot be empty")
    
    sentiment = predictor.predict(request.review_text, request.review_score)

    return SentimentResponse(
        review_text=request.review_text,
        review_score=request.review_score,
        sentiment=sentiment
    )





