# app/routes/prediction_routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# IMPORTANT: We import the 'ml_models' dictionary from main.py
# This is how we access the model that was loaded at startup.
from app.main import ml_models

# Define the router
router = APIRouter()

# Define the Pydantic model for the request body
class PredictionRequest(BaseModel):
    text: str

# Define the Pydantic model for the response
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"]
)
def predict_sentiment(request: PredictionRequest):
    """
    Accepts text input and returns a sentiment prediction using the loaded model.
    """
    # Check if the model is available
    if "sentiment_classifier" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable.")

    # Get the model from the dictionary
    model = ml_models["sentiment_classifier"]

    # --- This is where your specific model's logic goes ---
    # The following is just an example.
    # prediction = model.predict([request.text])
    # confidence = model.predict_proba([request.text]).max()

    # Example dummy logic:
    if "good" in request.text.lower():
        prediction = "Positive"
        confidence = 0.95
    else:
        prediction = "Negative"
        confidence = 0.88
    # --- End of model logic ---

    return PredictionResponse(prediction=prediction, confidence=confidence)