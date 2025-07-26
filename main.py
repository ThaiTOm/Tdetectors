# app/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib  # Or tensorflow, torch, etc.

from api.routes import prediction_routes

# A dictionary to hold our models
# This will be populated at startup
ml_models = {}

# The new 'lifespan' context manager is the recommended way to handle startup/shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code to run on startup ---
    print("INFO:     Loading ML model...")
    # Load the ML model from the /models directory
    ml_models["sentiment_classifier"] = joblib.load("models/sentiment_classifier.pkl")
    print("INFO:     ML model loaded.")
    
    yield  # The application runs while the 'yield' is active
    
    # --- Code to run on shutdown ---
    print("INFO:     Clearing ML models...")
    ml_models.clear()
    print("INFO:     ML models cleared. Application shutting down.")


# Create the FastAPI app instance with the lifespan handler
app = FastAPI(
    title="ML Model API",
    lifespan=lifespan
)

# Include the routes from the other file
# This makes our main.py file clean and easy to read
app.include_router(prediction_routes.router, prefix="/api/v1")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the ML Model API"}