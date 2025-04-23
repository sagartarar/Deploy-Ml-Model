# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from contextlib import asynccontextmanager
from pathlib import Path # Import pathlib

# --- Global variable for the model ---
ml_model = None

# --- Model Loading Logic ---
# Construct the absolute path to the model file using pathlib for robustness
APP_DIR = Path(__file__).resolve().parent # Directory of main.py (e.g., /path/to/project/app)
ROOT_DIR = APP_DIR.parent # Project root directory (e.g., /path/to/project)
MODEL_PATH = ROOT_DIR / "model" / "simple_model.joblib" # Path to model file

# --- Lifespan Context Manager for Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    global ml_model
    print(f"Lifespan: Attempting to load model from: {MODEL_PATH}") # Log the exact path being checked
    try:
        if not MODEL_PATH.is_file(): # Use pathlib's check
             # Provide more context in the error message
             raise FileNotFoundError(f"Model file not found at calculated path: {MODEL_PATH}. Ensure 'simple_model.joblib' exists in the '{ROOT_DIR / 'model'}' directory.")
        ml_model = joblib.load(MODEL_PATH)
        print(f"Lifespan: Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError as fnf_error:
        print(f"Lifespan Fatal Error: {fnf_error}")
        ml_model = None # Ensure model is None if loading fails
    except Exception as e:
        print(f"Lifespan Fatal Error loading model: {e}")
        ml_model = None # Ensure model is None if loading fails
    yield
    # Code to run on shutdown (e.g., cleanup resources)
    print("Lifespan: Cleaning up resources...")
    ml_model = None # Clear the model from memory

# Define the application with the lifespan manager
app = FastAPI(
    title="Deploy ML Model API",
    description="API for a simple pre-trained Scikit-learn Iris model.",
    version="0.1.0",
    lifespan=lifespan # Register the lifespan context manager
)

# --- Pydantic Models for Input/Output ---
# Use Field for examples and descriptions (requires Pydantic v2+)
class ModelInput(BaseModel):
    """Input features for prediction."""
    sepal_length: float = Field(..., example=5.1, description="Sepal length in cm")
    sepal_width: float = Field(..., example=3.5, description="Sepal width in cm")
    petal_length: float = Field(..., example=1.4, description="Petal length in cm")
    petal_width: float = Field(..., example=0.2, description="Petal width in cm")

class PredictionOutput(BaseModel):
    """Prediction result."""
    prediction: int = Field(..., example=0, description="Predicted class index (0, 1, or 2 for Iris)")
    class_name: str = Field(..., example="setosa", description="Predicted class name")

class ErrorOutput(BaseModel):
    """Error message structure."""
    detail: str

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint providing basic API info."""
    # **HIGHLIGHT: Ensure this message matches the test expectation**
    return {"message": "Welcome to the Deploy ML Model API. Use /docs for details."}

@app.post("/predict/",
          response_model=PredictionOutput,
          responses={
              400: {"model": ErrorOutput, "description": "Model not loaded"},
              500: {"model": ErrorOutput, "description": "Prediction error"}
          })
def predict(data: ModelInput):
    """
    Makes a prediction based on input features.
    Requires sepal_length, sepal_width, petal_length, petal_width.
    Returns the predicted class index and name.
    """
    # Access the globally loaded model
    if ml_model is None:
        # Log the path that was checked during startup for debugging
        print(f"Prediction failed: Model not loaded. Checked path during startup: {MODEL_PATH}")
        raise HTTPException(status_code=400, detail=f"Model is not loaded or failed to load. Check server logs. Path checked at startup: {MODEL_PATH}")

    try:
        # Convert input data to numpy array format expected by the model
        input_data = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        # Make prediction
        prediction_index = ml_model.predict(input_data)[0]
        prediction_proba = ml_model.predict_proba(input_data)[0]

        # Map index to Iris class name
        iris_class_names = ['setosa', 'versicolor', 'virginica']
        if 0 <= prediction_index < len(iris_class_names):
            class_name = iris_class_names[prediction_index]
        else:
            class_name = "unknown"

        print(f"Input: {input_data.tolist()}, Prediction Index: {prediction_index}, Class: {class_name}, Probabilities: {prediction_proba.tolist()}")

        return PredictionOutput(prediction=int(prediction_index), class_name=class_name)

    except Exception as e:
        print(f"Prediction failed for input {data.dict()}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model_status")
def model_status():
   """Check if the model is loaded."""
   # Check the global variable
   model_loaded_status = ml_model is not None
   print(f"Model status check: {'Loaded' if model_loaded_status else 'Not Loaded'}. Path checked at startup: {MODEL_PATH}")
   # Return the absolute path string for clarity in the response
   # **HIGHLIGHT: Ensure this key matches the test expectation**
   return {"model_loaded": model_loaded_status, "model_path_checked_at_startup": str(MODEL_PATH)}
