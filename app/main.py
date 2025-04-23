# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# Define the application
app = FastAPI(
    title="Simple ML API",
    description="API for a simple pre-trained Scikit-learn Iris model.",
    version="0.1.0",
)

# --- Model Loading ---
# Construct the absolute path to the model file relative to this script
MODEL_DIR = os.path.dirname(__file__) # Directory of main.py
MODEL_PATH = os.path.join(MODEL_DIR, '..', 'model', 'simple_model.joblib')

model = None

@app.on_event("startup")
async def load_model():
    """Load the model during startup."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
        # In a real app, you might want to prevent startup or handle this differently
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None # Ensure model is None if loading fails

# --- Pydantic Models for Input/Output ---
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
    return {"message": "Welcome to the Simple ML API. Use /docs for details."}

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
    if model is None:
        raise HTTPException(status_code=400, detail="Model is not loaded or failed to load.")

    try:
        # Convert input data to numpy array format expected by the model
        # The model expects a 2D array, hence the double brackets [[...]]
        input_data = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        # Make prediction
        prediction_index = model.predict(input_data)[0] # Get the first (and only) prediction index
        prediction_proba = model.predict_proba(input_data)[0] # Get probabilities

        # Map index to Iris class name (example)
        iris_class_names = ['setosa', 'versicolor', 'virginica']
        if 0 <= prediction_index < len(iris_class_names):
            class_name = iris_class_names[prediction_index]
        else:
            class_name = "unknown" # Handle unexpected index

        print(f"Input: {input_data.tolist()}, Prediction Index: {prediction_index}, Class: {class_name}, Probabilities: {prediction_proba.tolist()}")

        # Return the prediction
        return PredictionOutput(prediction=int(prediction_index), class_name=class_name)

    except Exception as e:
        print(f"Prediction failed for input {data.dict()}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model_status")
def model_status():
   """Check if the model is loaded."""
   return {"model_loaded": model is not None, "model_path_checked": MODEL_PATH}

# --- To run locally (optional, for testing) ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting Uvicorn server locally...")
#     # Note: Running this way might have issues with relative paths if not careful.
#     # It's generally better to run using 'uvicorn app.main:app' from the project root.
#     uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
