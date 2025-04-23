# tests/test_api.py
from fastapi.testclient import TestClient
# Ensure the app can be imported. Add project root to sys.path if needed,
# though pytest usually handles this if run from the root.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app # Import your FastAPI app instance

# Create a test client
client = TestClient(app)

# --- Test Cases ---

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Simple ML API. Use /docs for details."}

def test_model_status():
    """Test the model status endpoint."""
    response = client.get("/model_status")
    assert response.status_code == 200
    status = response.json()
    assert "model_loaded" in status
    # In a test environment, the startup event should load the model
    assert status["model_loaded"] is True

def test_predict_valid_input_setosa():
    """Test prediction with valid input expected to be class 0 (setosa)."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "class_name" in result
    assert isinstance(result["prediction"], int)
    assert result["prediction"] == 0 # Expecting setosa for this classic example
    assert result["class_name"] == "setosa"

def test_predict_valid_input_versicolor():
    """Test prediction with valid input expected to be class 1 (versicolor)."""
    payload = {
        "sepal_length": 6.0,
        "sepal_width": 2.7,
        "petal_length": 4.1,
        "petal_width": 1.3
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == 1 # Expecting versicolor
    assert result["class_name"] == "versicolor"

def test_predict_valid_input_virginica():
    """Test prediction with valid input expected to be class 2 (virginica)."""
    payload = {
        "sepal_length": 7.7,
        "sepal_width": 3.0,
        "petal_length": 6.1,
        "petal_width": 2.3
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == 2 # Expecting virginica
    assert result["class_name"] == "virginica"

def test_predict_invalid_input_missing_field():
    """Test prediction with a missing required field."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
        # Missing "petal_width"
    }
    response = client.post("/predict/", json=payload)
    # FastAPI returns 422 Unprocessable Entity for Pydantic validation errors
    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_invalid_input_wrong_type():
    """Test prediction with incorrect data type for a field."""
    payload = {
        "sepal_length": "five point one", # String instead of float
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()

