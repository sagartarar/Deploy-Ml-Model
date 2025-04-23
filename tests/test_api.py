# tests/test_api.py
from fastapi.testclient import TestClient
# Ensure the app can be imported. Add project root to sys.path if needed,
# though pytest usually handles this if run from the root.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FastAPI app instance
from app.main import app

# --- Test Cases ---

# Test that doesn't require the model (can run without lifespan context)
def test_read_root():
    """Test the root endpoint."""
    client = TestClient(app) # No lifespan needed here
    response = client.get("/")
    assert response.status_code == 200
    # **HIGHLIGHT: Ensure this message matches the API response**
    assert response.json() == {"message": "Welcome to the Deploy ML Model API. Use /docs for details."}

# Tests that require the model loaded via lifespan
def test_model_status():
    """Test the model status endpoint."""
    # Use TestClient as a context manager to handle lifespan
    with TestClient(app) as client:
        response = client.get("/model_status")
        assert response.status_code == 200
        status = response.json()
        assert "model_loaded" in status
        # Assert model is loaded within the 'with' block
        assert status["model_loaded"] is True, f"Model status reported not loaded. Path checked: {status.get('model_path_checked_at_startup')}"
        # **HIGHLIGHT: Ensure this key matches the API response**
        assert "model_path_checked_at_startup" in status

def test_predict_valid_input_setosa():
    """Test prediction with valid input expected to be class 0 (setosa)."""
    # Use TestClient as a context manager
    with TestClient(app) as client:
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict/", json=payload)
        assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}. Response: {response.text}"
        result = response.json()
        assert "prediction" in result
        assert "class_name" in result
        assert isinstance(result["prediction"], int)
        assert result["prediction"] == 0
        assert result["class_name"] == "setosa"

def test_predict_valid_input_versicolor():
    """Test prediction with valid input expected to be class 1 (versicolor)."""
    # Use TestClient as a context manager
    with TestClient(app) as client:
        payload = {
            "sepal_length": 6.0,
            "sepal_width": 2.7,
            "petal_length": 4.1,
            "petal_width": 1.3
        }
        response = client.post("/predict/", json=payload)
        assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}. Response: {response.text}"
        result = response.json()
        assert result["prediction"] == 1
        assert result["class_name"] == "versicolor"

def test_predict_valid_input_virginica():
    """Test prediction with valid input expected to be class 2 (virginica)."""
    # Use TestClient as a context manager
    with TestClient(app) as client:
        payload = {
            "sepal_length": 7.7,
            "sepal_width": 3.0,
            "petal_length": 6.1,
            "petal_width": 2.3
        }
        response = client.post("/predict/", json=payload)
        assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}. Response: {response.text}"
        result = response.json()
        assert result["prediction"] == 2
        assert result["class_name"] == "virginica"

# Tests for invalid input (don't strictly need the model, but using context manager is fine)
def test_predict_invalid_input_missing_field():
    """Test prediction with a missing required field."""
    # Use TestClient as a context manager (optional here, but consistent)
    with TestClient(app) as client:
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4
            # Missing "petal_width"
        }
        response = client.post("/predict/", json=payload)
        assert response.status_code == 422
        assert "detail" in response.json()

def test_predict_invalid_input_wrong_type():
    """Test prediction with incorrect data type for a field."""
     # Use TestClient as a context manager (optional here, but consistent)
    with TestClient(app) as client:
        payload = {
            "sepal_length": "five point one", # String instead of float
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict/", json=payload)
        assert response.status_code == 422
        assert "detail" in response.json()
