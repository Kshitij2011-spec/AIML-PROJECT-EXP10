"""Tests for prediction endpoints, input validation, and physical feature engineering."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_nominal_machine():
    """Nominal machine operating within standard parameters should predict Low risk and no failure."""
    payload = {
        "type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551.0,
        "torque": 42.8,
        "tool_wear": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_failure"] is False
    assert data["failure_probability"] < 0.30
    assert data["risk_level"] in ["Low", "Medium"]
    assert "engineered_features" in data

    eng = data["engineered_features"]
    assert round(eng["temperature_difference"], 1) == 10.5
    assert eng["wear_load"] == 0.0
    assert eng["mechanical_power_kw"] > 0


def test_predict_critical_tool_wear_failure():
    """Machine with severe tool wear (240 min) and high cutting torque should trigger failure prediction."""
    payload = {
        "type": "L",
        "air_temperature": 298.5,
        "process_temperature": 309.0,
        "rotational_speed": 1400.0,
        "torque": 68.0,
        "tool_wear": 235.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_failure"] is True
    assert data["failure_probability"] >= data["threshold_applied"]
    assert data["risk_level"] in ["High", "Critical"]
    assert len(data["risk_factors"]) > 0

    # Ensure tool wear or wear_load is flagged
    risk_factor_features = [f["feature"] for f in data["risk_factors"]]
    assert "Tool wear" in risk_factor_features or "wear_load" in risk_factor_features
    assert "URGENT" in data["maintenance_recommendation"] or "WARNING" in data["maintenance_recommendation"]


def test_predict_input_validation():
    """Verify that physically impossible or schema-violating inputs are rejected with 422."""
    # Invalid variant type 'Z'
    bad_payload = {
        "type": "Z",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551.0,
        "torque": 42.8,
        "tool_wear": 0.0,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

    # Negative torque
    bad_torque_payload = {
        "type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551.0,
        "torque": -15.0,
        "tool_wear": 0.0,
    }
    response = client.post("/predict", json=bad_torque_payload)
    assert response.status_code == 422


def test_batch_prediction():
    """Verify batch inference handles multiple records and returns aggregated metrics."""
    batch_payload = {
        "machines": [
            {
                "type": "L",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 0.0,
            },
            {
                "type": "H",
                "air_temperature": 298.9,
                "process_temperature": 309.2,
                "rotational_speed": 1340.0,
                "torque": 72.0,
                "tool_wear": 240.0,
            },
            {
                "type": "M",
                "air_temperature": 300.0,
                "process_temperature": 310.5,
                "rotational_speed": 1500.0,
                "torque": 40.0,
                "tool_wear": 50.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_records"] == 3
    assert len(data["predictions"]) == 3
    assert data["predicted_failures"] >= 1
    assert data["predicted_normals"] >= 1
    assert 0.0 <= data["highest_risk_score"] <= 1.0
