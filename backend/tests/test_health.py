"""Tests for health, root, metadata, and metrics endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "MachineGuard AI"
    assert data["status"] == "online"
    assert "AI4I 2020 synthetic" in data["data_honesty_statement"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "selected_threshold" in data
    assert 0.0 < data["selected_threshold"] < 1.0


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    meta = data["metadata"]
    assert meta["model_name"] == "MachineGuard AI - Industrial Predictive Maintenance"
    assert meta["algorithm"] == "RandomForestClassifier"
    assert "feature_importances" in meta
    assert len(meta["feature_importances"]) > 0
    assert "engineered_feature_definitions" in meta
    assert "dataset_statistics" in meta
    assert meta["dataset_statistics"]["total_records"] == 10000


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    metrics = data["metrics"]
    assert "ablation_comparison" in metrics
    assert "selected_model" in metrics
    selected = metrics["selected_model"]
    assert "test_metrics_tuned" in selected
    assert "pr_curve" in selected
    assert "roc_curve" in selected
    assert selected["test_metrics_tuned"]["pr_auc"] > 0.80
    assert selected["test_metrics_tuned"]["f1"] > 0.75


def test_api_model_info_endpoint():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "MachineGuard AI - Industrial Predictive Maintenance"
    assert data["test_samples"] == 1500
    assert "test_metrics" in data
    assert data["test_metrics"]["pr_auc"] == 0.8905
    assert data["test_metrics"]["roc_auc"] == 0.9876
    assert "feature_importances" in data
    assert len(data["feature_importances"]) > 0

