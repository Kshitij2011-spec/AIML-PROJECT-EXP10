"""Backend tests for academic components: Majority Baseline, 5-Fold Stratified CV, and Tree Explorer API."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_api_model_info_includes_academic_components():
    """Verify that /api/model-info returns baseline and cross-validation metrics."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()

    assert data["total_estimators"] == 150
    assert "baseline_comparison" in data
    assert "cross_validation" in data

    baseline = data["baseline_comparison"]
    assert baseline["strategy"] == "most_frequent"
    assert baseline["accuracy"] == 0.9660
    assert baseline["recall"] == 0.0
    assert baseline["precision"] == 0.0
    assert baseline["f1"] == 0.0
    assert baseline["macro_f1"] == 0.4914
    assert baseline["confusion_matrix"]["tn"] == 1449
    assert baseline["confusion_matrix"]["fn"] == 51

    cv = data["cross_validation"]
    assert cv["n_splits"] == 5
    assert cv["scoring"] == "f1_macro"
    assert "Logistic Regression (Engineered Features)" in cv["models"]
    assert "Random Forest (Engineered Features)" in cv["models"]

    rf_cv = cv["models"]["Random Forest (Engineered Features)"]
    assert len(rf_cv["fold_scores"]) == 5
    assert rf_cv["mean"] > 0.85
    assert rf_cv["std"] < 0.05


def test_tree_endpoint_tree_0():
    """Verify GET /api/model/tree/0 returns valid full tree structure."""
    response = client.get("/api/model/tree/0")
    assert response.status_code == 200
    data = response.json()

    assert data["tree_index"] == 0
    assert data["total_estimators"] == 150
    assert data["node_count"] > 100
    assert data["max_depth"] >= 10
    assert len(data["nodes"]) == data["node_count"]
    assert len(data["edges"]) > 0

    # Verify Root Node
    root = data["nodes"][0]
    assert root["id"] == 0
    assert root["depth"] == 0
    assert root["is_leaf"] is False
    assert root["feature"] is not None
    assert root["threshold"] is not None
    assert root["threshold_unscaled"] is not None
    assert root["samples"] > 0
    assert len(root["class_counts"]) == 2
    assert root["gini"] > 0.0
    assert root["left_child"] is not None
    assert root["right_child"] is not None

    # Verify Leaf Nodes Exist
    leaf_nodes = [n for n in data["nodes"] if n["is_leaf"]]
    assert len(leaf_nodes) > 0
    sample_leaf = leaf_nodes[0]
    assert sample_leaf["left_child"] is None
    assert sample_leaf["right_child"] is None
    assert sample_leaf["predicted_class"] in [0, 1]
    assert sample_leaf["predicted_class_name"] in ["Normal", "Failure"]

    # Verify Edge Conditions
    sample_edge = data["edges"][0]
    assert sample_edge["source"] == 0
    assert sample_edge["branch"] in ["left", "right"]
    assert len(sample_edge["condition"]) > 0


def test_tree_endpoint_depth_filtering():
    """Verify that max_depth filter restricts returned nodes."""
    response = client.get("/api/model/tree/0?max_depth=3")
    assert response.status_code == 200
    data = response.json()

    assert data["filtered_max_depth"] == 3
    # Check that all nodes have depth <= 3
    for node in data["nodes"]:
        assert node["depth"] <= 3

    # Nodes at depth 3 should be marked as leaves in filtered view
    depth_3_nodes = [n for n in data["nodes"] if n["depth"] == 3]
    for n in depth_3_nodes:
        assert n["is_leaf"] is True


def test_tree_endpoint_invalid_indices():
    """Verify that invalid tree indices return 404."""
    # Out of range positive
    response = client.get("/api/model/tree/999")
    assert response.status_code == 404

    # Negative index triggers validation error 422
    response_neg = client.get("/api/model/tree/-1")
    assert response_neg.status_code in [404, 422]
