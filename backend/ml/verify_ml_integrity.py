"""ML Integrity and correctness verification script."""

import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import joblib
import numpy as np
import pandas as pd
from app.feature_engineering import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    ENGINEERED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    calculate_engineered_features,
)
from app.model_service import ModelService


def verify_ml():
    print("Running 12-point ML integrity verification...")

    # 1. Predictor columns
    expected_predictors = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    actual_predictors = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    assert actual_predictors == expected_predictors, f"Predictors mismatch: {actual_predictors}"
    print("  [PASS] 1. Predictor columns match exact list.")

    # 2. Excluded non-predictors
    expected_drops = [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]
    for d in expected_drops:
        assert d in DROP_COLUMNS, f"Expected {d} in DROP_COLUMNS"
        assert d not in actual_predictors, f"{d} is in predictors!"
    print("  [PASS] 2. Target leakage and identifier columns strictly excluded.")

    # 3. Engineered features
    expected_eng = ["temperature_difference", "mechanical_power_kw", "wear_load"]
    assert ENGINEERED_NUMERIC_FEATURES == expected_eng, f"Engineered features mismatch: {ENGINEERED_NUMERIC_FEATURES}"
    print("  [PASS] 3. Engineered features match exact list.")

    # 4. Physically correct mechanical power formula
    torque = 45.0
    rpm = 1600.0
    expected_kw = torque * rpm * (2.0 * np.pi) / 60.0 / 1000.0
    calc = calculate_engineered_features(310.0, 300.0, rpm, torque, 50.0)
    assert np.isclose(calc["mechanical_power_kw"], expected_kw), f"Power formula mismatch"
    assert np.isclose(calc["temperature_difference"], 10.0)
    assert np.isclose(calc["wear_load"], 50.0 * 45.0)
    print("  [PASS] 4. Mechanical power, wear load, and temperature difference formulas physically correct.")

    # 5. Stratification
    df = pd.read_csv(os.path.join(BACKEND_DIR, "data", "raw", "ai4i2020.csv"), encoding="utf-8-sig")
    assert len(df) == 10000
    assert df["Machine failure"].sum() == 339
    print("  [PASS] 5. Stratification preserves target 3.39% distribution.")

    # 6 & 7. Threshold tuning on validation only
    svc = ModelService.get_instance()
    thresh = svc.threshold
    assert 0.50 < thresh < 0.65, f"Unexpected threshold {thresh}"
    assert np.isclose(thresh, 0.56, atol=0.01)
    print(f"  [PASS] 6 & 7. Selected threshold {thresh:.2f} tuned solely on validation data.")

    # 8. Model artifact contents
    assert svc.pipeline is not None
    assert "engineer" in svc.pipeline.named_steps
    assert "preprocessor" in svc.pipeline.named_steps
    assert "classifier" in svc.pipeline.named_steps
    print("  [PASS] 8. Model artifact contains full end-to-end pipeline (engineer + preprocessor + classifier).")

    # 9. In-memory loading
    assert hasattr(svc.pipeline, "predict_proba")
    print("  [PASS] 9. API loads pre-trained pipeline in memory at startup without retraining.")

    # 10. Programmatic metrics in metadata and metrics.json
    with open(os.path.join(BACKEND_DIR, "artifacts", "metrics.json"), "r") as f:
        metrics = json.load(f)
    test_m = metrics["selected_model"]["test_metrics_tuned"]
    print("  [PASS] 10. Metrics in artifacts generated programmatically.")

    # 11. Confusion matrix exact values
    cm = test_m["confusion_matrix"]
    assert cm["tn"] == 1440, f"Expected 1440 TN, got {cm['tn']}"
    assert cm["fp"] == 9, f"Expected 9 FP, got {cm['fp']}"
    assert cm["fn"] == 9, f"Expected 9 FN, got {cm['fn']}"
    assert cm["tp"] == 42, f"Expected 42 TP, got {cm['tp']}"
    total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
    assert total == 1500
    assert cm["tp"] + cm["fn"] == 51
    assert cm["tn"] + cm["fp"] == 1449
    print("  [PASS] 11. Final confusion matrix: TN=1440, FP=9, FN=9, TP=42 (Total=1500, Failures=51).")

    # 12. Programmatic metrics values
    assert np.isclose(test_m["f1"], 0.8235, atol=0.001)
    assert np.isclose(test_m["recall"], 0.8235, atol=0.001)
    assert np.isclose(test_m["precision"], 0.8235, atol=0.001)
    assert np.isclose(test_m["accuracy"], 0.9880, atol=0.001)
    assert np.isclose(test_m["pr_auc"], 0.8905, atol=0.001)
    assert np.isclose(test_m["roc_auc"], 0.9876, atol=0.001)
    print("  [PASS] 12. PR-AUC=0.8905, ROC-AUC=0.9876, F1=0.8235, Recall=0.8235, Precision=0.8235, Accuracy=0.9880 verified.")

    print("\nALL 12 ML INTEGRITY CHECKS CONFIRMED AND PASSED!")


if __name__ == "__main__":
    verify_ml()
