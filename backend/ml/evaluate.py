"""Evaluation and threshold tuning utilities for MachineGuard AI."""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics at a given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics, handling zero division gracefully
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    return {
        "threshold": float(threshold),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    candidate_thresholds: np.ndarray = None,
) -> Tuple[float, Dict[str, Any]]:
    """Tune probability threshold on validation data to maximize the target metric (default: F1-score).

    Never run this on test data!
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.05, 0.95, 91)

    best_thresh = 0.5
    best_score = -1.0
    best_metrics = None

    for thresh in candidate_thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(thresh))
        score = m[metric]
        # In case of tie, prefer threshold closest to 0.5 or higher recall
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
            best_metrics = m

    return best_thresh, best_metrics


def get_curve_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int = 50,
) -> Dict[str, List[Dict[str, float]]]:
    """Sample ROC and Precision-Recall curve points for visual presentation in the API / frontend."""
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    # Subsample to max_points for lightweight serialization
    def subsample(x_arr, y_arr, name_x, name_y):
        indices = np.linspace(0, len(x_arr) - 1, min(len(x_arr), max_points), dtype=int)
        return [
            {name_x: round(float(x_arr[i]), 4), name_y: round(float(y_arr[i]), 4)}
            for i in indices
        ]

    return {
        "roc_curve": subsample(fpr, tpr, "fpr", "tpr"),
        "pr_curve": subsample(recall, precision, "recall", "precision"),
    }
