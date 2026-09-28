"""Complete reproducible training, ablation evaluation, and artifact generation pipeline for MachineGuard AI."""

from datetime import datetime, timezone
import json
import os
import sys
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.feature_engineering import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    ENGINEERED_FEATURE_METADATA,
    ENGINEERED_NUMERIC_FEATURES,
    IndustrialFeatureEngineer,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from ml.evaluate import compute_metrics, find_optimal_threshold, get_curve_points

DATA_FILE = os.path.join(BASE_DIR, "data", "raw", "ai4i2020.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_ARTIFACT_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")


def load_dataset() -> pd.DataFrame:
    """Load canonical AI4I 2020 dataset."""
    if not os.path.exists(DATA_FILE):
        from ml.download_data import download_and_validate

        download_and_validate()

    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    return df


def compute_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for dataset honesty and frontend presentation."""
    total_records = len(df)
    total_failures = int(df[TARGET_COLUMN].sum())
    failure_rate = float(total_failures / total_records)

    # Feature summaries
    numeric_stats = {}
    for col in NUMERIC_FEATURES:
        series = df[col]
        numeric_stats[col] = {
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "median": round(float(series.median()), 2),
        }

    type_counts = df["Type"].value_counts().to_dict()

    return {
        "dataset_name": "AI4I 2020 Predictive Maintenance Dataset",
        "is_synthetic": True,
        "total_records": total_records,
        "total_failures": total_failures,
        "total_non_failures": total_records - total_failures,
        "failure_rate_pct": round(failure_rate * 100.0, 2),
        "type_distribution": type_counts,
        "numeric_feature_stats": numeric_stats,
    }


def build_pipeline(
    model: Any,
    include_engineered: bool = True,
) -> Pipeline:
    """Construct an end-to-end scikit-learn Pipeline with feature engineering and column transformations."""
    numeric_cols = NUMERIC_FEATURES + (
        ENGINEERED_NUMERIC_FEATURES if include_engineered else []
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    categories=[["L", "M", "H"]],
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    return Pipeline(
        [
            (
                "engineer",
                IndustrialFeatureEngineer(include_engineered=include_engineered),
            ),
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def extract_feature_names(
    pipeline: Pipeline,
    include_engineered: bool = True,
) -> List[str]:
    """Extract human-readable feature names post-transformation."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    numeric_cols = NUMERIC_FEATURES + (
        ENGINEERED_NUMERIC_FEATURES if include_engineered else []
    )
    return cat_names + numeric_cols


def run_training_pipeline() -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """Execute complete reproducible workflow: split, ablation study, model selection, threshold tuning, and test evaluation."""
    print("=" * 70)
    print("MachineGuard AI — Training & Evaluation Pipeline")
    print("=" * 70)

    # 1. Load data
    df = load_dataset()
    dataset_stats = compute_dataset_stats(df)
    print(f"Loaded dataset: {dataset_stats['total_records']} records, {dataset_stats['total_failures']} failures ({dataset_stats['failure_rate_pct']}%)")

    # 2. Strict Featurization Ordering & Split
    # Split 70% Train, 15% Validation, 15% Test (stratified on Machine failure)
    y = df[TARGET_COLUMN].values
    # Predictor dataframe excludes target and leakage columns
    X = df[[col for col in df.columns if col not in DROP_COLUMNS]]

    # Step A: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    # Step B: 50% of temp is validation (15% overall), 50% is test (15% overall)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    print(f"Data Split: Train={len(X_train)} (Failures={int(y_train.sum())}), Val={len(X_val)} (Failures={int(y_val.sum())}), Test={len(X_test)} (Failures={int(y_test.sum())})")

    # 3. Model candidate definition for ablation study
    candidates = {
        "Logistic Regression (Original Features)": {
            "model": LogisticRegression(
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
            ),
            "include_engineered": False,
        },
        "Logistic Regression (Engineered Features)": {
            "model": LogisticRegression(
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
            ),
            "include_engineered": True,
        },
        "Random Forest (Original Features)": {
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "include_engineered": False,
        },
        "Random Forest (Engineered Features)": {
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "include_engineered": True,
        },
    }

    ablation_results = {}
    fitted_pipelines = {}

    print("\n--- Running Ablation Study on Validation Set ---")
    for name, config in candidates.items():
        pipe = build_pipeline(
            config["model"],
            include_engineered=config["include_engineered"],
        )
        # Strict rule: Fit pipeline solely on training data
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        val_probs = pipe.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, val_probs, threshold=0.5)
        ablation_results[name] = {
            "engineered_features_included": config["include_engineered"],
            "model_type": type(config["model"]).__name__,
            "validation_metrics_default_threshold": metrics,
        }
        print(
            f"[{name}]\n"
            f"  Val F1: {metrics['f1']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f}"
        )

    # 4. Primary Model Selection & Threshold Tuning on Validation Data
    primary_name = "Random Forest (Engineered Features)"
    primary_pipe = fitted_pipelines[primary_name]

    print(f"\n--- Model Selected: {primary_name} ---")
    val_probs_primary = primary_pipe.predict_proba(X_val)[:, 1]

    # Tune probability threshold on validation data to maximize F1-score
    tuned_threshold, tuned_val_metrics = find_optimal_threshold(
        y_val,
        val_probs_primary,
        metric="f1",
    )
    default_val_metrics = compute_metrics(y_val, val_probs_primary, threshold=0.5)

    print(f"Validation Default Threshold (0.50) -> F1: {default_val_metrics['f1']:.4f}, Recall: {default_val_metrics['recall']:.4f}, Precision: {default_val_metrics['precision']:.4f}")
    print(f"Validation Tuned Threshold   ({tuned_threshold:.2f}) -> F1: {tuned_val_metrics['f1']:.4f}, Recall: {tuned_val_metrics['recall']:.4f}, Precision: {tuned_val_metrics['precision']:.4f}")

    # 5. Final Evaluation ONCE on Untouched Test Set
    print("\n--- Final Evaluation on Untouched Test Set ---")
    test_probs = primary_pipe.predict_proba(X_test)[:, 1]
    default_test_metrics = compute_metrics(y_test, test_probs, threshold=0.5)
    tuned_test_metrics = compute_metrics(y_test, test_probs, threshold=tuned_threshold)

    print(f"Test Default Threshold (0.50) -> F1: {default_test_metrics['f1']:.4f}, Recall: {default_test_metrics['recall']:.4f}, Precision: {default_test_metrics['precision']:.4f}, PR-AUC: {default_test_metrics['pr_auc']:.4f}")
    print(f"Test Tuned Threshold   ({tuned_threshold:.2f}) -> F1: {tuned_test_metrics['f1']:.4f}, Recall: {tuned_test_metrics['recall']:.4f}, Precision: {tuned_test_metrics['precision']:.4f}, PR-AUC: {tuned_test_metrics['pr_auc']:.4f}")

    # Curves for frontend visualization
    curve_points = get_curve_points(y_test, test_probs)

    # 6. Feature Importances
    rf_clf: RandomForestClassifier = primary_pipe.named_steps["classifier"]
    transformed_feature_names = extract_feature_names(primary_pipe, include_engineered=True)
    importances = rf_clf.feature_importances_
    feature_importance_list = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(
            zip(transformed_feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
    ]

    print("\nFeature Importances:")
    for item in feature_importance_list:
        print(f"  - {item['feature']:<25}: {item['importance']:.4f}")

    # 7. Assemble Metadata & Metrics dictionaries
    metadata = {
        "model_name": "MachineGuard AI - Industrial Predictive Maintenance",
        "model_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "RandomForestClassifier",
        "class_weight": "balanced",
        "selected_threshold": round(tuned_threshold, 4),
        "default_threshold": 0.5,
        "input_features": CATEGORICAL_FEATURES + NUMERIC_FEATURES,
        "engineered_features": ENGINEERED_NUMERIC_FEATURES,
        "engineered_feature_definitions": ENGINEERED_FEATURE_METADATA,
        "all_transformed_features": transformed_feature_names,
        "feature_importances": feature_importance_list,
        "dataset_statistics": {
            **dataset_stats,
            "splits": {
                "train_records": len(X_train),
                "val_records": len(X_val),
                "test_records": len(X_test),
                "train_failures": int(y_train.sum()),
                "val_failures": int(y_val.sum()),
                "test_failures": int(y_test.sum()),
            },
        },
        "data_honesty_statement": (
            "Educational prototype trained on the AI4I 2020 synthetic predictive-maintenance benchmark. "
            "Results are not validated for deployment on real industrial machinery."
        ),
    }

    metrics_report = {
        "ablation_comparison": ablation_results,
        "selected_model": {
            "name": primary_name,
            "tuned_threshold": round(tuned_threshold, 4),
            "validation_metrics_default": default_val_metrics,
            "validation_metrics_tuned": tuned_val_metrics,
            "test_metrics_default": default_test_metrics,
            "test_metrics_tuned": tuned_test_metrics,
            "roc_curve": curve_points["roc_curve"],
            "pr_curve": curve_points["pr_curve"],
        },
    }

    # 8. Save Artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print(f"\nSaving model artifact to: {MODEL_ARTIFACT_PATH}")
    joblib.dump(primary_pipe, MODEL_ARTIFACT_PATH)

    print(f"Saving metadata to: {METADATA_PATH}")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saving metrics report to: {METRICS_PATH}")
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    print("\nTraining and artifact export complete!")
    return primary_pipe, metadata, metrics_report


if __name__ == "__main__":
    run_training_pipeline()
