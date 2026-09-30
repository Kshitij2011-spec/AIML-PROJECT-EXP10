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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
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

    # 6. Baseline Evaluation on Untouched Test Set
    print("\n--- Evaluating Majority Class Baseline on Test Set ---")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)
    dummy_probs = dummy.predict_proba(X_test)[:, 1]
    dummy_cm = confusion_matrix(y_test, dummy_preds)
    dummy_tn, dummy_fp, dummy_fn, dummy_tp = dummy_cm.ravel()

    baseline_metrics = {
        "model_name": "Majority Class Baseline",
        "strategy": "most_frequent",
        "accuracy": round(float(accuracy_score(y_test, dummy_preds)), 4),
        "precision": round(float(precision_score(y_test, dummy_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, dummy_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, dummy_preds, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_test, dummy_preds, average="macro", zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_test, dummy_probs)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, dummy_probs)), 4),
        "confusion_matrix": {
            "tn": int(dummy_tn),
            "fp": int(dummy_fp),
            "fn": int(dummy_fn),
            "tp": int(dummy_tp),
        },
        "academic_note": (
            "The majority-class baseline achieves high accuracy (96.60%) simply because machine failures are rare (3.39%). "
            "Its failure recall is effectively zero (0.0%), demonstrating why accuracy alone is misleading in heavily imbalanced maintenance problems."
        ),
    }
    print(f"Majority Baseline -> Accuracy: {baseline_metrics['accuracy']:.4f} | Recall: {baseline_metrics['recall']:.4f} | Macro-F1: {baseline_metrics['macro_f1']:.4f}")

    # 7. 5-Fold Stratified Cross-Validation on Training Partition ONLY (Leakage-free)
    print("\n--- Running 5-Fold Stratified Cross-Validation on Train Split (f1_macro) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe_lr_cv = build_pipeline(
        LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        include_engineered=True,
    )
    pipe_rf_cv = build_pipeline(
        RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        include_engineered=True,
    )

    lr_cv_scores = cross_val_score(pipe_lr_cv, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    rf_cv_scores = cross_val_score(pipe_rf_cv, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

    cv_report = {
        "n_splits": 5,
        "scoring": "f1_macro",
        "partition": "train_only",
        "models": {
            "Logistic Regression (Engineered Features)": {
                "fold_scores": [round(float(s), 4) for s in lr_cv_scores],
                "mean": round(float(lr_cv_scores.mean()), 4),
                "std": round(float(lr_cv_scores.std()), 4),
            },
            "Random Forest (Engineered Features)": {
                "fold_scores": [round(float(s), 4) for s in rf_cv_scores],
                "mean": round(float(rf_cv_scores.mean()), 4),
                "std": round(float(rf_cv_scores.std()), 4),
            },
        },
    }
    print(f"LR (Engineered) CV Macro-F1: {cv_report['models']['Logistic Regression (Engineered Features)']['mean']:.4f} ± {cv_report['models']['Logistic Regression (Engineered Features)']['std']:.4f}")
    print(f"RF (Engineered) CV Macro-F1: {cv_report['models']['Random Forest (Engineered Features)']['mean']:.4f} ± {cv_report['models']['Random Forest (Engineered Features)']['std']:.4f}")

    # Curves for frontend visualization
    curve_points = get_curve_points(y_test, test_probs)

    # 8. Feature Importances
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

    # 9. Generate Tree Visualization Artifact (SVG and PNG)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        fig, ax = plt.subplots(figsize=(24, 12), dpi=150)
        plot_tree(
            rf_clf.estimators_[0],
            max_depth=3,
            feature_names=transformed_feature_names,
            class_names=["Normal", "Failure"],
            filled=True,
            rounded=True,
            impurity=True,
            proportion=False,
            ax=ax,
            fontsize=9,
        )
        ax.set_title("MachineGuard AI — Random Forest Estimator #0 (Top Decision Split Hierarchy)", fontsize=14, weight="bold")
        fig.tight_layout()
        svg_tree_path = os.path.join(ARTIFACTS_DIR, "random_forest_tree_0.svg")
        png_tree_path = os.path.join(ARTIFACTS_DIR, "random_forest_tree_0.png")
        fig.savefig(svg_tree_path, format="svg", bbox_inches="tight")
        fig.savefig(png_tree_path, format="png", bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved tree visualization artifacts:\n  - {svg_tree_path}\n  - {png_tree_path}")
    except Exception as e:
        print(f"Notice: Matplotlib tree plot skipped ({str(e)})")

    # 10. Assemble Metadata & Metrics dictionaries
    metadata = {
        "model_name": "MachineGuard AI - Industrial Predictive Maintenance",
        "model_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "RandomForestClassifier",
        "class_weight": "balanced",
        "total_estimators": len(rf_clf.estimators_),
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
        "baseline_comparison": baseline_metrics,
        "cross_validation": cv_report,
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

    # 11. Compile machine-readable academic report results artifact
    sample_tree = rf_clf.estimators_[0].tree_
    report_results = {
        "dataset": {
            **dataset_stats,
            "target_variable": TARGET_COLUMN,
            "usable_predictors": CATEGORICAL_FEATURES + NUMERIC_FEATURES,
            "engineered_features": ENGINEERED_NUMERIC_FEATURES,
            "strictly_excluded_leakage_columns": DROP_COLUMNS,
        },
        "data_split": {
            "train_samples": len(X_train),
            "train_failures": int(y_train.sum()),
            "val_samples": len(X_val),
            "val_failures": int(y_val.sum()),
            "test_samples": len(X_test),
            "test_failures": int(y_test.sum()),
            "stratified": True,
            "random_state": 42,
        },
        "baseline_results": baseline_metrics,
        "ablation_results": ablation_results,
        "cross_validation": cv_report,
        "threshold_tuning": {
            "metric_optimized": "f1",
            "tuning_partition": "validation_only",
            "default_threshold": 0.5,
            "selected_threshold": round(tuned_threshold, 4),
            "validation_metrics_default": default_val_metrics,
            "validation_metrics_tuned": tuned_val_metrics,
        },
        "final_test_metrics": {
            "default_threshold": default_test_metrics,
            "tuned_threshold": tuned_test_metrics,
        },
        "confusion_matrix": tuned_test_metrics["confusion_matrix"],
        "feature_importance": feature_importance_list,
        "model_configuration": {
            "algorithm": "RandomForestClassifier",
            "n_estimators": 150,
            "max_depth": 12,
            "class_weight": "balanced",
            "random_state": 42,
            "preprocessor": "ColumnTransformer(OneHotEncoder + StandardScaler)",
        },
        "tree_visualization_metadata": {
            "total_estimators": len(rf_clf.estimators_),
            "estimator_0_node_count": int(sample_tree.node_count),
            "estimator_0_max_depth": int(sample_tree.max_depth),
            "svg_artifact": "backend/artifacts/random_forest_tree_0.svg",
            "png_artifact": "backend/artifacts/random_forest_tree_0.png",
        },
        "data_honesty_statement": metadata["data_honesty_statement"],
    }

    # 12. Save Artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print(f"\nSaving model artifact to: {MODEL_ARTIFACT_PATH}")
    joblib.dump(primary_pipe, MODEL_ARTIFACT_PATH)

    print(f"Saving metadata to: {METADATA_PATH}")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saving metrics report to: {METRICS_PATH}")
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    report_path = os.path.join(ARTIFACTS_DIR, "report_results.json")
    print(f"Saving academic report results to: {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_results, f, indent=2)

    print("\nTraining, ablation evaluation, cross-validation, and artifact export complete!")
    return primary_pipe, metadata, metrics_report


if __name__ == "__main__":
    run_training_pipeline()
