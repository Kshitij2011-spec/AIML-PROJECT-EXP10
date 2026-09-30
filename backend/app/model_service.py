"""Model loading, inference, and explainability service for MachineGuard AI."""

import json
import os
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd
from app.feature_engineering import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURE_METADATA,
    NUMERIC_FEATURES,
    calculate_engineered_features,
)
from app.schemas import (
    BatchMachineInput,
    BatchPredictionResponse,
    EngineeredFeatureValues,
    MachineInput,
    PredictionResponse,
    RandomForestTreeResponse,
    RiskFactor,
    TreeEdge,
    TreeNode,
)

FEATURE_LABELS = {
    "Type_M": "Product Type: Medium",
    "Type_H": "Product Type: High",
    "Air temperature [K]": "Air Temperature",
    "Process temperature [K]": "Process Temperature",
    "Rotational speed [rpm]": "Rotational Speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool Wear",
    "temperature_difference": "Temperature Difference",
    "mechanical_power_kw": "Mechanical Power",
    "wear_load": "Wear Load",
}

FEATURE_UNITS = {
    "Air temperature [K]": "K",
    "Process temperature [K]": "K",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "Nm",
    "Tool wear [min]": "min",
    "temperature_difference": "K",
    "mechanical_power_kw": "kW",
    "wear_load": "min·Nm",
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")


class ModelService:
    """Service holding the loaded model pipeline, metadata, and decision threshold in memory."""

    _instance: Optional["ModelService"] = None

    def __init__(self):
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.threshold: float = 0.5
        self.load_artifacts()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def load_artifacts(self) -> None:
        """Load trained pipeline and metadata artifacts at startup."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
            print("Artifacts missing. Invoking training pipeline...")
            from ml.train import run_training_pipeline

            run_training_pipeline()

        print(f"Loading model pipeline from {MODEL_PATH} ...")
        self.pipeline = joblib.load(MODEL_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        self.threshold = float(self.metadata.get("selected_threshold", 0.5))
        self.feature_names = self.metadata.get(
            "all_transformed_features",
            [
                "Type_M",
                "Type_H",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "temperature_difference",
                "mechanical_power_kw",
                "wear_load",
            ],
        )
        try:
            scaler = self.pipeline.named_steps["preprocessor"].named_transformers_["num"]
            self.scaler_mean = list(scaler.mean_)
            self.scaler_scale = list(scaler.scale_)
        except Exception:
            self.scaler_mean = []
            self.scaler_scale = []

        print(f"ModelService ready. Active decision threshold: {self.threshold:.4f}")

    def _to_dataframe(self, item: MachineInput) -> pd.DataFrame:
        """Convert pydantic MachineInput to DataFrame matching canonical training column names."""
        row = {
            "Type": item.type,
            "Air temperature [K]": item.air_temperature_k,
            "Process temperature [K]": item.process_temperature_k,
            "Rotational speed [rpm]": item.rotational_speed_rpm,
            "Torque [Nm]": item.torque_nm,
            "Tool wear [min]": item.tool_wear_min,
        }
        return pd.DataFrame([row])

    def _analyze_risk_factors(
        self,
        item: MachineInput,
        eng: Dict[str, float],
    ) -> List[RiskFactor]:
        """Explainable physical risk diagnostics comparing inputs to benchmark operational boundaries."""
        factors: List[RiskFactor] = []

        # 1. Tool wear & Wear-load
        if item.tool_wear_min >= 210.0:
            factors.append(
                RiskFactor(
                    feature="Tool wear",
                    observed_value=item.tool_wear_min,
                    unit="min",
                    status="Critical",
                    message="Tool wear is critically high (>= 210 min). Imminent risk of tool wear failure (TWF).",
                )
            )
        elif item.tool_wear_min >= 180.0:
            factors.append(
                RiskFactor(
                    feature="Tool wear",
                    observed_value=item.tool_wear_min,
                    unit="min",
                    status="Elevated",
                    message="Tool wear is elevated (>= 180 min). Replacement recommended soon.",
                )
            )

        if eng["wear_load"] >= 11000.0:
            factors.append(
                RiskFactor(
                    feature="wear_load",
                    observed_value=round(eng["wear_load"], 1),
                    unit="min*Nm",
                    status="Critical",
                    message="High compound wear-load (> 11,000 min*Nm). Substantial risk of overstrain failure (OSF).",
                )
            )

        # 2. Torque & Rotational speed
        if item.torque_nm >= 65.0:
            factors.append(
                RiskFactor(
                    feature="Torque",
                    observed_value=item.torque_nm,
                    unit="Nm",
                    status="Critical",
                    message="Torque exceeds normal operating band (>= 65 Nm). High motor drive strain.",
                )
            )
        elif item.torque_nm <= 15.0:
            factors.append(
                RiskFactor(
                    feature="Torque",
                    observed_value=item.torque_nm,
                    unit="Nm",
                    status="Elevated",
                    message="Torque unusually low (<= 15 Nm). Potential spindle slipping or low load abnormality.",
                )
            )

        if item.rotational_speed_rpm <= 1350.0:
            factors.append(
                RiskFactor(
                    feature="Rotational speed",
                    observed_value=item.rotational_speed_rpm,
                    unit="rpm",
                    status="Elevated",
                    message="Low rotational speed (<= 1350 rpm) coupled with machining load increases thermal friction.",
                )
            )
        elif item.rotational_speed_rpm >= 2600.0:
            factors.append(
                RiskFactor(
                    feature="Rotational speed",
                    observed_value=item.rotational_speed_rpm,
                    unit="rpm",
                    status="Elevated",
                    message="Excessive spindle speed (>= 2600 rpm) indicates low cutting resistance or overspeed.",
                )
            )

        # 3. Mechanical Power
        if eng["mechanical_power_kw"] >= 9.0:
            factors.append(
                RiskFactor(
                    feature="mechanical_power_kw",
                    observed_value=round(eng["mechanical_power_kw"], 2),
                    unit="kW",
                    status="Critical",
                    message="Mechanical power exceeds 9.0 kW. Risk of power failure (PWF).",
                )
            )
        elif eng["mechanical_power_kw"] <= 3.2:
            factors.append(
                RiskFactor(
                    feature="mechanical_power_kw",
                    observed_value=round(eng["mechanical_power_kw"], 2),
                    unit="kW",
                    status="Elevated",
                    message="Mechanical power is below 3.2 kW under active cutting conditions.",
                )
            )

        # 4. Temperature difference
        if eng["temperature_difference"] < 8.6:
            factors.append(
                RiskFactor(
                    feature="temperature_difference",
                    observed_value=round(eng["temperature_difference"], 2),
                    unit="K",
                    status="Critical",
                    message="Process-to-air temperature gradient < 8.6 K. Heat dissipation failure (HDF) condition.",
                )
            )

        return factors

    def _build_recommendation(
        self,
        is_failure: bool,
        prob: float,
        risk_factors: List[RiskFactor],
    ) -> str:
        """Provide prescriptive maintenance advice."""
        if not is_failure and prob < 0.20:
            return "Operating parameters within nominal thresholds. Continue standard preventative maintenance schedule."

        if not is_failure and prob < 0.50:
            return "Minor stress indicators observed. Schedule inspection during next routine changeover."

        # High or Critical Risk
        critical_factors = [f.feature for f in risk_factors if f.status == "Critical"]
        if "Tool wear" in critical_factors or "wear_load" in critical_factors:
            return "URGENT: Replace cutting tool insert immediately to prevent tool rupture and workpiece damage."
        if "mechanical_power_kw" in critical_factors or "Torque" in critical_factors:
            return "WARNING: Reduce spindle torque and feed rate immediately to prevent motor overload or drive stall."
        if "temperature_difference" in critical_factors:
            return "WARNING: Inspect cooling lubrication flow and heat exchanger to resolve thermal dissipation bottleneck."

        return "ATTENTION: Machine operating at high failure probability. Inspect mechanical spindle, tool wear, and torque load."

    def predict(self, item: MachineInput) -> PredictionResponse:
        """Run single prediction through loaded pipeline."""
        df = self._to_dataframe(item)

        # Predict probability
        probs = self.pipeline.predict_proba(df)[0]
        failure_prob = float(probs[1])

        # Binary decision using validation-tuned threshold
        is_failure = bool(failure_prob >= self.threshold)

        # Categorical risk level
        if failure_prob < 0.20:
            risk_level = "Low"
        elif failure_prob < 0.50:
            risk_level = "Medium"
        elif failure_prob < 0.75:
            risk_level = "High"
        else:
            risk_level = "Critical"

        # Engineered features calculation
        eng = calculate_engineered_features(
            process_temp_k=item.process_temperature_k,
            air_temp_k=item.air_temperature_k,
            rotational_speed_rpm=item.rotational_speed_rpm,
            torque_nm=item.torque_nm,
            tool_wear_min=item.tool_wear_min,
        )

        risk_factors = self._analyze_risk_factors(item, eng)
        rec = self._build_recommendation(is_failure, failure_prob, risk_factors)

        return PredictionResponse(
            is_failure=is_failure,
            failure_probability=round(failure_prob, 4),
            threshold_applied=round(self.threshold, 4),
            risk_level=risk_level,
            engineered_features=EngineeredFeatureValues(**eng),
            risk_factors=risk_factors,
            maintenance_recommendation=rec,
            data_honesty_statement=self.metadata.get(
                "data_honesty_statement",
                "Educational prototype trained on the AI4I 2020 synthetic predictive-maintenance benchmark.",
            ),
        )

    def predict_batch(self, batch: BatchMachineInput) -> BatchPredictionResponse:
        """Process batch of machine inputs."""
        predictions = [self.predict(m) for m in batch.machines]
        failures = sum(1 for p in predictions if p.is_failure)
        normals = len(predictions) - failures
        highest = max((p.failure_probability for p in predictions), default=0.0)

        return BatchPredictionResponse(
            total_records=len(predictions),
            predicted_failures=failures,
            predicted_normals=normals,
            highest_risk_score=highest,
            predictions=predictions,
        )

    def get_tree_structure(
        self,
        tree_index: int = 0,
        max_depth: Optional[int] = None,
    ) -> RandomForestTreeResponse:
        """Extract exact structure, thresholds, impurity, and samples for an estimator tree."""
        if not self.pipeline or "classifier" not in self.pipeline.named_steps:
            raise RuntimeError("Model pipeline is not properly initialized.")

        rf = self.pipeline.named_steps["classifier"]
        if not hasattr(rf, "estimators_"):
            raise RuntimeError("Model does not expose ensemble estimators.")

        total_estimators = len(rf.estimators_)
        if tree_index < 0 or tree_index >= total_estimators:
            raise IndexError(
                f"Tree index {tree_index} out of range (0 to {total_estimators - 1})."
            )

        estimator = rf.estimators_[tree_index]
        tree = estimator.tree_
        total_tree_nodes = tree.node_count

        # Compute depth for each node using BFS
        node_depth: Dict[int, int] = {}
        queue = [(0, 0)]
        while queue:
            nid, d = queue.pop(0)
            node_depth[nid] = d
            l = int(tree.children_left[nid])
            r = int(tree.children_right[nid])
            if l != -1:
                queue.append((l, d + 1))
            if r != -1:
                queue.append((r, d + 1))

        # Determine which node ids are included based on max_depth
        if max_depth is not None:
            included_ids = {nid for nid, d in node_depth.items() if d <= max_depth}
        else:
            included_ids = set(range(total_tree_nodes))

        nodes: List[TreeNode] = []
        edges: List[TreeEdge] = []

        for nid in sorted(included_ids):
            d = node_depth.get(nid, 0)
            raw_left = int(tree.children_left[nid])
            raw_right = int(tree.children_right[nid])
            is_true_leaf = (raw_left == -1 and raw_right == -1)
            is_depth_cutoff = (max_depth is not None and d == max_depth)
            is_terminal = is_true_leaf or is_depth_cutoff

            # Node values and class predictions
            val0 = float(tree.value[nid][0][0])
            val1 = float(tree.value[nid][0][1])
            total_w = val0 + val1 if (val0 + val1) > 0 else 1.0
            samples = int(tree.n_node_samples[nid])
            norm_samples = round(samples * (val0 / total_w))
            fail_samples = samples - norm_samples
            pred_class = 1 if val1 > val0 else 0
            pred_name = "Failure" if pred_class == 1 else "Normal"
            gini = round(float(tree.impurity[nid]), 4)

            feat_name: Optional[str] = None
            feat_label: Optional[str] = None
            unit: Optional[str] = None
            raw_thresh: Optional[float] = None
            unscaled_thresh: Optional[float] = None
            cond_l: Optional[str] = None
            cond_r: Optional[str] = None

            if not is_terminal and not is_true_leaf:
                f_idx = int(tree.feature[nid])
                if 0 <= f_idx < len(self.feature_names):
                    feat_name = self.feature_names[f_idx]
                    feat_label = FEATURE_LABELS.get(feat_name, feat_name)
                    unit = FEATURE_UNITS.get(feat_name, "")
                    raw_thresh = round(float(tree.threshold[nid]), 4)
                    if f_idx >= 2 and len(self.scaler_scale) > (f_idx - 2):
                        unscaled = float(tree.threshold[nid]) * self.scaler_scale[f_idx - 2] + self.scaler_mean[f_idx - 2]
                        unscaled_thresh = round(unscaled, 2)
                        cond_l = f"≤ {unscaled_thresh} {unit}".strip()
                        cond_r = f"> {unscaled_thresh} {unit}".strip()
                    else:
                        unscaled_thresh = 0.5
                        variant = feat_label.replace("Product Type: ", "")
                        cond_l = f"≠ {variant}"
                        cond_r = f"= {variant}"

            actual_left = raw_left if (not is_terminal and raw_left in included_ids) else None
            actual_right = raw_right if (not is_terminal and raw_right in included_ids) else None

            nodes.append(
                TreeNode(
                    id=nid,
                    depth=d,
                    is_leaf=is_terminal,
                    feature=feat_name,
                    feature_label=feat_label,
                    threshold=raw_thresh,
                    threshold_unscaled=unscaled_thresh,
                    unit=unit,
                    condition_left=cond_l,
                    condition_right=cond_r,
                    gini=gini,
                    samples=samples,
                    class_counts=[norm_samples, fail_samples],
                    class_proportions=[round(val0 / total_w, 4), round(val1 / total_w, 4)],
                    predicted_class=pred_class,
                    predicted_class_name=pred_name,
                    left_child=actual_left,
                    right_child=actual_right,
                )
            )

            if actual_left is not None and cond_l:
                edges.append(
                    TreeEdge(
                        source=nid,
                        target=actual_left,
                        branch="left",
                        condition=cond_l,
                    )
                )
            if actual_right is not None and cond_r:
                edges.append(
                    TreeEdge(
                        source=nid,
                        target=actual_right,
                        branch="right",
                        condition=cond_r,
                    )
                )

        return RandomForestTreeResponse(
            tree_index=tree_index,
            total_estimators=total_estimators,
            node_count=len(nodes),
            max_depth=int(tree.max_depth),
            filtered_max_depth=max_depth,
            nodes=nodes,
            edges=edges,
        )
