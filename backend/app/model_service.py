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
    RiskFactor,
)

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
