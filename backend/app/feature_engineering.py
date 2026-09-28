"""Feature engineering definitions and scikit-learn compatible transformers for MachineGuard AI."""

import math
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Raw column definitions
CATEGORICAL_FEATURES: List[str] = ["Type"]

NUMERIC_FEATURES: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Engineered feature definitions
ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "temperature_difference",
    "mechanical_power_kw",
    "wear_load",
]

# Leakage and identifier columns that must NEVER be used as predictors
DROP_COLUMNS: List[str] = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

TARGET_COLUMN: str = "Machine failure"

# Feature metadata documentation for reproducibility and UI inspection
ENGINEERED_FEATURE_METADATA = {
    "temperature_difference": {
        "formula": "Process temperature [K] - Air temperature [K]",
        "unit": "K",
        "description": "Thermal dissipation gradient between machine process and ambient air. Elevated differences indicate heat dissipation stress (HDF risk).",
    },
    "mechanical_power_kw": {
        "formula": "Torque [Nm] * Rotational speed [rpm] * 2 * pi / 60 / 1000",
        "unit": "kW",
        "description": "Actual mechanical power transferred to the workpiece in kilowatts. Extreme values correlate with power failure (PWF risk).",
    },
    "wear_load": {
        "formula": "Tool wear [min] * Torque [Nm]",
        "unit": "min*Nm",
        "description": "Compound mechanical abrasion indicator. High torque exerted on an already degraded tool correlates with tool wear failure (TWF) and overstrain (OSF).",
    },
}


def calculate_engineered_features(
    process_temp_k: float,
    air_temp_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> Dict[str, float]:
    """Helper to compute engineered features from scalar inputs."""
    temp_diff = float(process_temp_k - air_temp_k)
    mech_power = float(torque_nm * rotational_speed_rpm * (2.0 * math.pi) / 60000.0)
    wear_load = float(tool_wear_min * torque_nm)
    return {
        "temperature_difference": temp_diff,
        "mechanical_power_kw": mech_power,
        "wear_load": wear_load,
    }


class IndustrialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for engineering industrial predictive maintenance features.

    When `include_engineered=True`, adds:
      1. temperature_difference = Process temperature [K] - Air temperature [K]
      2. mechanical_power_kw = Torque [Nm] * Rotational speed [rpm] * 2 * pi / 60 / 1000
      3. wear_load = Tool wear [min] * Torque [Nm]
    When `include_engineered=False`, passes through only the original usable features (used for ablation).
    """

    def __init__(self, include_engineered: bool = True):
        self.include_engineered = include_engineered

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        else:
            # If a numpy array or record list is passed
            expected_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
            df = pd.DataFrame(X, columns=expected_cols)

        # Standardize column types if present
        if "Type" in df.columns:
            df["Type"] = df["Type"].astype(str)
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if self.include_engineered:
            # 1. Temperature difference (Process - Air)
            df["temperature_difference"] = (
                df["Process temperature [K]"] - df["Air temperature [K]"]
            )
            # 2. Mechanical Power (kW) = Torque (Nm) * Speed (rad/s) / 1000
            # Speed (rad/s) = rpm * 2 * pi / 60
            df["mechanical_power_kw"] = (
                df["Torque [Nm]"]
                * df["Rotational speed [rpm]"]
                * (2.0 * np.pi)
                / 60000.0
            )
            # 3. Wear-load = Tool wear (min) * Torque (Nm)
            df["wear_load"] = df["Tool wear [min]"] * df["Torque [Nm]"]

        return df

    def get_feature_names_out(self, input_features=None):
        base = CATEGORICAL_FEATURES + NUMERIC_FEATURES
        if self.include_engineered:
            return np.array(base + ENGINEERED_NUMERIC_FEATURES)
        return np.array(base)
