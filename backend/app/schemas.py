"""Pydantic schemas for MachineGuard AI API."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class MachineInput(BaseModel):
    """Input payload for a single machine prediction."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["L", "M", "H"] = Field(
        ...,
        description="Machine product variant: Low (L), Medium (M), or High (H) quality tier.",
        examples=["M"],
    )
    air_temperature_k: float = Field(
        ...,
        alias="air_temperature",
        ge=280.0,
        le=320.0,
        description="Air temperature in Kelvin [K]. Typically between 295K and 305K.",
        examples=[298.1],
    )
    process_temperature_k: float = Field(
        ...,
        alias="process_temperature",
        ge=290.0,
        le=330.0,
        description="Process temperature in Kelvin [K]. Typically between 305K and 315K.",
        examples=[308.6],
    )
    rotational_speed_rpm: float = Field(
        ...,
        alias="rotational_speed",
        ge=500.0,
        le=3500.0,
        description="Rotational speed of the spindle in rotations per minute [rpm]. Typically 1200-2800 rpm.",
        examples=[1551.0],
    )
    torque_nm: float = Field(
        ...,
        alias="torque",
        ge=0.0,
        le=120.0,
        description="Torque exerted by the motor in Newton-meters [Nm]. Typically 10-80 Nm.",
        examples=[42.8],
    )
    tool_wear_min: float = Field(
        ...,
        alias="tool_wear",
        ge=0.0,
        le=400.0,
        description="Tool wear in minutes of machining operation [min]. Tool replacement typically scheduled < 200 min.",
        examples=[110.0],
    )


class BatchMachineInput(BaseModel):
    """Batch prediction request."""

    machines: List[MachineInput] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of machine operating parameter records.",
    )


class EngineeredFeatureValues(BaseModel):
    """Engineered physical features calculated from raw telemetry."""

    temperature_difference: float = Field(
        ...,
        description="Process temperature minus Air temperature in Kelvin [K].",
    )
    mechanical_power_kw: float = Field(
        ...,
        description="Calculated mechanical power in kilowatts [kW].",
    )
    wear_load: float = Field(
        ...,
        description="Compound wear-load index (Tool wear * Torque) in min*Nm.",
    )


class RiskFactor(BaseModel):
    """Identified operational risk factor contributing to elevated failure probability."""

    feature: str
    observed_value: float
    unit: str
    status: Literal["Normal", "Elevated", "Critical"]
    message: str


class PredictionResponse(BaseModel):
    """Single machine failure prediction response."""

    is_failure: bool = Field(
        ...,
        description="Binary failure prediction based on the tuned decision threshold.",
    )
    failure_probability: float = Field(
        ...,
        description="Calibrated probability of machine failure (0.0 to 1.0).",
    )
    threshold_applied: float = Field(
        ...,
        description="Decision threshold applied for classification (tuned on validation split).",
    )
    risk_level: Literal["Low", "Medium", "High", "Critical"] = Field(
        ...,
        description="Categorical risk tier based on failure probability.",
    )
    engineered_features: EngineeredFeatureValues
    risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Key physical parameters identified as operating in dangerous stress bands.",
    )
    maintenance_recommendation: str = Field(
        ...,
        description="Actionable maintenance guidance.",
    )
    data_honesty_statement: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response summary."""

    total_records: int
    predicted_failures: int
    predicted_normals: int
    highest_risk_score: float
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """System health and status response."""

    status: str
    model_loaded: bool
    model_version: str
    selected_threshold: float
    data_honesty: str


class ModelMetadataResponse(BaseModel):
    """Model architecture and dataset metadata."""

    metadata: Dict[str, Any]


class ModelMetricsResponse(BaseModel):
    """Model evaluation and ablation metrics."""

    metrics: Dict[str, Any]


class TreeNode(BaseModel):
    """Individual node within a Random Forest decision tree."""

    id: int
    depth: int
    is_leaf: bool
    feature: Optional[str] = None
    feature_label: Optional[str] = None
    threshold: Optional[float] = None
    threshold_unscaled: Optional[float] = None
    unit: Optional[str] = None
    condition_left: Optional[str] = None
    condition_right: Optional[str] = None
    gini: float
    samples: int
    class_counts: List[int]
    class_proportions: List[float]
    predicted_class: int
    predicted_class_name: str
    left_child: Optional[int] = None
    right_child: Optional[int] = None


class TreeEdge(BaseModel):
    """Parent-to-child split connection in a decision tree."""

    source: int
    target: int
    branch: Literal["left", "right"]
    condition: str


class RandomForestTreeResponse(BaseModel):
    """Real extracted tree structure from the trained Random Forest model."""

    tree_index: int
    total_estimators: int
    node_count: int
    max_depth: int
    filtered_max_depth: Optional[int] = None
    nodes: List[TreeNode]
    edges: List[TreeEdge]
