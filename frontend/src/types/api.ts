export type MachineType = "L" | "M" | "H";

export interface MachineInputData {
  type: MachineType;
  air_temperature: number;
  process_temperature: number;
  rotational_speed: number;
  torque: number;
  tool_wear: number;
}

export interface EngineeredFeatures {
  temperature_difference: number;
  mechanical_power_kw: number;
  wear_load: number;
}

export type RiskTier = "Low" | "Medium" | "High" | "Critical";

export interface RiskFactor {
  feature: string;
  observed_value: number;
  unit: string;
  status: "Normal" | "Elevated" | "Critical";
  message: string;
}

export interface PredictionResult {
  is_failure: boolean;
  failure_probability: number;
  threshold_applied: number;
  risk_level: RiskTier;
  engineered_features: EngineeredFeatures;
  risk_factors: RiskFactor[];
  maintenance_recommendation: string;
  data_honesty_statement: string;
}

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface TestMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  confusion_matrix: ConfusionMatrix;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface FeatureDefinition {
  formula: string;
  unit: string;
  description: string;
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  algorithm: string;
  selected_threshold: number;
  default_threshold: number;
  test_samples: number;
  test_metrics: TestMetrics;
  feature_importances: FeatureImportanceItem[];
  engineered_feature_definitions: Record<string, FeatureDefinition>;
  data_honesty_statement: string;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_version: string;
  selected_threshold: number;
  data_honesty: string;
}
