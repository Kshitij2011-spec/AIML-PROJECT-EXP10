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

export interface BaselineMetrics {
  model_name: string;
  strategy: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  macro_f1: number;
  pr_auc: number;
  roc_auc: number;
  confusion_matrix: ConfusionMatrix;
  academic_note?: string;
}

export interface CrossValidationModelMetrics {
  fold_scores: number[];
  mean: number;
  std: number;
}

export interface CrossValidationResult {
  n_splits: number;
  scoring: string;
  partition: string;
  models: {
    "Logistic Regression (Engineered Features)": CrossValidationModelMetrics;
    "Random Forest (Engineered Features)": CrossValidationModelMetrics;
    [key: string]: CrossValidationModelMetrics;
  };
}

export interface AblationMetrics {
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  confusion_matrix: ConfusionMatrix;
}

export interface AblationModelEntry {
  engineered_features_included: boolean;
  model_type: string;
  validation_metrics_default_threshold: AblationMetrics;
}

export interface TreeNode {
  id: number;
  depth: number;
  is_leaf: boolean;
  feature?: string | null;
  feature_label?: string | null;
  threshold?: number | null;
  threshold_unscaled?: number | null;
  unit?: string | null;
  condition_left?: string | null;
  condition_right?: string | null;
  gini: number;
  samples: number;
  class_counts: number[];
  class_proportions: number[];
  predicted_class: number;
  predicted_class_name: "Normal" | "Failure" | string;
  left_child?: number | null;
  right_child?: number | null;
}

export interface TreeEdge {
  source: number;
  target: number;
  branch: "left" | "right";
  condition: string;
}

export interface RandomForestTreeResponse {
  tree_index: number;
  total_estimators: number;
  node_count: number;
  max_depth: number;
  filtered_max_depth?: number | null;
  nodes: TreeNode[];
  edges: TreeEdge[];
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
  total_estimators?: number;
  selected_threshold: number;
  default_threshold: number;
  test_samples: number;
  test_metrics: TestMetrics;
  baseline_comparison?: BaselineMetrics;
  cross_validation?: CrossValidationResult;
  ablation_comparison?: Record<string, AblationModelEntry>;
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
