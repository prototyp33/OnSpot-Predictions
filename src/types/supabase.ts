export interface Model {
  model_id: string;
  model_type: string;
  training_date: string;
  parameters: Record<string, any>;
  metrics: Record<string, number>;
  description?: string;
  version?: string;
  status?: string;
}

export interface Prediction {
  model_id: string;
  location_id: string;
  timestamp: string;
  predicted_occupancy: number;
  actual_occupancy: number;
  confidence?: number;
  error_margin?: number;
}

export interface DriftAnalysis {
  model_id: string;
  feature_name: string;
  drift_score: number;
  p_value: number;
  timestamp: string;
  baseline_timestamp?: string;
  alert_threshold?: number;
}

export interface ModelMetric {
  model_id: string;
  metric_name: string;
  value: number;
  timestamp: string;
  context?: string;
  window_size?: number;
}

export type Tables = {
  models: Model;
  predictions: Prediction;
  drift_analysis: DriftAnalysis;
  model_metrics: ModelMetric;
}; 