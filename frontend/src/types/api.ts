export type LoadType = "Light_Load" | "Medium_Load" | "Maximum_Load";

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string;
  model_name: string;
  load_time_ms: number | null;
  memory_rss_mb: number;
  uptime_seconds: number;
}

export interface ModelInfoResponse {
  model_version: string;
  model_name: string;
  feature_set: string;
  best_params: Record<string, unknown>;
  classes: LoadType[];
  train_rows: number | null;
  input_features: string[];
  engineered_features: {
    timestamp_features: string[];
    measurement_features: string[];
    anomaly_indicator_features: string[];
  };
  input_schema: Record<string, unknown>;
}

export interface LoadRecordInput {
  Date_Time: string;
  Usage_kWh: number | null;
  "Lagging_Current_Reactive.Power_kVarh": number | null;
  Leading_Current_Reactive_Power_kVarh: number | null;
  "CO2(tCO2)": number | null;
  Lagging_Current_Power_Factor: number | null;
  Leading_Current_Power_Factor: number | null;
  NSM: number | null;
}

export interface PredictionResponse {
  Date_Time: string;
  prediction: LoadType;
  probabilities: Record<string, number>;
  model_version: string;
  inference_time_ms: number;
}

export interface BatchPredictionRow {
  Date_Time: string;
  prediction: LoadType;
  probabilities: Record<string, number>;
}

export interface BatchPredictionResponse {
  total_rows: number;
  model_version: string;
  inference_time_ms: number;
  class_distribution: Record<string, number>;
  predictions: BatchPredictionRow[];
}

export interface MetricsResponse {
  test_sample_count: number;
  train_dev_sample_count: number;
  model_configuration: {
    name: string;
    feature_set: string;
    best_cv_macro_f1: number;
    best_params: Record<string, unknown>;
    search_time_sec: number;
  };
  summary_metrics: {
    accuracy: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
    weighted_precision: number;
    weighted_recall: number;
    weighted_f1: number;
    f1_Light_Load: number;
    f1_Maximum_Load: number;
    f1_Medium_Load: number;
  };
  per_class_report: Record<
    string,
    { precision?: number; recall?: number; "f1-score"?: number; support?: number } | number
  >;
  confusion_matrix: {
    labels: string[];
    matrix: number[][];
  };
}

export interface AnalyticsResponse {
  model_version: string;
  data_quality?: {
    row_count: number;
    column_count: number;
    date_range: { min: string; max: string };
    class_counts: Record<string, number>;
    missing_counts: Record<string, number>;
    duplicate_timestamps: number;
  };
  model_comparison?: Array<{
    model: string;
    feature_set: string;
    status: string;
    accuracy: number | null;
    macro_precision: number | null;
    macro_recall: number | null;
    macro_f1: number | null;
    weighted_f1: number | null;
    train_time_sec: number | null;
  }>;
  explainability?: {
    permutation_importance_val: Array<{
      feature: string;
      importance_mean: number;
      importance_std: number;
    }>;
    test_error_summary: {
      total_test_rows: number;
      total_errors: number;
      overall_error_rate: number;
      weekday_error_rate: number;
      weekend_error_rate: number;
      confusion_pairs: Record<string, number>;
    };
  };
  historical_summary?: {
    hourly_load_distribution: Array<Record<string, number>>;
    monthly_load_distribution: Array<Record<string, number>>;
    usage_kwh_by_load_type: Array<{
      Load_Type: string;
      count: number;
      mean: number;
      median: number;
      min: number;
      max: number;
    }>;
  };
}

