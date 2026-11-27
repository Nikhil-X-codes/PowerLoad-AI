import axios, { AxiosError } from "axios";
import type {
  AnalyticsResponse,
  BatchPredictionResponse,
  HealthResponse,
  LoadRecordInput,
  MetricsResponse,
  ModelInfoResponse,
  PredictionResponse,
} from "../types/api";

const baseURL = import.meta.env.VITE_API_BASE_URL || "";

export const apiClient = axios.create({
  baseURL,
  timeout: 45000,
});

export function isAbortError(err: unknown): boolean {
  return (
    (err instanceof AxiosError && err.code === "ERR_CANCELED") ||
    (err instanceof DOMException && err.name === "AbortError")
  );
}

export function extractApiErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = (err.response?.data as { detail?: unknown } | undefined)?.detail;
    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }
    if (Array.isArray(detail)) {
      return detail.map((item) => item?.msg || JSON.stringify(item)).join("; ");
    }
    if (err.code === "ECONNABORTED") {
      return "Request timed out while waiting for the backend (possible cold start). Please retry.";
    }
    if (err.code === "ERR_NETWORK") {
      return "Cannot reach the FastAPI backend. If the server is starting up, wait a few seconds and retry.";
    }
    return err.message;
  }
  if (err instanceof Error) {
    return err.message;
  }
  return "Unexpected error occurred.";
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>("/api/v1/health", { signal });
  return data;
}

export async function fetchModelInfo(signal?: AbortSignal): Promise<ModelInfoResponse> {
  const { data } = await apiClient.get<ModelInfoResponse>("/api/v1/model/info", { signal });
  return data;
}

export async function predictSingle(
  payload: LoadRecordInput,
  signal?: AbortSignal
): Promise<PredictionResponse> {
  const { data } = await apiClient.post<PredictionResponse>("/api/v1/predict", payload, {
    signal,
  });
  return data;
}

export async function predictBatchCsv(
  file: File,
  signal?: AbortSignal
): Promise<BatchPredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await apiClient.post<BatchPredictionResponse>(
    "/api/v1/predict/batch",
    form,
    {
      headers: { "Content-Type": "multipart/form-data" },
      signal,
    }
  );
  return data;
}

export async function fetchMetrics(signal?: AbortSignal): Promise<MetricsResponse> {
  const { data } = await apiClient.get<MetricsResponse>("/api/v1/metrics", { signal });
  return data;
}

export async function fetchAnalytics(signal?: AbortSignal): Promise<AnalyticsResponse> {
  const { data } = await apiClient.get<AnalyticsResponse>("/api/v1/analytics", { signal });
  return data;
}

