import axios, { AxiosProgressEvent } from "axios";
import type {
  ComparisonResponse,
  HealthResponse,
  ModelCatalog,
  PerSubregionMetrics,
  SessionInfo,
  UploadAck,
  UploadKind,
} from "@/lib/types";

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120_000,
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    const detail =
      err?.response?.data?.detail ??
      err?.message ??
      "Request failed";
    return Promise.reject(new Error(typeof detail === "string" ? detail : JSON.stringify(detail)));
  },
);

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>("/health");
  return data;
}

export async function getModels(): Promise<ModelCatalog> {
  const { data } = await api.get<ModelCatalog>("/models");
  return data;
}

export async function createSession(): Promise<SessionInfo> {
  const { data } = await api.post<SessionInfo>("/sessions");
  return data;
}

export async function getSession(sid: string): Promise<SessionInfo> {
  const { data } = await api.get<SessionInfo>(`/sessions/${sid}`);
  return data;
}

export async function deleteSession(sid: string): Promise<void> {
  await api.delete(`/sessions/${sid}`);
}

export async function uploadFile(
  sid: string,
  kind: UploadKind,
  file: File,
  onProgress?: (pct: number) => void,
): Promise<UploadAck> {
  const form = new FormData();
  form.append("kind", kind);
  form.append("file", file);
  const { data } = await api.post<UploadAck>(`/sessions/${sid}/upload`, form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e: AxiosProgressEvent) => {
      if (e.total && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
  return data;
}

export async function runPrediction(sid: string, model_key: string): Promise<PerSubregionMetrics> {
  const { data } = await api.post<PerSubregionMetrics>(`/sessions/${sid}/predict`, {
    model_key,
  });
  return data;
}

export async function getPredictionMetrics(
  sid: string,
  modelKey: string,
): Promise<PerSubregionMetrics> {
  const { data } = await api.get<PerSubregionMetrics>(
    `/sessions/${sid}/predictions/${modelKey}/metrics`,
  );
  return data;
}

export async function compareModels(
  sid: string,
  model_keys: string[],
): Promise<ComparisonResponse> {
  const { data } = await api.post<ComparisonResponse>(`/sessions/${sid}/compare`, { model_keys });
  return data;
}

export async function compareAll(sid: string): Promise<ComparisonResponse> {
  const { data } = await api.post<ComparisonResponse>(`/sessions/${sid}/compare-all`, {});
  return data;
}

export function uploadUrl(sid: string, kind: UploadKind) {
  return `${API_BASE_URL}/sessions/${sid}/uploads/${kind}`;
}

export function maskUrl(sid: string, modelKey: string) {
  return `${API_BASE_URL}/sessions/${sid}/predictions/${modelKey}/mask.nii.gz`;
}

export function reportUrl(sid: string, modelKeys: string[]) {
  const qs = modelKeys.length ? `?models=${modelKeys.join(",")}` : "";
  return `${API_BASE_URL}/sessions/${sid}/report.pdf${qs}`;
}
