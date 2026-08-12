const API_BASE = (
  import.meta.env.VITE_API_URL ||
  "https://truthlens-ai-production-2093.up.railway.app"
).replace(/\/$/, "");

import type {
  AnalysisResult,
  AnalyticsData,
  BatchResult,
  DatasetSampleRow,
  DatasetStatsResponse,
  DemoArticle,
  HealthStatus,
  HistoryDetail,
  HistoryPage,
  ModelPerformanceData,
  SettingsData,
} from "../types";


export function apiUrl(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${normalized}`;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options.headers as Record<string, string>) || {}),
  };

  const response = await fetch(apiUrl(path), { ...options, headers });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.error) message = body.error;
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  analyze: (headline: string, article: string, save = true) =>
    apiFetch<AnalysisResult>("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ headline, article, save }),
    }),

  analyzeHeadline: (headline: string) =>
    apiFetch<AnalysisResult>("/api/analyze/headline", {
      method: "POST",
      body: JSON.stringify({ headline }),
    }),

  batchAnalyze: (rows: { headline: string; article: string }[]) =>
    apiFetch<BatchResult>("/api/batch/analyze", {
      method: "POST",
      body: JSON.stringify({ rows }),
    }),

  history: (params: URLSearchParams) => apiFetch<HistoryPage>(`/api/history?${params.toString()}`),
  historyItem: (id: number) => apiFetch<HistoryDetail>(`/api/history/${id}`),
  historyReport: (id: number) => apiFetch<Record<string, unknown>>(`/api/history/report/${id}`),
  deleteHistoryItem: (id: number) =>
    apiFetch<{ status: string }>(`/api/history/${id}`, { method: "DELETE" }),
  clearHistory: () =>
    apiFetch<{ status: string; deleted: number }>("/api/history/clear", { method: "DELETE" }),
  reanalyze: (id: number) =>
    apiFetch<AnalysisResult>(`/api/history/${id}/reanalyze`, { method: "POST" }),
  exportHistory: () => apiFetch<{ items: HistoryDetail[] }>("/api/history/export"),

  analytics: (params?: { bucket?: string; days?: number }) => {
    const q = new URLSearchParams();
    if (params?.bucket) q.set("bucket", params.bucket);
    if (params?.days) q.set("days", String(params.days));
    const suffix = q.toString() ? `?${q.toString()}` : "";
    return apiFetch<AnalyticsData>(`/api/analytics${suffix}`);
  },

  modelPerformance: () => apiFetch<ModelPerformanceData>("/api/model-performance"),
  datasetStats: () => apiFetch<DatasetStatsResponse>("/api/dataset/stats"),
  datasetSamples: () => apiFetch<{ items: DatasetSampleRow[] }>("/api/dataset/samples"),
  samples: () => apiFetch<{ items: DemoArticle[] }>("/api/samples"),
  health: () => apiFetch<HealthStatus>("/api/health"),
  settings: () => apiFetch<SettingsData>("/api/settings"),
  updateSettings: (patch: Partial<SettingsData>) =>
    apiFetch<SettingsData>("/api/settings", { method: "PUT", body: JSON.stringify(patch) }),
};
