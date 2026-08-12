import { useState } from "react";
import { FileText, CheckCircle2, XCircle, HelpCircle, Gauge } from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card, CardHeader } from "../components/ui/Card";
import { StatCard } from "../components/ui/StatCard";
import { EmptyState } from "../components/ui/EmptyState";
import { Skeleton } from "../components/ui/Skeleton";
import { Button } from "../components/ui/Button";
import { PredictionDonut } from "../components/charts/PredictionDonut";
import { TrendLine } from "../components/charts/TrendLine";
import { HistogramBars } from "../components/charts/HistogramBars";
import { useApi } from "../hooks/useApi";
import { api } from "../api/client";
import type { AnalyticsData, PredictionLabel } from "../types";
import { formatPercent } from "../lib/utils";

export function Analytics() {
  const [days, setDays] = useState(14);
  const { data, loading } = useApi<AnalyticsData>(() => api.analytics({ days }), [days]);
  const hasData = (data?.total_analyses ?? 0) > 0;

  const modelRows = Object.entries(data?.model_counts ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <div>
      <PageHeader
        title="Analytics"
        subtitle="Aggregated statistics computed from your real prediction history."
        actions={
          <div className="flex gap-2">
            {[7, 14, 30].map((d) => (
              <Button key={d} size="sm" variant={days === d ? "primary" : "outline"} onClick={() => setDays(d)}>
                {d}d
              </Button>
            ))}
          </div>
        }
      />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <StatCard label="Total Analyses" value={loading ? "…" : (data?.total_analyses ?? 0).toLocaleString()} icon={FileText} tone="primary" />
        <StatCard label="Real" value={loading ? "…" : (data?.label_counts?.REAL ?? 0).toLocaleString()} icon={CheckCircle2} tone="real" />
        <StatCard label="Fake" value={loading ? "…" : (data?.label_counts?.FAKE ?? 0).toLocaleString()} icon={XCircle} tone="fake" />
        <StatCard label="Uncertain" value={loading ? "…" : (data?.label_counts?.UNCERTAIN ?? 0).toLocaleString()} icon={HelpCircle} tone="uncertain" />
        <StatCard label="Average Confidence" value={loading ? "…" : formatPercent(data?.average_confidence ?? 0)} icon={Gauge} />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader title="Prediction Distribution" />
          <div className="p-4">
            {loading ? (
              <Skeleton className="h-[240px] w-full" />
            ) : !hasData ? (
              <EmptyState compact title="No analyses yet" />
            ) : (
              <PredictionDonut
                data={(Object.keys(data!.label_counts) as PredictionLabel[]).map((label) => ({
                  name: label,
                  value: data!.label_counts[label],
                }))}
              />
            )}
          </div>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader title="Predictions Over Time" subtitle={`Daily counts over the last ${days} days`} />
          <div className="p-4">
            {loading ? (
              <Skeleton className="h-[260px] w-full" />
            ) : !hasData ? (
              <EmptyState compact title="No trend data" />
            ) : (
              <TrendLine data={data!.trend} />
            )}
          </div>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Confidence Distribution" subtitle="Number of analyses per confidence band" />
          <div className="p-4">
            {loading ? (
              <Skeleton className="h-[240px] w-full" />
            ) : !hasData ? (
              <EmptyState compact title="No confidence data" />
            ) : (
              <HistogramBars data={data!.confidence_histogram} />
            )}
          </div>
        </Card>

        <Card>
          <CardHeader title="Model Usage" subtitle="Predictions per model" />
          <div className="p-4">
            {loading ? (
              <Skeleton className="h-[240px] w-full" />
            ) : !modelRows.length ? (
              <EmptyState compact title="No model data" />
            ) : (
              <div className="space-y-3">
                {modelRows.map(([model, count]) => {
                  const max = Math.max(...modelRows.map(([, c]) => c));
                  return (
                    <div key={model}>
                      <div className="mb-1 flex justify-between text-xs">
                        <span className="font-medium text-slate-700 dark:text-slate-200">{model}</span>
                        <span className="tabular-nums text-slate-400">{count.toLocaleString()}</span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                        <div className="h-full rounded-full bg-primary-500" style={{ width: `${(count / max) * 100}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
