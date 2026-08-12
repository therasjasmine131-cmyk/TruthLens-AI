import { Link } from "react-router-dom";
import {
  FileText,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Gauge,
  ScanSearch,
  Server,
  Database,
  Cpu,
  ArrowRight,
  Layers,
} from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card, CardHeader } from "../components/ui/Card";
import { StatCard } from "../components/ui/StatCard";
import { EmptyState } from "../components/ui/EmptyState";
import { Skeleton } from "../components/ui/Skeleton";
import { Button } from "../components/ui/Button";
import { PredictionBadge } from "../components/ui/PredictionBadge";
import { PredictionDonut } from "../components/charts/PredictionDonut";
import { TrendLine } from "../components/charts/TrendLine";
import { HistogramBars } from "../components/charts/HistogramBars";
import { useApi } from "../hooks/useApi";
import { api } from "../api/client";
import type { AnalyticsData, HealthStatus, HistoryItem, PredictionLabel } from "../types";
import { formatDateTime, formatPercent } from "../lib/utils";

export function Dashboard() {
  const { data: analytics, loading: loadingAnalytics } = useApi<AnalyticsData>(() => api.analytics({ days: 14 }), []);
  const { data: recent, loading: loadingRecent } = useApi<{ items: HistoryItem[] }>(() => api.history(new URLSearchParams({ per_page: "6" })), []);
  const { data: health } = useApi<HealthStatus>(() => api.health(), []);

  const hasData = (analytics?.total_analyses ?? 0) > 0;

  return (
    <div>
      <PageHeader
        title="News Intelligence Dashboard"
        subtitle="Monitor your news analysis activity and model performance."
        actions={
          <Link to="/analyze">
            <Button icon={<ScanSearch size={15} />}>Quick Analyze</Button>
          </Link>
        }
      />

      {/* ---- KPIs ---- */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <StatCard
          label="Total Analyses"
          value={loadingAnalytics ? "…" : (analytics?.total_analyses ?? 0).toLocaleString()}
          icon={FileText}
          tone="primary"
        />
        <StatCard
          label="Real Predictions"
          value={loadingAnalytics ? "…" : (analytics?.label_counts?.REAL ?? 0).toLocaleString()}
          icon={CheckCircle2}
          tone="real"
        />
        <StatCard
          label="Fake Predictions"
          value={loadingAnalytics ? "…" : (analytics?.label_counts?.FAKE ?? 0).toLocaleString()}
          icon={XCircle}
          tone="fake"
        />
        <StatCard
          label="Uncertain"
          value={loadingAnalytics ? "…" : (analytics?.label_counts?.UNCERTAIN ?? 0).toLocaleString()}
          icon={HelpCircle}
          tone="uncertain"
        />
        <StatCard
          label="Average Confidence"
          value={loadingAnalytics ? "…" : formatPercent(analytics?.average_confidence ?? 0)}
          icon={Gauge}
        />
      </div>

      {/* ---- Charts row ---- */}
      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader title="Prediction Distribution" subtitle="Share of predictions by class" />
          <div className="p-4">
            {loadingAnalytics ? (
              <Skeleton className="mx-auto h-[240px] w-full" />
            ) : !hasData ? (
              <EmptyState
                compact
                title="No analyses yet"
                description="Analyze your first article to start building your analytics."
                action={<Link to="/analyze"><Button size="sm" icon={<ScanSearch size={14} />}>Analyze News</Button></Link>}
              />
            ) : (
              <PredictionDonut
                data={(Object.keys(analytics?.label_counts ?? {}) as PredictionLabel[]).map((label) => ({
                  name: label,
                  value: analytics!.label_counts[label],
                }))}
              />
            )}
          </div>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader title="Prediction Trend" subtitle="Analyses over the last 14 days" />
          <div className="p-4">
            {loadingAnalytics ? (
              <Skeleton className="h-[260px] w-full" />
            ) : !hasData ? (
              <EmptyState
                compact
                title="No trend data"
                description="Run some analyses to see how your prediction activity evolves over time."
              />
            ) : (
              <TrendLine data={analytics!.trend} />
            )}
          </div>
        </Card>
      </div>

      {/* ---- Confidence + recent ---- */}
      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader title="Confidence Distribution" subtitle="Counts by confidence band" />
          <div className="p-4">
            {loadingAnalytics ? (
              <Skeleton className="h-[240px] w-full" />
            ) : !hasData ? (
              <EmptyState compact title="No confidence data" />
            ) : (
              <HistogramBars data={analytics!.confidence_histogram} />
            )}
          </div>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader
            title="Recent Analyses"
            subtitle="Latest predictions from the database"
            action={
              <Link to="/history" className="flex items-center gap-1 text-xs font-medium text-primary-600 hover:text-primary-700 dark:text-primary-400">
                View all <ArrowRight size={13} />
              </Link>
            }
          />
          <div className="overflow-x-auto">
            {loadingRecent ? (
              <div className="p-4"><Skeleton className="h-40 w-full" /></div>
            ) : !recent?.items.length ? (
              <EmptyState compact title="No predictions stored yet" />
            ) : (
              <table className="w-full text-left text-xs">
                <thead className="border-b border-slate-200 text-[11px] uppercase tracking-wide text-slate-400 dark:border-slate-800">
                  <tr>
                    <th className="px-4 py-2.5 font-medium">Headline</th>
                    <th className="px-4 py-2.5 font-medium">Prediction</th>
                    <th className="px-4 py-2.5 font-medium">Confidence</th>
                    <th className="px-4 py-2.5 font-medium">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.items.map((item) => (
                    <tr key={item.id} className="border-b border-slate-100 last:border-0 dark:border-slate-800/60">
                      <td className="max-w-[260px] truncate px-4 py-2.5 font-medium text-slate-700 dark:text-slate-200">
                        {item.headline || "(no headline)"}
                      </td>
                      <td className="px-4 py-2.5">
                        <PredictionBadge label={item.prediction} size="sm" />
                      </td>
                      <td className="px-4 py-2.5 tabular-nums text-slate-600 dark:text-slate-300">
                        {formatPercent(item.confidence)}
                      </td>
                      <td className="px-4 py-2.5 text-slate-400">{formatDateTime(item.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Card>
      </div>

      {/* ---- Model health ---- */}
      <div className="mt-6">
        <Card>
          <CardHeader title="Model Health" subtitle="Live status of the deployed ML stack" />
          <div className="grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-4">
            <HealthTile
              icon={Server}
              label="Backend"
              value={health?.backend?.status === "healthy" ? "HEALTHY" : "OFFLINE"}
              ok={health?.backend?.status === "healthy"}
            />
            <HealthTile
              icon={Database}
              label="Database"
              value={health?.database?.healthy ? "CONNECTED" : "ERROR"}
              ok={health?.database?.healthy === true}
            />
            <HealthTile
              icon={Cpu}
              label="ML Model"
              value={health?.model?.status === "online" ? "ONLINE" : health?.model?.status === "error" ? "ERROR" : "OFFLINE"}
              ok={health?.model?.status === "online"}
              sub={health?.model_name}
            />
            <HealthTile
              icon={Layers}
              label="TF-IDF Vectorizer"
              value={health?.vectorizer?.ready ? "READY" : "UNAVAILABLE"}
              ok={health?.vectorizer?.ready === true}
            />
          </div>
        </Card>
      </div>
    </div>
  );
}

function HealthTile({
  icon: Icon,
  label,
  value,
  ok,
  sub,
}: {
  icon: typeof Server;
  label: string;
  value: string;
  ok: boolean;
  sub?: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-slate-200 px-4 py-3 dark:border-slate-800">
      <div className={`rounded-lg p-2 ${ok ? "bg-emerald-50 text-emerald-600 dark:bg-emerald-950 dark:text-emerald-400" : "bg-rose-50 text-rose-600 dark:bg-rose-950 dark:text-rose-400"}`}>
        <Icon size={16} />
      </div>
      <div className="min-w-0">
        <p className="text-[11px] text-slate-500 dark:text-slate-400">{label}</p>
        <p className={`text-sm font-semibold ${ok ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
          {value}
        </p>
        {sub && <p className="truncate text-[10px] text-slate-400">{sub}</p>}
      </div>
    </div>
  );
}
