import { useState } from "react";
import { Database, CheckCircle2, XCircle, AlertTriangle, Copy, FileText, Ruler } from "lucide-react";
import { useApi } from "../hooks/useApi";
import { api } from "../api/client";
import { PageHeader } from "../components/ui/PageHeader";
import { Card, CardHeader } from "../components/ui/Card";
import { StatCard } from "../components/ui/StatCard";
import { EmptyState } from "../components/ui/EmptyState";
import { Skeleton } from "../components/ui/Skeleton";
import { PredictionDonut } from "../components/charts/PredictionDonut";
import { HistogramBars } from "../components/charts/HistogramBars";
import type { DatasetStatsResponse, DatasetSampleRow } from "../types";

export function DatasetExplorer() {
  const { data, loading, error } = useApi<DatasetStatsResponse>(() => api.datasetStats(), []);
  const { data: sampleData } = useApi<{ items: DatasetSampleRow[] }>(() => api.datasetSamples(), []);
  const [filter, setFilter] = useState<"ALL" | "REAL" | "FAKE">("ALL");

  if (loading) {
    return (
      <div>
        <PageHeader title="Dataset Explorer" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <PageHeader title="Dataset Explorer" />
        <Card className="p-8">
          <EmptyState title="Dataset not found" description={error} />
        </Card>
      </div>
    );
  }

  const s = data!.stats;
  const rows = (sampleData?.items ?? []).filter((r) => filter === "ALL" || r.label === filter);

  return (
    <div>
      <PageHeader
        title="Dataset Explorer"
        subtitle={`Source: ${s.source}${data?.using_raw ? " (full ISOT)" : ""}`}
      />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Total Records" value={s.total_records.toLocaleString()} icon={Database} tone="primary" />
        <StatCard label="REAL Articles" value={s.real_count.toLocaleString()} icon={CheckCircle2} tone="real" />
        <StatCard label="FAKE Articles" value={s.fake_count.toLocaleString()} icon={XCircle} tone="fake" />
        <StatCard label="Class Balance" value={`${s.fake_count ? Math.round((s.real_count / (s.real_count + s.fake_count)) * 100) : 0}% / ${s.fake_count ? Math.round((s.fake_count / (s.real_count + s.fake_count)) * 100) : 0}%`} icon={AlertTriangle} />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader title="Class Distribution" />
          <div className="p-4">
            <PredictionDonut
              height={220}
              data={[
                { name: "REAL", value: s.real_count },
                { name: "FAKE", value: s.fake_count },
              ]}
            />
          </div>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader title="Article Length Distribution" subtitle="Character counts across the dataset" />
          <div className="p-4">
            <HistogramBars data={s.length_histogram} color="#6366f1" />
          </div>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader title="Data Quality" />
          <div className="grid grid-cols-1 gap-3 p-5">
            <QualityRow label="Missing Values" value={s.missing_values.toLocaleString()} icon={AlertTriangle} ok={s.missing_values === 0} />
            <QualityRow label="Duplicate Articles" value={s.duplicate_count.toLocaleString()} icon={Copy} ok={s.duplicate_count === 0} />
            <QualityRow label="Average Length" value={`${s.average_article_length.toLocaleString()} chars`} icon={Ruler} ok />
            <QualityRow label="Sample Preview" value={`${rows.length} rows shown`} icon={FileText} ok />
          </div>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader
            title="Dataset Preview"
            subtitle="A small, safe preview of records"
            action={
              <div className="flex gap-1.5">
                {(["ALL", "REAL", "FAKE"] as const).map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={`rounded-md px-2.5 py-1 text-[11px] font-medium ${
                      filter === f
                        ? "bg-primary-600 text-white"
                        : "bg-slate-100 text-slate-500 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300"
                    }`}
                  >
                    {f}
                  </button>
                ))}
              </div>
            }
          />
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-xs">
              <thead className="border-b border-slate-200 text-[11px] uppercase tracking-wide text-slate-400 dark:border-slate-800">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Label</th>
                  <th className="px-4 py-2.5 font-medium">Headline</th>
                  <th className="px-4 py-2.5 font-medium">Subject</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={i} className="border-b border-slate-100 last:border-0 dark:border-slate-800/60">
                    <td className="px-4 py-2.5">
                      <span className={`chip ${row.label === "REAL" ? "border-emerald-200 bg-emerald-50 text-emerald-600 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-400" : "border-rose-200 bg-rose-50 text-rose-600 dark:border-rose-800 dark:bg-rose-950 dark:text-rose-400"}`}>
                        {row.label}
                      </span>
                    </td>
                    <td className="max-w-[320px] truncate px-4 py-2.5 font-medium text-slate-700 dark:text-slate-200" title={row.headline}>
                      {row.headline}
                    </td>
                    <td className="px-4 py-2.5 text-slate-400">{row.subject}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <p className="mt-4 text-[11px] text-slate-400 dark:text-slate-500">
        Statistics are computed from the local dataset files (ml/data/raw/True.csv and Fake.csv, or the
        bundled sample). No fabricated values.
      </p>
    </div>
  );
}

function QualityRow({ label, value, icon: Icon, ok }: { label: string; value: string; icon: typeof AlertTriangle; ok: boolean }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-slate-200 px-3 py-2.5 dark:border-slate-800">
      <span className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
        <Icon size={14} className={ok ? "text-emerald-500" : "text-amber-500"} />
        {label}
      </span>
      <span className={`text-xs font-semibold tabular-nums ${ok ? "text-slate-800 dark:text-slate-100" : "text-amber-600"}`}>
        {value}
      </span>
    </div>
  );
}
