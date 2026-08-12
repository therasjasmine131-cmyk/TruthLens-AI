import { useEffect, useState } from "react";
import { GitCompareArrows } from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Select } from "../components/ui/Input";
import { EmptyState } from "../components/ui/EmptyState";
import { PredictionBadge } from "../components/ui/PredictionBadge";
import { useApi } from "../hooks/useApi";
import { api } from "../api/client";
import type { HistoryItem } from "../types";
import { formatDateTime, formatPercent } from "../lib/utils";

const COMPARE_KEYS: { key: string; label: string }[] = [
  { key: "prediction", label: "Prediction" },
  { key: "confidence", label: "Confidence" },
  { key: "word_count", label: "Word Count" },
  { key: "character_count", label: "Character Count" },
  { key: "sentence_count", label: "Sentence Count" },
  { key: "model_name", label: "Model" },
];

export function Compare() {
  const { data: history, loading } = useApi<{ items: HistoryItem[] }>(
    () => api.history(new URLSearchParams({ per_page: "100" })),
    []
  );
  const [leftId, setLeftId] = useState<number | "">("");
  const [rightId, setRightId] = useState<number | "">("");

  const left = history?.items.find((i) => i.id === leftId) ?? null;
  const right = history?.items.find((i) => i.id === rightId) ?? null;

  useEffect(() => {
    if (!leftId && (history?.items?.length ?? 0)) setLeftId(history?.items?.[0]?.id ?? "");
    if (!rightId && (history?.items?.length ?? 0) > 1) setRightId(history?.items?.[1]?.id ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history]);

  const options = (exclude: number) =>
    (history?.items ?? []).filter((i) => i.id !== exclude).map((i) => ({
      value: String(i.id),
      label: `${i.headline ? i.headline.slice(0, 60) : "(no headline)"} · ${i.prediction} · ${formatDateTime(i.created_at)}`,
    }));

  return (
    <div>
      <PageHeader
        title="Compare Analyses"
        subtitle="Select two stored analyses and view them side by side."
      />

      {loading ? (
        <Card className="p-8 text-center text-sm text-slate-400">Loading history…</Card>
      ) : !history?.items.length ? (
        <Card className="p-8">
          <EmptyState
            icon={<GitCompareArrows size={22} />}
            title="No analyses to compare"
            description="Run at least two analyses and they will be available here."
          />
        </Card>
      ) : (
        <>
          <Card className="mb-6 p-5">
            <div className="grid gap-4 sm:grid-cols-2">
              <Select
                label="Analysis A"
                value={leftId ? String(leftId) : ""}
                onChange={(e) => setLeftId(Number(e.target.value))}
                options={options(rightId ? Number(rightId) : -1)}
              />
              <Select
                label="Analysis B"
                value={rightId ? String(rightId) : ""}
                onChange={(e) => setRightId(Number(e.target.value))}
                options={options(leftId ? Number(leftId) : -1)}
              />
            </div>
          </Card>

          {left && right ? (
            <>
              <div className="grid gap-6 lg:grid-cols-2 animate-fade-in">
                <ComparisonCard title="Analysis A" item={left} />
                <ComparisonCard title="Analysis B" item={right} />
              </div>

              <Card className="mt-6 overflow-hidden">
            <div className="border-b border-slate-200 px-5 py-3 dark:border-slate-800">
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Side-by-side comparison</h3>
            </div>
            <table className="w-full text-left text-xs">
              <tbody>
                {COMPARE_KEYS.map(({ key, label }) => {
                  const a = (left as unknown as Record<string, unknown>)?.[key];
                  const b = (right as unknown as Record<string, unknown>)?.[key];
                  const differ = key === "prediction" ? a !== b : Math.abs(Number(a ?? 0) - Number(b ?? 0)) > 0.0001;
                  return (
                    <tr key={key} className="border-b border-slate-100 last:border-0 dark:border-slate-800/60">
                      <td className="w-1/4 px-5 py-3 font-medium text-slate-500 dark:text-slate-400">{label}</td>
                      <td className={`w-1/4 px-5 py-3 ${differ ? "bg-emerald-50/50 dark:bg-emerald-950/20" : ""}`}>
                        {key === "prediction" ? <PredictionBadge label={a as HistoryItem["prediction"]} size="sm" /> : key === "confidence" ? formatPercent(Number(a)) : String(a ?? "—")}
                      </td>
                      <td className={`w-1/4 px-5 py-3 ${differ ? "bg-rose-50/50 dark:bg-rose-950/20" : ""}`}>
                        {key === "prediction" ? <PredictionBadge label={b as HistoryItem["prediction"]} size="sm" /> : key === "confidence" ? formatPercent(Number(b)) : String(b ?? "—")}
                      </td>
                      <td className="w-1/4 px-5 py-3 text-right">
                        {differ ? <span className="chip border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400">differs</span> : <span className="chip border-slate-200 bg-slate-50 text-slate-500 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-400">same</span>}
                      </td>
                    </tr>
                  );
                })}
                <tr className="border-b border-slate-100 last:border-0 dark:border-slate-800/60">
                  <td className="px-5 py-3 font-medium text-slate-500 dark:text-slate-400">Top Keywords</td>
                  <td className="px-5 py-3">{left.top_keywords.slice(0, 5).join(", ") || "—"}</td>
                  <td className="px-5 py-3">{right.top_keywords.slice(0, 5).join(", ") || "—"}</td>
                  <td />
                </tr>
                <tr>
                  <td className="px-5 py-3 font-medium text-slate-500 dark:text-slate-400">Date</td>
                  <td className="px-5 py-3">{formatDateTime(left.created_at)}</td>
                  <td className="px-5 py-3">{formatDateTime(right.created_at)}</td>
                  <td />
                </tr>
              </tbody>
              </table>
              </Card>
            </>
          ) : null}
        </>
      )}
    </div>
  );
}

function ComparisonCard({ title, item }: { title: string; item: HistoryItem }) {
  const rows = [
    ["Prediction", <PredictionBadge key="p" label={item.prediction} size="sm" />],
    ["Confidence", formatPercent(item.confidence)],
    ["Words", item.word_count.toLocaleString()],
    ["Characters", item.character_count.toLocaleString()],
    ["Sentences", item.sentence_count.toLocaleString()],
    ["Model", item.model_name],
    ["Date", formatDateTime(item.created_at)],
  ] as const;
  return (
    <Card className="overflow-hidden">
      <div className="border-b border-slate-200 px-5 py-3 dark:border-slate-800">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{title}</h3>
      </div>
      <div className="max-h-[320px] overflow-y-auto px-5 py-2">
        {rows.map(([label, value]) => (
          <div key={String(label)} className="flex items-center justify-between border-b border-slate-100 py-2.5 text-xs last:border-0 dark:border-slate-800/60">
            <span className="text-slate-500 dark:text-slate-400">{label}</span>
            <span className="font-medium text-slate-800 dark:text-slate-100">{value}</span>
          </div>
        ))}
        <div className="py-2.5">
          <p className="mb-1 text-xs text-slate-500 dark:text-slate-400">Top keywords</p>
          <p className="text-xs font-medium text-slate-800 dark:text-slate-100">
            {item.top_keywords.length ? item.top_keywords.slice(0, 8).join(", ") : "—"}
          </p>
        </div>
      </div>
    </Card>
  );
}
