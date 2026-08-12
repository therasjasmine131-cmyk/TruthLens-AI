import { useParams, useNavigate } from "react-router-dom";
import { Printer, ArrowLeft, FileDown, FlaskConical } from "lucide-react";
import { useApi } from "../hooks/useApi";
import { api } from "../api/client";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import type { PredictionLabel } from "../types";
import { formatDateTime, formatPercent } from "../lib/utils";

interface ReportData {
  history_id: number;
  headline: string;
  article_text: string;
  prediction: PredictionLabel;
  confidence: number;
  confidence_level: string;
  probabilities: { real: number; fake: number; uncertain: number };
  model_raw?: { p_real: number; p_fake: number };
  article_stats: Record<string, number>;
  keywords: { term: string; score?: number }[];
  explanation: { note?: string; features?: { term: string; contribution: number }[] };
  model_info: Record<string, unknown>;
  created_at: string | null;
  disclaimer: string;
}

const STAT_LABELS: Record<string, string> = {
  word_count: "Word Count",
  character_count: "Character Count",
  sentence_count: "Sentence Count",
  average_sentence_length: "Avg Sentence Length",
  unique_words: "Unique Words",
  vocabulary_richness: "Vocabulary Richness",
  capitalized_words: "Capitalized Words",
  exclamation_marks: "Exclamation Marks",
  question_marks: "Question Marks",
};

export function Report() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data, loading, error } = useApi<ReportData>(
    () => api.historyReport(Number(id)) as unknown as Promise<ReportData>,
    [id]
  );

  const downloadHtml = () => {
    if (!data) return;
    const el = document.getElementById("report-root");
    if (!el) return;
    const content = el.outerHTML;
    const doc = `<!doctype html><html><head><meta charset="utf-8"><title>TruthLens AI Report</title>
<style>body{font-family:Inter,system-ui,sans-serif;max-width:820px;margin:32px auto;padding:0 24px;color:#0f172a;font-size:14px;line-height:1.6}
table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #e2e8f0;padding:8px 10px;text-align:left;font-size:13px}
h1{font-size:20px}h2{font-size:15px;margin-top:24px;border-bottom:2px solid #eef2ff;padding-bottom:6px}
.badge{display:inline-block;padding:4px 12px;border-radius:8px;font-weight:700;color:#fff}.REAL{background:#059669}.FAKE{background:#dc2626}.UNCERTAIN{background:#d97706}
.banner{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:12px;margin-top:24px;font-size:12px;color:#92400e}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.metric{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px}
.metric b{display:block;font-size:16px}.metric span{font-size:11px;color:#64748b}
@media print{body{margin:0;padding:0}}</style></head><body>${content}</body></html>`;
    const blob = new Blob([doc], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `truthlens-report-${data.history_id}.html`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div>
        <PageHeader title="Analysis Report" />
        <Card className="p-10 text-center text-sm text-slate-400">Loading report…</Card>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div>
        <PageHeader title="Analysis Report" />
        <Card className="p-8">
          <EmptyState title="Report not found" description={error ?? "This analysis may have been deleted."} />
        </Card>
      </div>
    );
  }

  const modelMetrics = (data.model_info?.metrics ?? {}) as Record<string, number | undefined>;

  return (
    <div>
      <PageHeader
        title="Export Analysis Report"
        subtitle="A professional, print-ready summary of this analysis."
        actions={
          <>
            <Button variant="outline" size="sm" icon={<ArrowLeft size={14} />} onClick={() => navigate("/history")}>
              Back to history
            </Button>
            <Button variant="outline" size="sm" icon={<FileDown size={14} />} onClick={downloadHtml}>
              Download HTML
            </Button>
            <Button size="sm" icon={<Printer size={14} />} onClick={() => window.print()}>
              Print / Save as PDF
            </Button>
          </>
        }
      />

      <Card className="print:shadow-none print:border-0">
        <div id="report-root" className="p-8">
          {/* Header */}
          <div className="flex items-center justify-between border-b-2 border-primary-100 pb-5 dark:border-primary-900">
            <div className="flex items-center gap-2.5">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-600 text-white">
                <FlaskConical size={20} />
              </div>
              <div>
                <p className="text-base font-bold">TruthLens AI</p>
                <p className="text-[11px] text-slate-500">AI-powered news credibility analysis</p>
              </div>
            </div>
            <div className="text-right text-[11px] text-slate-500">
              <p>Report #{data.history_id}</p>
              <p>{formatDateTime(data.created_at)}</p>
            </div>
          </div>

          {/* Headline + verdict */}
          <div className="mt-6">
            <h1 className="text-lg font-bold leading-snug">{data.headline || "(no headline)"}</h1>
            <p className="mt-1 line-clamp-6 max-h-40 overflow-y-auto rounded-lg bg-slate-50 p-3 text-xs leading-relaxed text-slate-600 dark:bg-slate-900 dark:text-slate-300">
              {data.article_text}
            </p>
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-4">
            <span className={`badge ${data.prediction}`}>{data.prediction}</span>
            <div>
              <p className="text-3xl font-bold tabular-nums">{formatPercent(data.confidence)}</p>
              <p className="text-xs text-slate-500">{data.confidence_level}</p>
            </div>
          </div>

          {/* Probability distribution */}
          <h2>Probability Distribution</h2>
          <div className="grid grid-cols-3 gap-3">
            <div className="metric"><span>REAL</span><b>{formatPercent(data.probabilities.real)}</b></div>
            <div className="metric"><span>FAKE</span><b>{formatPercent(data.probabilities.fake)}</b></div>
            <div className="metric"><span>UNCERTAIN</span><b>{formatPercent(data.probabilities.uncertain)}</b></div>
          </div>

          {/* Stats */}
          <h2>Article Statistics</h2>
          <table>
            <tbody>
              {Object.entries(data.article_stats ?? {}).map(([key, value]) => (
                <tr key={key}>
                  <td className="w-1/2">{STAT_LABELS[key] ?? key}</td>
                  <td className="tabular-nums">{typeof value === "number" && value >= 100 ? value.toLocaleString() : value}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Keywords */}
          <h2>TF-IDF Keywords</h2>
          <div className="flex flex-wrap gap-2">
            {data.keywords?.length ? (
              data.keywords.map((k) => (
                <span key={k.term} className="rounded-md border border-slate-200 px-2 py-1 text-xs dark:border-slate-700">
                  {k.term} <span className="text-slate-400 tabular-nums">{(k.score ?? 0).toFixed(3)}</span>
                </span>
              ))
            ) : (
              <p className="text-xs text-slate-400">No keywords extracted.</p>
            )}
          </div>

          {/* Explanation */}
          <h2>Important Features</h2>
          <table>
            <thead>
              <tr><th>Term</th><th>Contribution</th></tr>
            </thead>
            <tbody>
              {(data.explanation?.features ?? []).slice(0, 10).map((f) => (
                <tr key={f.term}>
                  <td>{f.term}</td>
                  <td className="tabular-nums">{f.contribution.toFixed(5)}</td>
                </tr>
              ))}
              {!(data.explanation?.features ?? []).length && (
                <tr><td colSpan={2} className="text-slate-400">No explanation data.</td></tr>
              )}
            </tbody>
          </table>

          {/* Model info */}
          <h2>Model Information</h2>
          <table>
            <tbody>
              <tr><td className="w-1/2">Model</td><td>{String(data.model_info?.name ?? "—")}</td></tr>
              <tr><td>Vectorizer</td><td>{String(data.model_info?.vectorizer ?? "—")}</td></tr>
              <tr><td>Training Dataset</td><td>{String(data.model_info?.dataset_source ?? "—")}</td></tr>
              <tr><td>Training Samples</td><td>{Number(data.model_info?.train_samples ?? 0).toLocaleString()}</td></tr>
              <tr><td>Test Samples</td><td>{Number(data.model_info?.test_samples ?? 0).toLocaleString()}</td></tr>
              <tr><td>Number of Features</td><td>{Number(data.model_info?.n_features ?? 0).toLocaleString()}</td></tr>
              <tr><td>Accuracy</td><td>{modelMetrics.accuracy != null ? formatPercent(modelMetrics.accuracy) : "—"}</td></tr>
              <tr><td>F1 Score</td><td>{modelMetrics.f1 != null ? formatPercent(modelMetrics.f1) : "—"}</td></tr>
            </tbody>
          </table>

          {/* Disclaimer */}
          <div className="banner">
            <strong>Disclaimer.</strong> {data.disclaimer}
          </div>
          <p className="mt-6 text-center text-[10px] text-slate-400">
            Generated by TruthLens AI on {formatDateTime(new Date().toISOString())}
          </p>
        </div>
      </Card>
    </div>
  );
}
