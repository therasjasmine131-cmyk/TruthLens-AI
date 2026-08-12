import { useRef, useState } from "react";
import Papa from "papaparse";
import { UploadCloud, Download, FileSpreadsheet, AlertTriangle } from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card, CardHeader } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import { ProgressBar } from "../components/ui/Progress";
import { PredictionBadge } from "../components/ui/PredictionBadge";
import { useToast } from "../context/ToastContext";
import { api } from "../api/client";
import type { BatchResult, PredictionLabel } from "../types";
import { downloadBlob, formatPercent } from "../lib/utils";

const CHUNK_SIZE = 20;

interface ParsedRow {
  headline: string;
  article: string;
}

export function Batch() {
  const { toast } = useToast();
  const fileRef = useRef<HTMLInputElement>(null);
  const [rows, setRows] = useState<ParsedRow[]>([]);
  const [fileName, setFileName] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [result, setResult] = useState<BatchResult | null>(null);

  const onFile = (file: File) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: "greedy",
      complete: (output) => {
        if (output.errors.length) {
          toast("error", "CSV parse issues", `${output.errors.length} row(s) skipped by the parser.`);
        }
        const parsed = output.data
          .filter((r) => Object.keys(r).some((k) => (r[k] ?? "").trim()))
          .map((r) => {
            const headline = String(r.headline ?? r.title ?? "").trim();
            const article = String(r.article ?? r.text ?? r.body ?? "").trim();
            return { headline, article };
          })
          .filter((r) => r.headline || r.article);
        setRows(parsed);
        setFileName(file.name);
        setResult(null);
        if (!parsed.length) toast("error", "No valid rows", "Expected columns: headline, article.");
      },
    });
  };

  const run = async () => {
    if (!rows.length) return;
    setRunning(true);
    setResult(null);
    const allResults: BatchResult["results"] = [];
    let errors = 0;
    let real = 0;
    let fake = 0;
    let uncertain = 0;

    for (let i = 0; i < rows.length; i += CHUNK_SIZE) {
      const chunk = rows.slice(i, i + CHUNK_SIZE);
      try {
        const res = await api.batchAnalyze(
          chunk.map((r, offset) => ({ headline: r.headline, article: r.article, _index: i + offset }) as unknown as { headline: string; article: string }),
        );
        res.results.forEach((r, offset) => {
          const item = { ...r, index: i + offset };
          allResults.push(item);
          if (item.error) errors += 1;
          else if (item.prediction === "REAL") real += 1;
          else if (item.prediction === "FAKE") fake += 1;
          else if (item.prediction === "UNCERTAIN") uncertain += 1;
        });
      } catch (err) {
        chunk.forEach((_, offset) => {
          allResults.push({ index: i + offset, error: err instanceof Error ? err.message : "chunk failed" });
        });
        errors += chunk.length;
      }
      setProgress({ done: Math.min(i + CHUNK_SIZE, rows.length), total: rows.length });
    }

    setResult({ total: rows.length, completed: rows.length - errors, errors, real, fake, uncertain, results: allResults });
    setRunning(false);
  };

  const downloadResults = () => {
    if (!result) return;
    const header = "index,headline,prediction,real_probability,fake_probability,uncertain_probability,confidence,error";
    const lines = result.results.map((r) =>
      [
        r.index,
        `"${(r.headline ?? "").replace(/"/g, '""')}"`,
        r.prediction ?? "",
        r.probabilities?.real?.toFixed(4) ?? "",
        r.probabilities?.fake?.toFixed(4) ?? "",
        r.probabilities?.uncertain?.toFixed(4) ?? "",
        r.confidence?.toFixed(4) ?? "",
        `"${(r.error ?? "").replace(/"/g, '""')}"`,
      ].join(","),
    );
    downloadBlob([header, ...lines].join("\n"), "truthlens-batch-results.csv");
    toast("success", "Results downloaded");
  };

  return (
    <div>
      <PageHeader
        title="Batch Analysis"
        subtitle="Upload a CSV with headline and article columns to analyze many articles at once."
      />

      <Card>
        <CardHeader
          title="Upload CSV"
          subtitle="Required columns: headline, article (title/text aliases accepted). Max 2000 rows."
          action={
            <>
              <input
                ref={fileRef}
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
              />
              <Button size="sm" variant="outline" icon={<UploadCloud size={14} />} onClick={() => fileRef.current?.click()}>
                Choose file
              </Button>
            </>
          }
        />
        <div className="flex flex-col items-center gap-2 p-8 text-center">
          {!fileName ? (
            <EmptyState
              compact
              icon={<FileSpreadsheet size={22} />}
              title="No file selected"
              description="Upload a CSV with headline and article columns to get started."
            />
          ) : (
            <>
              <p className="text-sm font-medium text-slate-700 dark:text-slate-200">{fileName}</p>
              <p className="text-xs text-slate-400">{rows.length.toLocaleString()} valid rows detected</p>
              <div className="mt-2 flex gap-2">
                <Button size="sm" disabled={running || !rows.length} loading={running} onClick={run}>
                  {running ? "Analyzing…" : "Run Batch Analysis"}
                </Button>
                <Button size="sm" variant="outline" onClick={() => { setRows([]); setFileName(null); setResult(null); }}>
                  Reset
                </Button>
              </div>
            </>
          )}
        </div>

        {running && (
          <div className="border-t border-slate-200 px-5 py-4 dark:border-slate-800">
            <div className="mb-2 flex justify-between text-xs text-slate-500">
              <span>Processing rows…</span>
              <span className="tabular-nums">
                {progress.done.toLocaleString()} / {progress.total.toLocaleString()}
              </span>
            </div>
            <ProgressBar value={progress.total ? (progress.done / progress.total) * 100 : 0} />
          </div>
        )}
      </Card>

      {result && (
        <div className="mt-6 space-y-6 animate-fade-in">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
            <SummaryCard label="Total Records" value={result.total.toLocaleString()} tone="text-slate-900 dark:text-slate-50" />
            <SummaryCard label="Completed" value={result.completed.toLocaleString()} tone="text-emerald-600 dark:text-emerald-400" />
            <SummaryCard label="Real" value={result.real.toLocaleString()} tone="text-emerald-600 dark:text-emerald-400" />
            <SummaryCard label="Fake" value={result.fake.toLocaleString()} tone="text-rose-600 dark:text-rose-400" />
            <SummaryCard label="Errors" value={result.errors.toLocaleString()} tone={result.errors ? "text-rose-600" : "text-slate-400"} />
          </div>

          <Card>
            <CardHeader
              title="Results"
              subtitle="Detailed per-row outcomes"
              action={
                <Button size="sm" icon={<Download size={14} />} onClick={downloadResults}>
                  Download Results CSV
                </Button>
              }
            />
            <div className="max-h-[480px] overflow-auto">
              <table className="w-full min-w-[640px] text-left text-xs">
                <thead className="sticky top-0 border-b border-slate-200 bg-white text-[11px] uppercase tracking-wide text-slate-400 dark:border-slate-800 dark:bg-[#111a2e]">
                  <tr>
                    <th className="px-4 py-2.5 font-medium">#</th>
                    <th className="px-4 py-2.5 font-medium">Headline</th>
                    <th className="px-4 py-2.5 font-medium">Prediction</th>
                    <th className="px-4 py-2.5 font-medium">Confidence</th>
                    <th className="px-4 py-2.5 font-medium">Real / Fake / Uncert.</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.map((r) => (
                    <tr key={r.index} className="border-b border-slate-100 last:border-0 dark:border-slate-800/60">
                      <td className="px-4 py-2 tabular-nums text-slate-400">{r.index + 1}</td>
                      <td className="max-w-[260px] truncate px-4 py-2 font-medium text-slate-700 dark:text-slate-200">
                        {r.error ? (
                          <span className="flex items-center gap-1 text-rose-500">
                            <AlertTriangle size={12} /> {r.error}
                          </span>
                        ) : (
                          r.headline || "(no headline)"
                        )}
                      </td>
                      <td className="px-4 py-2">
                        {r.prediction && <PredictionBadge label={r.prediction as PredictionLabel} size="sm" />}
                      </td>
                      <td className="px-4 py-2 tabular-nums text-slate-600 dark:text-slate-300">
                        {r.confidence != null ? formatPercent(r.confidence) : "—"}
                      </td>
                      <td className="px-4 py-2 tabular-nums text-slate-400">
                        {r.probabilities
                          ? `${formatPercent(r.probabilities.real)} / ${formatPercent(r.probabilities.fake)} / ${formatPercent(r.probabilities.uncertain)}`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

function SummaryCard({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="card p-4">
      <p className="text-xs text-slate-500 dark:text-slate-400">{label}</p>
      <p className={`mt-1 text-2xl font-bold tabular-nums ${tone}`}>{value}</p>
    </div>
  );
}
