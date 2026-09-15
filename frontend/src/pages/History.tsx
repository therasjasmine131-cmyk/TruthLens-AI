import { useCallback, useEffect, useState } from "react";
import {
  Search,
  Trash2,
  Eye,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Download,
} from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card, CardHeader } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Input, Select } from "../components/ui/Input";
import { EmptyState } from "../components/ui/EmptyState";
import { Modal } from "../components/ui/Modal";
import { PredictionBadge } from "../components/ui/PredictionBadge";
import { ResultPanel } from "../components/analyze/ResultPanel";
import { useApi } from "../hooks/useApi";
import { useToast } from "../context/ToastContext";
import { api } from "../api/client";
import type {
  AnalysisResult,
  ArticleStats,
  Explanation,
  HistoryDetail,
  HistoryItem,
  HistoryPage,
  Keyword,
  ModelInfo,
  PredictionLabel,
} from "../types";
import { downloadBlob, formatDateTime, formatPercent } from "../lib/utils";

const PER_PAGE = 10;

function DetailResult({ detail }: { detail: HistoryDetail }) {
  const meta = detail.analysis_metadata as {
    confidence_level?: string;
    keywords?: Keyword[];
    article_stats?: ArticleStats;
    explanation?: Explanation;
    model_info?: ModelInfo;
    model_raw?: { p_real: number; p_fake: number };
    probabilities?: AnalysisResult["probabilities"];
  };
  const result: AnalysisResult = {
    prediction: detail.prediction,
    confidence: detail.confidence,
    confidence_level: meta.confidence_level ?? "—",
    probabilities: meta.probabilities ?? {
      real: detail.real_probability,
      fake: detail.fake_probability,
      uncertain: detail.uncertain_probability,
    },
    model_raw: meta.model_raw ?? { p_real: detail.real_probability, p_fake: detail.fake_probability },
    model: detail.model_name,
    model_info: meta.model_info ?? { name: detail.model_name },
    keywords: meta.keywords ?? [],
    article_stats: meta.article_stats ?? {
      word_count: detail.word_count,
      character_count: detail.character_count,
      sentence_count: detail.sentence_count,
      average_sentence_length: 0,
      unique_words: 0,
      vocabulary_richness: 0,
      capitalized_words: 0,
      exclamation_marks: 0,
      question_marks: 0,
    },
    explanation: meta.explanation ?? {
      method: "stored",
      model_class: "",
      features: [],
      note: "Explanation data was not stored for this record.",
      direction_label: "",
    },
    disclaimer:
      "This is an ML-based prediction, not proof of factual truth. TruthLens AI reflects statistical patterns learned from a training dataset and can be wrong. Always verify important claims using reliable sources.",
    saved: true,
    history_id: detail.id,
  };
  return <ResultPanel compact result={result} />;
}

export function History() {
  const { toast } = useToast();
  const [params, setParams] = useState<URLSearchParams>(() =>
    new URLSearchParams({ per_page: String(PER_PAGE), sort_by: "created_at", order: "desc" }),
  );
  const [detail, setDetail] = useState<HistoryDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<HistoryItem | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");

  const fetchHistory = useCallback(() => api.history(params), [params]);
  const { data, loading, error, refresh } = useApi<HistoryPage>(fetchHistory, [params]);

  const updateParams = (mutate: (p: URLSearchParams) => void) => {
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      mutate(next);
      return next;
    });
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchInput);
    }, 350);
    return () => clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => {
    updateParams((p) => {
      if (debouncedSearch) p.set("search", debouncedSearch);
      else p.delete("search");
      p.set("page", "1");
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSearch]);

  const openDetail = async (id: number) => {
    setDetailLoading(true);
    try {
      setDetail(await api.historyItem(id));
    } catch (err) {
      toast("error", "Could not load record", err instanceof Error ? err.message : undefined);
    } finally {
      setDetailLoading(false);
    }
  };

  const deleteRecord = async () => {
    if (!deleteTarget) return;
    try {
      await api.deleteHistoryItem(deleteTarget.id);
      toast("success", "Record deleted");
      setDeleteTarget(null);
      refresh();
    } catch (err) {
      toast("error", "Delete failed", err instanceof Error ? err.message : undefined);
    }
  };

  const reanalyze = async (id: number) => {
    try {
      const result = await api.reanalyze(id);
      toast("success", "Re-analyzed", `${result.prediction} at ${formatPercent(result.confidence)}`);
      refresh();
    } catch (err) {
      toast("error", "Re-analysis failed", err instanceof Error ? err.message : undefined);
    }
  };

  const clearAll = async () => {
    try {
      const res = await api.clearHistory();
      toast("success", "History cleared", `${res.deleted} records removed`);
      setConfirmClear(false);
      refresh();
    } catch (err) {
      toast("error", "Clear failed", err instanceof Error ? err.message : undefined);
    }
  };

  const exportCsv = async () => {
    try {
      const data = await api.exportHistory();
      const rows = data.items.map((item) => ({
        date: item.created_at,
        headline: item.headline,
        prediction: item.prediction,
        confidence: item.confidence,
        real_probability: item.real_probability,
        fake_probability: item.fake_probability,
        uncertain_probability: item.uncertain_probability,
        model: item.model_name,
        word_count: item.word_count,
      }));
      if (!rows.length) {
        toast("info", "Nothing to export");
        return;
      }
      const header = Object.keys(rows[0]).join(",");
      const body = rows
        .map((r) => Object.values(r).map((v) => `"${String(v).replace(/"/g, '""')}"`).join(","))
        .join("\n");
      downloadBlob(`${header}\n${body}`, "truthlens-history.csv");
      toast("success", "History exported", `${rows.length} records`);
    } catch (err) {
      toast("error", "Export failed", err instanceof Error ? err.message : undefined);
    }
  };

  return (
    <div>
      <PageHeader
        title="Prediction History"
        subtitle="Search, filter and review every stored analysis."
        actions={
          <>
            <Button variant="outline" size="sm" icon={<Download size={14} />} onClick={exportCsv}>
              Export
            </Button>
            <Button
              variant="danger"
              size="sm"
              icon={<Trash2 size={14} />}
              onClick={() => setConfirmClear(true)}
              disabled={!data?.total}
            >
              Clear history
            </Button>
          </>
        }
      />

      <Card>
        <CardHeader title="Filters" />
        <div className="grid gap-3 p-4 lg:grid-cols-5">
          <div className="relative lg:col-span-2">
            <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
            <Input
              className="pl-9"
              placeholder="Search headline or article…"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
            />
          </div>
          <Select
            label=""
            value={params.get("prediction") ?? ""}
            onChange={(e) =>
              updateParams((p) => {
                e.target.value ? p.set("prediction", e.target.value) : p.delete("prediction");
                p.set("page", "1");
              })
            }
            options={[
              { value: "", label: "All predictions" },
              { value: "REAL", label: "REAL" },
              { value: "FAKE", label: "FAKE" },
              { value: "UNCERTAIN", label: "UNCERTAIN" },
            ]}
          />
          <Select
            label=""
            value={params.get("min_confidence") ?? ""}
            onChange={(e) =>
              updateParams((p) => {
                e.target.value ? p.set("min_confidence", e.target.value) : p.delete("min_confidence");
                p.set("page", "1");
              })
            }
            options={[
              { value: "", label: "Any confidence" },
              { value: "0.9", label: "≥ 90%" },
              { value: "0.75", label: "≥ 75%" },
              { value: "0.5", label: "≥ 50%" },
            ]}
          />
          <Select
            label=""
            value={params.get("sort_by") ?? "created_at"}
            onChange={(e) => updateParams((p) => p.set("sort_by", e.target.value))}
            options={[
              { value: "created_at", label: "Sort: Date" },
              { value: "confidence", label: "Sort: Confidence" },
              { value: "word_count", label: "Sort: Word count" },
              { value: "headline", label: "Sort: Headline" },
            ]}
          />
        </div>

        <div className="overflow-x-auto">
          {loading ? (
            <div className="p-6 text-center text-sm text-slate-400">Loading history…</div>
          ) : error ? (
            <div className="p-6 text-center text-sm text-rose-500">{error}</div>
          ) : !data?.items.length ? (
            <EmptyState
              title={debouncedSearch || params.get("prediction") ? "No matching records" : "No analyses yet"}
              description={
                debouncedSearch || params.get("prediction")
                  ? "Try adjusting your search or filters."
                  : "Analyze your first article to build up prediction history."
              }
            />
          ) : (
            <table className="w-full min-w-[760px] text-left text-xs">
              <thead className="border-b border-slate-200 text-[11px] uppercase tracking-wide text-slate-400 dark:border-slate-800">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Date</th>
                  <th className="px-4 py-2.5 font-medium">Headline</th>
                  <th className="px-4 py-2.5 font-medium">Prediction</th>
                  <th className="px-4 py-2.5 font-medium">Confidence</th>
                  <th className="px-4 py-2.5 font-medium">Model</th>
                  <th className="px-4 py-2.5 font-medium">Words</th>
                  <th className="px-4 py-2.5 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((item) => (
                  <tr key={item.id} className="border-b border-slate-100 transition-colors hover:bg-slate-50 dark:border-slate-800/60 dark:hover:bg-slate-800/30">
                    <td className="whitespace-nowrap px-4 py-3 text-slate-400">{formatDateTime(item.created_at)}</td>
                    <td className="max-w-[240px] truncate px-4 py-3 font-medium text-slate-700 dark:text-slate-200">
                      {item.headline || "(no headline)"}
                    </td>
                    <td className="px-4 py-3">
                      <PredictionBadge label={item.prediction as PredictionLabel} size="sm" />
                    </td>
                    <td className="px-4 py-3 tabular-nums text-slate-600 dark:text-slate-300">
                      {formatPercent(item.confidence)}
                    </td>
                    <td className="px-4 py-3 text-slate-500 dark:text-slate-400">{item.model_name}</td>
                    <td className="px-4 py-3 tabular-nums text-slate-500 dark:text-slate-400">{item.word_count.toLocaleString()}</td>
                    <td className="px-4 py-3">
                      <div className="flex justify-end gap-1">
                        <button
                          onClick={() => openDetail(item.id)}
                          className="rounded p-1.5 text-slate-400 hover:bg-slate-100 hover:text-primary-600 dark:hover:bg-slate-800"
                          title="View analysis"
                          aria-label="View analysis"
                        >
                          <Eye size={15} />
                        </button>
                        <button
                          onClick={() => reanalyze(item.id)}
                          className="rounded p-1.5 text-slate-400 hover:bg-slate-100 hover:text-primary-600 dark:hover:bg-slate-800"
                          title="Re-analyze"
                          aria-label="Re-analyze"
                        >
                          <RefreshCw size={15} />
                        </button>
                        <button
                          onClick={() => setDeleteTarget(item)}
                          className="rounded p-1.5 text-slate-400 hover:bg-slate-100 hover:text-rose-600 dark:hover:bg-slate-800"
                          title="Delete"
                          aria-label="Delete"
                        >
                          <Trash2 size={15} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {data && data.pages > 1 && (
          <div className="flex items-center justify-between border-t border-slate-200 px-4 py-3 text-xs dark:border-slate-800">
            <span className="text-slate-400">
              Page {data.page} of {data.pages} · {data.total.toLocaleString()} records
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={!data.has_prev}
                icon={<ChevronLeft size={14} />}
                onClick={() => updateParams((p) => p.set("page", String(data.page - 1)))}
              >
                Prev
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={!data.has_next}
                onClick={() => updateParams((p) => p.set("page", String(data.page + 1)))}
              >
                Next <ChevronRight size={14} />
              </Button>
            </div>
          </div>
        )}
      </Card>

      {/* ---- Detail modal ---- */}
      <Modal open={!!detail || detailLoading} onClose={() => setDetail(null)} title="Complete Analysis" maxWidth="xl">
        {detailLoading && <div className="py-10 text-center text-sm text-slate-400">Loading…</div>}
        {detail && !detailLoading && (
          <div>
            <p className="mb-3 text-xs text-slate-400">{formatDateTime(detail.created_at)} · {detail.model_name}</p>
            <div className="mb-4 max-h-40 overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs leading-relaxed text-slate-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300">
              {detail.article_text || detail.headline}
            </div>
            <DetailResult detail={detail} />
          </div>
        )}
      </Modal>

      {/* ---- Delete confirmation ---- */}
      <Modal
        open={!!deleteTarget}
        onClose={() => setDeleteTarget(null)}
        title="Delete this analysis?"
        footer={
          <>
            <Button variant="outline" size="sm" onClick={() => setDeleteTarget(null)}>
              Cancel
            </Button>
            <Button variant="danger" size="sm" onClick={deleteRecord}>
              Delete
            </Button>
          </>
        }
      >
        <p className="text-sm text-slate-600 dark:text-slate-300">
          "<span className="font-medium">{deleteTarget?.headline || "(no headline)"}</span>" and its full
          analysis will be permanently removed.
        </p>
      </Modal>

      {/* ---- Clear confirmation ---- */}
      <Modal
        open={confirmClear}
        onClose={() => setConfirmClear(false)}
        title="Clear entire history?"
        footer={
          <>
            <Button variant="outline" size="sm" onClick={() => setConfirmClear(false)}>
              Cancel
            </Button>
            <Button variant="danger" size="sm" onClick={clearAll}>
              Clear all {data?.total ?? 0} records
            </Button>
          </>
        }
      >
        <p className="text-sm text-slate-600 dark:text-slate-300">
          This permanently deletes every stored analysis. This cannot be undone.
        </p>
      </Modal>
    </div>
  );
}
