import { useState } from "react";
import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import {
  FileBarChart,
  GitCompareArrows,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Brain,
  Info,
  AlertTriangle,
  Globe2,
} from "lucide-react";
import type { AnalysisResult } from "../../types";
import { Card } from "../ui/Card";
import { Button } from "../ui/Button";
import { HorizontalBars, ExplanationBars } from "../charts/HorizontalBars";
import { VerificationSection } from "./VerificationSection";
import { formatPercent } from "../../lib/utils";

const STAT_META: { key: keyof AnalysisResult["article_stats"]; label: string }[] = [
  { key: "word_count", label: "Word Count" },
  { key: "character_count", label: "Character Count" },
  { key: "sentence_count", label: "Sentence Count" },
  { key: "average_sentence_length", label: "Avg Sentence Length" },
  { key: "unique_words", label: "Unique Words" },
  { key: "vocabulary_richness", label: "Vocabulary Richness" },
  { key: "capitalized_words", label: "Capitalized Words" },
  { key: "exclamation_marks", label: "Exclamation Marks" },
  { key: "question_marks", label: "Question Marks" },
];

const FLOW_STEPS = [
  "User input — headline and/or article text",
  "Language detection — auto / English / Tamil / Tanglish",
  "Preprocessing — clean, tokenize, truncate to 320 tokens",
  "Claim extraction — split the text into atomic checkable claims",
  "Evidence retrieval — search a knowledge base and live sources, never inventing sources",
  "Evidence scoring — relevance, source credibility, temporal freshness per claim",
  "AI analysis #1 — judge each claim SUPPORT / CONTRADICT / INSUFFICIENT from evidence only",
  "Adversarial AI review #2 — independent AI critic hunts for errors in AI #1",
  "Final decision engine — Gemini decides TRUE or FALSE using live data and its own knowledge",
  "Overall verdict — Gemini's final answer is shown; the local model's suggestion is displayed beside it, never as the answer",
];

export function ResultPanel({
  result,
  onViewHistory,
  compact,
}: {
  result: AnalysisResult;
  onViewHistory?: () => void;
  compact?: boolean;
}) {
  const navigate = useNavigate();
  const [showKeywordInfo, setShowKeywordInfo] = useState(false);

  const stats = result.article_stats;
  const keywords = result.keywords.slice(0, 10);
  const feats = result.explanation?.features ?? [];

  const final = pickFinalVerdict(result);
  const finalVerdict = final?.verdict ?? null;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* ---- Headline-only caveat ---- */}
      {result.headline_only && (
        <div className="flex items-start gap-2.5 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-xs leading-relaxed text-amber-800 dark:border-amber-700/60 dark:bg-amber-950/40 dark:text-amber-200">
          <AlertTriangle size={15} className="mt-0.5 shrink-0" />
          <span>
            {result.caveat ??
              "Headline-only analysis — with only a headline, the model has very little text to judge. Verify with the full article and reliable sources."}
          </span>
        </div>
      )}

      {/* ---- Final verdict hero = AI verdict ---- */}
      {finalVerdict && (
        <div className={`rounded-xl border p-5 ${VERDICT_STYLES[finalVerdict].card}`}>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className={`text-[11px] font-semibold uppercase tracking-wide ${VERDICT_STYLES[finalVerdict].muted}`}>
                Final Verdict · AI decided
              </p>
              <div className="mt-1 flex flex-wrap items-center gap-3">
                <p className={`text-2xl font-bold ${VERDICT_STYLES[finalVerdict].text}`}>
                  {finalVerdict}
                </p>
                {final?.confidence != null && (
                  <span className={`rounded-md px-2 py-1 text-sm font-bold tabular-nums ${VERDICT_STYLES[finalVerdict].pill}`}>
                    {Math.round(final.confidence * 100)}% confidence
                  </span>
                )}
              </div>
              <p className={`mt-1 max-w-xl text-xs leading-relaxed ${VERDICT_STYLES[finalVerdict].muted}`}>
                {VERDICT_STYLES[finalVerdict].blurb}
              </p>
              {result.verification?.gemini_validation?.label && (
                <p className={`mt-2 flex items-start gap-1.5 rounded-md px-2.5 py-1.5 text-[11px] leading-relaxed ${VERDICT_STYLES[finalVerdict].pill}`}>
                  <span className="mt-px shrink-0">✓</span>
                  <span>
                    <strong>Validated by AI:</strong>{" "}
                    {result.verification.gemini_validation.reasoning ||
                      "The AI reviewed this verdict and explained its reasoning."}
                  </span>
                </p>
              )}
            </div>
            <div className={`flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium ${VERDICT_STYLES[finalVerdict].pill}`}>
              {VERDICT_STYLES[finalVerdict].icon}
              Evidence-led · never fabricates sources
            </div>
          </div>
        </div>
      )}

      {/* ---- AI verdict (headline) ---- */}
      {result.ai_verdict && (
        <Card className="overflow-hidden">
          <div className="flex items-center gap-2 border-b border-slate-200 px-5 py-4 dark:border-slate-800">
            <Brain size={16} className="text-primary-500" />
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">AI Verdict</h3>
              <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                {result.ai_verdict.source === "gemini"
                    ? "Independent AI cross-check against global reporting and evidence"
                    : "Based on retrieved evidence and independent AI analysis"}
              </p>
            </div>
          </div>
          <div className="flex flex-col gap-4 p-5 sm:flex-row sm:items-start">
            <div className="flex flex-col items-start gap-2 sm:min-w-[200px]">
              <LiveCheckBadge label={result.ai_verdict.verdict} confidence={result.ai_verdict.confidence} />
              <p className="text-[11px] text-slate-400 dark:text-slate-500">
                {result.ai_verdict.source === "gemini" ? "Source: independent AI review" : "Source: Evidence + AI"}
              </p>
            </div>
            <p className="flex-1 text-xs leading-relaxed text-slate-600 dark:text-slate-300">
              {result.ai_verdict.reasoning || "No reasoning returned."}
            </p>
          </div>
        </Card>
      )}

      {/* ---- Live knowledge check (Gemini) ---- */}
      {!result.ai_verdict && result.live_check && result.live_check.label && (
        <Card>
          <div className="flex items-center gap-2 border-b border-slate-200 px-5 py-4 dark:border-slate-800">
            <Globe2 size={16} className="text-primary-500" />
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Live Knowledge Check
              </h3>
              <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                AI cross-check against global reporting and evidence
              </p>
            </div>
          </div>
          <div className="flex flex-col gap-3 p-5 sm:flex-row sm:items-center">
            <LiveCheckBadge label={result.live_check.label} confidence={result.live_check.confidence} />
            <p className="flex-1 text-xs leading-relaxed text-slate-600 dark:text-slate-300">
              {result.live_check.reasoning || "No reasoning returned."}
            </p>
          </div>
        </Card>
      )}

      {/* ---- Evidence-based verification ---- */}
      {result.verification && <VerificationSection verification={result.verification} />}

      {/* ---- How it works (one flow map) ---- */}
      <Card className="overflow-hidden">
        <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            How TruthLens Works
          </h3>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
            One end-to-end pipeline — every step is executed and logged on the backend.
          </p>
        </div>
        <ol className="grid gap-2 p-5 sm:grid-cols-2">
          {FLOW_STEPS.map((step, idx) => (
            <li
              key={step}
              className="flex items-start gap-2 rounded-lg border border-slate-100 bg-slate-50/60 px-3 py-2 text-xs leading-relaxed text-slate-600 dark:border-slate-800 dark:bg-slate-900/40 dark:text-slate-300"
            >
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-[10px] font-bold text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
                {idx + 1}
              </span>
              {step}
            </li>
          ))}
        </ol>
      </Card>

      {/* ---- Actions ---- */}
      {!compact && (
        <div className="flex flex-wrap gap-2">
          {result.history_id && (
            <>
              <Button variant="outline" size="sm" icon={<FileBarChart size={14} />} onClick={() => navigate(`/report/${result.history_id}`)}>
                Export Report
              </Button>
              <Button variant="outline" size="sm" icon={<GitCompareArrows size={14} />} onClick={() => navigate("/compare")}>
                Compare Analyses
              </Button>
            </>
          )}
          {onViewHistory && (
            <Button variant="ghost" size="sm" onClick={onViewHistory}>
              View in history
            </Button>
          )}
        </div>
      )}

      {/* ---- Statistics ---- */}
      <Card>
        <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Article Statistics</h3>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">Computed from the submitted text.</p>
        </div>
        <div className="grid grid-cols-2 gap-px bg-slate-100 dark:bg-slate-800 sm:grid-cols-3">
          {STAT_META.map(({ key, label }) => (
            <div key={key} className="bg-white px-4 py-3 dark:bg-[#111a2e]">
              <p className="text-[11px] text-slate-500 dark:text-slate-400">{label}</p>
              <p className="mt-0.5 text-lg font-semibold tabular-nums text-slate-900 dark:text-slate-50">
                {typeof stats[key] === "number" && Math.abs(stats[key]) >= 100
                  ? (stats[key] as number).toLocaleString()
                  : String(stats[key])}
              </p>
            </div>
          ))}
        </div>
      </Card>

      {/* ---- NLP insights ---- */}
      <Card>
        <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">NLP Insights</h3>
              <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                Neural keywords extracted from this article · {keywords.length} terms · vocabulary size{" "}
                {result.model_info?.n_features?.toLocaleString() ?? "—"}
              </p>
            </div>
            <button
              onClick={() => setShowKeywordInfo((v) => !v)}
              className="flex items-center gap-1 text-xs font-medium text-primary-600 hover:text-primary-700 dark:text-primary-400"
            >
              {showKeywordInfo ? "Hide explanation" : "How are keywords chosen?"}
              {showKeywordInfo ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
          </div>
        </div>
        <div className="p-5">
          {showKeywordInfo && (
            <div className="mb-5 rounded-lg border border-primary-200 bg-primary-50 px-4 py-3 text-xs leading-relaxed text-primary-900 dark:border-primary-800 dark:bg-primary-950/40 dark:text-primary-200">
              <strong>How are these chosen?</strong> The neural network reads the article once
              left-to-right and once right-to-left (a bidirectional GRU). Each word moves the
              network's internal state by some amount; the words that move it the most are
              ranked as the keywords the network paid attention to for its REAL/FAKE call.
            </div>
          )}
          <HorizontalBars
            items={keywords.map((k) => ({
              label: k.term,
              value: k.score,
              sub: k.score.toFixed(3),
              color: "#4f46e5",
            }))}
          />
        </div>
      </Card>

      {/* ---- Explainability ---- */}
      <Card>
        <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <div className="flex items-center gap-2">
            <Brain size={16} className="text-primary-500" />
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Model-Associated Features
              </h3>
              <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                {result.explanation?.direction_label}
              </p>
            </div>
          </div>
        </div>
        <div className="p-5">
          {feats.length ? (
            <ExplanationBars items={feats} />
          ) : (
            <p className="text-xs text-slate-400">No significant features detected in this text.</p>
          )}
          <p className="mt-4 flex items-start gap-1.5 text-[11px] leading-relaxed text-slate-400 dark:text-slate-500">
            <Info size={13} className="mt-0.5 shrink-0" />
            {result.explanation?.note}
          </p>
        </div>
      </Card>

      {/* ---- Model info ---- */}
      <Card>
        <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Model Information</h3>
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-3 p-5 pb-2 text-xs sm:grid-cols-3 lg:grid-cols-4">
          <ModelField label="Model Used" value={result.model} />
          <ModelField label="Tokenizer" value={result.model_info?.vectorizer ?? "—"} />
          <ModelField label="Training Dataset" value={result.model_info?.dataset_source ?? "—"} />
          <ModelField label="Training Samples" value={result.model_info?.train_samples?.toLocaleString() ?? "—"} />
          <ModelField label="Testing Samples" value={result.model_info?.test_samples?.toLocaleString() ?? "—"} />
          <ModelField label="Vocabulary Size" value={result.model_info?.n_features?.toLocaleString() ?? "—"} />
          <ModelField
            label="Training Date"
            value={result.model_info?.training_date ? new Date(result.model_info.training_date).toLocaleDateString() : "—"}
          />
          <ModelField label="Explainability" value={result.model_info?.explainability ?? "—"} />
        </div>

        <div className="border-t border-slate-200 px-5 py-4 dark:border-slate-800">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Model Performance on Test Dataset
          </h3>
          <p className="mt-0.5 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
            These scores were measured on a held-out test set that was not used to train the
            model (a leakage-safe grouped split, with the token vocabulary built from the
            training split only). They describe the model's performance on the training corpus,{" "}
            <strong className="font-medium">not</strong> a
            guarantee of accuracy on real-world news.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-3 px-5 pb-5 text-xs sm:grid-cols-3 lg:grid-cols-4">
          <ModelField label="Accuracy" value={result.model_info?.metrics?.accuracy != null ? formatPercent(result.model_info.metrics.accuracy) : "—"} />
          <ModelField label="Precision" value={result.model_info?.metrics?.precision != null ? formatPercent(result.model_info.metrics.precision) : "—"} />
          <ModelField label="Recall" value={result.model_info?.metrics?.recall != null ? formatPercent(result.model_info.metrics.recall) : "—"} />
          <ModelField label="F1 Score" value={result.model_info?.metrics?.f1 != null ? formatPercent(result.model_info.metrics.f1) : "—"} />
          <ModelField label="ROC-AUC" value={result.model_info?.metrics?.roc_auc != null ? (result.model_info.metrics.roc_auc).toFixed(4) : "—"} />
        </div>
      </Card>
    </div>
  );
}

function ModelField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[11px] text-slate-400 dark:text-slate-500">{label}</p>
      <p className="mt-0.5 font-medium text-slate-800 dark:text-slate-200">{value}</p>
    </div>
  );
}

function LiveCheckBadge({ label, confidence }: { label: string; confidence: number }) {
  const styles: Record<string, string> = {
    REAL: "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-700/60 dark:bg-emerald-950/40 dark:text-emerald-300",
    FAKE: "border-rose-300 bg-rose-50 text-rose-700 dark:border-rose-700/60 dark:bg-rose-950/40 dark:text-rose-300",
    UNVERIFIED: "border-slate-300 bg-slate-50 text-slate-600 dark:border-slate-700 dark:bg-slate-800/60 dark:text-slate-300",
  };
  const Icon = label === "REAL" ? CheckCircle2 : label === "FAKE" ? XCircle : HelpCircle;
  return (
    <div className={`flex shrink-0 items-center gap-2 self-start rounded-full border px-3 py-1.5 text-xs font-semibold ${styles[label] ?? styles.UNVERIFIED}`}>
      <Icon size={14} />
      {label === "UNVERIFIED" ? "Cannot Verify" : label}
      <span className="font-medium opacity-80">· {Math.round(confidence * 100)}%</span>
    </div>
  );
}

const VERDICT_STYLES: Record<
  string,
  { card: string; text: string; muted: string; pill: string; blurb: string; icon: ReactNode }
> = {
  TRUE: {
    card: "border-emerald-400/60 bg-emerald-50/70 dark:border-emerald-700/50 dark:bg-emerald-950/30",
    text: "text-emerald-700 dark:text-emerald-400",
    muted: "text-emerald-700/70 dark:text-emerald-400/70",
    pill: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
    blurb: "Credible evidence supports this claim.",
    icon: <CheckCircle2 size={15} />,
  },
  FALSE: {
    card: "border-rose-400/60 bg-rose-50/70 dark:border-rose-700/50 dark:bg-rose-950/30",
    text: "text-rose-700 dark:text-rose-400",
    muted: "text-rose-700/70 dark:text-rose-400/70",
    pill: "bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-300",
    blurb: "Credible evidence contradicts this claim.",
    icon: <XCircle size={15} />,
  },
  UNVERIFIED: {
    card: "border-amber-400/60 bg-amber-50/70 dark:border-amber-700/50 dark:bg-amber-950/30",
    text: "text-amber-700 dark:text-amber-400",
    muted: "text-amber-700/70 dark:text-amber-400/70",
    pill: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
    blurb: "No credible evidence could confirm or refute this claim, so it stays unverified.",
    icon: <HelpCircle size={15} />,
  },
};

function pickFinalVerdict(result: AnalysisResult): { verdict: string; confidence?: number } | null {
  // The AI (Gemini) verdict is THE final answer. Everything else - the local
  // neural suggestion, the evidence-pipeline score - is a secondary signal.
  const ai = result.ai_verdict;
  if (ai?.verdict) {
    const verdict = ai.verdict === "REAL" ? "TRUE" : ai.verdict === "FAKE" ? "FALSE" : "UNVERIFIED";
    return { verdict, confidence: ai.confidence };
  }
  const gemini = result.verification?.gemini_validation;
  if (gemini?.label) {
    const verdict = gemini.label === "REAL" ? "TRUE" : "FALSE";
    return { verdict, confidence: gemini.confidence };
  }
  const live = result.live_check;
  if (live?.label) {
    const verdict = live.label === "REAL" ? "TRUE" : live.label === "FAKE" ? "FALSE" : "UNVERIFIED";
    return { verdict, confidence: live.confidence };
  }
  const fromOverview = result.verdict;
  if (fromOverview) {
    return {
      verdict: fromOverview === "FALSE" ? "FALSE" : fromOverview === "UNVERIFIED" ? "UNVERIFIED" : "TRUE",
      confidence: result.confidence,
    };
  }
  return null;
}
