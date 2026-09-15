import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ShieldCheck,
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
import { PredictionBadge } from "../ui/PredictionBadge";
import { ConfidenceGauge } from "../charts/ConfidenceGauge";
import { HorizontalBars, ExplanationBars } from "../charts/HorizontalBars";
import { Disclaimer } from "../ui/PageHeader";
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

function ProbCard({
  label,
  value,
  icon,
  tone,
}: {
  label: string;
  value: number;
  icon: typeof ShieldCheck;
  tone: string;
}) {
  const Icon = icon;
  return (
    <div className={cn_("rounded-lg border bg-slate-50 px-4 py-3 dark:bg-slate-900/60", tone)}>
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-xs font-medium text-slate-500 dark:text-slate-400">
          <Icon size={13} />
          {label}
        </span>
        <span className="text-sm font-bold tabular-nums text-slate-900 dark:text-slate-50">
          {formatPercent(value)}
        </span>
      </div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
        <div
          className={cn_("h-full rounded-full transition-all duration-700", tone === "text-emerald-600 dark:text-emerald-400" ? "bg-emerald-500" : tone === "text-rose-600 dark:text-rose-400" ? "bg-rose-500" : "bg-amber-500")}
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
    </div>
  );
}

function cn_(...parts: (string | false | undefined | null)[]) {
  return parts.filter(Boolean).join(" ");
}

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
  const [showTfidf, setShowTfidf] = useState(false);

  const stats = result.article_stats;
  const probs = result.probabilities;
  const keywords = result.keywords.slice(0, 10);
  const feats = result.explanation?.features ?? [];

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

      {/* ---- AI verdict (headline) ---- */}
      {result.ai_verdict && (
        <Card className="overflow-hidden">
          <div className="flex items-center gap-2 border-b border-slate-200 px-5 py-4 dark:border-slate-800">
            <Brain size={16} className="text-primary-500" />
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">AI Verdict</h3>
              <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                {result.ai_verdict.source === "gemini"
                  ? "Google Gemini cross-check against its knowledge of real-world reporting"
                  : "Based on retrieved evidence and independent AI analysis"}
              </p>
            </div>
          </div>
          <div className="flex flex-col gap-4 p-5 sm:flex-row sm:items-start">
            <div className="flex flex-col items-start gap-2 sm:min-w-[200px]">
              <LiveCheckBadge label={result.ai_verdict.verdict} confidence={result.ai_verdict.confidence} />
              <p className="text-[11px] text-slate-400 dark:text-slate-500">
                {result.ai_verdict.source === "gemini" ? "Source: Google Gemini" : "Source: Evidence + AI"}
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
                Gemini cross-check against its knowledge of real-world reporting
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

      {/* ---- Secondary ML signal ---- */}
      {result.prediction && (
      <Card className="overflow-hidden">
        <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Secondary Signal — Trained ML Classifier
          </h3>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
            Stylistic pattern from the local model only. The AI verdict above is the primary judgment.
          </p>
        </div>
        <div className="grid gap-6 p-6 md:grid-cols-2">
          <div className="flex flex-col items-center justify-center gap-3">
            <div className="flex flex-col items-center gap-2">
              <PredictionBadge label={result.prediction} size="lg" />
              <p className="text-xs text-slate-500 dark:text-slate-400">{result.confidence_level}</p>
            </div>
            <ConfidenceGauge confidence={result.confidence} prediction={result.prediction} />
            <p className="text-center text-[11px] text-slate-400 dark:text-slate-500">
              Model: {result.model} · {result.model_info?.vectorizer?.replace(/^TfidfVectorizer /, "")}
            </p>
          </div>

          <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Probability Distribution
            </p>
            <ProbCard label="REAL" value={probs.real} icon={CheckCircle2} tone="text-emerald-600 dark:text-emerald-400" />
            <ProbCard label="FAKE" value={probs.fake} icon={XCircle} tone="text-rose-600 dark:text-rose-400" />

            {result.prediction === "UNCERTAIN" && (
              <div className="flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-[11px] leading-relaxed text-amber-800 dark:border-amber-700/60 dark:bg-amber-950/40 dark:text-amber-200">
                <HelpCircle size={13} className="mt-0.5 shrink-0" />
                <span>
                  <strong>UNCERTAIN (abstain):</strong> the model's top-class
                  probability ({(result.confidence * 100).toFixed(1)}%) was below
                  the {Math.round((result.uncertain_threshold ?? 0.78) * 100)}%
                  tolerances, so TruthLens declined to call REAL or FAKE. This is
                  a decision, not a third probability.
                </span>
              </div>
            )}

            <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] text-slate-500 dark:border-slate-800 dark:bg-slate-900/60 dark:text-slate-400">
              Raw model probabilities (binary classifier): REAL{" "}
              {formatPercent(result.model_raw?.p_real ?? probs.real)}, FAKE{" "}
              {formatPercent(result.model_raw?.p_fake ?? probs.fake)} — together
              100%. UNCERTAIN is an abstain decision when the model is unsure; it
              is never manufactured by subtracting from these probabilities.
            </div>
          </div>
        </div>

        <div className="border-t border-slate-200 px-6 py-4 dark:border-slate-800">
          <Disclaimer text={result.disclaimer} />
        </div>
      </Card>
      )}

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
                TF-IDF keywords extracted from this article · {keywords.length} terms · vocabulary size{" "}
                {result.model_info?.n_features?.toLocaleString() ?? "—"}
              </p>
            </div>
            <button
              onClick={() => setShowTfidf((v) => !v)}
              className="flex items-center gap-1 text-xs font-medium text-primary-600 hover:text-primary-700 dark:text-primary-400"
            >
              {showTfidf ? "Hide explanation" : "What is TF-IDF?"}
              {showTfidf ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
          </div>
        </div>
        <div className="p-5">
          {showTfidf && (
            <div className="mb-5 rounded-lg border border-primary-200 bg-primary-50 px-4 py-3 text-xs leading-relaxed text-primary-900 dark:border-primary-800 dark:bg-primary-950/40 dark:text-primary-200">
              <strong>What is TF-IDF?</strong> Term Frequency–Inverse Document Frequency scores how
              important a word is within one article. A word is important if it appears often in this
              article (high term frequency) but is rare across the whole training corpus (low document
              frequency). Common words like "the" get low scores; distinctive words like "election"
              get high scores. The classifier uses these weighted word features to separate real from
              fake articles.
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
          <ModelField label="Vectorizer" value={result.model_info?.vectorizer ?? "—"} />
          <ModelField label="Training Dataset" value={result.model_info?.dataset_source ?? "—"} />
          <ModelField label="Training Samples" value={result.model_info?.train_samples?.toLocaleString() ?? "—"} />
          <ModelField label="Testing Samples" value={result.model_info?.test_samples?.toLocaleString() ?? "—"} />
          <ModelField label="Number of Features" value={result.model_info?.n_features?.toLocaleString() ?? "—"} />
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
            model (a stratified split, with TF-IDF fitted on the training folds only). They
            describe the model's performance on the training corpus, <strong className="font-medium">not</strong> a
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
