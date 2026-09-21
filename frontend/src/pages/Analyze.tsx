import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ClipboardPaste,
  Eraser,
  Wand2,
  ScanSearch,
  FileText,
  Sparkles,
  AlertTriangle,
} from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { TextArea, Input } from "../components/ui/Input";
import { EmptyState } from "../components/ui/EmptyState";
import { Spinner } from "../components/ui/Progress";
import { useToast } from "../context/ToastContext";
import { api } from "../api/client";
import { ResultPanel } from "../components/analyze/ResultPanel";
import { ProgressSteps } from "../components/analyze/ProgressSteps";
import { computeTextStats } from "../lib/utils";
import type { AnalysisResult, DemoArticle } from "../types";

const ANALYSIS_STEPS = [
  "Reading article",
  "Cleaning text",
  "Encoding tokens",
  "Running neural network",
  "Calculating confidence",
  "Preparing explanation",
];

const DEMO_LABELS: Record<string, string> = { REAL: "Dataset sample (real news)", FAKE: "Dataset sample (fake news)" };

export function Analyze() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [headline, setHeadline] = useState("");
  const [article, setArticle] = useState("");
  const [phase, setPhase] = useState<"idle" | "running" | "done">("idle");
  const [step, setStep] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [emptyNotice, setEmptyNotice] = useState<string | null>(null);
  const [aiError, setAiError] = useState<string | null>(null);

  const stats = useMemo(() => computeTextStats(headline, article), [headline, article]);

  const analyze = async (headlineOnly: boolean) => {
    // Never send empty text to the model.
    if (!headline.trim() && !article.trim()) {
      setResult(null);
      setPhase("idle");
      setAiError(null);
      setEmptyNotice("Please enter a headline or article to analyze.");
      toast("error", "Nothing to analyze", "Add an article or headline first.");
      return;
    }
    if (headlineOnly && !headline.trim()) {
      toast("error", "Headline required", "Enter a headline to analyze.");
      return;
    }
    setEmptyNotice(null);
    setAiError(null);
    setLoading(true);
    setPhase("running");
    setStep(0);
    const timer = setInterval(() => setStep((s) => Math.min(s + 1, ANALYSIS_STEPS.length - 1)), 300);
    try {
      const payload = headlineOnly
        ? await api.analyzeHeadline(headline)
        : await api.analyze(headline, article, true);
      clearInterval(timer);
      setStep(ANALYSIS_STEPS.length - 1);
      setResult(payload);
      setPhase("done");
    } catch (err) {
      clearInterval(timer);
      setPhase("idle");
      const message = err instanceof Error ? err.message : "Unknown error";
      setAiError(message);
      toast("error", "Analysis failed", message);
    } finally {
      setLoading(false);
    }
  };

  const loadDemo = async (demo: DemoArticle) => {
    setHeadline(demo.headline);
    setArticle(demo.article);
    setResult(null);
    setPhase("idle");
    setEmptyNotice(null);
    setAiError(null);
    toast("info", "Demo loaded", demo.disclaimer);
  };

  return (
    <div>
      <PageHeader
        title="Analyze News"
        subtitle="Enter a headline and article to run the AI credibility analysis."
      />

      <div className="grid gap-6 lg:grid-cols-2">
        {/* ---- Left: input ---- */}
        <Card className="self-start">
          <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Analyze an article</h2>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              The text is analyzed by a neural network (BiGRU) and cross-checked with live evidence.
            </p>
          </div>
          <div className="space-y-4 p-5">
            <Input
              label="Headline"
              placeholder="News headline (optional for full analysis)"
              value={headline}
              maxLength={500}
              onChange={(e) => setHeadline(e.target.value)}
            />

            <TextArea
              label="Article"
              name="article"
              placeholder="Paste the news article here..."
              value={article}
              maxLength={12000}
              onChange={(e) => setArticle(e.target.value)}
            />

            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-500 dark:text-slate-400">
              <span>
                <strong className="tabular-nums">{stats.chars.toLocaleString()}</strong> chars
              </span>
              <span>
                <strong className="tabular-nums">{stats.words.toLocaleString()}</strong> words
              </span>
              <span>
                <strong className="tabular-nums">{stats.sentences.toLocaleString()}</strong> sentences
              </span>
            </div>

            <div className="flex flex-wrap gap-2">
              <Button variant="outline" size="sm" icon={<ClipboardPaste size={14} />} onClick={async () => {
                try {
                  const text = await navigator.clipboard.readText();
                  setArticle(text);
                  toast("success", "Pasted from clipboard");
                } catch {
                  toast("error", "Clipboard unavailable", "Allow clipboard access or paste manually.");
                }
              }}>
                Paste
              </Button>
              <Button variant="outline" size="sm" icon={<Eraser size={14} />} onClick={() => {
                setHeadline("");
                setArticle("");
                setResult(null);
                setPhase("idle");
                setEmptyNotice(null);
              }}>
                Clear
              </Button>
              <Button variant="outline" size="sm" icon={<Wand2 size={14} />} onClick={() => {
                const demoPanel = document.getElementById("demo-panel");
                demoPanel?.scrollIntoView({ behavior: "smooth", block: "start" });
              }}>
                Load sample
              </Button>
            </div>

            <div className="flex flex-col gap-2 border-t border-slate-200 pt-4 dark:border-slate-800 sm:flex-row">
              <Button
                className="flex-1"
                size="lg"
                loading={loading}
                icon={<ScanSearch size={16} />}
                onClick={() => analyze(false)}
              >
                Analyze Article
              </Button>
              <Button
                variant="secondary"
                size="lg"
                loading={loading && !article.trim()}
                icon={<FileText size={16} />}
                onClick={() => analyze(true)}
                disabled={loading}
              >
                Headline Only
              </Button>
            </div>
          </div>
        </Card>

        {/* ---- Right: preview / result ---- */}
        <div className="min-w-0">
          {phase === "idle" && (
            <Card className="min-h-[480px]">
              {aiError ? (
                <div className="flex h-full flex-col items-center justify-center p-8 text-center">
                  <AlertTriangle size={22} className="mb-4 text-rose-500" />
                  <p className="text-sm font-semibold text-rose-700 dark:text-rose-400">
                    AI verification unavailable
                  </p>
                  <p className="mt-2 max-w-md text-xs text-slate-600 dark:text-slate-300">{aiError}</p>
                  <p className="mt-3 text-xs text-slate-400 dark:text-slate-500">
                    No verdict was generated. Please try again shortly.
                  </p>
                </div>
              ) : emptyNotice ? (
                <div className="flex h-full flex-col items-center justify-center p-8 text-center">
                  <AlertTriangle size={22} className="mb-4 text-amber-500" />
                  <p className="text-sm font-medium text-slate-700 dark:text-slate-200">{emptyNotice}</p>
                  <p className="mt-1 text-xs text-slate-400 dark:text-slate-500">
                    No model was run and nothing was saved.
                  </p>
                </div>
              ) : (
                <div className="flex h-full flex-col items-center justify-center p-8 text-center">
                  <EmptyState
                    compact
                    icon={<Sparkles size={22} />}
                    title="Your analysis will appear here"
                    description="Enter a headline or article and run the AI analysis to see the prediction, confidence, keywords and explanation."
                  />
                </div>
              )}
            </Card>
          )}

          {phase === "running" && (
            <Card className="flex min-h-[480px] flex-col items-center justify-center p-8">
              <Spinner className="mb-5 h-8 w-8" />
              <p className="mb-5 text-sm font-medium text-slate-700 dark:text-slate-200">
                Analyzing article…
              </p>
              <ProgressSteps steps={ANALYSIS_STEPS} activeStep={step} />
            </Card>
          )}

          {phase === "done" && result && (
            <ResultPanel result={result} onViewHistory={() => navigate("/history")} />
          )}
        </div>
      </div>

      {/* ---- Demo samples ---- */}
      <DemoPanel onLoad={loadDemo} />
    </div>
  );
}

function DemoPanel({ onLoad }: { onLoad: (demo: DemoArticle) => void }) {
  const [demos, setDemos] = useState<DemoArticle[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const data = await api.samples();
      setDemos(data.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load samples");
    }
  };

  return (
    <Card className="mt-6" id="demo-panel">
      <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Try Demo</h2>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              Educational demonstration only — not a real news source.
            </p>
          </div>
          {!demos && !error && (
            <Button size="sm" variant="outline" onClick={load} icon={<Wand2 size={14} />}>
              Load samples
            </Button>
          )}
        </div>
      </div>
      <div className="p-5">
        {error && <p className="text-sm text-rose-600">{error}</p>}
        {demos && (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {demos.map((demo) => (
              <button
                key={demo.id}
                onClick={() => onLoad(demo)}
                className="group flex flex-col rounded-lg border border-slate-200 p-4 text-left transition-all hover:border-primary-300 hover:shadow-card-hover dark:border-slate-800 dark:hover:border-primary-700"
              >
                <span className="chip mb-2 self-start text-[10px] text-slate-500 dark:text-slate-400">
                  {DEMO_LABELS[demo.label] || demo.label}
                </span>
                <p className="line-clamp-2 text-xs font-medium text-slate-700 dark:text-slate-200">
                  {demo.headline}
                </p>
                <p className="mt-2 line-clamp-3 text-[11px] text-slate-500 dark:text-slate-400">
                  {demo.article.slice(0, 300)}
                </p>
              </button>
            ))}
          </div>
        )}
      </div>
    </Card>
  );
}
