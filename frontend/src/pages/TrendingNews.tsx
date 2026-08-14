import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Newspaper, RefreshCw, ExternalLink, ScanSearch, Bot, Clock } from "lucide-react";
import { PageHeader, Disclaimer } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import { Spinner } from "../components/ui/Progress";
import { useToast } from "../context/ToastContext";
import { api } from "../api/client";
import type { AiTextLabel, NewsArticle, TrendingNewsResponse } from "../types";

const COUNTRIES = [
  { code: "us", name: "United States" },
  { code: "gb", name: "United Kingdom" },
  { code: "ca", name: "Canada" },
  { code: "au", name: "Australia" },
  { code: "de", name: "Germany" },
  { code: "fr", name: "France" },
  { code: "in", name: "India" },
  { code: "jp", name: "Japan" },
  { code: "br", name: "Brazil" },
  { code: "mx", name: "Mexico" },
  { code: "za", name: "South Africa" },
  { code: "ng", name: "Nigeria" },
];

const LABEL_STYLES: Record<AiTextLabel, string> = {
  "Likely AI": "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  "Likely Human": "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  Uncertain: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
};

interface AiCheck {
  loading: boolean;
  label?: AiTextLabel;
  score?: number;
  error?: string;
}

export function TrendingNews() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [country, setCountry] = useState("us");
  const [data, setData] = useState<TrendingNewsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aiChecks, setAiChecks] = useState<Record<number, AiCheck>>({});
  const [analyzingId, setAnalyzingId] = useState<number | null>(null);

  const load = async (nextCountry: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.trendingNews(nextCountry);
      setData(result);
      setAiChecks({});
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load trending news.");
    } finally {
      setLoading(false);
    }
  };

  const analyze = async (article: NewsArticle) => {
    setAnalyzingId(article.id);
    try {
      const body = (article.article || article.description || "").trim();
      const result = body
        ? await api.analyze(article.headline, body, true)
        : await api.analyzeHeadline(article.headline);
      toast("success", "Analysis complete", `Credibility: ${result.prediction}`);
      if (result.history_id) {
        navigate(`/report/${result.history_id}`);
      } else {
        navigate("/analyze");
      }
    } catch (err) {
      toast("error", "Analysis failed", err instanceof Error ? err.message : "Unknown error");
    } finally {
      setAnalyzingId(null);
    }
  };

  const checkAi = async (article: NewsArticle) => {
    setAiChecks((prev) => ({ ...prev, [article.id]: { loading: true } }));
    try {
      const text = [article.headline, article.description, article.article].filter(Boolean).join("\n");
      const result = await api.detectAiText(text.slice(0, 4000));
      if (result.error) throw new Error(result.error);
      setAiChecks((prev) => ({
        ...prev,
        [article.id]: { loading: false, label: result.label, score: result.ai_generated_score },
      }));
    } catch (err) {
      setAiChecks((prev) => ({
        ...prev,
        [article.id]: { loading: false, error: err instanceof Error ? err.message : "AI check failed" },
      }));
    }
  };

  const articles = useMemo(() => data?.items ?? [], [data]);

  return (
    <div>
      <PageHeader
        title="Trending News"
        subtitle="Top headlines from NewsAPI — run a credibility or AI-text check on real articles."
        actions={
          <div className="flex items-center gap-2">
            <select
              value={country}
              onChange={(e) => {
                setCountry(e.target.value);
                load(e.target.value);
              }}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500/30 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
            >
              {COUNTRIES.map((c) => (
                <option key={c.code} value={c.code}>
                  {c.name}
                </option>
              ))}
            </select>
            <Button variant="outline" size="md" icon={<RefreshCw size={14} />} onClick={() => load(country)}>
              Refresh
            </Button>
          </div>
        }
      />

      <Disclaimer text="Trending headlines are fetched from NewsAPI.org and are shown for educational analysis only. They are not claims about accuracy." />

      {loading && !data && (
        <Card className="mt-6 flex min-h-[320px] flex-col items-center justify-center p-8">
          <Spinner className="mb-4 h-8 w-8" />
          <p className="text-sm text-slate-500 dark:text-slate-400">Fetching latest headlines…</p>
        </Card>
      )}

      {error && !data && (
        <Card className="mt-6">
          <EmptyState
            compact
            icon={<Newspaper size={22} />}
            title="Could not load trending news"
            description={error}
          />
        </Card>
      )}

      {data && articles.length === 0 && (
        <Card className="mt-6">
          <EmptyState compact icon={<Newspaper size={22} />} title="No headlines found" description="Try a different country." />
        </Card>
      )}

      <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {articles.map((article) => (
          <ArticleCard
            key={article.id}
            article={article}
            analyzing={analyzingId === article.id}
            aiCheck={aiChecks[article.id]}
            onAnalyze={() => analyze(article)}
            onCheckAi={() => checkAi(article)}
          />
        ))}
      </div>
    </div>
  );
}

function ArticleCard({
  article,
  analyzing,
  aiCheck,
  onAnalyze,
  onCheckAi,
}: {
  article: NewsArticle;
  analyzing: boolean;
  aiCheck?: AiCheck;
  onAnalyze: () => void;
  onCheckAi: () => void;
}) {
  return (
    <Card className="flex flex-col">
      {article.image_url ? (
        <div className="h-36 w-full overflow-hidden rounded-t-lg">
          <img
            src={article.image_url}
            alt=""
            className="h-full w-full object-cover"
            onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
          />
        </div>
      ) : (
        <div className="flex h-24 items-center justify-center rounded-t-lg bg-slate-100 dark:bg-slate-800/60">
          <Newspaper size={22} className="text-slate-400" />
        </div>
      )}

      <div className="flex flex-1 flex-col p-4">
        <div className="mb-2 flex items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400">
          <span className="chip">{article.source}</span>
          {article.published_at && (
            <span className="inline-flex items-center gap-1">
              <Clock size={11} />
              {new Date(article.published_at).toLocaleDateString()}
            </span>
          )}
        </div>

        <h3 className="line-clamp-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
          {article.headline}
        </h3>
        {article.description && (
          <p className="mt-1.5 line-clamp-3 text-xs text-slate-500 dark:text-slate-400">
            {article.description}
          </p>
        )}

        <div className="mt-4 flex flex-1 flex-col justify-end gap-2">
          {aiCheck && aiCheck.label && (
            <div className={`inline-flex items-center gap-2 self-start rounded-full px-2.5 py-1 text-[11px] font-medium ${LABEL_STYLES[aiCheck.label]}`}>
              <Bot size={12} />
              AI text: {aiCheck.label} · {Math.round((aiCheck.score ?? 0) * 100)}%
            </div>
          )}
          {aiCheck?.error && <p className="text-[11px] text-rose-600">{aiCheck.error}</p>}

          <div className="flex flex-wrap gap-2">
            <Button size="sm" loading={analyzing} icon={<ScanSearch size={13} />} onClick={onAnalyze}>
              Analyze
            </Button>
            <Button size="sm" variant="outline" loading={aiCheck?.loading} icon={<Bot size={13} />} onClick={onCheckAi}>
              AI check
            </Button>
            {article.url && (
              <a
                href={article.url}
                target="_blank"
                rel="noreferrer"
                className="ml-auto inline-flex items-center gap-1 text-[11px] font-medium text-primary-600 hover:underline dark:text-primary-400"
              >
                Source <ExternalLink size={11} />
              </a>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
