import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function truncate(text: string, length = 60): string {
  if (text.length <= length) return text;
  return `${text.slice(0, length - 1).trimEnd()}…`;
}

export function confidenceColor(value: number): string {
  if (value >= 0.75) return "text-emerald-600 dark:text-emerald-400";
  if (value >= 0.5) return "text-amber-600 dark:text-amber-400";
  return "text-slate-500 dark:text-slate-400";
}

export interface LiveStats {
  chars: number;
  words: number;
  sentences: number;
  unique_words: number;
  vocabulary_richness: number;
  average_sentence_length: number;
  capitalized_words: number;
  exclamation_marks: number;
  question_marks: number;
}

/**
 * Compute text statistics from the exact combined submitted text
 * (headline + article) -- the same source string sent to the model, and the
 * same algorithm the backend uses. Keeping a single source of truth prevents
 * the on-screen counter and the result statistics from disagreeing.
 */
export function computeTextStats(headline: string, article: string): LiveStats {
  const full = `${headline} ${article}`.trim();
  const words = full ? full.split(/\s+/).filter(Boolean) : [];
  const sentences = full.split(/[.!?]+/).filter((s) => s.trim().length > 0);
  const unique = new Set(words.map((w) => w.toLowerCase()));
  const caps = full.match(/\b[A-Z][A-Za-z]*\b/g) ?? [];
  return {
    chars: full.length,
    words: words.length,
    sentences: sentences.length,
    unique_words: unique.size,
    vocabulary_richness: words.length ? unique.size / words.length : 0,
    average_sentence_length: sentences.length ? words.length / sentences.length : 0,
    capitalized_words: caps.length,
    exclamation_marks: (full.match(/!/g) ?? []).length,
    question_marks: (full.match(/\?/g) ?? []).length,
  };
}

export function downloadBlob(content: string, filename: string, mime = "text/csv;charset=utf-8;"): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
