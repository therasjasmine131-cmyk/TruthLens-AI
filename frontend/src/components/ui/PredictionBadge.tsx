import type { PredictionLabel } from "../../types";
import { cn } from "../../lib/utils";

const LABEL_STYLES: Record<PredictionLabel, { text: string; bg: string; dot: string }> = {
  REAL: {
    text: "text-emerald-600 dark:text-emerald-400",
    bg: "bg-emerald-50 border-emerald-200 dark:bg-emerald-950/60 dark:border-emerald-800",
    dot: "bg-emerald-500",
  },
  FAKE: {
    text: "text-rose-600 dark:text-rose-400",
    bg: "bg-rose-50 border-rose-200 dark:bg-rose-950/60 dark:border-rose-800",
    dot: "bg-rose-500",
  },
  UNCERTAIN: {
    text: "text-amber-600 dark:text-amber-400",
    bg: "bg-amber-50 border-amber-200 dark:bg-amber-950/60 dark:border-amber-800",
    dot: "bg-amber-500",
  },
};

export function PredictionBadge({
  label,
  className,
  size = "md",
}: {
  label: PredictionLabel;
  className?: string;
  size?: "sm" | "md" | "lg";
}) {
  const style = LABEL_STYLES[label];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border font-semibold",
        style.bg,
        size === "lg" ? "px-3 py-1 text-sm" : "px-2 py-0.5 text-xs",
        className,
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", style.dot)} />
      <span className={style.text}>{label}</span>
    </span>
  );
}

export function predictionColor(label: PredictionLabel): string {
  if (label === "REAL") return "#059669";
  if (label === "FAKE") return "#dc2626";
  return "#d97706";
}
