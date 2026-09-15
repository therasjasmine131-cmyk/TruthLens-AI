import { cn } from "../../lib/utils";

interface HorizontalBarsProps {
  items: { label: string; value: number; sub?: string; color?: string }[];
  emptyMessage?: string;
}

export function HorizontalBars({ items, emptyMessage = "No data" }: HorizontalBarsProps) {
  if (!items.length) {
    return <p className="py-6 text-center text-xs text-slate-400">{emptyMessage}</p>;
  }
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  return (
    <div className="flex flex-col gap-2.5">
      {items.map((item) => {
        const width = (Math.abs(item.value) / max) * 100;
        return (
          <div key={item.label} className="group">
            <div className="mb-1 flex items-center justify-between gap-2 text-xs">
              <span className="truncate font-medium text-slate-700 dark:text-slate-200">{item.label}</span>
              <span className="shrink-0 tabular-nums text-slate-400">{item.sub ?? item.value.toFixed(3)}</span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${width}%`, backgroundColor: item.color ?? "#4f46e5" }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function ExplanationBars({
  items,
}: {
  items: { term: string; contribution: number; influence: string }[];
}) {
  const max = Math.max(...items.map((i) => Math.abs(i.contribution)), 1e-9);
  return (
    <div className="flex flex-col gap-2.5">
      {items.map((item) => {
        const width = (Math.abs(item.contribution) / max) * 100;
        const isPositive = item.influence === "positive";
        const isNegative = item.influence === "negative";
        return (
          <div key={item.term}>
            <div className="mb-1 flex items-center justify-between gap-2 text-xs">
              <span className="truncate font-medium text-slate-700 dark:text-slate-200">{item.term}</span>
              <span
                className={cn(
                  "shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
                  isPositive && "bg-emerald-50 text-emerald-600 dark:bg-emerald-950 dark:text-emerald-400",
                  isNegative && "bg-rose-50 text-rose-600 dark:bg-rose-950 dark:text-rose-400",
                  item.influence === "low" && "bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400",
                )}
              >
                {isPositive ? "REAL-leaning" : isNegative ? "FAKE-leaning" : "low"}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className={cn("h-full rounded-full transition-all duration-700")}
                style={{
                  width: `${width}%`,
                  backgroundColor: isPositive ? "#059669" : isNegative ? "#dc2626" : "#94a3b8",
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
