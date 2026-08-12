import { cn } from "../../lib/utils";

export function ProgressBar({ value, className }: { value: number; className?: string }) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800", className)}>
      <div
        className="h-full rounded-full bg-primary-500 transition-all duration-500"
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-primary-600 dark:border-slate-700 dark:border-t-primary-400",
        className,
      )}
      aria-label="Loading"
    />
  );
}
