import { cn } from "../../lib/utils";

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-slate-200/70 dark:bg-slate-800", className)} />;
}

export function SkeletonCard({ rows = 3 }: { rows?: number }) {
  return (
    <div className="card p-5">
      <Skeleton className="mb-4 h-4 w-32" />
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="mb-2 h-3 w-full" />
      ))}
    </div>
  );
}
