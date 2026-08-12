import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "../../lib/utils";

interface StatCardProps {
  label: string;
  value: ReactNode;
  icon: LucideIcon;
  hint?: string;
  tone?: "default" | "real" | "fake" | "uncertain" | "primary";
}

const TONES: Record<string, string> = {
  default: "text-slate-500 dark:text-slate-400",
  primary: "text-primary-600 dark:text-primary-400",
  real: "text-emerald-600 dark:text-emerald-400",
  fake: "text-rose-600 dark:text-rose-400",
  uncertain: "text-amber-600 dark:text-amber-400",
};

export function StatCard({ label, value, icon: Icon, hint, tone = "default" }: StatCardProps) {
  return (
    <div className="card card-hover p-4">
      <div className="flex items-start justify-between">
        <div className="min-w-0">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">{label}</p>
          <p className="mt-1.5 text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-50">
            {value}
          </p>
          {hint && <p className="mt-1 text-[11px] text-slate-400 dark:text-slate-500">{hint}</p>}
        </div>
        <div className={cn("rounded-lg bg-slate-100 p-2 dark:bg-slate-800", TONES[tone])}>
          <Icon className="h-4.5 w-4.5" size={18} />
        </div>
      </div>
    </div>
  );
}
