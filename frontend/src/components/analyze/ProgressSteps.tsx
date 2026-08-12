import { cn } from "../../lib/utils";

export function ProgressSteps({ steps, activeStep }: { steps: string[]; activeStep: number }) {
  return (
    <ol className="w-full max-w-sm space-y-2">
      {steps.map((label, i) => {
        const done = i < activeStep;
        const active = i === activeStep;
        return (
          <li key={label} className="flex items-center gap-3">
            <span
              className={cn(
                "flex h-5 w-5 items-center justify-center rounded-full border text-[10px] font-semibold transition-colors",
                done && "border-emerald-500 bg-emerald-500 text-white",
                active && "border-primary-500 text-primary-600 dark:text-primary-400",
                !done && !active && "border-slate-300 text-slate-400 dark:border-slate-700",
              )}
            >
              {done ? "✓" : i + 1}
            </span>
            <span
              className={cn(
                "text-xs transition-colors",
                active ? "font-semibold text-slate-800 dark:text-slate-100" : "text-slate-400",
              )}
            >
              {label}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
