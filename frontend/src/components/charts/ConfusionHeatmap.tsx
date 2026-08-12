import { cn } from "../../lib/utils";

interface ConfusionHeatmapProps {
  matrix: number[][];
  labels: string[];
  rowsLabel?: string;
  colsLabel?: string;
}

export function ConfusionHeatmap({ matrix, labels }: ConfusionHeatmapProps) {
  if (!matrix.length) return null;
  const max = Math.max(...matrix.flat());

  const colorFor = (value: number) => {
    const ratio = max ? value / max : 0;
    if (ratio >= 0.9) return "bg-primary-600 text-white";
    if (ratio >= 0.6) return "bg-primary-400 text-white";
    if (ratio >= 0.3) return "bg-primary-200 text-primary-900 dark:bg-primary-800 dark:text-primary-50";
    return "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300";
  };

  return (
    <div className="overflow-x-auto">
      <table className="mx-auto border-separate border-spacing-1">
        <thead>
          <tr>
            <th className="px-2 pb-2 text-right text-[10px] font-medium uppercase tracking-wide text-slate-400">
              Actual ↓ / Predicted →
            </th>
            {labels.map((label) => (
              <th key={label} className="px-2 pb-2 text-center text-[10px] font-semibold uppercase text-slate-500 dark:text-slate-400">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="pr-2 text-right text-[10px] font-semibold uppercase text-slate-500 dark:text-slate-400">
                {labels[i]}
              </td>
              {row.map((value, j) => (
                <td key={j}>
                  <div
                    className={cn(
                      "flex h-16 w-24 items-center justify-center rounded-md text-sm font-bold tabular-nums",
                      colorFor(value),
                    )}
                    title={`${labels[i]} (actual) vs ${labels[j]} (predicted)`}
                  >
                    {value.toLocaleString()}
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
