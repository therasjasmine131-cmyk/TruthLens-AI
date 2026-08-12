import { useEffect, useState } from "react";
import type { PredictionLabel } from "../../types";
import { predictionColor } from "../ui/PredictionBadge";

interface ConfidenceGaugeProps {
  confidence: number;
  prediction: PredictionLabel;
}

const RADIUS = 82;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export function ConfidenceGauge({ confidence, prediction }: ConfidenceGaugeProps) {
  const [offset, setOffset] = useState(CIRCUMFERENCE);
  const pct = Math.round(confidence * 1000) / 10;
  const target = CIRCUMFERENCE * (1 - confidence);

  useEffect(() => {
    const frame = requestAnimationFrame(() => setOffset(target));
    return () => cancelAnimationFrame(frame);
  }, [target]);

  return (
    <div className="relative inline-flex flex-col items-center">
      <svg width="200" height="120" viewBox="0 0 200 120" className="overflow-visible">
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          strokeWidth="12"
          strokeLinecap="round"
          className="stroke-slate-200 dark:stroke-slate-800"
        />
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          strokeWidth="12"
          strokeLinecap="round"
          stroke={predictionColor(prediction)}
          style={{ strokeDasharray: CIRCUMFERENCE, strokeDashoffset: offset }}
          className="transition-[stroke-dashoffset] duration-1000 ease-out"
        />
      </svg>
      <div className="absolute bottom-0 flex flex-col items-center">
        <span className="text-4xl font-bold tabular-nums tracking-tight text-slate-900 dark:text-slate-50">
          {pct.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}
