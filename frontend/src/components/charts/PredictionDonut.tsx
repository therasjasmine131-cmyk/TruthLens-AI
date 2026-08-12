import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
} from "recharts";
import type { PredictionLabel } from "../../types";

const COLORS: Record<PredictionLabel, string> = {
  REAL: "#059669",
  FAKE: "#dc2626",
  UNCERTAIN: "#d97706",
};

interface DonutProps {
  data: { name: PredictionLabel; value: number }[];
  height?: number;
}

export function PredictionDonut({ data, height = 240 }: DonutProps) {
  const total = data.reduce((sum, d) => sum + d.value, 0);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          innerRadius={55}
          outerRadius={85}
          paddingAngle={3}
          strokeWidth={0}
        >
          {data.map((entry) => (
            <Cell key={entry.name} fill={COLORS[entry.name]} />
          ))}
        </Pie>
        <Tooltip
          formatter={(value: number) => [
            `${value.toLocaleString()} (${total ? ((value / total) * 100).toFixed(1) : 0}%)`,
            "Analyses",
          ]}
          contentStyle={{
            borderRadius: 8,
            border: "1px solid #e2e8f0",
            fontSize: 12,
          }}
        />
        <Legend iconType="circle" iconSize={8} formatter={(v) => <span className="text-xs text-slate-600 dark:text-slate-300">{v}</span>} />
      </PieChart>
    </ResponsiveContainer>
  );
}
