import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

interface ComparisonProps {
  data: {
    name: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  }[];
  height?: number;
}

const METRIC_COLORS = ["#4f46e5", "#059669", "#d97706", "#dc2626"];

export function ModelComparisonBars({ data, height = 300 }: ComparisonProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200 dark:stroke-slate-800" />
        <XAxis dataKey="name" tick={{ fontSize: 11 }} className="fill-slate-500 dark:fill-slate-400" interval={0} />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          tick={{ fontSize: 11 }}
          className="fill-slate-500 dark:fill-slate-400"
        />
        <Tooltip
          formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
          contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12, background: "#ffffff" }}
        />
        <Legend iconType="circle" iconSize={8} formatter={(v) => <span className="text-xs text-slate-600 dark:text-slate-300">{v}</span>} />
        {["accuracy", "precision", "recall", "f1"].map((key, i) => (
          <Bar key={key} dataKey={key} name={key.toUpperCase()} fill={METRIC_COLORS[i]} radius={[3, 3, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
