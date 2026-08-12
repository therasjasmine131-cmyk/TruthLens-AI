import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from "recharts";

interface HistogramProps {
  data: { bucket: string; count: number }[];
  color?: string;
  height?: number;
}

export function HistogramBars({ data, color = "#4f46e5", height = 240 }: HistogramProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200 dark:stroke-slate-800" />
        <XAxis dataKey="bucket" tick={{ fontSize: 11 }} className="fill-slate-500 dark:fill-slate-400" />
        <YAxis allowDecimals={false} tick={{ fontSize: 11 }} className="fill-slate-500 dark:fill-slate-400" />
        <Tooltip
          cursor={{ fill: "rgba(100,116,139,0.08)" }}
          contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12, background: "#ffffff" }}
        />
        <Bar dataKey="count" name="Analyses" radius={[4, 4, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.count > 0 ? color : "#cbd5e1"} fillOpacity={entry.count > 0 ? 1 : 0.4} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
