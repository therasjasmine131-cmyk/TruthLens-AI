import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

interface RocProps {
  data: { fpr: number[]; tpr: number[] };
  label: string;
  height?: number;
}

export function RocChart({ data, label, height = 240 }: RocProps) {
  const points = data.fpr.map((fpr, i) => ({ fpr: +fpr.toFixed(4), tpr: +data.tpr[i].toFixed(4) }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={points} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200 dark:stroke-slate-800" />
        <XAxis
          dataKey="fpr"
          type="number"
          domain={[0, 1]}
          tick={{ fontSize: 11 }}
          className="fill-slate-500 dark:fill-slate-400"
        />
        <YAxis
          type="number"
          domain={[0, 1]}
          tick={{ fontSize: 11 }}
          className="fill-slate-500 dark:fill-slate-400"
        />
        <Tooltip
          formatter={(value: number) => value.toFixed(3)}
          contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12, background: "#ffffff" }}
        />
        <Line type="monotone" dataKey="tpr" name={`TPR (${label})`} stroke="#4f46e5" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
