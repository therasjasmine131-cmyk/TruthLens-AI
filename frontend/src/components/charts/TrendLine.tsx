import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

interface TrendPoint {
  bucket: string;
  real: number;
  fake: number;
  uncertain: number;
  total: number;
}

export function TrendLine({ data }: { data: TrendPoint[] }) {
  const formatted = data.map((d) => ({
    ...d,
    label: new Date(d.bucket).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={formatted} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <defs>
          <linearGradient id="gradReal" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#059669" stopOpacity={0.35} />
            <stop offset="100%" stopColor="#059669" stopOpacity={0.02} />
          </linearGradient>
          <linearGradient id="gradFake" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#dc2626" stopOpacity={0.35} />
            <stop offset="100%" stopColor="#dc2626" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200 dark:stroke-slate-800" />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 11 }}
          className="fill-slate-500 dark:fill-slate-400"
          minTickGap={24}
        />
        <YAxis
          allowDecimals={false}
          tick={{ fontSize: 11 }}
          className="fill-slate-500 dark:fill-slate-400"
        />
        <Tooltip
          contentStyle={{
            borderRadius: 8,
            border: "1px solid #e2e8f0",
            fontSize: 12,
            background: "#ffffff",
          }}
          labelStyle={{ fontWeight: 600, marginBottom: 4 }}
        />
        <Area type="monotone" dataKey="real" name="REAL" stroke="#059669" fill="url(#gradReal)" strokeWidth={2} />
        <Area type="monotone" dataKey="fake" name="FAKE" stroke="#dc2626" fill="url(#gradFake)" strokeWidth={2} />
        <Area type="monotone" dataKey="uncertain" name="UNCERTAIN" stroke="#d97706" fill="transparent" strokeWidth={2} strokeDasharray="4 3" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
