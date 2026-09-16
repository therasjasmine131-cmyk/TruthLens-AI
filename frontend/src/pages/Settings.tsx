import { useState } from "react";
import {
  Monitor,
  Sun,
  Moon,
  Trash2,
  Download,
  Info,
  FlaskConical,
  SlidersHorizontal,
} from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Modal } from "../components/ui/Modal";
import { useApi } from "../hooks/useApi";
import { useTheme, type ThemePreference } from "../context/ThemeContext";
import { useToast } from "../context/ToastContext";
import { api } from "../api/client";
import type { SettingsData } from "../types";
import { cn } from "../lib/utils";

const THEME_OPTIONS: { value: ThemePreference; label: string; icon: typeof Sun }[] = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
];

export function Settings() {
  const { toast } = useToast();
  const { preference, setPreference } = useTheme();
  const { data, refresh } = useApi<SettingsData>(() => api.settings(), []);
  const [confirmClear, setConfirmClear] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [maxArticle, setMaxArticle] = useState<string>("12000");
  const [thresholds, setThresholds] = useState({ very_high_min: 0.9, high_min: 0.75, moderate_min: 0.5 });

  const saveSettings = async () => {
    try {
      const values: Record<string, unknown> = {
        max_article_length: Number(maxArticle),
        confidence_levels: thresholds,
      };
      await api.updateSettings(values as Partial<SettingsData>);
      toast("success", "Settings saved");
      refresh();
    } catch (err) {
      toast("error", "Could not save settings", err instanceof Error ? err.message : undefined);
    }
  };

  const clearHistory = async () => {
    setClearing(true);
    try {
      const res = await api.clearHistory();
      toast("success", "History cleared", `${res.deleted} records removed`);
      setConfirmClear(false);
    } catch (err) {
      toast("error", "Clear failed", err instanceof Error ? err.message : undefined);
    } finally {
      setClearing(false);
    }
  };

  const exportHistory = async () => {
    try {
      const res = await api.exportHistory();
      const blob = new Blob([JSON.stringify(res, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "truthlens-history.json";
      link.click();
      URL.revokeObjectURL(url);
      toast("success", "History exported", `${res.items.length} records`);
    } catch (err) {
      toast("error", "Export failed", err instanceof Error ? err.message : undefined);
    }
  };

  return (
    <div>
      <PageHeader title="Settings" subtitle="Application preferences and data management." />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
              <Monitor size={15} className="text-primary-500" /> Theme
            </h3>
          </div>
          <div className="p-5">
            <div className="grid grid-cols-3 gap-2">
              {THEME_OPTIONS.map(({ value, label, icon: Icon }) => (
                <button
                  key={value}
                  onClick={() => setPreference(value)}
                  className={cn(
                    "flex flex-col items-center gap-2 rounded-lg border px-3 py-4 text-xs font-medium transition-all",
                    preference === value
                      ? "border-primary-500 bg-primary-50 text-primary-700 dark:bg-primary-950 dark:text-primary-300"
                      : "border-slate-200 text-slate-500 hover:border-slate-300 dark:border-slate-800 dark:text-slate-400",
                  )}
                >
                  <Icon size={18} />
                  {label}
                </button>
              ))}
            </div>
            <p className="mt-3 text-[11px] text-slate-400">
              Preference is persisted locally and applied instantly.
            </p>
          </div>
        </Card>

        <Card>
          <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
              <SlidersHorizontal size={15} className="text-primary-500" /> Analysis settings
            </h3>
          </div>
          <div className="space-y-4 p-5">
            <div>
              <label className="label" htmlFor="max-article">
                Maximum article length (characters)
              </label>
              <input
                id="max-article"
                className="input"
                type="number"
                min={1000}
                max={100000}
                value={maxArticle}
                onChange={(e) => setMaxArticle(e.target.value)}
              />
              <p className="mt-1 text-[11px] text-slate-400">Current server value: {data?.max_article_length?.toLocaleString() ?? "—"}</p>
            </div>

            <div>
              <label className="label">Confidence thresholds (fractions)</label>
              <div className="grid grid-cols-3 gap-3">
                {(
                  [
                    ["very_high_min", "Very High ≥"],
                    ["high_min", "High ≥"],
                    ["moderate_min", "Moderate ≥"],
                  ] as const
                ).map(([key, label]) => (
                  <div key={key}>
                    <label className="mb-1 block text-[11px] text-slate-500">{label}</label>
                    <input
                      className="input"
                      type="number"
                      step={0.05}
                      min={0}
                      max={1}
                      value={thresholds[key]}
                      onChange={(e) =>
                        setThresholds((prev) => ({ ...prev, [key]: Number(e.target.value) }))
                      }
                    />
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-slate-400">
                Confidence level labels: Low 0–49%, Moderate 50–74%, High 75–89%, Very High 90–100%.
                Defaults: {data?.confidence_levels?.very_high_min ?? 0.9} / {data?.confidence_levels?.high_min ?? 0.75} / {data?.confidence_levels?.moderate_min ?? 0.5}.
              </p>
            </div>

            <div className="flex justify-end gap-2 border-t border-slate-200 pt-4 dark:border-slate-800">
              <Button variant="outline" size="sm" onClick={() => { setMaxArticle(String(data?.max_article_length ?? 12000)); setThresholds(data?.confidence_levels ?? { very_high_min: 0.9, high_min: 0.75, moderate_min: 0.5 }); }}>
                Reset
              </Button>
              <Button size="sm" onClick={saveSettings}>
                Save settings
              </Button>
            </div>
          </div>
        </Card>

        <Card>
          <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
              <Trash2 size={15} className="text-primary-500" /> Data management
            </h3>
          </div>
          <div className="space-y-3 p-5">
            <Button variant="outline" size="sm" className="w-full justify-start" icon={<Download size={14} />} onClick={exportHistory}>
              Export history (JSON)
            </Button>
            <Button variant="danger" size="sm" className="w-full justify-start" icon={<Trash2 size={14} />} onClick={() => setConfirmClear(true)}>
              Clear prediction history
            </Button>
          </div>
        </Card>

        <Card>
          <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
              <Info size={15} className="text-primary-500" /> About
            </h3>
          </div>
          <div className="space-y-4 p-5">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-primary-600 text-white">
                <FlaskConical size={20} />
              </div>
              <div>
                <p className="text-sm font-bold text-slate-900 dark:text-slate-50">TruthLens AI</p>
                <p className="text-xs text-slate-500">Version 1.0.0</p>
              </div>
            </div>
            <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">
              TruthLens AI is an educational machine-learning platform that estimates news credibility
              with a neural network (Embedding → BiGRU) and evidence-based verification. Predictions
              are estimates, not proof.
              Built with Flask, NumPy, React and Tailwind CSS.
            </p>
          </div>
        </Card>
      </div>

      <Modal
        open={confirmClear}
        onClose={() => setConfirmClear(false)}
        title="Clear prediction history?"
        footer={
          <>
            <Button variant="outline" size="sm" onClick={() => setConfirmClear(false)}>
              Cancel
            </Button>
            <Button variant="danger" size="sm" loading={clearing} onClick={clearHistory}>
              Clear history
            </Button>
          </>
        }
      >
        <p className="text-sm text-slate-600 dark:text-slate-300">
          This permanently deletes every stored analysis from the database.
        </p>
      </Modal>
    </div>
  );
}
