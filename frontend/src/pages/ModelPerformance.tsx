import { useApi } from "../hooks/useApi";
import { api } from "../api/client";
import { PageHeader } from "../components/ui/PageHeader";
import { Card, CardHeader } from "../components/ui/Card";
import { StatCard } from "../components/ui/StatCard";
import { EmptyState } from "../components/ui/EmptyState";
import { Skeleton } from "../components/ui/Skeleton";
import { Button } from "../components/ui/Button";
import { ModelComparisonBars } from "../components/charts/ModelComparisonBars";
import { ConfusionHeatmap } from "../components/charts/ConfusionHeatmap";
import { RocChart } from "../components/charts/RocChart";
import {
  Target,
  Crosshair,
  Radar,
  Scale,
  LineChart,
  Download,
} from "lucide-react";
import type { ModelPerformanceData, ModelMetrics } from "../types";
import { formatPercent } from "../lib/utils";

export function ModelPerformance() {
  const { data, loading, error } = useApi<ModelPerformanceData>(() => api.modelPerformance(), []);

  if (loading) {
    return (
      <div>
        <PageHeader title="Model Performance" subtitle="Measured metrics from training and evaluation." />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return <EmptyState title="Could not load model data" description={error} />;
  }

  if (!data?.trained) {
    return (
      <div>
        <PageHeader title="Model Performance" />
        <Card className="p-8">
          <EmptyState
            title="Train the model to view performance metrics."
            description={data?.message ?? "Run `python ml/train.py` and restart the backend."}
          />
        </Card>
      </div>
    );
  }

  const bm = (data.best_metrics ?? {}) as ModelMetrics;
  const comparisonData = data.models.map((m) => ({
    name: m.name,
    accuracy: m.accuracy ?? 0,
    precision: m.precision ?? 0,
    recall: m.recall ?? 0,
    f1: m.f1 ?? 0,
  }));

  const kpis = [
    { label: "Accuracy", value: bm.accuracy, icon: Target },
    { label: "Precision", value: bm.precision, icon: Crosshair },
    { label: "Recall", value: bm.recall, icon: Radar },
    { label: "F1 Score", value: bm.f1, icon: Scale },
    { label: "ROC-AUC", value: bm.roc_auc, icon: LineChart, raw: true },
  ];

  return (
    <div>
      <PageHeader
        title="Model Performance"
        subtitle={`Measured on the held-out test set · best model: ${data.best_model}`}
        actions={
          <Button
            variant="outline"
            size="sm"
            icon={<Download size={14} />}
            onClick={() => {
              const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
              const url = URL.createObjectURL(blob);
              const link = document.createElement("a");
              link.href = url;
              link.download = "truthlens-model-performance.json";
              link.click();
              URL.revokeObjectURL(url);
            }}
          >
            Export metrics
          </Button>
        }
      />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        {kpis.map((kpi) => (
          <StatCard
            key={kpi.label}
            label={kpi.label}
            value={kpi.value != null ? (kpi.raw ? kpi.value.toFixed(4) : formatPercent(kpi.value)) : "—"}
            icon={kpi.icon}
            tone="primary"
          />
        ))}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Model Comparison" subtitle="F1, precision, recall and accuracy per model" />
          <div className="p-4">
            <ModelComparisonBars data={comparisonData} />
          </div>
        </Card>

        <Card>
          <CardHeader title="Confusion Matrix" subtitle={`Actual vs predicted on the test set (${data.best_model})`} />
          <div className="p-5">
            <ConfusionHeatmap
              matrix={data.best_confusion_matrix ?? []}
              labels={data.class_labels ?? ["FAKE", "REAL"]}
            />
            <div className="mt-4 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
              <MatrixNote label="Correctly classified" value="Diagonal cells" tone="text-emerald-600 dark:text-emerald-400" />
              <MatrixNote label="False positives" value="Col 1, row 0" tone="text-rose-600 dark:text-rose-400" />
              <MatrixNote label="False negatives" value="Col 0, row 1" tone="text-rose-600 dark:text-rose-400" />
              <MatrixNote label="Test support" value={`${(bm.support ?? 0).toLocaleString()} docs`} tone="text-slate-500 dark:text-slate-400" />
            </div>
          </div>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="ROC Curve" subtitle={`Best model (${data.best_model}) — AUC ${bm.roc_auc != null ? bm.roc_auc.toFixed(4) : "—"}`} />
          <div className="p-4">
            {bm.roc_curve ? <RocChart data={bm.roc_curve} label={data.best_model ?? "best"} /> : <EmptyState compact title="ROC data unavailable" />}
          </div>
        </Card>

        <Card>
          <CardHeader title="Training Configuration" />
          <div className="grid grid-cols-2 gap-x-4 gap-y-4 p-5 text-xs">
            <ConfigField label="Dataset Source" value={data.dataset_source ?? "—"} />
            <ConfigField label="Class Labels" value={data.class_labels?.join(" / ") ?? "—"} />
            <ConfigField label="Training Samples" value={(data.train_samples ?? 0).toLocaleString()} />
            <ConfigField label="Test Samples" value={(data.test_samples ?? 0).toLocaleString()} />
            <ConfigField label="Vocabulary Size" value={(data.n_features ?? 0).toLocaleString()} />
            <ConfigField label="Training Date" value={data.training_date ? new Date(data.training_date).toLocaleDateString() : "—"} />
            <ConfigField label="Model" value={data.best_model ?? "—"} />
            <ConfigField label="Split Method" value="Leakage-safe grouped split" />
          </div>
          <p className="px-5 pb-5 text-[11px] text-slate-400 dark:text-slate-500">
            All metrics are measured on the held-out test set and stored in models/fake_news_neural_network — nothing is
            fabricated.
          </p>
        </Card>
      </div>

      <div className="mt-6">
        <Card>
          <CardHeader title="Per-Model ROC Curves" subtitle="ROC curves for all four classifiers" />
          <div className="grid gap-4 p-4 lg:grid-cols-2">
            {data.models.filter((m) => m.roc_curve).map((m) => (
              <div key={m.name} className="rounded-lg border border-slate-200 p-3 dark:border-slate-800">
                <p className="mb-2 text-xs font-semibold text-slate-700 dark:text-slate-200">
                  {m.name} · AUC {m.roc_auc != null ? m.roc_auc.toFixed(4) : "—"}
                </p>
                <RocChart data={m.roc_curve!} label={m.name} height={180} />
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function MatrixNote({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div>
      <p className="text-[11px] text-slate-400">{label}</p>
      <p className={`font-medium ${tone}`}>{value}</p>
    </div>
  );
}

function ConfigField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[11px] text-slate-400 dark:text-slate-500">{label}</p>
      <p className="mt-0.5 font-medium text-slate-800 dark:text-slate-200">{value}</p>
    </div>
  );
}
