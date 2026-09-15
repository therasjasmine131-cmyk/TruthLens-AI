import { lazy, Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { Dashboard } from "./pages/Dashboard";

const Analyze = lazy(() => import("./pages/Analyze").then((m) => ({ default: m.Analyze })));
const History = lazy(() => import("./pages/History").then((m) => ({ default: m.History })));
const Analytics = lazy(() => import("./pages/Analytics").then((m) => ({ default: m.Analytics })));
const ModelPerformance = lazy(() => import("./pages/ModelPerformance").then((m) => ({ default: m.ModelPerformance })));
const DatasetExplorer = lazy(() => import("./pages/DatasetExplorer").then((m) => ({ default: m.DatasetExplorer })));
const Batch = lazy(() => import("./pages/Batch").then((m) => ({ default: m.Batch })));
const Compare = lazy(() => import("./pages/Compare").then((m) => ({ default: m.Compare })));
const Methodology = lazy(() => import("./pages/Methodology").then((m) => ({ default: m.Methodology })));
const Settings = lazy(() => import("./pages/Settings").then((m) => ({ default: m.Settings })));
const Report = lazy(() => import("./pages/Report").then((m) => ({ default: m.Report })));
const TrendingNews = lazy(() => import("./pages/TrendingNews").then((m) => ({ default: m.TrendingNews })));
const NotFound = lazy(() => import("./pages/NotFound").then((m) => ({ default: m.NotFound })));

function PageFallback() {
  return (
    <div className="flex min-h-[40vh] items-center justify-center">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-300 border-t-primary-600 dark:border-slate-700 dark:border-t-primary-400" />
    </div>
  );
}

export default function App() {
  return (
    <Suspense fallback={<PageFallback />}>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/history" element={<History />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/model-performance" element={<ModelPerformance />} />
          <Route path="/dataset" element={<DatasetExplorer />} />
          <Route path="/batch" element={<Batch />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/trending" element={<TrendingNews />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/report/:id" element={<Report />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  );
}