import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { Analyze } from "./pages/Analyze";

const History = lazy(() => import("./pages/History").then((m) => ({ default: m.History })));
const Compare = lazy(() => import("./pages/Compare").then((m) => ({ default: m.Compare })));
const Report = lazy(() => import("./pages/Report").then((m) => ({ default: m.Report })));

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
          <Route path="/" element={<Analyze />} />
          <Route path="/history" element={<History />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/report/:id" element={<Report />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}