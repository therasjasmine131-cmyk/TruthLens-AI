import { Routes, Route } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { Dashboard } from "./pages/Dashboard";
import { Analyze } from "./pages/Analyze";
import { History } from "./pages/History";
import { Analytics } from "./pages/Analytics";
import { ModelPerformance } from "./pages/ModelPerformance";
import { DatasetExplorer } from "./pages/DatasetExplorer";
import { Batch } from "./pages/Batch";
import { Compare } from "./pages/Compare";
import { Methodology } from "./pages/Methodology";
import { Settings } from "./pages/Settings";
import { Report } from "./pages/Report";
import { TrendingNews } from "./pages/TrendingNews";
import { NotFound } from "./pages/NotFound";

export default function App() {
  return (
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
  );
}
