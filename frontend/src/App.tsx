import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { OverviewPage } from "./pages/OverviewPage";
import { SinglePredictionPage } from "./pages/SinglePredictionPage";
import { BatchPredictionPage } from "./pages/BatchPredictionPage";
import { ModelPerformancePage } from "./pages/ModelPerformancePage";
import { ExplainabilityPage } from "./pages/ExplainabilityPage";

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="predict" element={<SinglePredictionPage />} />
          <Route path="batch" element={<BatchPredictionPage />} />
          <Route path="performance" element={<ModelPerformancePage />} />
          <Route path="explainability" element={<ExplainabilityPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
