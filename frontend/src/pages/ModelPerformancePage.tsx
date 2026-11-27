import { useCallback, useEffect, useState } from "react";
import { extractApiErrorMessage, fetchAnalytics, fetchMetrics, isAbortError } from "../services/api";
import type { AnalyticsResponse, MetricsResponse } from "../types/api";
import { ErrorAlert, LoadingPanel, StatCard } from "../components/UiStates";

export function ModelPerformancePage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const [m, a] = await Promise.all([fetchMetrics(signal), fetchAnalytics(signal)]);
      setMetrics(m);
      setAnalytics(a);
    } catch (err) {
      if (!isAbortError(err)) {
        setError(extractApiErrorMessage(err));
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadData(controller.signal);
    return () => controller.abort();
  }, [loadData]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Model Performance</h1>
          <p className="mt-1 text-sm text-slate-500">
            Saved held-out December evaluation metrics, confusion matrix, and candidate comparison
          </p>
        </div>
        <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600">
          Historical Evaluation Artifact (Dec 2018)
        </span>
      </div>

      {loading && <LoadingPanel message="Loading evaluation metrics..." />}
      {error && <ErrorAlert message={error} onRetry={() => void loadData()} />}

      {!loading && !error && metrics && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard
              label="Accuracy"
              value={`${(metrics.summary_metrics.accuracy * 100).toFixed(2)}%`}
              sublabel={`${metrics.test_sample_count.toLocaleString()} test rows`}
            />
            <StatCard
              label="Macro Precision"
              value={metrics.summary_metrics.macro_precision.toFixed(4)}
            />
            <StatCard
              label="Macro Recall"
              value={metrics.summary_metrics.macro_recall.toFixed(4)}
            />
            <StatCard
              label="Macro F1"
              value={metrics.summary_metrics.macro_f1.toFixed(4)}
              sublabel={`Weighted F1: ${metrics.summary_metrics.weighted_f1.toFixed(4)}`}
            />
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
              <h2 className="mb-4 text-base font-semibold text-slate-900">Per-Class Metrics</h2>
              <table className="w-full text-left text-sm">
                <thead className="border-b border-slate-200 text-xs uppercase text-slate-500">
                  <tr>
                    <th className="py-2">Class</th>
                    <th className="py-2">Precision</th>
                    <th className="py-2">Recall</th>
                    <th className="py-2">F1</th>
                    <th className="py-2">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {metrics.confusion_matrix.labels.map((label) => {
                    const row = metrics.per_class_report[label];
                    if (!row || typeof row === "number") return null;
                    return (
                      <tr key={label}>
                        <td className="py-2.5 font-medium text-slate-800">{label}</td>
                        <td className="py-2.5 text-slate-600">{(row.precision ?? 0).toFixed(4)}</td>
                        <td className="py-2.5 text-slate-600">{(row.recall ?? 0).toFixed(4)}</td>
                        <td className="py-2.5 font-semibold text-slate-900">
                          {(row["f1-score"] ?? 0).toFixed(4)}
                        </td>
                        <td className="py-2.5 text-slate-600">{row.support}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
              <h2 className="mb-4 text-base font-semibold text-slate-900">Confusion Matrix</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-center text-xs">
                  <thead>
                    <tr className="text-slate-500">
                      <th className="p-2 text-left">Actual \ Pred</th>
                      {metrics.confusion_matrix.labels.map((l) => (
                        <th key={l} className="p-2">
                          {l.replace("_Load", "")}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.confusion_matrix.matrix.map((row, rIdx) => (
                      <tr key={metrics.confusion_matrix.labels[rIdx]}>
                        <td className="p-2 text-left font-medium text-slate-700">
                          {metrics.confusion_matrix.labels[rIdx].replace("_Load", "")}
                        </td>
                        {row.map((val, cIdx) => {
                          const isDiag = rIdx === cIdx;
                          return (
                            <td
                              key={cIdx}
                              className={`rounded border border-white p-3 font-mono font-semibold ${
                                isDiag
                                  ? "bg-emerald-50 text-emerald-800"
                                  : val > 0
                                  ? "bg-amber-50 text-amber-800"
                                  : "bg-slate-50 text-slate-400"
                              }`}
                            >
                              {val}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {analytics?.model_comparison && (
            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
              <h2 className="mb-4 text-base font-semibold text-slate-900">
                Candidate Model Comparison (Macro F1 — Validation)
              </h2>
              <div className="mb-6 space-y-2.5">
                {analytics.model_comparison
                  .filter(
                    (r) =>
                      r.feature_set === "combined" &&
                      r.status === "ok" &&
                      typeof r.macro_f1 === "number"
                  )
                  .sort((a, b) => (b.macro_f1 ?? 0) - (a.macro_f1 ?? 0))
                  .map((row) => (
                    <div key={`${row.model}-${row.feature_set}`}>
                      <div className="mb-1 flex justify-between text-xs">
                        <span className="font-medium text-slate-700">{row.model}</span>
                        <span className="font-mono text-slate-600">
                          {(row.macro_f1 ?? 0).toFixed(4)}
                        </span>
                      </div>
                      <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full rounded-full bg-blue-600"
                          style={{ width: `${Math.min(100, (row.macro_f1 ?? 0) * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-slate-200 text-xs uppercase text-slate-500">
                    <tr>
                      <th className="py-2 pr-4">Model</th>
                      <th className="py-2 pr-4">Status</th>
                      <th className="py-2 pr-4">Accuracy</th>
                      <th className="py-2 pr-4">Macro F1</th>
                      <th className="py-2 pr-4">Weighted F1</th>
                      <th className="py-2">Train Time (s)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {analytics.model_comparison.map((row, idx) => (
                      <tr key={`${row.model}-${row.feature_set}-${idx}`}>
                        <td className="py-2 pr-4 font-medium text-slate-800">{row.model}</td>
                        <td className="py-2 pr-4 text-xs text-slate-600">{row.status}</td>
                        <td className="py-2 pr-4 text-slate-600">
                          {typeof row.accuracy === "number"
                            ? `${(row.accuracy * 100).toFixed(2)}%`
                            : "—"}
                        </td>
                        <td className="py-2 pr-4 font-semibold text-slate-900">
                          {typeof row.macro_f1 === "number" ? row.macro_f1.toFixed(4) : "—"}
                        </td>
                        <td className="py-2 pr-4 text-slate-600">
                          {typeof row.weighted_f1 === "number"
                            ? row.weighted_f1.toFixed(4)
                            : "—"}
                        </td>
                        <td className="py-2 text-slate-500">
                          {typeof row.train_time_sec === "number"
                            ? row.train_time_sec.toFixed(2)
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

