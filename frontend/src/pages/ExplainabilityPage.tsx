import { useCallback, useEffect, useState } from "react";
import { extractApiErrorMessage, fetchAnalytics, isAbortError } from "../services/api";
import type { AnalyticsResponse } from "../types/api";
import { ErrorAlert, LoadingPanel } from "../components/UiStates";

export function ExplainabilityPage() {
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const a = await fetchAnalytics(signal);
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

  const importances = analytics?.explainability?.permutation_importance_val ?? [];
  const maxImp = Math.max(...importances.map((i) => Math.abs(i.importance_mean)), 0.001);
  const errSummary = analytics?.explainability?.test_error_summary;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Explainability</h1>
          <p className="mt-1 text-sm text-slate-500">
            Validation permutation importance (Macro F1 drop), influential features, and limitations
          </p>
        </div>
        <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600">
          Historical Validation Artifact
        </span>
      </div>

      {loading && <LoadingPanel message="Loading explainability diagnostics..." />}
      {error && <ErrorAlert message={error} onRetry={() => void loadData()} />}

      {!loading && !error && (
        <>
          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
            <h2 className="mb-4 text-base font-semibold text-slate-900">
              Permutation Feature Importance (Validation)
            </h2>
            <div className="space-y-3">
              {importances.map((item) => {
                const widthPct = Math.max(0, (item.importance_mean / maxImp) * 100);
                return (
                  <div key={item.feature}>
                    <div className="mb-1 flex justify-between text-xs">
                      <span className="font-mono font-medium text-slate-700">{item.feature}</span>
                      <span className="text-slate-500">
                        {item.importance_mean.toFixed(5)} ± {item.importance_std.toFixed(5)}
                      </span>
                    </div>
                    <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                      <div
                        className="h-full rounded-full bg-blue-600"
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {errSummary && (
              <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
                <h2 className="mb-3 text-base font-semibold text-slate-900">
                  December Error Analysis
                </h2>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="rounded-lg bg-slate-50 p-3">
                    <div className="text-xs text-slate-500">Weekday Error Rate</div>
                    <div className="mt-1 text-lg font-bold text-slate-900">
                      {(errSummary.weekday_error_rate * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="rounded-lg bg-amber-50 p-3">
                    <div className="text-xs text-amber-700">Weekend Error Rate</div>
                    <div className="mt-1 text-lg font-bold text-amber-900">
                      {(errSummary.weekend_error_rate * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
                <div className="mt-4 space-y-1.5 text-xs text-slate-600">
                  {Object.entries(errSummary.confusion_pairs).map(([pair, cnt]) => (
                    <div key={pair} className="flex justify-between">
                      <span>{pair}</span>
                      <strong className="text-slate-900">{cnt}</strong>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
              <h2 className="mb-3 text-base font-semibold text-slate-900">
                Most Influential Features & Limitations
              </h2>
              <ul className="list-disc space-y-2 pl-5 text-sm text-slate-600">
                <li>
                  <strong>Top driver — <code>Date_Time</code> (0.1839):</strong> Industrial shift
                  schedules, weekday vs. weekend patterns, and time-of-day strongly govern load
                  tier transitions.
                </li>
                <li>
                  <strong>Power &amp; emissions — <code>Usage_kWh</code> (0.0111) &amp; <code>CO2(tCO2)</code> (0.0062):</strong>{" "}
                  Active energy draw and correlated CO2 readings separate peak industrial runs from
                  idle periods.
                </li>
                <li>
                  <strong>Schedule &amp; holiday drift:</strong> Weekend error rate (13.54%) exceeds
                  weekday error rate (3.42%) due to year-end schedule shifts.
                </li>
                <li>
                  <strong>Contemporaneous classification:</strong> Classifies <code>Load_Type</code>{" "}
                  from current measurements and timestamps rather than future demand forecasting.
                </li>
              </ul>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

