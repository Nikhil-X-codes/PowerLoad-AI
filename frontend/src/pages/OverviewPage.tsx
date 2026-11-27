import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import {
  extractApiErrorMessage,
  fetchAnalytics,
  fetchHealth,
  fetchMetrics,
  isAbortError,
} from "../services/api";
import type { AnalyticsResponse, HealthResponse, MetricsResponse } from "../types/api";
import {
  AreaTrendChart,
  DonutChart,
  ErrorAlert,
  LoadingPanel,
  StatCard,
} from "../components/UiStates";

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function OverviewPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const [h, m, a] = await Promise.all([
        fetchHealth(signal),
        fetchMetrics(signal),
        fetchAnalytics(signal),
      ]);
      setHealth(h);
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

  const classCounts = analytics?.data_quality?.class_counts ?? {};
  const donutSegments = [
    { label: "Light_Load", value: classCounts.Light_Load ?? 0, color: "#10b981" },
    { label: "Medium_Load", value: classCounts.Medium_Load ?? 0, color: "#f59e0b" },
    { label: "Maximum_Load", value: classCounts.Maximum_Load ?? 0, color: "#ef4444" },
  ];

  const monthlyRows = analytics?.historical_summary?.monthly_load_distribution ?? [];
  const monthlyTotals = monthlyRows.map(
    (r) => (r.Light_Load ?? 0) + (r.Medium_Load ?? 0) + (r.Maximum_Load ?? 0)
  );
  const monthlyLabels = monthlyRows.map((r) => MONTH_NAMES[(r.month ?? 1) - 1] ?? String(r.month));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Energy Overview</h1>
        <p className="mt-1 text-sm text-slate-500">
          Electricity load classification dashboard
        </p>
      </div>

      {loading && <LoadingPanel message="Loading dataset summary and model metrics..." />}
      {error && <ErrorAlert message={error} onRetry={() => void loadData()} />}

      {!loading && !error && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard
              label="Total records"
              value={
                analytics?.data_quality?.row_count
                  ? analytics.data_quality.row_count.toLocaleString()
                  : "—"
              }
              sublabel="2018 15-min intervals"
            />
            <StatCard
              label="Accuracy"
              value={
                metrics
                  ? `${(metrics.summary_metrics.accuracy * 100).toFixed(2)}%`
                  : "—"
              }
              sublabel={`Held-out Dec (${metrics?.test_sample_count.toLocaleString() ?? 0} rows)`}
            />
            <StatCard
              label="Macro F1"
              value={metrics ? metrics.summary_metrics.macro_f1.toFixed(4) : "—"}
              sublabel={metrics?.model_configuration.name ?? "RandomForest"}
            />
            <StatCard
              label="Model status"
              value={health?.model_loaded ? "Ready" : "Unavailable"}
              sublabel={
                health ? `v${health.model_version} · ${health.load_time_ms ?? 0} ms load` : undefined
              }
            />
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs lg:col-span-2">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-base font-semibold text-slate-900">
                    Historical Electricity Consumption Volume
                  </h2>
                  <p className="text-xs text-slate-500">
                    Historical dataset statistics across Jan–Dec 2018 (distinct from live predictions)
                  </p>
                </div>
                <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-[11px] font-medium text-slate-600">
                  Historical
                </span>
              </div>
              <AreaTrendChart points={monthlyTotals} xLabels={monthlyLabels} />
              {analytics?.historical_summary?.usage_kwh_by_load_type && (
                <div className="mt-4 grid grid-cols-1 gap-3 border-t border-slate-100 pt-4 sm:grid-cols-3">
                  {analytics.historical_summary.usage_kwh_by_load_type.map((u) => (
                    <div key={u.Load_Type} className="rounded-lg bg-slate-50 p-3 text-xs">
                      <div className="font-semibold text-slate-800">{u.Load_Type}</div>
                      <div className="mt-1 text-slate-600">
                        Mean: <strong>{u.mean.toFixed(1)} kWh</strong> · Median:{" "}
                        <strong>{u.median.toFixed(1)} kWh</strong>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
              <h2 className="mb-4 text-base font-semibold text-slate-900">Load Distribution</h2>
              <DonutChart segments={donutSegments} />
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
            <h2 className="mb-4 text-base font-semibold text-slate-900">Quick Actions</h2>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Link
                to="/predict"
                className="group flex items-center justify-between rounded-xl border border-blue-100 bg-blue-50/70 p-4 transition-colors hover:bg-blue-100/70"
              >
                <div>
                  <div className="text-sm font-semibold text-blue-700">Single Prediction</div>
                  <div className="mt-0.5 text-xs text-slate-600">
                    Classify an electricity measurement
                  </div>
                </div>
                <ArrowRight className="h-4 w-4 text-blue-600 transition-transform group-hover:translate-x-0.5" />
              </Link>

              <Link
                to="/batch"
                className="group flex items-center justify-between rounded-xl border border-emerald-100 bg-emerald-50/70 p-4 transition-colors hover:bg-emerald-100/70"
              >
                <div>
                  <div className="text-sm font-semibold text-emerald-700">Batch Prediction</div>
                  <div className="mt-0.5 text-xs text-slate-600">
                    Upload and analyze a CSV file
                  </div>
                </div>
                <ArrowRight className="h-4 w-4 text-emerald-600 transition-transform group-hover:translate-x-0.5" />
              </Link>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

