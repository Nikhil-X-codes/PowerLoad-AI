import React, { useEffect, useState } from "react";
import { Loader2, Play } from "lucide-react";
import { extractApiErrorMessage, predictSingle } from "../services/api";
import type { LoadRecordInput, PredictionResponse } from "../types/api";
import { ErrorAlert, LoadBadge } from "../components/UiStates";

const PRESETS: Record<string, LoadRecordInput> = {
  Light: {
    Date_Time: "15-12-2018 02:30",
    Usage_kWh: 3.2,
    "Lagging_Current_Reactive.Power_kVarh": 4.1,
    Leading_Current_Reactive_Power_kVarh: 0.0,
    "CO2(tCO2)": 0.0,
    Lagging_Current_Power_Factor: 61.5,
    Leading_Current_Power_Factor: 100.0,
    NSM: 9000,
  },
  Medium: {
    Date_Time: "15-12-2018 19:15",
    Usage_kWh: 34.8,
    "Lagging_Current_Reactive.Power_kVarh": 11.4,
    Leading_Current_Reactive_Power_kVarh: 0.0,
    "CO2(tCO2)": 0.02,
    Lagging_Current_Power_Factor: 95.0,
    Leading_Current_Power_Factor: 100.0,
    NSM: 69300,
  },
  Maximum: {
    Date_Time: "15-12-2018 14:30",
    Usage_kWh: 92.4,
    "Lagging_Current_Reactive.Power_kVarh": 32.0,
    Leading_Current_Reactive_Power_kVarh: 0.0,
    "CO2(tCO2)": 0.04,
    Lagging_Current_Power_Factor: 94.5,
    Leading_Current_Power_Factor: 100.0,
    NSM: 52200,
  },
};

const PROB_BAR_COLORS: Record<string, string> = {
  Light_Load: "bg-emerald-500",
  Medium_Load: "bg-amber-500",
  Maximum_Load: "bg-rose-500",
};

function deriveNsmFromDateTime(dateStr: string): number | null {
  const trimmed = dateStr.trim();
  const timeMatch = trimmed.match(/(\d{2}):(\d{2})(?::(\d{2}))?$/);
  if (!timeMatch) return null;
  const hh = Number(timeMatch[1]);
  const mm = Number(timeMatch[2]);
  const ss = Number(timeMatch[3] ?? "0");
  if (hh < 0 || hh > 23 || mm < 0 || mm > 59 || ss < 0 || ss > 59) return null;
  return hh * 3600 + mm * 60 + ss;
}

function validateFormRanges(form: LoadRecordInput): string | null {
  if (!form.Date_Time.trim()) return "Date_Time is required.";
  if (deriveNsmFromDateTime(form.Date_Time) === null) {
    return "Date_Time must include a valid time (e.g., 15-12-2018 14:30 or 2018-12-15 14:30:00).";
  }
  const nonNegFields: Array<[keyof LoadRecordInput, string]> = [
    ["Usage_kWh", "Usage (kWh)"],
    ["Lagging_Current_Reactive.Power_kVarh", "Lagging Reactive Power (kVarh)"],
    ["Leading_Current_Reactive_Power_kVarh", "Leading Reactive Power (kVarh)"],
    ["CO2(tCO2)", "CO2 (tCO2)"],
  ];
  for (const [key, label] of nonNegFields) {
    const val = form[key];
    if (typeof val === "number" && (Number.isNaN(val) || val < 0)) {
      return `${label} must be 0 or greater.`;
    }
  }
  for (const [key, label] of [
    ["Lagging_Current_Power_Factor", "Lagging Power Factor (%)"],
    ["Leading_Current_Power_Factor", "Leading Power Factor (%)"],
  ] as const) {
    const val = form[key];
    if (typeof val === "number" && (Number.isNaN(val) || val < 0 || val > 100)) {
      return `${label} must be between 0 and 100.`;
    }
  }
  if (typeof form.NSM === "number" && (Number.isNaN(form.NSM) || form.NSM < 0 || form.NSM > 86400)) {
    return "NSM must be between 0 and 86400 seconds.";
  }
  return null;
}

export function SinglePredictionPage() {
  const [form, setForm] = useState<LoadRecordInput>(PRESETS.Medium);
  const [autoDeriveNsm, setAutoDeriveNsm] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    if (!autoDeriveNsm) return;
    const derived = deriveNsmFromDateTime(form.Date_Time);
    if (derived !== null && derived !== form.NSM) {
      setForm((prev) => ({ ...prev, NSM: derived }));
    }
  }, [autoDeriveNsm, form.Date_Time, form.NSM]);

  const updateNumberField = (key: keyof LoadRecordInput, raw: string) => {
    setForm((prev) => ({
      ...prev,
      [key]: raw.trim() === "" ? null : Number(raw),
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const validationError = validateFormRanges(form);
    if (validationError) {
      setError(validationError);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await predictSingle(form);
      setResult(res);
    } catch (err) {
      setError(extractApiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const numericFields: Array<{
    key: keyof LoadRecordInput;
    label: string;
    min?: number;
    max?: number;
  }> = [
    { key: "Usage_kWh", label: "Usage (kWh)", min: 0 },
    {
      key: "Lagging_Current_Reactive.Power_kVarh",
      label: "Lagging Reactive Power (kVarh)",
      min: 0,
    },
    {
      key: "Leading_Current_Reactive_Power_kVarh",
      label: "Leading Reactive Power (kVarh)",
      min: 0,
    },
    { key: "CO2(tCO2)", label: "CO2 (tCO2)", min: 0 },
    {
      key: "Lagging_Current_Power_Factor",
      label: "Lagging Power Factor (%)",
      min: 0,
      max: 100,
    },
    {
      key: "Leading_Current_Power_Factor",
      label: "Leading Power Factor (%)",
      min: 0,
      max: 100,
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Single Prediction</h1>
          <p className="mt-1 text-sm text-slate-500">
            Enter power system measurements to classify load type
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-slate-500">Presets:</span>
          {Object.entries(PRESETS).map(([name, preset]) => (
            <button
              key={name}
              type="button"
              onClick={() => {
                setError(null);
                setForm(preset);
              }}
              className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100"
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <form
          noValidate
          onSubmit={handleSubmit}
          className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs lg:col-span-3"
        >
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="sm:col-span-2">
              <label className="block text-xs font-medium text-slate-600">
                Date_Time (DD-MM-YYYY HH:MM)
              </label>
              <input
                type="text"
                required
                value={form.Date_Time}
                onChange={(e) => setForm((p) => ({ ...p, Date_Time: e.target.value }))}
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-slate-900 focus:outline-none"
              />
            </div>

            {numericFields.map(({ key, label, min, max }) => {
              const val = form[key];
              return (
                <div key={key}>
                  <label className="block text-xs font-medium text-slate-600">{label}</label>
                  <input
                    type="number"
                    step="any"
                    min={min}
                    max={max}
                    value={val === null ? "" : String(val)}
                    onChange={(e) => updateNumberField(key, e.target.value)}
                    placeholder="Optional (null)"
                    className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-slate-900 focus:outline-none"
                  />
                </div>
              );
            })}

            <div className="sm:col-span-2">
              <div className="flex items-center justify-between">
                <label className="block text-xs font-medium text-slate-600">
                  NSM (Seconds from Midnight, 0–86400)
                </label>
                <label className="flex items-center gap-1.5 text-xs text-slate-500 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={autoDeriveNsm}
                    onChange={(e) => setAutoDeriveNsm(e.target.checked)}
                    className="rounded border-slate-300"
                  />
                  <span>Auto-derive from Date_Time</span>
                </label>
              </div>
              <input
                type="number"
                step="1"
                min={0}
                max={86400}
                disabled={autoDeriveNsm}
                value={form.NSM === null ? "" : String(form.NSM)}
                onChange={(e) => updateNumberField("NSM", e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm disabled:bg-slate-50 disabled:text-slate-500 focus:border-slate-900 focus:outline-none"
              />
            </div>
          </div>

          <div className="mt-6 flex justify-end">
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Predicting...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  <span>Predict Load Type</span>
                </>
              )}
            </button>
          </div>
        </form>

        <div className="space-y-4 lg:col-span-2">
          {error && <ErrorAlert message={error} />}

          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              Prediction Result
            </h2>
            {!result ? (
              <p className="mt-4 text-sm text-slate-500">
                Submit measurements to view the predicted class and probabilities.
              </p>
            ) : (
              <div className="mt-4 space-y-5">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600">Predicted Class</span>
                  <LoadBadge label={result.prediction} />
                </div>

                <div className="space-y-3">
                  {Object.entries(result.probabilities).map(([cls, prob]) => (
                    <div key={cls}>
                      <div className="mb-1 flex justify-between text-xs font-medium text-slate-700">
                        <span>{cls}</span>
                        <span>{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                        <div
                          className={`h-full rounded-full transition-all ${
                            PROB_BAR_COLORS[cls] || "bg-slate-900"
                          }`}
                          style={{ width: `${Math.min(100, prob * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex justify-between border-t border-slate-100 pt-3 text-xs text-slate-500">
                  <span>Model v{result.model_version}</span>
                  <span>{result.inference_time_ms} ms</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

