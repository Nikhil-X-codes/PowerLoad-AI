import React, { useState } from "react";
import { ChevronLeft, ChevronRight, Download, FileText, Loader2, Upload } from "lucide-react";
import { extractApiErrorMessage, predictBatchCsv } from "../services/api";
import type { BatchPredictionResponse } from "../types/api";
import { ErrorAlert, LoadBadge } from "../components/UiStates";

const REQUIRED_CSV_HEADERS = [
  "Date_Time",
  "Usage_kWh",
  "Lagging_Current_Reactive.Power_kVarh",
  "Leading_Current_Reactive_Power_kVarh",
  "CO2(tCO2)",
  "Lagging_Current_Power_Factor",
  "Leading_Current_Power_Factor",
  "NSM",
];

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const PAGE_SIZE = 25;

const SAMPLE_CSV = [
  REQUIRED_CSV_HEADERS.join(","),
  "15-12-2018 02:30,3.2,4.1,0.0,0.0,61.5,100.0,9000",
  "15-12-2018 19:15,34.8,11.4,0.0,0.02,95.0,100.0,69300",
  "15-12-2018 14:30,92.4,32.0,0.0,0.04,94.5,100.0,52200",
].join("\n");

export function BatchPredictionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BatchPredictionResponse | null>(null);
  const [page, setPage] = useState(1);

  const validateAndSelectFile = async (candidate: File | null) => {
    setError(null);
    if (!candidate) {
      setFile(null);
      return;
    }
    if (!candidate.name.toLowerCase().endsWith(".csv")) {
      setError("Only .csv files are supported.");
      return;
    }
    if (candidate.size === 0) {
      setError("Selected CSV file is empty (0 bytes).");
      return;
    }
    if (candidate.size > MAX_FILE_BYTES) {
      setError("CSV file exceeds the 10 MB upload limit.");
      return;
    }
    try {
      const headerSlice = await candidate.slice(0, 4096).text();
      const firstLine = headerSlice.split(/\r?\n/)[0] ?? "";
      const cols = firstLine.split(",").map((c) => c.trim().replace(/^"|"$/g, ""));
      const missing = REQUIRED_CSV_HEADERS.filter((req) => !cols.includes(req));
      if (missing.length > 0) {
        setError(`Missing required CSV columns: ${missing.join(", ")}`);
        return;
      }
    } catch {
      setError("Could not read the selected CSV file.");
      return;
    }
    setFile(candidate);
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictBatchCsv(file);
      setResult(res);
      setPage(1);
    } catch (err) {
      setError(extractApiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadTemplate = () => {
    const blob = new Blob([SAMPLE_CSV], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "load_batch_template.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadCsv = () => {
    if (!result) return;
    const header = [
      "Date_Time",
      "predicted_load_type",
      "prob_Light_Load",
      "prob_Medium_Load",
      "prob_Maximum_Load",
    ];
    const lines = [
      header.join(","),
      ...result.predictions.map((row) =>
        [
          row.Date_Time,
          row.prediction,
          row.probabilities.Light_Load ?? "",
          row.probabilities.Medium_Load ?? "",
          row.probabilities.Maximum_Load ?? "",
        ].join(",")
      ),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "batch_predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const totalPages = result ? Math.max(1, Math.ceil(result.predictions.length / PAGE_SIZE)) : 1;
  const pagedRows = result
    ? result.predictions.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
    : [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Batch Prediction</h1>
          <p className="mt-1 text-sm text-slate-500">
            Upload a CSV containing the 8 measurement columns to generate load predictions
          </p>
        </div>
        <button
          type="button"
          onClick={handleDownloadTemplate}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100"
        >
          <FileText className="h-3.5 w-3.5" />
          <span>Sample CSV Template</span>
        </button>
      </div>

      <form
        onSubmit={handleUpload}
        className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs"
      >
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            void validateAndSelectFile(e.dataTransfer.files?.[0] ?? null);
          }}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed p-8 text-center transition-colors ${
            dragOver
              ? "border-slate-900 bg-slate-100"
              : "border-slate-300 bg-slate-50 hover:border-slate-400"
          }`}
        >
          <Upload className="mb-2 h-6 w-6 text-slate-500" />
          <span className="text-sm font-medium text-slate-700">
            {file
              ? `${file.name} (${(file.size / 1024).toFixed(1)} KB)`
              : "Drag & drop a .csv file here, or click to browse"}
          </span>
          <span className="mt-1 text-xs text-slate-500">
            Max 10 MB · Up to 50,000 rows
          </span>
          <input
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => void validateAndSelectFile(e.target.files?.[0] ?? null)}
          />
        </label>

        <div className="mt-4 flex justify-end">
          <button
            type="submit"
            disabled={!file || loading}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            <span>{loading ? "Processing CSV..." : "Run Batch Prediction"}</span>
          </button>
        </div>
      </form>

      {error && <ErrorAlert message={error} />}

      {result && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
            <div className="flex flex-wrap items-center gap-4 text-sm">
              <span className="font-semibold text-slate-900">
                {result.total_rows.toLocaleString()} rows
              </span>
              <span className="text-slate-500">{result.inference_time_ms} ms</span>
              {Object.entries(result.class_distribution).map(([cls, count]) => (
                <span key={cls} className="text-xs text-slate-600">
                  {cls}: <strong>{count}</strong> (
                  {result.total_rows ? ((count / result.total_rows) * 100).toFixed(1) : "0.0"}%)
                </span>
              ))}
            </div>
            <button
              type="button"
              onClick={handleDownloadCsv}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Download CSV</span>
            </button>
          </div>

          <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-xs">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                  <tr>
                    <th className="px-4 py-3">Date_Time</th>
                    <th className="px-4 py-3">Prediction</th>
                    <th className="px-4 py-3">Light</th>
                    <th className="px-4 py-3">Medium</th>
                    <th className="px-4 py-3">Maximum</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {pagedRows.map((row, idx) => (
                    <tr key={`${row.Date_Time}-${idx}`} className="hover:bg-slate-50">
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-700">
                        {row.Date_Time}
                      </td>
                      <td className="px-4 py-2.5">
                        <LoadBadge label={row.prediction} />
                      </td>
                      <td className="px-4 py-2.5 text-xs text-slate-600">
                        {((row.probabilities.Light_Load ?? 0) * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-2.5 text-xs text-slate-600">
                        {((row.probabilities.Medium_Load ?? 0) * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-2.5 text-xs text-slate-600">
                        {((row.probabilities.Maximum_Load ?? 0) * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-between border-t border-slate-100 bg-slate-50 px-4 py-2.5 text-xs text-slate-600">
              <span>
                Page {page} of {totalPages} ({result.predictions.length.toLocaleString()} total rows)
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={page <= 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  className="inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-2.5 py-1 disabled:opacity-40"
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                  <span>Prev</span>
                </button>
                <button
                  type="button"
                  disabled={page >= totalPages}
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  className="inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-2.5 py-1 disabled:opacity-40"
                >
                  <span>Next</span>
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

