import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import * as api from "../services/api";
import { OverviewPage } from "../pages/OverviewPage";
import { SinglePredictionPage } from "../pages/SinglePredictionPage";
import { BatchPredictionPage } from "../pages/BatchPredictionPage";

describe("PowerLoad AI Frontend", () => {
  it("renders OverviewPage with real API stats", async () => {
    vi.spyOn(api, "fetchHealth").mockResolvedValue({
      status: "ok",
      model_loaded: true,
      model_version: "1.0.0",
      model_name: "RandomForest",
      load_time_ms: 150,
      memory_rss_mb: 195,
      uptime_seconds: 10,
    });
    vi.spyOn(api, "fetchMetrics").mockResolvedValue({
      test_sample_count: 2977,
      train_dev_sample_count: 32064,
      model_configuration: {
        name: "RandomForest",
        feature_set: "combined",
        best_cv_macro_f1: 0.9082,
        best_params: {},
        search_time_sec: 29.9,
      },
      summary_metrics: {
        accuracy: 0.9332,
        macro_precision: 0.9091,
        macro_recall: 0.9476,
        macro_f1: 0.9261,
        weighted_precision: 0.9383,
        weighted_recall: 0.9332,
        weighted_f1: 0.9338,
        f1_Light_Load: 0.9461,
        f1_Maximum_Load: 0.914,
        f1_Medium_Load: 0.9184,
      },
      per_class_report: {},
      confusion_matrix: { labels: ["Light_Load", "Maximum_Load", "Medium_Load"], matrix: [] },
    });
    vi.spyOn(api, "fetchAnalytics").mockResolvedValue({
      model_version: "1.0.0",
      data_quality: {
        row_count: 35041,
        column_count: 9,
        date_range: { min: "2018-01-01", max: "2018-12-31" },
        class_counts: { Light_Load: 18073, Medium_Load: 9696, Maximum_Load: 7272 },
        missing_counts: {},
        duplicate_timestamps: 1,
      },
    });

    render(
      <MemoryRouter>
        <OverviewPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText("35,041")).toBeInTheDocument();
      expect(screen.getByText("93.32%")).toBeInTheDocument();
      expect(screen.getByText("0.9261")).toBeInTheDocument();
    });
  });

  it("validates ranges and submits SinglePredictionPage", async () => {
    const predictSpy = vi.spyOn(api, "predictSingle").mockResolvedValue({
      Date_Time: "2018-12-15 19:15:00",
      prediction: "Medium_Load",
      probabilities: { Light_Load: 0.08, Medium_Load: 0.87, Maximum_Load: 0.05 },
      model_version: "1.0.0",
      inference_time_ms: 12,
    });

    render(
      <MemoryRouter>
        <SinglePredictionPage />
      </MemoryRouter>
    );

    const submitBtn = screen.getByRole("button", { name: /Predict Load Type/i });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(predictSpy).toHaveBeenCalledTimes(1);
      expect(screen.getAllByText("Medium_Load").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("87.0%")).toBeInTheDocument();
    });
  });

  it("rejects serialized model files on BatchPredictionPage", async () => {
    render(
      <MemoryRouter>
        <BatchPredictionPage />
      </MemoryRouter>
    );

    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const evilFile = new File(["payload"], "evil.joblib", { type: "application/octet-stream" });
    fireEvent.change(input, { target: { files: [evilFile] } });

    await waitFor(() => {
      expect(screen.getByText(/Only \.csv files are supported/i)).toBeInTheDocument();
    });
  });
});

