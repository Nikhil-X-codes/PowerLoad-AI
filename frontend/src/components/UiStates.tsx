import { AlertCircle, Loader2 } from "lucide-react";

export function StatCard({
  label,
  value,
  sublabel,
}: {
  label: string;
  value: string;
  sublabel?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
      <p className="text-sm font-medium text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-bold tracking-tight text-slate-900">{value}</p>
      {sublabel && <p className="mt-1 text-xs text-slate-500">{sublabel}</p>}
    </div>
  );
}

export function LoadingPanel({ message = "Loading data..." }: { message?: string }) {
  return (
    <div className="flex items-center justify-center gap-2.5 rounded-xl border border-slate-200 bg-white p-10 text-sm text-slate-600">
      <Loader2 className="h-4 w-4 animate-spin text-slate-700" />
      <span>{message}</span>
    </div>
  );
}

export function ErrorAlert({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-start justify-between gap-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
      <div className="flex items-start gap-2.5">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />
        <span>{message}</span>
      </div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="shrink-0 rounded-md border border-red-300 bg-white px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-100"
        >
          Retry
        </button>
      )}
    </div>
  );
}

const LOAD_STYLES: Record<string, string> = {
  Light_Load: "bg-emerald-50 text-emerald-700 border-emerald-200",
  Medium_Load: "bg-amber-50 text-amber-700 border-amber-200",
  Maximum_Load: "bg-rose-50 text-rose-700 border-rose-200",
};

export function LoadBadge({ label }: { label: string }) {
  const cls = LOAD_STYLES[label] || "bg-slate-100 text-slate-700 border-slate-200";
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${cls}`}>
      {label}
    </span>
  );
}

export function DonutChart({
  segments,
}: {
  segments: Array<{ label: string; value: number; color: string }>;
}) {
  const total = segments.reduce((acc, s) => acc + s.value, 0) || 1;
  const radius = 56;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;

  return (
    <div className="flex flex-col items-center gap-4">
      <svg width="150" height="150" viewBox="0 0 150 150" className="-rotate-90">
        <circle
          cx="75"
          cy="75"
          r={radius}
          fill="transparent"
          stroke="#f1f5f9"
          strokeWidth="22"
        />
        {segments.map((seg) => {
          const frac = seg.value / total;
          const dash = frac * circumference;
          const el = (
            <circle
              key={seg.label}
              cx="75"
              cy="75"
              r={radius}
              fill="transparent"
              stroke={seg.color}
              strokeWidth="22"
              strokeDasharray={`${dash} ${circumference - dash}`}
              strokeDashoffset={-offset}
            />
          );
          offset += dash;
          return el;
        })}
      </svg>
      <div className="flex flex-wrap justify-center gap-4 text-xs text-slate-600">
        {segments.map((seg) => (
          <div key={seg.label} className="flex items-center gap-1.5">
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: seg.color }}
            />
            <span>
              {seg.label.replace("_Load", "")} ({((seg.value / total) * 100).toFixed(1)}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function AreaTrendChart({
  points,
  xLabels,
}: {
  points: number[];
  xLabels: string[];
}) {
  if (!points.length) return null;
  const width = 520;
  const height = 180;
  const padX = 16;
  const padY = 18;
  const maxVal = Math.max(...points, 1);
  const minVal = Math.min(...points, 0);
  const span = Math.max(maxVal - minVal, 1);

  const coords = points.map((v, idx) => {
    const x =
      points.length === 1
        ? width / 2
        : padX + (idx / (points.length - 1)) * (width - 2 * padX);
    const y = height - padY - ((v - minVal) / span) * (height - 2 * padY);
    return { x, y };
  });

  const linePath = coords.map((c, i) => `${i === 0 ? "M" : "L"} ${c.x} ${c.y}`).join(" ");
  const areaPath = `${linePath} L ${coords[coords.length - 1].x} ${height - padY} L ${coords[0].x} ${height - padY} Z`;

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-48 w-full">
        <defs>
          <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#2563eb" stopOpacity="0.24" />
            <stop offset="100%" stopColor="#2563eb" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map((ratio) => {
          const y = padY + ratio * (height - 2 * padY);
          return (
            <line
              key={ratio}
              x1={padX}
              x2={width - padX}
              y1={y}
              y2={y}
              stroke="#e2e8f0"
              strokeWidth="1"
            />
          );
        })}
        <path d={areaPath} fill="url(#trendGrad)" />
        <path d={linePath} fill="none" stroke="#2563eb" strokeWidth="2.5" strokeLinejoin="round" />
      </svg>
      <div className="mt-1 flex justify-between px-2 text-xs text-slate-400">
        <span>{xLabels[0] ?? ""}</span>
        <span>{xLabels[Math.floor(xLabels.length / 2)] ?? ""}</span>
        <span>{xLabels[xLabels.length - 1] ?? ""}</span>
      </div>
    </div>
  );
}
