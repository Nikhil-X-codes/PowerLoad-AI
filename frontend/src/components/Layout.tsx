import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import {
  Activity,
  BarChart3,
  FileSpreadsheet,
  LayoutDashboard,
  Menu,
  SlidersHorizontal,
  X,
  Zap,
} from "lucide-react";
import { fetchHealth } from "../services/api";
import type { HealthResponse } from "../types/api";

const NAV_ITEMS = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/predict", label: "Single Prediction", icon: Activity },
  { to: "/batch", label: "Batch Prediction", icon: FileSpreadsheet },
  { to: "/performance", label: "Model Performance", icon: BarChart3 },
  { to: "/explainability", label: "Explainability", icon: SlidersHorizontal },
];

export function Layout() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetchHealth(controller.signal)
      .then((res) => setHealth(res))
      .catch(() => setHealth(null));
    return () => controller.abort();
  }, []);

  const isOnline = Boolean(health?.model_loaded);

  const sidebarContent = (
    <div className="flex h-full flex-col justify-between bg-slate-900 px-4 py-6 text-slate-100">
      <div>
        <div className="mb-8 flex items-center gap-2.5 px-2">
          <Zap className="h-5 w-5 text-emerald-400" />
          <div>
            <div className="text-base font-bold tracking-tight text-emerald-400">
              PowerLoad AI
            </div>
            <div className="text-xs text-slate-400">Energy intelligence</div>
          </div>
        </div>

        <nav className="space-y-1">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-slate-800 text-white"
                    : "text-slate-300 hover:bg-slate-800/60 hover:text-white"
                }`
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="border-t border-slate-800 pt-4 px-2">
        <div className="flex items-center gap-2 text-xs text-slate-300">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              isOnline ? "bg-emerald-400" : "bg-amber-400"
            }`}
          />
          <span>{isOnline ? "Model online" : "Backend offline"}</span>
        </div>
        {health && (
          <div className="mt-1 text-[11px] text-slate-500">
            v{health.model_version} · {health.memory_rss_mb} MB
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="flex min-h-screen bg-slate-50">
      <aside className="hidden w-60 shrink-0 lg:block">{sidebarContent}</aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-40 flex lg:hidden">
          <div className="w-64 shrink-0">{sidebarContent}</div>
          <button
            type="button"
            aria-label="Close navigation"
            className="flex-1 bg-slate-950/50"
            onClick={() => setMobileOpen(false)}
          />
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 lg:hidden">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-emerald-600" />
            <span className="text-sm font-bold text-slate-900">PowerLoad AI</span>
          </div>
          <button
            type="button"
            onClick={() => setMobileOpen((v) => !v)}
            className="rounded-md p-1.5 text-slate-600 hover:bg-slate-100"
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </header>

        <main className="mx-auto w-full max-w-6xl flex-1 p-5 sm:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
