import React from "react";
import { Activity, RefreshCw } from "lucide-react";
import { HealthStatus } from "../types/api";

interface HeaderProps {
  health: HealthStatus | null;
  loading: boolean;
  onRefreshHealth: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, loading, onRefreshHealth }) => {
  const isOnline = health?.status === "healthy" && health.model_loaded;

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-900 text-white shadow-sm">
            <Activity className="h-5 w-5 text-brand-500" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-semibold tracking-tight text-slate-900">
                MachineGuard AI
              </h1>
              <span className="rounded bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
                {health?.model_version ? `v${health.model_version}` : "v1.0.0"}
              </span>
            </div>
            <p className="text-xs font-medium text-slate-500">
              Predictive Maintenance Decision Support
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="inline-flex items-center rounded-md border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-800">
            Educational Prototype
          </span>

          <div
            className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium ${
              isOnline
                ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                : "border-rose-200 bg-rose-50 text-rose-800"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                isOnline ? "bg-emerald-500" : "bg-rose-500 animate-pulse"
              }`}
            />
            <span>{isOnline ? "API Online" : "API Unavailable"}</span>
          </div>

          <button
            type="button"
            onClick={onRefreshHealth}
            disabled={loading}
            aria-label="Refresh API Status"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-400 disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>
    </header>
  );
};
