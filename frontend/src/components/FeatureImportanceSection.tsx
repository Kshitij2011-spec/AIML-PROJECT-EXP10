import React from "react";
import { Info } from "lucide-react";
import { FeatureImportanceItem } from "../types/api";

interface FeatureImportanceProps {
  importances: FeatureImportanceItem[];
  loading: boolean;
}

export const FeatureImportanceSection: React.FC<FeatureImportanceProps> = ({
  importances,
  loading,
}) => {
  if (loading || !importances || importances.length === 0) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm animate-pulse">
        <div className="h-4 w-40 bg-slate-200 rounded mb-4" />
        <div className="space-y-3">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-6 bg-slate-100 rounded" />
          ))}
        </div>
      </div>
    );
  }

  // Find max value to normalize bar widths
  const maxImportance = Math.max(...importances.map((item) => item.importance), 0.01);

  // Clean label mapping for presentation
  const formatFeatureLabel = (raw: string): { label: string; isEngineered: boolean } => {
    switch (raw) {
      case "Rotational speed [rpm]":
        return { label: "Rotational Speed [rpm]", isEngineered: false };
      case "mechanical_power_kw":
        return { label: "Mechanical Power [kW]", isEngineered: true };
      case "Torque [Nm]":
        return { label: "Torque [Nm]", isEngineered: false };
      case "Tool wear [min]":
        return { label: "Tool Wear [min]", isEngineered: false };
      case "wear_load":
        return { label: "Wear Load [min*Nm]", isEngineered: true };
      case "temperature_difference":
        return { label: "Temperature Difference [K]", isEngineered: true };
      case "Air temperature [K]":
        return { label: "Air Temperature [K]", isEngineered: false };
      case "Process temperature [K]":
        return { label: "Process Temperature [K]", isEngineered: false };
      case "Type_M":
        return { label: "Machine Type (M)", isEngineered: false };
      case "Type_H":
        return { label: "Machine Type (H)", isEngineered: false };
      default:
        return { label: raw, isEngineered: false };
    }
  };

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-4 border-b border-slate-100 pb-3">
        <h3 className="text-sm font-semibold text-slate-900">Feature Importance</h3>
        <p className="mt-0.5 text-xs text-slate-500">
          Random Forest Gini importance values across input and physics-engineered variables.
        </p>
      </div>

      {/* Ranked Horizontal Bars */}
      <div className="space-y-2.5">
        {importances.map((item) => {
          const { label, isEngineered } = formatFeatureLabel(item.feature);
          const percentWidth = ((item.importance / maxImportance) * 100).toFixed(1);
          const sharePercent = (item.importance * 100).toFixed(1);

          return (
            <div key={item.feature} className="group">
              <div className="flex items-center justify-between text-xs mb-1">
                <div className="flex items-center gap-1.5">
                  <span className="font-medium text-slate-800">{label}</span>
                  {isEngineered && (
                    <span className="inline-flex items-center gap-0.5 rounded bg-brand-50 px-1.5 py-0.2 text-[10px] font-semibold text-brand-700">
                      Engineered
                    </span>
                  )}
                </div>
                <span className="text-xs font-semibold text-slate-700">
                  {sharePercent}%
                </span>
              </div>

              {/* Progress bar container */}
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    isEngineered ? "bg-brand-500" : "bg-slate-700"
                  }`}
                  style={{ width: `${percentWidth}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Explanatory Note */}
      <div className="mt-4 flex items-start gap-2 rounded-lg bg-slate-50 p-3 text-[11px] text-slate-600 border border-slate-100">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-400" />
        <p leading-relaxed>
          Global feature importance shows which inputs contributed most to the Random Forest's overall predictions during training. It does not explain an individual prediction.
        </p>
      </div>
    </div>
  );
};
