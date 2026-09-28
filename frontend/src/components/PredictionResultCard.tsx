import React from "react";
import { AlertTriangle, CheckCircle, HelpCircle, ShieldAlert, Cpu } from "lucide-react";
import { MachineInputData, PredictionResult, RiskTier } from "../types/api";

interface ResultCardProps {
  prediction: PredictionResult | null;
  currentInput: MachineInputData | null;
  loading: boolean;
  modelVersion: string;
}

export const PredictionResultCard: React.FC<ResultCardProps> = ({
  prediction,
  currentInput,
  loading,
  modelVersion,
}) => {
  // Empty State before running
  if (!prediction && !loading) {
    return (
      <div className="flex h-full min-h-[420px] flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center shadow-xs">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-100 text-slate-400 mb-3">
          <Cpu className="h-6 w-6" />
        </div>
        <h3 className="text-sm font-semibold text-slate-800">Failure Risk</h3>
        <p className="mt-1 max-w-xs text-xs text-slate-500">
          Enter machine conditions and run a prediction.
        </p>
        <span className="mt-4 inline-flex items-center gap-1 rounded bg-slate-100 px-2 py-1 text-[11px] text-slate-500">
          Model Version: {modelVersion || "v1.0.0"}
        </span>
      </div>
    );
  }

  // Loading Skeleton State
  if (loading) {
    return (
      <div className="flex h-full min-h-[420px] flex-col rounded-xl border border-slate-200 bg-white p-6 shadow-sm animate-pulse">
        <div className="h-4 w-28 bg-slate-200 rounded mb-4" />
        <div className="h-10 w-36 bg-slate-200 rounded mb-3" />
        <div className="h-6 w-24 bg-slate-200 rounded mb-6" />
        <div className="space-y-3 pt-4 border-t border-slate-100">
          <div className="h-4 w-full bg-slate-100 rounded" />
          <div className="h-4 w-5/6 bg-slate-100 rounded" />
          <div className="h-4 w-4/6 bg-slate-100 rounded" />
        </div>
      </div>
    );
  }

  if (!prediction) return null;

  const probPercent = (prediction.failure_probability * 100).toFixed(1);

  // Badge styling according to risk tier
  const riskBadgeStyles: Record<RiskTier, { border: string; bg: string; text: string; icon: React.ReactNode }> = {
    Low: {
      border: "border-emerald-200",
      bg: "bg-emerald-50",
      text: "text-emerald-800",
      icon: <CheckCircle className="h-3.5 w-3.5 text-emerald-600" />,
    },
    Medium: {
      border: "border-amber-200",
      bg: "bg-amber-50",
      text: "text-amber-800",
      icon: <AlertTriangle className="h-3.5 w-3.5 text-amber-600" />,
    },
    High: {
      border: "border-orange-200",
      bg: "bg-orange-50",
      text: "text-orange-800",
      icon: <ShieldAlert className="h-3.5 w-3.5 text-orange-600" />,
    },
    Critical: {
      border: "border-rose-200",
      bg: "bg-rose-50",
      text: "text-rose-800",
      icon: <AlertTriangle className="h-3.5 w-3.5 text-rose-600" />,
    },
  };

  const currentBadge = riskBadgeStyles[prediction.risk_level] || riskBadgeStyles.Low;

  return (
    <div className="flex h-full flex-col justify-between rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div>
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <h2 className="text-base font-semibold text-slate-900">Failure Risk</h2>
          <span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">
            Threshold: {prediction.threshold_applied}
          </span>
        </div>

        {/* Primary Probability & Risk Level Display */}
        <div className="mt-4 flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <span className="text-xs font-medium text-slate-500">Failure Probability</span>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold tracking-tight text-slate-900">
                {probPercent}%
              </span>
              <span className="text-xs text-slate-400">
                ({prediction.failure_probability.toFixed(4)})
              </span>
            </div>
          </div>

          <div>
            <span className="text-xs font-medium text-slate-500 block mb-1">Risk Level</span>
            <span
              className={`inline-flex items-center gap-1.5 rounded-md border px-3 py-1 text-xs font-semibold ${currentBadge.border} ${currentBadge.bg} ${currentBadge.text}`}
            >
              {currentBadge.icon}
              {prediction.risk_level.toUpperCase()}
            </span>
          </div>
        </div>

        {/* Probability Bar */}
        <div className="mt-3">
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
            <div
              className={`h-full transition-all duration-500 ${
                prediction.risk_level === "Low"
                  ? "bg-emerald-500"
                  : prediction.risk_level === "Medium"
                  ? "bg-amber-500"
                  : prediction.risk_level === "High"
                  ? "bg-orange-500"
                  : "bg-rose-500"
              }`}
              style={{ width: `${Math.min(Math.max(prediction.failure_probability * 100, 2), 100)}%` }}
            />
          </div>
          <div className="mt-1 flex justify-between text-[10px] text-slate-400">
            <span>0% (Nominal)</span>
            <span className="text-slate-500 font-medium">Cutoff: {(prediction.threshold_applied * 100).toFixed(0)}%</span>
            <span>100% (High Risk)</span>
          </div>
        </div>

        {/* Model Decision (Non-Certainty Language) */}
        <div className="mt-4 rounded-lg bg-slate-50 p-3.5 border border-slate-100">
          <p className="text-xs font-semibold text-slate-800">
            {prediction.is_failure ? "Failure Signal Detected" : "Normal Operational Parameters Indicated"}
          </p>
          <p className="mt-0.5 text-xs text-slate-600 leading-relaxed">
            {prediction.is_failure
              ? "The model indicates elevated failure risk based on the supplied operating conditions."
              : "The model indicates operating conditions are within standard operational limits."}
          </p>
        </div>

        {/* Result Interpretation — What this means */}
        <div className="mt-4">
          <h4 className="flex items-center gap-1.5 text-xs font-semibold text-slate-700">
            <HelpCircle className="h-3.5 w-3.5 text-slate-400" />
            What This Means
          </h4>
          <p className="mt-1 text-xs text-slate-600 leading-relaxed">
            {prediction.is_failure
              ? "The model estimates an elevated probability of machine failure for these operating conditions. Use this result as a maintenance screening signal and inspect the machine before taking action."
              : "Operating conditions demonstrate stability relative to the training distribution. Continue standard periodic maintenance monitoring without immediate intervention."}
          </p>
          {prediction.maintenance_recommendation && (
            <p className="mt-2 text-xs font-medium text-slate-800 border-l-2 border-slate-400 pl-2">
              {prediction.maintenance_recommendation}
            </p>
          )}
        </div>

        {/* Risk Factors Breakdown if present */}
        {prediction.risk_factors && prediction.risk_factors.length > 0 && (
          <div className="mt-4 border-t border-slate-100 pt-3">
            <h4 className="text-xs font-semibold text-slate-700 mb-2">
              Physical Stress Indicators ({prediction.risk_factors.length})
            </h4>
            <div className="space-y-1.5">
              {prediction.risk_factors.map((rf, idx) => (
                <div
                  key={idx}
                  className="flex items-start justify-between rounded border border-slate-100 bg-slate-50/50 p-2 text-xs"
                >
                  <div>
                    <span className="font-medium text-slate-800">{rf.feature}</span>
                    <span className="ml-1.5 text-slate-500">
                      ({rf.observed_value} {rf.unit})
                    </span>
                    <p className="mt-0.5 text-[11px] text-slate-600">{rf.message}</p>
                  </div>
                  <span
                    className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                      rf.status === "Critical"
                        ? "bg-rose-100 text-rose-800"
                        : "bg-amber-100 text-amber-800"
                    }`}
                  >
                    {rf.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Input Summary Key-Value Data */}
      {currentInput && (
        <div className="mt-5 border-t border-slate-100 pt-3.5">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-2">
            Current Machine Conditions
          </span>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="rounded bg-slate-50 p-1.5">
              <span className="text-[10px] text-slate-400 block">Type</span>
              <span className="font-medium text-slate-800">Variant {currentInput.type}</span>
            </div>
            <div className="rounded bg-slate-50 p-1.5">
              <span className="text-[10px] text-slate-400 block">Air Temp</span>
              <span className="font-medium text-slate-800">{currentInput.air_temperature} K</span>
            </div>
            <div className="rounded bg-slate-50 p-1.5">
              <span className="text-[10px] text-slate-400 block">Process Temp</span>
              <span className="font-medium text-slate-800">{currentInput.process_temperature} K</span>
            </div>
            <div className="rounded bg-slate-50 p-1.5">
              <span className="text-[10px] text-slate-400 block">Speed</span>
              <span className="font-medium text-slate-800">{currentInput.rotational_speed} rpm</span>
            </div>
            <div className="rounded bg-slate-50 p-1.5">
              <span className="text-[10px] text-slate-400 block">Torque</span>
              <span className="font-medium text-slate-800">{currentInput.torque} Nm</span>
            </div>
            <div className="rounded bg-slate-50 p-1.5">
              <span className="text-[10px] text-slate-400 block">Tool Wear</span>
              <span className="font-medium text-slate-800">{currentInput.tool_wear} min</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
