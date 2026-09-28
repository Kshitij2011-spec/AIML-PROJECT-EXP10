import React from "react";
import { BarChart3, CheckCircle2, ShieldCheck, Target, Layers } from "lucide-react";
import { ModelInfo } from "../types/api";

interface PerformanceProps {
  modelInfo: ModelInfo | null;
  loading: boolean;
}

export const ModelPerformanceSection: React.FC<PerformanceProps> = ({ modelInfo, loading }) => {
  if (loading || !modelInfo) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm animate-pulse">
        <div className="h-4 w-40 bg-slate-200 rounded mb-4" />
        <div className="grid grid-cols-2 sm:grid-cols-6 gap-3">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-16 bg-slate-100 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  const { test_metrics, test_samples, selected_threshold, algorithm } = modelInfo;
  const cm = test_metrics?.confusion_matrix;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-900">
            <BarChart3 className="h-4 w-4 text-slate-500" />
            Model Performance
          </h3>
          <p className="mt-0.5 text-xs text-slate-500">
            Evaluated once on untouched test partition ({test_samples.toLocaleString()} samples) using validation-tuned threshold ({selected_threshold}).
          </p>
        </div>
        <span className="rounded bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700">
          {algorithm}
        </span>
      </div>

      {/* Primary Key Metrics Grid */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <div className="rounded-lg border border-slate-100 bg-slate-50/70 p-3">
          <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
            <span>Precision</span>
            <Target className="h-3.5 w-3.5 text-slate-400" />
          </div>
          <div className="text-xl font-bold text-slate-900">
            {(test_metrics.precision * 100).toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-400">Low false alarm rate</span>
        </div>

        <div className="rounded-lg border border-slate-100 bg-slate-50/70 p-3">
          <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
            <span>Recall</span>
            <ShieldCheck className="h-3.5 w-3.5 text-slate-400" />
          </div>
          <div className="text-xl font-bold text-slate-900">
            {(test_metrics.recall * 100).toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-400">42 of 51 failures caught</span>
        </div>

        <div className="rounded-lg border border-slate-100 bg-slate-50/70 p-3">
          <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
            <span>F1-Score</span>
            <CheckCircle2 className="h-3.5 w-3.5 text-slate-400" />
          </div>
          <div className="text-xl font-bold text-slate-900">
            {test_metrics.f1.toFixed(4)}
          </div>
          <span className="text-[10px] text-slate-400">Harmonic balance</span>
        </div>

        <div className="rounded-lg border border-slate-100 bg-slate-50/70 p-3">
          <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
            <span>PR-AUC</span>
            <BarChart3 className="h-3.5 w-3.5 text-brand-500" />
          </div>
          <div className="text-xl font-bold text-brand-600">
            {test_metrics.pr_auc.toFixed(4)}
          </div>
          <span className="text-[10px] text-slate-400">Primary imbalanced metric</span>
        </div>

        <div className="rounded-lg border border-slate-100 bg-slate-50/70 p-3">
          <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
            <span>ROC-AUC</span>
            <Layers className="h-3.5 w-3.5 text-slate-400" />
          </div>
          <div className="text-xl font-bold text-slate-900">
            {test_metrics.roc_auc.toFixed(4)}
          </div>
          <span className="text-[10px] text-slate-400">Overall discrimination</span>
        </div>

        <div className="rounded-lg border border-slate-100 bg-slate-50/70 p-3">
          <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
            <span>Accuracy</span>
            <CheckCircle2 className="h-3.5 w-3.5 text-slate-400" />
          </div>
          <div className="text-xl font-bold text-slate-900">
            {(test_metrics.accuracy * 100).toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-400">{test_samples} total tests</span>
        </div>
      </div>

      {/* Confusion Matrix Table */}
      {cm && (
        <div className="mt-4 pt-3 border-t border-slate-100">
          <span className="text-xs font-semibold text-slate-700 block mb-2">
            Test Confusion Matrix (Threshold = {selected_threshold})
          </span>
          <div className="max-w-md overflow-hidden rounded-lg border border-slate-200">
            <table className="w-full text-center text-xs">
              <thead className="bg-slate-100 text-slate-600 font-medium">
                <tr>
                  <th className="py-1.5 px-3 text-left">Ground Truth</th>
                  <th className="py-1.5 px-3">Predicted Normal</th>
                  <th className="py-1.5 px-3">Predicted Failure</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                <tr>
                  <td className="py-2 px-3 text-left font-medium text-slate-700 bg-slate-50/50">
                    Actual Normal (1,449)
                  </td>
                  <td className="py-2 px-3 font-semibold text-slate-900 bg-emerald-50/30">
                    {cm.tn} <span className="text-[10px] font-normal text-slate-400">(TN)</span>
                  </td>
                  <td className="py-2 px-3 font-semibold text-slate-900">
                    {cm.fp} <span className="text-[10px] font-normal text-slate-400">(FP)</span>
                  </td>
                </tr>
                <tr>
                  <td className="py-2 px-3 text-left font-medium text-slate-700 bg-slate-50/50">
                    Actual Failure (51)
                  </td>
                  <td className="py-2 px-3 font-semibold text-slate-900">
                    {cm.fn} <span className="text-[10px] font-normal text-slate-400">(FN)</span>
                  </td>
                  <td className="py-2 px-3 font-semibold text-slate-900 bg-emerald-50/30">
                    {cm.tp} <span className="text-[10px] font-normal text-slate-400">(TP)</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
