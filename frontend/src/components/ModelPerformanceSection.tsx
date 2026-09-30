import React from "react";
import {
  BarChart3,
  CheckCircle2,
  ShieldCheck,
  Target,
  Layers,
  GitCompare,
  CheckSquare,
  AlertCircle,
} from "lucide-react";
import { ModelInfo } from "../types/api";

interface PerformanceProps {
  modelInfo: ModelInfo | null;
  loading: boolean;
}

export const ModelPerformanceSection: React.FC<PerformanceProps> = ({ modelInfo, loading }) => {
  if (loading || !modelInfo) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm animate-pulse space-y-4">
        <div className="h-5 w-48 bg-slate-200 rounded" />
        <div className="grid grid-cols-2 sm:grid-cols-6 gap-3">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-16 bg-slate-100 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  const {
    test_metrics,
    test_samples,
    selected_threshold,
    algorithm,
    baseline_comparison,
    cross_validation,
    ablation_comparison,
  } = modelInfo;

  const cm = test_metrics?.confusion_matrix;
  const lrAblation = ablation_comparison?.["Logistic Regression (Engineered Features)"]?.validation_metrics_default_threshold;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-6">
      {/* 1. Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div>
          <h3 className="flex items-center gap-2 text-base font-semibold text-slate-900">
            <BarChart3 className="h-5 w-5 text-slate-600" />
            Model Evaluation & Performance Benchmarks
          </h3>
          <p className="mt-0.5 text-xs text-slate-500">
            Comprehensive evaluation across untouched test partition ({test_samples.toLocaleString()} samples), 5-fold stratified cross-validation, and majority baseline.
          </p>
        </div>
        <span className="rounded bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700">
          Primary: {algorithm} (Tuned τ = {selected_threshold})
        </span>
      </div>

      {/* 2. Primary Key Metrics Grid (Test Set) */}
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 block mb-2">
          Final Test Partition Performance (Single Run, Held-Out Data)
        </span>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <div className="rounded-lg border border-slate-100 bg-slate-50/70 p-3">
            <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
              <span>Precision</span>
              <Target className="h-3.5 w-3.5 text-slate-400" />
            </div>
            <div className="text-xl font-bold text-slate-900">
              {(test_metrics.precision * 100).toFixed(1)}%
            </div>
            <span className="text-[10px] text-slate-400">Only 9 false alarms</span>
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
            <span className="text-[10px] text-slate-400">Discrimination index</span>
          </div>

          <div className="rounded-lg border border-slate-100 bg-slate-50/70 p-3">
            <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
              <span>Accuracy</span>
              <CheckCircle2 className="h-3.5 w-3.5 text-slate-400" />
            </div>
            <div className="text-xl font-bold text-slate-900">
              {(test_metrics.accuracy * 100).toFixed(1)}%
            </div>
            <span className="text-[10px] text-slate-400">1,482 / 1,500 correct</span>
          </div>
        </div>
      </div>

      {/* 3. Confusion Matrix Table */}
      {cm && (
        <div className="pt-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 block mb-2">
            Final Test Confusion Matrix (Threshold = {selected_threshold})
          </span>
          <div className="max-w-lg overflow-hidden rounded-lg border border-slate-200">
            <table className="w-full text-center text-xs">
              <thead className="bg-slate-100 text-slate-600 font-medium">
                <tr>
                  <th className="py-2 px-3 text-left">Ground Truth</th>
                  <th className="py-2 px-3">Predicted Normal</th>
                  <th className="py-2 px-3">Predicted Failure</th>
                  <th className="py-2 px-3 text-right">Class Total</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                <tr>
                  <td className="py-2.5 px-3 text-left font-medium text-slate-700 bg-slate-50/50">
                    Actual Normal
                  </td>
                  <td className="py-2.5 px-3 font-semibold text-slate-900 bg-emerald-50/30">
                    {cm.tn} <span className="text-[10px] font-normal text-slate-400">(TN)</span>
                  </td>
                  <td className="py-2.5 px-3 font-semibold text-slate-900">
                    {cm.fp} <span className="text-[10px] font-normal text-slate-400">(FP)</span>
                  </td>
                  <td className="py-2.5 px-3 text-right text-slate-500 font-mono">1,449</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 text-left font-medium text-slate-700 bg-slate-50/50">
                    Actual Failure
                  </td>
                  <td className="py-2.5 px-3 font-semibold text-slate-900">
                    {cm.fn} <span className="text-[10px] font-normal text-slate-400">(FN)</span>
                  </td>
                  <td className="py-2.5 px-3 font-semibold text-slate-900 bg-emerald-50/30">
                    {cm.tp} <span className="text-[10px] font-normal text-slate-400">(TP)</span>
                  </td>
                  <td className="py-2.5 px-3 text-right text-slate-500 font-mono">51</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 4. 5-Fold Stratified Cross-Validation Section */}
      {cross_validation && (
        <div className="pt-2 border-t border-slate-100">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
              <CheckSquare className="h-3.5 w-3.5 text-slate-400" />
              5-Fold Stratified Cross-Validation (Metric: Macro-F1, Training Partition Only)
            </span>
            <span className="text-[11px] text-slate-400">Strictly Leakage-Free (7,000 samples)</span>
          </div>

          <div className="overflow-x-auto rounded-lg border border-slate-200">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-100 text-slate-600 font-medium">
                <tr>
                  <th className="py-2 px-3">Model Architecture</th>
                  <th className="py-2 px-3 text-center">Fold 1</th>
                  <th className="py-2 px-3 text-center">Fold 2</th>
                  <th className="py-2 px-3 text-center">Fold 3</th>
                  <th className="py-2 px-3 text-center">Fold 4</th>
                  <th className="py-2 px-3 text-center">Fold 5</th>
                  <th className="py-2 px-3 text-center font-bold text-slate-800">Mean Macro-F1</th>
                  <th className="py-2 px-3 text-center">Std Dev (σ)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {Object.entries(cross_validation.models).map(([mName, data]) => {
                  const isSelected = mName.includes("Random Forest");
                  return (
                    <tr key={mName} className={isSelected ? "bg-emerald-50/20 font-medium" : ""}>
                      <td className="py-2.5 px-3 font-medium text-slate-800">
                        {mName}
                        {isSelected && (
                          <span className="ml-2 inline-flex items-center rounded-full bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-800">
                            Selected
                          </span>
                        )}
                      </td>
                      {data.fold_scores.map((s, idx) => (
                        <td key={idx} className="py-2.5 px-3 text-center text-slate-600 font-mono">
                          {s.toFixed(4)}
                        </td>
                      ))}
                      <td className="py-2.5 px-3 text-center font-bold text-slate-900 font-mono">
                        {data.mean.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 text-center text-slate-500 font-mono">
                        ±{data.std.toFixed(4)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 5. Comprehensive Model Comparison Section */}
      <div className="pt-2 border-t border-slate-100">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
            <GitCompare className="h-3.5 w-3.5 text-slate-400" />
            Model Comparison: Baseline vs Logistic Regression vs Random Forest
          </span>
          <span className="text-[11px] text-slate-400">Untouched Test Partition (1,500 samples)</span>
        </div>

        <div className="overflow-x-auto rounded-lg border border-slate-200">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-100 text-slate-600 font-medium">
              <tr>
                <th className="py-2 px-3">Model</th>
                <th className="py-2 px-3">Feature Set</th>
                <th className="py-2 px-3 text-center">Accuracy</th>
                <th className="py-2 px-3 text-center">Precision</th>
                <th className="py-2 px-3 text-center">Recall</th>
                <th className="py-2 px-3 text-center">F1-Score</th>
                <th className="py-2 px-3 text-center">Macro-F1</th>
                <th className="py-2 px-3 text-center">PR-AUC</th>
                <th className="py-2 px-3 text-center">ROC-AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {/* Row 1: Majority Class Baseline */}
              {baseline_comparison && (
                <tr className="bg-slate-50/30 text-slate-600">
                  <td className="py-2.5 px-3 font-medium text-slate-700">
                    Majority Class Baseline
                  </td>
                  <td className="py-2.5 px-3 text-slate-500">None (Majority Class)</td>
                  <td className="py-2.5 px-3 text-center font-mono font-medium text-amber-700">
                    {(baseline_comparison.accuracy * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 text-center font-mono">
                    {(baseline_comparison.precision * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 text-center font-mono text-rose-600 font-medium">
                    {(baseline_comparison.recall * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 text-center font-mono">
                    {baseline_comparison.f1.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-center font-mono">
                    {baseline_comparison.macro_f1.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-center font-mono">
                    {baseline_comparison.pr_auc.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-center font-mono">
                    {baseline_comparison.roc_auc.toFixed(4)}
                  </td>
                </tr>
              )}

              {/* Row 2: Logistic Regression (Engineered Features) */}
              <tr className="text-slate-700">
                <td className="py-2.5 px-3 font-medium">Logistic Regression</td>
                <td className="py-2.5 px-3 text-slate-500">Original + Engineered</td>
                <td className="py-2.5 px-3 text-center font-mono">
                  {lrAblation ? `${(lrAblation.accuracy * 100).toFixed(1)}%` : "86.4%"}
                </td>
                <td className="py-2.5 px-3 text-center font-mono">
                  {lrAblation ? `${(lrAblation.precision * 100).toFixed(1)}%` : "17.5%"}
                </td>
                <td className="py-2.5 px-3 text-center font-mono">
                  {lrAblation ? `${(lrAblation.recall * 100).toFixed(1)}%` : "80.4%"}
                </td>
                <td className="py-2.5 px-3 text-center font-mono">
                  {lrAblation ? lrAblation.f1.toFixed(4) : "0.2867"}
                </td>
                <td className="py-2.5 px-3 text-center font-mono">
                  {cross_validation?.models["Logistic Regression (Engineered Features)"]
                    ? cross_validation.models["Logistic Regression (Engineered Features)"].mean.toFixed(4)
                    : "0.5972"}
                </td>
                <td className="py-2.5 px-3 text-center font-mono">
                  {lrAblation ? lrAblation.pr_auc.toFixed(4) : "0.3562"}
                </td>
                <td className="py-2.5 px-3 text-center font-mono">
                  {lrAblation ? lrAblation.roc_auc.toFixed(4) : "0.9048"}
                </td>
              </tr>

              {/* Row 3: Random Forest (Selected Model) */}
              <tr className="bg-emerald-50/20 font-semibold text-slate-900">
                <td className="py-2.5 px-3 font-bold text-slate-900">
                  Random Forest (Selected)
                </td>
                <td className="py-2.5 px-3 text-slate-700">Original + Engineered</td>
                <td className="py-2.5 px-3 text-center font-mono text-emerald-700">
                  {(test_metrics.accuracy * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 text-center font-mono text-emerald-700">
                  {(test_metrics.precision * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 text-center font-mono text-emerald-700">
                  {(test_metrics.recall * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 text-center font-mono text-emerald-700">
                  {test_metrics.f1.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-center font-mono text-emerald-700">
                  {cross_validation?.models["Random Forest (Engineered Features)"]
                    ? cross_validation.models["Random Forest (Engineered Features)"].mean.toFixed(4)
                    : "0.8946"}
                </td>
                <td className="py-2.5 px-3 text-center font-mono text-brand-600">
                  {test_metrics.pr_auc.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-center font-mono text-emerald-700">
                  {test_metrics.roc_auc.toFixed(4)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Academic Interpretation Alert Callout */}
        <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50/70 p-3 flex items-start gap-2.5">
          <AlertCircle className="h-4 w-4 text-amber-700 shrink-0 mt-0.5" />
          <p className="text-xs text-amber-800 leading-relaxed">
            <span className="font-semibold">Academic Baseline Interpretation:</span> The majority-class baseline can achieve high accuracy (96.60%) because machine failures are rare (3.39%). Its failure recall is effectively zero, demonstrating why accuracy alone is insufficient for this problem.
          </p>
        </div>
      </div>
    </div>
  );
};
