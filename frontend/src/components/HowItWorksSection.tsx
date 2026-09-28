import React from "react";
import { ArrowRight, Settings2, Binary, Cpu, ShieldAlert, Wrench } from "lucide-react";

export const HowItWorksSection: React.FC = () => {
  const steps = [
    {
      step: "01",
      title: "Machine Conditions",
      subtitle: "6 Telemetry Inputs",
      desc: "Spindle speed, torque, air/process temps, tool wear, and machine quality tier.",
      icon: <Settings2 className="h-4 w-4 text-slate-700" />,
    },
    {
      step: "02",
      title: "Preprocessing",
      subtitle: "Encode & Standardize",
      desc: "OneHotEncoding for categorical machine tier and StandardScaler for numerical telemetry.",
      icon: <Binary className="h-4 w-4 text-slate-700" />,
    },
    {
      step: "03",
      title: "Feature Engineering",
      subtitle: "Physics Derivations",
      desc: "Temperature Difference (ΔT), Mechanical Power (kW), and Wear-Load index.",
      icon: <Wrench className="h-4 w-4 text-brand-600" />,
      highlight: true,
    },
    {
      step: "04",
      title: "Random Forest",
      subtitle: "150 Decision Trees",
      desc: "Balanced class weights handle the 3.39% failure imbalance to output calibrated probabilities.",
      icon: <Cpu className="h-4 w-4 text-slate-700" />,
    },
    {
      step: "05",
      title: "Failure Risk",
      subtitle: "Threshold-Tuned Signal",
      desc: "Validation-tuned 0.56 decision boundary classifies risk tier and prescribes maintenance.",
      icon: <ShieldAlert className="h-4 w-4 text-slate-700" />,
    },
  ];

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-4 border-b border-slate-100 pb-3">
        <h3 className="text-sm font-semibold text-slate-900">How It Works</h3>
        <p className="mt-0.5 text-xs text-slate-500">
          End-to-end inference pipeline executing entirely within the loaded scikit-learn model artifact.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-5">
        {steps.map((s, idx) => (
          <div
            key={s.step}
            className={`relative flex flex-col justify-between rounded-lg border p-3.5 transition-colors ${
              s.highlight
                ? "border-brand-200 bg-brand-50/30"
                : "border-slate-100 bg-slate-50/50"
            }`}
          >
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold text-slate-400">{s.step}</span>
                <div className="flex h-7 w-7 items-center justify-center rounded-md bg-white border border-slate-200 shadow-2xs">
                  {s.icon}
                </div>
              </div>
              <h4 className="text-xs font-semibold text-slate-900">{s.title}</h4>
              <p className="text-[11px] font-medium text-slate-500 mb-1">{s.subtitle}</p>
              <p className="text-[11px] text-slate-600 leading-relaxed">{s.desc}</p>
            </div>

            {idx < steps.length - 1 && (
              <div className="hidden sm:block absolute -right-2.5 top-1/2 -translate-y-1/2 z-10">
                <ArrowRight className="h-3.5 w-3.5 text-slate-300" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
