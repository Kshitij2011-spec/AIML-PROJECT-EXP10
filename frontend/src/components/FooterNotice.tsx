import React from "react";
import { AlertCircle } from "lucide-react";

export const FooterNotice: React.FC = () => {
  return (
    <footer className="mt-8 border-t border-slate-200 bg-white py-6">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-start gap-2 max-w-3xl">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
            <div>
              <p className="font-medium text-slate-700">
                Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI Repository ID: 601).
              </p>
              <p className="mt-0.5 leading-relaxed text-slate-500">
                Educational prototype trained on a synthetic predictive-maintenance benchmark. Results are not validated for deployment on real industrial machinery.
              </p>
            </div>
          </div>

          <div className="shrink-0 text-slate-400 text-[11px]">
            MachineGuard AI &copy; 2026 Academic AIML Mini-Project
          </div>
        </div>
      </div>
    </footer>
  );
};
