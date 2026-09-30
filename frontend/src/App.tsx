import React, { useEffect, useState } from "react";
import { Header } from "./components/Header";
import { MachineConditionsForm } from "./components/MachineConditionsForm";
import { PredictionResultCard } from "./components/PredictionResultCard";
import { ModelPerformanceSection } from "./components/ModelPerformanceSection";
import { TreeExplorerSection } from "./components/TreeExplorerSection";
import { FeatureImportanceSection } from "./components/FeatureImportanceSection";
import { HowItWorksSection } from "./components/HowItWorksSection";
import { FooterNotice } from "./components/FooterNotice";
import { api } from "./lib/api";
import { DEFAULT_INPUT } from "./lib/constants";
import { HealthStatus, MachineInputData, ModelInfo, PredictionResult } from "./types/api";

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);

  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [modelInfoLoading, setModelInfoLoading] = useState<boolean>(true);

  const [input, setInput] = useState<MachineInputData>(DEFAULT_INPUT);
  const [currentInput, setCurrentInput] = useState<MachineInputData | null>(null);

  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predictLoading, setPredictLoading] = useState<boolean>(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  const fetchHealth = async () => {
    setHealthLoading(true);
    try {
      const res = await api.getHealth();
      setHealth(res);
    } catch {
      setHealth({
        status: "unavailable",
        model_loaded: false,
        model_version: "1.0.0",
        selected_threshold: 0.56,
        data_honesty: "",
      });
    } finally {
      setHealthLoading(false);
    }
  };

  const fetchModelInfo = async () => {
    setModelInfoLoading(true);
    try {
      const res = await api.getModelInfo();
      setModelInfo(res);
    } catch {
      // Fallback or leave null
      setModelInfo(null);
    } finally {
      setModelInfoLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    fetchModelInfo();
  }, []);

  const handleRunPrediction = async () => {
    setPredictLoading(true);
    setPredictError(null);
    try {
      const res = await api.predict(input);
      setPrediction(res);
      setCurrentInput({ ...input });
    } catch (err: any) {
      setPredictError(err.message || "Prediction service is currently unavailable.");
    } finally {
      setPredictLoading(false);
    }
  };

  const handleReset = () => {
    setInput(DEFAULT_INPUT);
    setPrediction(null);
    setCurrentInput(null);
    setPredictError(null);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 font-sans">
      {/* Primary Header */}
      <Header
        health={health}
        loading={healthLoading}
        onRefreshHealth={fetchHealth}
      />

      <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-6 sm:px-6">
        {/* Compact Page Introduction */}
        <div className="mb-6">
          <p className="text-sm font-medium text-slate-600">
            Predict machine failure risk from current machine operating conditions.
          </p>
        </div>

        {/* Main Workspace (Two-column layout on desktop) */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start mb-8">
          {/* Left Column: Machine Conditions Form */}
          <div className="lg:col-span-7">
            <MachineConditionsForm
              input={input}
              onChange={setInput}
              onSubmit={handleRunPrediction}
              onReset={handleReset}
              loading={predictLoading}
              error={predictError}
            />
          </div>

          {/* Right Column: Prediction Result */}
          <div className="lg:col-span-5 h-full">
            <PredictionResultCard
              prediction={prediction}
              currentInput={currentInput}
              loading={predictLoading}
              modelVersion={health?.model_version || modelInfo?.model_version || "1.0.0"}
            />
          </div>
        </div>

        {/* Model Information & Diagnostics Sections */}
        <div className="space-y-6">
          <ModelPerformanceSection
            modelInfo={modelInfo}
            loading={modelInfoLoading}
          />

          <TreeExplorerSection />

          <FeatureImportanceSection
            importances={modelInfo?.feature_importances || []}
            loading={modelInfoLoading}
          />

          <HowItWorksSection />
        </div>
      </main>

      {/* Dataset & Academic Notice */}
      <FooterNotice />
    </div>
  );
};

export default App;
