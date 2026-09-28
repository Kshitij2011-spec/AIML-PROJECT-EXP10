import React, { useState } from "react";
import { Sliders, RotateCcw, Play, AlertCircle } from "lucide-react";
import { MachineInputData, MachineType } from "../types/api";
import { EXAMPLE_PRESETS, FIELD_DEFINITIONS } from "../lib/constants";

interface FormProps {
  input: MachineInputData;
  onChange: (input: MachineInputData) => void;
  onSubmit: () => void;
  onReset: () => void;
  loading: boolean;
  error: string | null;
}

interface ValidationErrors {
  air_temperature?: string;
  process_temperature?: string;
  rotational_speed?: string;
  torque?: string;
  tool_wear?: string;
}

export const MachineConditionsForm: React.FC<FormProps> = ({
  input,
  onChange,
  onSubmit,
  onReset,
  loading,
  error,
}) => {
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({});

  const validate = (): boolean => {
    const errors: ValidationErrors = {};
    const { air_temperature, process_temperature, rotational_speed, torque, tool_wear } = input;

    if (air_temperature < FIELD_DEFINITIONS.air_temperature.min || air_temperature > FIELD_DEFINITIONS.air_temperature.max) {
      errors.air_temperature = `Range: ${FIELD_DEFINITIONS.air_temperature.min} – ${FIELD_DEFINITIONS.air_temperature.max} K`;
    }
    if (process_temperature < FIELD_DEFINITIONS.process_temperature.min || process_temperature > FIELD_DEFINITIONS.process_temperature.max) {
      errors.process_temperature = `Range: ${FIELD_DEFINITIONS.process_temperature.min} – ${FIELD_DEFINITIONS.process_temperature.max} K`;
    }
    if (rotational_speed < FIELD_DEFINITIONS.rotational_speed.min || rotational_speed > FIELD_DEFINITIONS.rotational_speed.max) {
      errors.rotational_speed = `Range: ${FIELD_DEFINITIONS.rotational_speed.min} – ${FIELD_DEFINITIONS.rotational_speed.max} rpm`;
    }
    if (torque < FIELD_DEFINITIONS.torque.min || torque > FIELD_DEFINITIONS.torque.max) {
      errors.torque = `Range: ${FIELD_DEFINITIONS.torque.min} – ${FIELD_DEFINITIONS.torque.max} Nm`;
    }
    if (tool_wear < FIELD_DEFINITIONS.tool_wear.min || tool_wear > FIELD_DEFINITIONS.tool_wear.max) {
      errors.tool_wear = `Range: ${FIELD_DEFINITIONS.tool_wear.min} – ${FIELD_DEFINITIONS.tool_wear.max} min`;
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      onSubmit();
    }
  };

  const handleFieldChange = (field: keyof MachineInputData, value: any) => {
    const updated = { ...input, [field]: value };
    onChange(updated);
    if (validationErrors[field as keyof ValidationErrors]) {
      setValidationErrors((prev) => ({ ...prev, [field]: undefined }));
    }
  };

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <h2 className="flex items-center gap-2 text-base font-semibold text-slate-900">
            <Sliders className="h-4 w-4 text-slate-500" />
            Machine Conditions
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Enter the current operating conditions of the machine.
          </p>
        </div>

        {/* Example Presets */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs font-medium text-slate-400 mr-1">Presets:</span>
          {EXAMPLE_PRESETS.map((preset) => (
            <button
              key={preset.name}
              type="button"
              onClick={() => {
                onChange(preset.data);
                setValidationErrors({});
              }}
              title={preset.description}
              className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 hover:text-slate-900 focus:outline-none focus:ring-1 focus:ring-slate-400"
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} noValidate>
        {/* Machine Type Selection */}
        <div className="mb-4">
          <label htmlFor="machine-type-select" className="block text-xs font-semibold text-slate-700 mb-1.5">
            Machine Type <span className="font-normal text-slate-400">(Quality Tier)</span>
          </label>
          <div className="grid grid-cols-3 gap-2">
            {(["L", "M", "H"] as MachineType[]).map((t) => (
              <button
                key={t}
                type="button"
                id={`type-btn-${t}`}
                onClick={() => handleFieldChange("type", t)}
                className={`flex flex-col items-center justify-center rounded-lg border py-2 px-3 text-xs transition-colors ${
                  input.type === t
                    ? "border-slate-900 bg-slate-900 font-semibold text-white shadow-xs"
                    : "border-slate-200 bg-white font-medium text-slate-700 hover:bg-slate-50 hover:text-slate-900"
                }`}
              >
                <span>Type {t}</span>
                <span className={`text-[10px] ${input.type === t ? "text-slate-300" : "text-slate-400"}`}>
                  {t === "L" ? "Low (60%)" : t === "M" ? "Medium (30%)" : "High (10%)"}
                </span>
              </button>
            ))}
          </div>
          {/* Accessible hidden select for testing and screen readers */}
          <select
            id="machine-type-select"
            aria-label="Machine Type"
            value={input.type}
            onChange={(e) => handleFieldChange("type", e.target.value as MachineType)}
            className="sr-only"
          >
            <option value="L">L</option>
            <option value="M">M</option>
            <option value="H">H</option>
          </select>
        </div>

        {/* Telemetry Inputs Grid */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {/* Air Temperature */}
          <div>
            <div className="flex items-center justify-between">
              <label htmlFor="air-temp" className="block text-xs font-semibold text-slate-700">
                Air Temperature [K]
              </label>
              <span className="text-[11px] text-slate-400">Observed: ~295–304 K</span>
            </div>
            <div className="mt-1 relative rounded-md shadow-xs">
              <input
                type="number"
                id="air-temp"
                name="air_temperature"
                aria-label="Air Temperature [K]"
                step={FIELD_DEFINITIONS.air_temperature.step}
                min={FIELD_DEFINITIONS.air_temperature.min}
                max={FIELD_DEFINITIONS.air_temperature.max}
                value={input.air_temperature}
                onChange={(e) => handleFieldChange("air_temperature", parseFloat(e.target.value) || 0)}
                className={`block w-full rounded-lg border px-3 py-2 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 ${
                  validationErrors.air_temperature
                    ? "border-rose-300 focus:border-rose-500 focus:ring-rose-200"
                    : "border-slate-200 focus:border-slate-500 focus:ring-slate-100"
                }`}
              />
            </div>
            {validationErrors.air_temperature && (
              <p className="mt-1 text-[11px] text-rose-600">{validationErrors.air_temperature}</p>
            )}
          </div>

          {/* Process Temperature */}
          <div>
            <div className="flex items-center justify-between">
              <label htmlFor="process-temp" className="block text-xs font-semibold text-slate-700">
                Process Temperature [K]
              </label>
              <span className="text-[11px] text-slate-400">Observed: ~305–314 K</span>
            </div>
            <div className="mt-1 relative rounded-md shadow-xs">
              <input
                type="number"
                id="process-temp"
                name="process_temperature"
                aria-label="Process Temperature [K]"
                step={FIELD_DEFINITIONS.process_temperature.step}
                min={FIELD_DEFINITIONS.process_temperature.min}
                max={FIELD_DEFINITIONS.process_temperature.max}
                value={input.process_temperature}
                onChange={(e) => handleFieldChange("process_temperature", parseFloat(e.target.value) || 0)}
                className={`block w-full rounded-lg border px-3 py-2 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 ${
                  validationErrors.process_temperature
                    ? "border-rose-300 focus:border-rose-500 focus:ring-rose-200"
                    : "border-slate-200 focus:border-slate-500 focus:ring-slate-100"
                }`}
              />
            </div>
            {validationErrors.process_temperature && (
              <p className="mt-1 text-[11px] text-rose-600">{validationErrors.process_temperature}</p>
            )}
          </div>

          {/* Rotational Speed */}
          <div>
            <div className="flex items-center justify-between">
              <label htmlFor="rot-speed" className="block text-xs font-semibold text-slate-700">
                Rotational Speed [rpm]
              </label>
              <span className="text-[11px] text-slate-400">Observed: ~1200–2800 rpm</span>
            </div>
            <div className="mt-1 relative rounded-md shadow-xs">
              <input
                type="number"
                id="rot-speed"
                name="rotational_speed"
                aria-label="Rotational Speed [rpm]"
                step={FIELD_DEFINITIONS.rotational_speed.step}
                min={FIELD_DEFINITIONS.rotational_speed.min}
                max={FIELD_DEFINITIONS.rotational_speed.max}
                value={input.rotational_speed}
                onChange={(e) => handleFieldChange("rotational_speed", parseFloat(e.target.value) || 0)}
                className={`block w-full rounded-lg border px-3 py-2 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 ${
                  validationErrors.rotational_speed
                    ? "border-rose-300 focus:border-rose-500 focus:ring-rose-200"
                    : "border-slate-200 focus:border-slate-500 focus:ring-slate-100"
                }`}
              />
            </div>
            {validationErrors.rotational_speed && (
              <p className="mt-1 text-[11px] text-rose-600">{validationErrors.rotational_speed}</p>
            )}
          </div>

          {/* Torque */}
          <div>
            <div className="flex items-center justify-between">
              <label htmlFor="torque" className="block text-xs font-semibold text-slate-700">
                Torque [Nm]
              </label>
              <span className="text-[11px] text-slate-400">Observed: ~10–80 Nm</span>
            </div>
            <div className="mt-1 relative rounded-md shadow-xs">
              <input
                type="number"
                id="torque"
                name="torque"
                aria-label="Torque [Nm]"
                step={FIELD_DEFINITIONS.torque.step}
                min={FIELD_DEFINITIONS.torque.min}
                max={FIELD_DEFINITIONS.torque.max}
                value={input.torque}
                onChange={(e) => handleFieldChange("torque", parseFloat(e.target.value) || 0)}
                className={`block w-full rounded-lg border px-3 py-2 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 ${
                  validationErrors.torque
                    ? "border-rose-300 focus:border-rose-500 focus:ring-rose-200"
                    : "border-slate-200 focus:border-slate-500 focus:ring-slate-100"
                }`}
              />
            </div>
            {validationErrors.torque && (
              <p className="mt-1 text-[11px] text-rose-600">{validationErrors.torque}</p>
            )}
          </div>

          {/* Tool Wear */}
          <div className="sm:col-span-2">
            <div className="flex items-center justify-between">
              <label htmlFor="tool-wear" className="block text-xs font-semibold text-slate-700">
                Tool Wear [min]
              </label>
              <span className="text-[11px] text-slate-400">Replacement limit: ~200–240 min</span>
            </div>
            <div className="mt-1 relative rounded-md shadow-xs">
              <input
                type="number"
                id="tool-wear"
                name="tool_wear"
                aria-label="Tool Wear [min]"
                step={FIELD_DEFINITIONS.tool_wear.step}
                min={FIELD_DEFINITIONS.tool_wear.min}
                max={FIELD_DEFINITIONS.tool_wear.max}
                value={input.tool_wear}
                onChange={(e) => handleFieldChange("tool_wear", parseFloat(e.target.value) || 0)}
                className={`block w-full rounded-lg border px-3 py-2 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 ${
                  validationErrors.tool_wear
                    ? "border-rose-300 focus:border-rose-500 focus:ring-rose-200"
                    : "border-slate-200 focus:border-slate-500 focus:ring-slate-100"
                }`}
              />
            </div>
            {validationErrors.tool_wear && (
              <p className="mt-1 text-[11px] text-rose-600">{validationErrors.tool_wear}</p>
            )}
          </div>
        </div>

        {/* Global Error Notice if API fails */}
        {error && (
          <div className="mt-4 flex items-start gap-2.5 rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
            <div>
              <p className="font-medium">Prediction Request Failed</p>
              <p className="mt-0.5 text-rose-700">{error}</p>
            </div>
          </div>
        )}

        {/* Form Actions */}
        <div className="mt-6 flex items-center justify-between gap-3 border-t border-slate-100 pt-4">
          <button
            type="button"
            onClick={() => {
              onReset();
              setValidationErrors({});
            }}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-medium text-slate-600 hover:bg-slate-50 hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-400 disabled:opacity-50"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset
          </button>

          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-slate-900 px-5 py-2 text-xs font-semibold text-white shadow-sm hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 disabled:bg-slate-400"
          >
            <Play className={`h-3.5 w-3.5 fill-current ${loading ? "animate-pulse" : ""}`} />
            {loading ? "Predicting…" : "Run Prediction"}
          </button>
        </div>
      </form>
    </div>
  );
};
