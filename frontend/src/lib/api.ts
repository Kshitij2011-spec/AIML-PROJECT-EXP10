import { HealthStatus, MachineInputData, ModelInfo, PredictionResult } from "../types/api";

const BASE_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

class ApiError extends Error {
  statusCode?: number;
  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
  }
}

export const api = {
  async getHealth(): Promise<HealthStatus> {
    try {
      const res = await fetch(`${BASE_URL}/health`);
      if (!res.ok) {
        throw new ApiError(`Health check failed with status ${res.status}`, res.status);
      }
      return await res.json();
    } catch (err: unknown) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        "Prediction service is currently unavailable. Check the backend connection and try again."
      );
    }
  },

  async getModelInfo(): Promise<ModelInfo> {
    try {
      const res = await fetch(`${BASE_URL}/api/model-info`);
      if (!res.ok) {
        throw new ApiError(`Failed to fetch model info: HTTP ${res.status}`, res.status);
      }
      return await res.json();
    } catch (err: unknown) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        "Unable to retrieve model information. Check backend connectivity."
      );
    }
  },

  async predict(input: MachineInputData): Promise<PredictionResult> {
    try {
      const res = await fetch(`${BASE_URL}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(input),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => null);
        const detail = errorData?.detail;
        if (typeof detail === "string") {
          throw new ApiError(detail, res.status);
        } else if (Array.isArray(detail)) {
          // Pydantic validation error list
          const messages = detail.map((d: { msg?: string; loc?: string[] }) => d.msg || "Invalid input").join("; ");
          throw new ApiError(`Validation error: ${messages}`, res.status);
        }
        throw new ApiError(`Prediction request failed with status ${res.status}`, res.status);
      }

      return await res.json();
    } catch (err: unknown) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        "Prediction service is currently unavailable. Check the backend connection and try again."
      );
    }
  },
};
