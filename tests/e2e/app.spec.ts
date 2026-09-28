import { test, expect } from "@playwright/test";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "*",
  "Access-Control-Allow-Headers": "*",
};

test.describe("MachineGuard AI E2E Suite", () => {
  test.beforeEach(async ({ page }) => {
    // Intercept OPTIONS preflight requests
    await page.route("**/*", async (route) => {
      if (route.request().method() === "OPTIONS") {
        await route.fulfill({
          status: 204,
          headers: corsHeaders,
        });
      } else {
        await route.fallback();
      }
    });

    // Mock health and model-info by default so tests are fast and deterministic
    await page.route("**/health", async (route) => {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({
          status: "healthy",
          model_loaded: true,
          model_version: "1.0.0",
          selected_threshold: 0.56,
          data_honesty: "Educational prototype trained on the AI4I 2020 synthetic predictive-maintenance benchmark.",
        }),
      });
    });

    await page.route("**/api/model-info", async (route) => {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({
          model_name: "MachineGuard AI - Industrial Predictive Maintenance",
          model_version: "1.0.0",
          algorithm: "RandomForestClassifier",
          selected_threshold: 0.56,
          default_threshold: 0.5,
          test_samples: 1500,
          test_metrics: {
            accuracy: 0.988,
            precision: 0.8235,
            recall: 0.8235,
            f1: 0.8235,
            roc_auc: 0.9876,
            pr_auc: 0.8905,
            confusion_matrix: { tn: 1440, fp: 9, fn: 9, tp: 42 },
          },
          feature_importances: [
            { feature: "Rotational speed [rpm]", importance: 0.201 },
            { feature: "mechanical_power_kw", importance: 0.1943 },
            { feature: "Torque [Nm]", importance: 0.1814 },
            { feature: "Tool wear [min]", importance: 0.1412 },
            { feature: "wear_load", importance: 0.1159 },
            { feature: "temperature_difference", importance: 0.0938 },
            { feature: "Air temperature [K]", importance: 0.0405 },
            { feature: "Process temperature [K]", importance: 0.0245 },
            { feature: "Type_M", importance: 0.0051 },
            { feature: "Type_H", importance: 0.0023 },
          ],
          engineered_feature_definitions: {},
          dataset_statistics: {},
          ablation_comparison: {},
          data_honesty_statement: "Educational prototype.",
        }),
      });
    });

    await page.goto("/");
  });

  test("Application loads with header, version, and educational indicator", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "MachineGuard AI", level: 1 })).toBeVisible();
    await expect(page.getByText("Predictive Maintenance Decision Support")).toBeVisible();
    await expect(page.getByText("Educational Prototype", { exact: true })).toBeVisible();
    await expect(page.getByText("API Online")).toBeVisible();
    await expect(page.getByText("Predict machine failure risk from current machine operating conditions.")).toBeVisible();
  });

  test("Prediction form presents all six inputs accessible by label", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Machine Conditions", level: 2 })).toBeVisible();
    await expect(page.getByLabel("Machine Type")).toBeAttached();
    await expect(page.getByLabel("Air Temperature [K]")).toBeVisible();
    await expect(page.getByLabel("Process Temperature [K]")).toBeVisible();
    await expect(page.getByLabel("Rotational Speed [rpm]")).toBeVisible();
    await expect(page.getByLabel("Torque [Nm]")).toBeVisible();
    await expect(page.getByLabel("Tool Wear [min]")).toBeVisible();
  });

  test("User can select machine type and fill numerical fields", async ({ page }) => {
    // Select Type H
    await page.locator("#type-btn-H").click();
    await expect(page.locator("#machine-type-select")).toHaveValue("H");

    // Fill numbers
    const airInput = page.getByLabel("Air Temperature [K]");
    await airInput.fill("300.5");
    await expect(airInput).toHaveValue("300.5");

    const speedInput = page.getByLabel("Rotational Speed [rpm]");
    await speedInput.fill("1620");
    await expect(speedInput).toHaveValue("1620");
  });

  test("Example presets correctly populate the form inputs", async ({ page }) => {
    // Click High Tool Wear preset
    await page.getByRole("button", { name: "High Tool Wear" }).click();
    await expect(page.getByLabel("Tool Wear [min]")).toHaveValue("235");

    // Click Typical Operation preset
    await page.getByRole("button", { name: "Typical Operation" }).click();
    await expect(page.getByLabel("Tool Wear [min]")).toHaveValue("25");
  });

  test("Invalid numerical inputs display visible field-level validation messages", async ({ page }) => {
    // Fill air temperature with out of range value (e.g. 500 K)
    const airInput = page.getByLabel("Air Temperature [K]");
    await airInput.fill("500");

    // Click Run Prediction
    await page.getByRole("button", { name: /Run Prediction/i }).click();

    // Verify validation message is visible
    await expect(page.getByText(/Range: 280/i)).toBeVisible();
  });

  test("Prediction submission shows result with probability, risk level, and threshold", async ({ page }) => {
    // Intercept prediction POST request with deterministic mock
    await page.route("**/api/predict", async (route) => {
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({
          is_failure: true,
          failure_probability: 0.784,
          threshold_applied: 0.56,
          risk_level: "High",
          engineered_features: {
            temperature_difference: 10.5,
            mechanical_power_kw: 8.85,
            wear_load: 13630.0,
          },
          risk_factors: [
            {
              feature: "Tool wear",
              observed_value: 235.0,
              unit: "min",
              status: "Critical",
              message: "Tool wear is critically high (>= 210 min). Imminent risk of tool wear failure (TWF).",
            },
          ],
          maintenance_recommendation: "URGENT: Replace cutting tool insert immediately.",
          data_honesty_statement: "Educational prototype.",
        }),
      });
    });

    // Click Run Prediction
    await page.getByRole("button", { name: /Run Prediction/i }).click();

    // Expect probability text to appear
    await expect(page.getByText(/Failure Probability/i)).toBeVisible();
    await expect(page.getByText(/78.4%/i)).toBeVisible();
    await expect(page.getByText("HIGH", { exact: true })).toBeVisible();
    await expect(page.getByText(/Failure Signal Detected/i)).toBeVisible();
    await expect(page.getByText(/Threshold: 0.56/i)).toBeVisible();
    await expect(page.getByText(/URGENT: Replace cutting tool insert immediately/i)).toBeVisible();
  });

  test("Reset button restores default form state and clears prediction result", async ({ page }) => {
    // Modify a value
    const airInput = page.getByLabel("Air Temperature [K]");
    await airInput.fill("315");

    // Click Reset
    await page.getByRole("button", { name: "Reset" }).click();

    // Should be restored to default (298.1)
    await expect(airInput).toHaveValue("298.1");
    await expect(page.getByText("Enter machine conditions and run a prediction.")).toBeVisible();
  });

  test("Backend unavailability displays visible and calm error UI", async ({ page }) => {
    // Mock 500 error from predict endpoint
    await page.route("**/api/predict", async (route) => {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Internal inference error" }),
      });
    });

    // Run prediction
    await page.getByRole("button", { name: /Run Prediction/i }).click();

    // Verify error UI is displayed
    await expect(page.getByText(/Prediction Request Failed/i)).toBeVisible();
  });
});
