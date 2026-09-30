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
          total_estimators: 150,
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
          baseline_comparison: {
            model_name: "Majority Class Baseline",
            strategy: "most_frequent",
            accuracy: 0.966,
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            macro_f1: 0.4914,
            pr_auc: 0.034,
            roc_auc: 0.5,
            confusion_matrix: { tn: 1449, fp: 0, fn: 51, tp: 0 },
            academic_note:
              "The majority-class baseline achieves high accuracy (96.60%) simply because machine failures are rare (3.39%). Its failure recall is effectively zero, demonstrating why accuracy alone is insufficient for this problem.",
          },
          cross_validation: {
            n_splits: 5,
            scoring: "f1_macro",
            partition: "train_only",
            models: {
              "Logistic Regression (Engineered Features)": {
                fold_scores: [0.5992, 0.58, 0.601, 0.6106, 0.5951],
                mean: 0.5972,
                std: 0.01,
              },
              "Random Forest (Engineered Features)": {
                fold_scores: [0.8898, 0.8876, 0.9067, 0.8691, 0.9199],
                mean: 0.8946,
                std: 0.0174,
              },
            },
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

    // Mock tree endpoint by default
    await page.route("**/api/model/tree/*", async (route) => {
      const url = route.request().url();
      if (url.includes("/tree/999")) {
        await route.fulfill({
          status: 404,
          headers: corsHeaders,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Tree index 999 out of range" }),
        });
        return;
      }

      const match = url.match(/\/tree\/(\d+)/);
      const treeIdx = match ? parseInt(match[1], 10) : 0;

      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({
          tree_index: treeIdx,
          total_estimators: 150,
          node_count: 3,
          max_depth: 3,
          filtered_max_depth: 3,
          nodes: [
            {
              id: 0,
              depth: 0,
              is_leaf: false,
              feature: "Rotational speed [rpm]",
              feature_label: "Rotational Speed",
              threshold: -0.8474,
              threshold_unscaled: 1386.5,
              unit: "rpm",
              condition_left: "≤ 1386.5 rpm",
              condition_right: "> 1386.5 rpm",
              gini: 0.5,
              samples: 2972,
              class_counts: [2862, 110],
              class_proportions: [0.5034, 0.4966],
              predicted_class: 0,
              predicted_class_name: "Normal",
              left_child: 1,
              right_child: 2,
            },
            {
              id: 1,
              depth: 1,
              is_leaf: true,
              feature: null,
              feature_label: null,
              threshold: null,
              threshold_unscaled: null,
              unit: null,
              condition_left: null,
              condition_right: null,
              gini: 0.05,
              samples: 521,
              class_counts: [500, 21],
              class_proportions: [0.96, 0.04],
              predicted_class: 0,
              predicted_class_name: "Normal",
              left_child: null,
              right_child: null,
            },
            {
              id: 2,
              depth: 1,
              is_leaf: true,
              feature: null,
              feature_label: null,
              threshold: null,
              threshold_unscaled: null,
              unit: null,
              condition_left: null,
              condition_right: null,
              gini: 0.12,
              samples: 2451,
              class_counts: [2362, 89],
              class_proportions: [0.963, 0.037],
              predicted_class: 1,
              predicted_class_name: "Failure",
              left_child: null,
              right_child: null,
            },
          ],
          edges: [
            {
              source: 0,
              target: 1,
              branch: "left",
              condition: "≤ 1386.5 rpm",
            },
            {
              source: 0,
              target: 2,
              branch: "right",
              condition: "> 1386.5 rpm",
            },
          ],
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
    await page.getByRole("button", { name: "Reset", exact: true }).click();

    // Should be restored to default (298.1)
    await expect(airInput).toHaveValue("298.1");
    await expect(page.getByText("Enter machine conditions and run a prediction.")).toBeVisible();
  });

  test("Model Performance shows 5-fold cross-validation and baseline comparison", async ({ page }) => {
    // Verify Model Evaluation header
    await expect(page.getByText("Model Evaluation & Performance Benchmarks")).toBeVisible();

    // Verify 5-fold cross validation table appears
    await expect(page.getByText(/5-Fold Stratified Cross-Validation/i).first()).toBeVisible();
    await expect(page.getByText(/Mean Macro-F1/i)).toBeVisible();
    await expect(page.getByText("Random Forest (Engineered Features)")).toBeVisible();

    // Verify Model Comparison table appears
    await expect(page.getByText(/Model Comparison: Baseline vs Logistic Regression vs Random Forest/i)).toBeVisible();
    await expect(page.getByText("Majority Class Baseline")).toBeVisible();
    await expect(page.getByText(/The majority-class baseline can achieve high accuracy/i)).toBeVisible();
  });

  test("Tree Explorer renders with tree selector, metadata, and decision/leaf nodes", async ({ page }) => {
    const explorer = page.locator("#tree-explorer");
    // Verify Tree Explorer Header
    await expect(explorer.getByText("Random Forest Tree Explorer")).toBeVisible();
    await expect(explorer.getByText("Total Trees:")).toBeVisible();
    await expect(explorer.getByText("Tree 1", { exact: true })).toBeVisible();

    // Verify decision node and leaf node exist in rendered canvas
    await expect(explorer.getByText("ROOT", { exact: true })).toBeVisible();
    await expect(explorer.getByText("Rotational Speed", { exact: true })).toBeVisible();
    await expect(explorer.getByText("LEAF #1", { exact: true })).toBeVisible();
    await expect(explorer.getByText("LEAF #2", { exact: true })).toBeVisible();
    await expect(explorer.getByText("NORMAL", { exact: true })).toBeVisible();
    await expect(explorer.getByText("FAILURE", { exact: true })).toBeVisible();

    // Verify How to Read the Decision Tree explanation
    await expect(explorer.getByText("How to Read the Decision Tree")).toBeVisible();
    await expect(explorer.getByText(/Each decision node applies a learned split/i)).toBeVisible();
  });

  test("Changing tree selector triggers another tree API request", async ({ page }) => {
    let tree1Requested = false;
    await page.route("**/api/model/tree/1*", async (route) => {
      tree1Requested = true;
      await route.fulfill({
        status: 200,
        headers: corsHeaders,
        contentType: "application/json",
        body: JSON.stringify({
          tree_index: 1,
          total_estimators: 150,
          node_count: 3,
          max_depth: 2,
          nodes: [
            {
              id: 0,
              depth: 0,
              is_leaf: false,
              feature: "Torque [Nm]",
              feature_label: "Torque",
              threshold: 0.12,
              threshold_unscaled: 41.2,
              unit: "Nm",
              condition_left: "≤ 41.2 Nm",
              condition_right: "> 41.2 Nm",
              gini: 0.48,
              samples: 3000,
              class_counts: [2900, 100],
              class_proportions: [0.5, 0.5],
              predicted_class: 0,
              predicted_class_name: "Normal",
              left_child: 1,
              right_child: 2,
            },
            {
              id: 1,
              depth: 1,
              is_leaf: true,
              gini: 0.01,
              samples: 1200,
              class_counts: [1190, 10],
              class_proportions: [0.99, 0.01],
              predicted_class: 0,
              predicted_class_name: "Normal",
              left_child: null,
              right_child: null,
            },
            {
              id: 2,
              depth: 1,
              is_leaf: true,
              gini: 0.08,
              samples: 1800,
              class_counts: [1710, 90],
              class_proportions: [0.95, 0.05],
              predicted_class: 1,
              predicted_class_name: "Failure",
              left_child: null,
              right_child: null,
            },
          ],
          edges: [],
        }),
      });
    });

    const explorer = page.locator("#tree-explorer");

    // Click Next Tree button
    await explorer.getByRole("button", { name: "Next Tree" }).click();

    // Verify Tree 2 is displayed and Torque node is visible
    await expect(explorer.getByText("Tree 2", { exact: true })).toBeVisible();
    await expect(explorer.getByText("Torque", { exact: true })).toBeVisible();
    expect(tree1Requested).toBe(true);
  });
});
