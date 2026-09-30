# MachineGuard AI — AI-Based Predictive Maintenance of Industrial Machines

> **Educational Prototype Notice:**  
> Educational prototype trained on the AI4I 2020 synthetic predictive-maintenance benchmark. Results are not validated for deployment on real industrial machinery.

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Vite](https://img.shields.io/badge/Vite-6.0+-646CFF.svg)](https://vitejs.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Problem Statement

Industrial manufacturing relies heavily on computer numerical control (CNC) milling, turning, and cutting machinery. Unplanned spindle and tooling failures lead to expensive production outages, damaged workpieces, and dangerous mechanical fragmentation. Traditional preventative maintenance relies on rigid calendar intervals or fixed hours of operation, which frequently replaces functional components prematurely or fails to prevent unexpected breakdowns under severe mechanical or thermal stress.

**MachineGuard AI** addresses this challenge by framing predictive maintenance as an imbalanced binary classification and calibrated probability estimation problem: predicting whether an operating industrial machine will experience a failure based on dynamic sensor telemetry (temperatures, spindle rotational speed, torque) and cumulative cutting wear.

---

## 2. Objectives

1. **Target Leakage Prevention**: Enforce strict architectural isolation by removing post-failure symptom flags and serial identifiers.
2. **Physics-Informed Feature Engineering**: Augment basic machine telemetry with thermodynamic and mechanical interaction features without hardcoding artificial heuristic rules.
3. **Rigorous Academic Baseline Comparison**: Implement a deterministic majority-class baseline to demonstrate why accuracy is a flawed evaluation metric in imbalanced industrial safety regimes.
4. **Empirical Ablation Evaluation**: Validate the individual and combined impact of feature engineering across linear and ensemble paradigms.
5. **5-Fold Stratified Cross-Validation**: Evaluate model variance and generalization stability strictly on the training partition without test or validation set contamination.
6. **Validation-Guided Threshold Optimization**: Maximize minority-class $F_1$-score on held-out validation data rather than using the arbitrary $0.50$ decision threshold.
7. **Constituent Decision Tree Inspection**: Provide interactive, node-by-node architectural transparency into constituent Random Forest estimators (features, split thresholds, impurities, sample counts, and class distributions).
8. **Production-Ready Deployment**: Deliver an interactive web application, containerized FastAPI backend, and comprehensive automated test suite (pytest + Playwright E2E).

---

## 3. Dataset Identification

* **Benchmark**: UCI Machine Learning Repository — [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (Dataset ID: 601).
* **Records**: 10,000 synthetic operational cycles reflecting real-world industrial CNC cutting tools.
* **Target Variable**: `Machine failure` (binary 0 = Normal Operation, 1 = Machine Failure).
* **Class Distribution (Severe Imbalance)**:
  - Nominal / Non-Failure ($0$): **9,661 samples** ($96.61\%$)
  - Failure ($1$): **339 samples** ($3.39\%$)
* **Product Variant (`Type`)**:
  - `L` (Low Quality / High Volume): 6,000 records ($60.00\%$)
  - `M` (Medium Quality / Standard): 2,997 records ($29.97\%$)
  - `H` (High Quality / Premium): 1,003 records ($10.03\%$)

---

## 4. Data Preprocessing

To ensure zero information leakage from future distributions:
1. **Categorical Encoding**: Product variant `Type` (`L`, `M`, `H`) is transformed using `OneHotEncoder(categories=[["L", "M", "H"]], drop="first", sparse_output=False)` producing binary indicators `Type_M` and `Type_H`.
2. **Feature Standardization**: Continuous telemetry features and engineered variables are scaled via `StandardScaler()`.
3. **Strict Fitting Isolation**: Scalers and encoders are fitted **strictly on the 70% training split** and subsequently applied to transform validation and test splits without refitting.

---

## 5. Leakage Prevention

The AI4I dataset includes row identifiers and diagnostic failure-mode indicator columns:
* Identifiers: `UDI` (record index), `Product ID` (serial string).
* Post-Failure Root Cause Flags: `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), `RNF` (Random Failure).

> **CRITICAL ARCHITECTURAL RULE:** These five failure-mode flags represent post-failure root-cause labels generated when a failure has already occurred. Using them as inputs would introduce catastrophic target leakage. They are **strictly excluded** from all training and inference pipelines.

### Usable Telemetry Predictors:
1. `Type`: Machine product variant (`L`, `M`, `H`).
2. `Air temperature [K]`: Ambient factory temperature ($295.3 - 304.5\text{ K}$, $\mu = 300.00\text{ K}$).
3. `Process temperature [K]`: Thermal cutting chamber temperature ($305.7 - 313.8\text{ K}$, $\mu = 310.01\text{ K}$).
4. `Rotational speed [rpm]`: Spindle kinematic speed ($1168 - 2886\text{ rpm}$, $\mu = 1538.78\text{ rpm}$).
5. `Torque [Nm]`: Spindle shaft resistance ($3.8 - 76.6\text{ Nm}$, $\mu = 39.99\text{ Nm}$).
6. `Tool wear [min]`: Cumulative active cutting time ($0 - 253\text{ min}$, $\mu = 107.95\text{ min}$).

---

## 6. Feature Engineering

Rather than hardcoding arbitrary heuristics, three physically grounded engineering variables were derived from thermodynamics and rotational dynamics:

1. **`temperature_difference`** ($K$):
   $$\Delta T = T_{\text{process}} - T_{\text{air}}$$
   *Physical Meaning:* The thermal dissipation gradient across the tool-workpiece interface. Narrow or negative gradients indicate heat-sink saturation, while extreme gradients reflect severe friction heat buildup.

2. **`mechanical_power_kw`** ($kW$):
   $$P_{\text{mech}} = \frac{\tau \times \omega}{1000} = \frac{\text{Torque [Nm]} \times \text{Rotational Speed [rpm]} \times 2\pi}{60 \times 1000}$$
   *Physical Meaning:* Instantaneous mechanical power delivered to the machine spindle. Excursions above $9.0\text{ kW}$ or below $3.0\text{ kW}$ characterize severe motor overload or stall.

3. **`wear_load`** ($min \cdot Nm$):
   $$\text{WearLoad} = \text{Tool wear [min]} \times \text{Torque [Nm]}$$
   *Physical Meaning:* Compound mechanical stress. High cutting torque exerted on an insert with extensive microscopic fatigue fractures triggers immediate breakage and overstrain failure.

---

## 7. Algorithms

We examine both linear and tree-based paradigms:
1. **Majority Class Baseline (`DummyClassifier`)**: Deterministic baseline predicting the majority class (`most_frequent`), setting the lower bound for utility.
2. **Regularized Logistic Regression (`LogisticRegression`)**: Linear decision boundary model with $L_2$ regularization (`class_weight="balanced"`, `max_iter=1000`, `random_state=42`).
3. **Random Forest Classifier (`RandomForestClassifier`)**: Ensemble of 150 bagged decision trees with balanced bootstrap weights (`n_estimators=150`, `max_depth=12`, `class_weight="balanced"`, `random_state=42`).

---

## 8. Baseline Comparison

To demonstrate why raw accuracy is deeply misleading in rare-event predictive maintenance ($3.39\%$ failure rate), the Majority Class Baseline was evaluated on the untouched 1,500-sample test set:

| Model Configuration | Strategy / Type | Accuracy | Precision | Recall | F1-Score | Macro-F1 | PR-AUC | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | `most_frequent` | 0.9660 | 0.0000 | 0.0000 | 0.0000 | 0.4914 | 0.0340 | 0.5000 |
| **Logistic Regression (Eng.)** | Linear ($L_2$) | 0.8160 | 0.1340 | 0.7647 | 0.2281 | 0.5623 | 0.3168 | 0.8659 |
| **Random Forest (Final Model)** | Ensemble (150 trees) | **0.9880** | **0.8235** | **0.8235** | **0.8235** | **0.9085** | **0.8905** | **0.9876** |

### Academic Insight:
> *"The majority-class baseline achieves **96.60% accuracy** simply because machine failures are rare. Its failure recall is **0.00%**, detecting zero true failures out of 51 test failures. This proves why accuracy alone is completely insufficient for industrial predictive maintenance."*

---

## 9. Ablation Study

An empirical ablation study on the **1,500-sample Validation Set** isolated the impact of domain-engineered features across both model architectures:

| Model | Engineered Features | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Val PR-AUC | Val ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | No | 0.8347 | 0.1365 | 0.7255 | 0.2298 | 0.3834 | 0.8749 |
| **Logistic Regression** | **Yes** | 0.8640 | 0.1745 | 0.8039 | 0.2867 | 0.3562 | 0.9048 |
| **Random Forest** | No | 0.9720 | 0.5738 | 0.6863 | 0.6250 | 0.6639 | 0.9731 |
| **Random Forest** | **Yes** | **0.9873** | **0.8077** | **0.8235** | **0.8155** | **0.8479** | **0.9782** |

### Key Findings:
* For the Random Forest, adding physical engineered features improved Validation $F_1$ from **0.6250 to 0.8155 (+19.05% absolute increase)**.
* **PR-AUC increased by +18.40%** (from 0.6639 to 0.8479).
* Precision increased from 0.5738 to 0.8077, reducing false alarms by **61.5%** while simultaneously improving failure recall.

---

## 10. Cross-Validation

To verify that model performance is robust and not an artifact of a single train/validation split, **5-Fold Stratified Cross-Validation** was executed:

* **Splits**: $k = 5$, `shuffle=True`, `random_state=42`.
* **Partition**: Evaluated **strictly on the 7,000-sample training partition** (validation and test sets remained completely untouched).
* **Scoring Metric**: `f1_macro` (evaluates performance across both minority failure and majority nominal classes).

| Model Configuration | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean Macro-F1 | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Eng.)** | 0.5992 | 0.5800 | 0.6010 | 0.6106 | 0.5951 | **0.5972** | $\pm 0.0100$ |
| **Random Forest (Eng.)** | 0.8898 | 0.8876 | 0.9067 | 0.8691 | 0.9199 | **0.8946** | $\pm 0.0174$ |

---

## 11. Threshold Optimization

In imbalanced classification ($3.39\%$ failure incidence), the default $0.50$ decision threshold is suboptimal. The classification threshold was optimized **strictly on the 1,500-sample Validation Set**:

* **Default Threshold ($0.50$)**: Val $F_1 = 0.8155$ | Val Recall = $0.8235$ | Val Precision = $0.8077$
* **Selected Optimal Threshold ($0.56$)**: **Val $F_1 = 0.8367$** | Val Recall = $0.8039$ | **Val Precision = $0.8723$**

---

## 12. Final Test Evaluation

The selected Random Forest pipeline was evaluated **exactly once** on the untouched 1,500-sample Test Set ($1,449$ normal, $51$ failures):

| Metric | Default Threshold (0.50) | Tuned Threshold (0.56) |
| :--- | :---: | :---: |
| **Accuracy** | 0.9853 ($98.53\%$) | **0.9880 ($98.80\%$)** |
| **Precision** | 0.7636 ($76.36\%$) | **0.8235 ($82.35\%$)** |
| **Recall (Sensitivity)** | 0.8235 ($82.35\%$) | **0.8235 ($82.35\%$)** |
| **F1-Score** | 0.7925 | **0.8235** |
| **PR-AUC (Average Precision)** | **0.8905** | **0.8905** |
| **ROC-AUC** | **0.9876** | **0.9876** |

---

## 13. Confusion Matrix

On the held-out test partition at the optimal $0.56$ threshold:

```
                      Predicted Normal    Predicted Failure
Actual Normal (0)          1440                  9          (Specificity: 99.38%)
Actual Failure (1)            9                 42          (Recall:      82.35%)
```

* **True Negatives ($TN$)**: 1,440 normal operating cycles correctly identified.
* **False Positives ($FP$)**: Only 9 false maintenance dispatches out of 1,449 normal machines.
* **False Negatives ($FN$)**: 9 missed failure events.
* **True Positives ($TP$)**: 42 catastrophic machine failures successfully caught in advance.

---

## 14. Feature Importance

Global Gini feature importances extracted from the fitted ensemble confirm that domain-engineered features account for **40.4%** of the model's total discriminatory power:

| Rank | Feature | Type | Importance | Physical Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Rotational speed [rpm]` | Telemetry | 20.10% | Primary kinematic cutting parameter |
| 2 | **`mechanical_power_kw`** | **Engineered** | **19.43%** | Motor electrical-to-mechanical load coupling |
| 3 | `Torque [Nm]` | Telemetry | 18.14% | Spindle cutting resistance and torque load |
| 4 | `Tool wear [min]` | Telemetry | 14.12% | Cumulative insert micro-fracture life |
| 5 | **`wear_load`** | **Engineered** | **11.59%** | Compound cutting abrasion index |
| 6 | **`temperature_difference`** | **Engineered** | **9.38%** | Heat dissipation efficiency gradient |
| 7 | `Air temperature [K]` | Telemetry | 4.05% | Ambient baseline temperature |
| 8 | `Process temperature [K]` | Telemetry | 2.45% | Cutting chamber temperature |
| 9 | `Type_M` | Encoded | 0.51% | Medium quality product variant |
| 10 | `Type_H` | Encoded | 0.23% | High quality product variant |

---

## 15. Random Forest Tree Visualization

To fulfill academic transparency requirements, MachineGuard AI includes an interactive **Random Forest Tree Explorer** backed by dedicated API endpoints (`GET /api/model/tree/{tree_index}`):

* **Direct Estimator Access**: Traverses the `estimator.tree_` structure directly from `RandomForestClassifier.estimators_`.
* **Real Model Split Data**: Exposes actual features, physical thresholds (both normalized $z$-scores and unscaled engineering units like $kW$, $rpm$, $Nm$), Gini impurities, sample counts, and class distributions.
* **Tree Metrics**:
  - Total Ensemble Estimators: **150 trees**
  - Estimator 0: **245 total nodes**, **Max Depth = 12**
  - Estimator 1: **249 total nodes**, **Max Depth = 12**
  - Estimator 2: **201 total nodes**, **Max Depth = 11**
* **Visualization Controls**: Interactive pan and zoom canvas, depth filter (levels 2, 3, 4, 5, or Full Tree), fit-to-screen, and previous/next tree navigation.
* **Static Publication Artifacts**: Exported vector graphic `backend/artifacts/random_forest_tree_0.svg` and raster `backend/artifacts/random_forest_tree_0.png`.

---

## 16. Practical Applications

1. **Condition-Based Maintenance (CBM)**: Replace rigid calendar-based maintenance with telemetry-informed interventions.
2. **Real-Time Edge SCADA Ingestion**: The serialized pipeline (~15 MB memory footprint) performs single-record inference in under 5 milliseconds on industrial IoT gateways.
3. **Tool Insert Lifespan Extension**: Identify optimal insert replacement points before catastrophic tool fractures destroy expensive machine castings.

---

## 17. Societal Impact

* **Worker Safety**: Protects machine operators from high-speed spindle fragmentation and explosive tool breakage.
* **Industrial Energy Conservation**: Degraded cutting tools consume excessive electrical energy; early detection optimizes motor load and reduces manufacturing power draw.
* **Resource Sustainability**: Reduces premature scrappage of specialized carbide and diamond-tipped cutting tooling.

---

## 18. Limitations

1. **Synthetic Benchmark Characteristics**: The AI4I 2020 dataset models failure mechanics deterministically. Real-world machine tools experience vibrational harmonics, lubrication breakdown, and seasonal humidity drift not captured here.
2. **Absence of Continuous Regression Target**: The AI4I dataset contains no continuous Remaining Useful Life (RUL) target; inventing an artificial regression target would compromise academic validity.
3. **Sensor Fault Tolerance**: The model assumes operational, calibrated sensors. In production environments, sensor failure validation must precede inference.

---

## 19. Web Application

Built with **React 18, Vite, TypeScript, and Tailwind CSS**:
* **Real-time Diagnostic Dashboard**: Immediate probability gauge, risk level classification, and actionable prescriptive engineering recommendations.
* **Model Performance Suite**: Displays final test metrics, 5-fold cross-validation tables, model comparison (Baseline vs LR vs RF), and feature importance distributions.
* **Random Forest Tree Explorer**: Visual canvas inspecting constituent decision trees with pan, zoom, depth filtering, and node breakdown.
* **Accessible Engineering UX**: Strict dark/light slate palette, high-contrast badges, keyboard accessibility, and responsive layouts.

---

## 20. API

FastAPI backend with Pydantic v2 schemas:
* `GET /health` / `GET /api/health`: Service health check, model status, and active decision threshold.
* `GET /api/model-info`: Aggregated metadata, baseline metrics, 5-fold CV results, ablation table, feature importances, and dataset statistics.
* `GET /api/model/tree/{tree_index}?max_depth=3`: Direct inspection of constituent Random Forest decision tree nodes, edges, and split criteria.
* `POST /api/predict`: Real-time machine failure prediction, risk category, and prescriptive guidance.
* `POST /predict/batch`: High-throughput batch inference.

---

## 21. Testing

Automated testing across unit, integration, and end-to-end suites:
* **Backend Suite (pytest)**: 16 automated tests verifying endpoints, input validation, threshold enforcement, baseline metrics, 5-fold CV structure, and tree node extraction (`backend/tests/`).
* **ML Integrity Verification**: 12-point integrity check confirming predictor schemas, leakage absence, feature formulas, stratification, and artifact consistency (`backend/ml/verify_ml_integrity.py`).
* **End-to-End Suite (Playwright)**: 10 automated browser tests verifying page load, input validation, form presets, prediction workflow, model performance tables, tree explorer rendering, and tree navigation (`tests/e2e/app.spec.ts`).

---

## 22. Deployment

* **Backend**: Hosted on **Render** as a Python 3 web service (`https://machineguard-ai-backend.onrender.com`).
  - Command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
* **Frontend**: Hosted on **Vercel** (`https://machineguard-ai-chi.vercel.app`).
  - Framework: Vite / React
  - Output: `dist`

---

## 23. References

1. Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. Third International Conference on Artificial Intelligence for Industries (AI4I 2020).
2. UCI Machine Learning Repository: *AI4I 2020 Predictive Maintenance Dataset*, DOI: 10.24432/C5HS5C.
3. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
4. Breiman, L. (2001). *Random Forests*. Machine Learning 45(1), pp. 5-32.
