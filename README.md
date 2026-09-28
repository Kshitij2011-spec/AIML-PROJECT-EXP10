# MachineGuard AI — AI-Based Predictive Maintenance of Industrial Machines

> **Educational Prototype Notice:**  
> Educational prototype trained on the AI4I 2020 synthetic predictive-maintenance benchmark. Results are not validated for deployment on real industrial machinery.

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview

**MachineGuard AI** is an academic AI/ML engineering mini-project that predicts whether an industrial milling or cutting machine is at risk of operational failure based on telemetry and operating conditions.

The system is built to demonstrate end-to-end technical rigor:
- **Reproducible Machine Learning**: Strict train/validation/test featurization ordering.
- **Physics-Informed Feature Engineering**: Derived physical parameters modeling thermal dynamics and mechanical strain.
- **Ablation Comparison**: Empirical validation demonstrating how engineered features improve discriminatory power.
- **Threshold Optimization**: Decision threshold tuned on validation data for minority-class recall and F1.
- **Inference Service**: Production-ready FastAPI service pre-loading sklearn pipelines without runtime retraining.

---

## 2. Canonical Dataset

* **Source**: UCI Machine Learning Repository — [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (ID: 601)
* **Nature**: 10,000 synthetic records reflecting real-world industrial machinery failure patterns.
* **Target Variable**: `Machine failure` (binary 0 / 1)
* **Severe Class Imbalance**:
  - Non-failures (0): **9,661** (96.61%)
  - Failures (1): **339** (3.39%)

### Target Leakage Prevention
The raw dataset contains several failure-mode indicators:
- `TWF` (Tool Wear Failure)
- `HDF` (Heat Dissipation Failure)
- `PWF` (Power Failure)
- `OSF` (Overstrain Failure)
- `RNF` (Random Failure)
- Identifiers: `UDI`, `Product ID`

> **Critical Leakage Rule:** These columns describe specific post-failure root causes or arbitrary identifiers. Using them as model inputs constitutes severe target leakage. They are **strictly excluded** from all model inputs.

### Usable Features
- `Type`: Machine quality variant (`L` = Low 50%, `M` = Medium 30%, `H` = High 20%)
- `Air temperature [K]`: Ambient temperature
- `Process temperature [K]`: Cutting operation temperature
- `Rotational speed [rpm]`: Spindle speed
- `Torque [Nm]`: Spindle torque
- `Tool wear [min]`: Cumulative active tool cutting time

---

## 3. Physics-Informed Feature Engineering

To capture known mechanical and thermal failure boundaries without hardcoding arbitrary rules:

1. **`temperature_difference`** ($K$):
   $$\Delta T = T_{\text{process}} - T_{\text{air}}$$
   *Rationale:* Thermal dissipation gradient. Narrow or excessive gradients reflect cooling failure or thermal saturation.

2. **`mechanical_power_kw`** ($kW$):
   $$P_{\text{mech}} = \frac{\tau \times \omega}{1000} = \frac{\text{Torque [Nm]} \times \text{Rotational Speed [rpm]} \times 2\pi}{60000}$$
   *Rationale:* Actual mechanical power delivered to the spindle. Excursions above 9.0 kW or below 3.0 kW correlate with power failure conditions.

3. **`wear_load`** ($min \cdot Nm$):
   $$\text{WearLoad} = \text{Tool wear [min]} \times \text{Torque [Nm]}$$
   *Rationale:* Compound mechanical abrasion index. High cutting resistance applied against a tool near its fatigue limit triggers sudden rupture and overstrain.

---

## 4. Ablation Study & Empirical Results

We conducted an ablation study across linear (`Logistic Regression`) and tree-based (`Random Forest`) models, with and without engineered physical features.

All pipelines were evaluated on the **1,500-sample Validation Set** (stratified 70% Train, 15% Validation, 15% Test):

| Model Configuration | Feature Set | Val Precision | Val Recall | Val F1-Score | Val PR-AUC | Val ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Baseline) | Original Usable | 0.1365 | 0.7255 | 0.2298 | 0.3834 | 0.8749 |
| **Logistic Regression** (Baseline) | Original + Engineered | 0.1745 | 0.8039 | 0.2867 | 0.3562 | 0.9048 |
| **Random Forest** | Original Usable | 0.5738 | 0.6863 | 0.6250 | 0.6639 | 0.9731 |
| **Random Forest** (Selected) | **Original + Engineered** | **0.8077** | **0.8235** | **0.8155** | **0.8479** | **0.9782** |

### Key Ablation Insights
* Adding engineered features produced an **+19.05% absolute increase in F1-score** (0.6250 -> 0.8155) on Random Forest.
* **PR-AUC jumped from 0.6639 to 0.8479 (+18.40%)**.
* False alarms dropped precipitously: Precision rose from 0.5738 to 0.8077 while maintaining high recall (82.35%).

---

## 5. Validation-Guided Threshold Optimization

Because the failure class represents only 3.39% of all data, the default decision threshold of 0.50 was tuned **strictly on the Validation partition** (evaluating thresholds from 0.05 to 0.95):

* **Default Threshold (0.50)**: Val F1 = 0.8155 | Val Recall = 0.8235 | Val Precision = 0.8077
* **Optimal Tuned Threshold (0.56)**: **Val F1 = 0.8367** | Val Recall = 0.8039 | **Val Precision = 0.8723**

---

## 6. Final Evaluation on Untouched Test Set

The selected pipeline was evaluated **exactly once** on the untouched 1,500-sample Test Set:

| Metric | Default Threshold (0.50) | Tuned Threshold (0.56) |
| :--- | :---: | :---: |
| **PR-AUC (Average Precision)** | **0.8905** | **0.8905** |
| **ROC-AUC** | **0.9876** | **0.9876** |
| **F1-Score** | 0.7925 | **0.8235** |
| **Recall (Sensitivity)** | 0.8235 | **0.8235** (42 / 51 failures caught) |
| **Precision** | 0.7636 | **0.8235** (only 9 false alarms out of 1,449) |
| **Accuracy** | 98.53% | **98.80%** |

### Test Confusion Matrix (Tuned Threshold = 0.56)
```
                  Predicted Normal    Predicted Failure
Actual Normal          1440                  9          (Specificity: 99.38%)
Actual Failure            9                 42          (Recall:      82.35%)
```

---

## 7. Feature Importance Analysis

Gini importances from the primary Random Forest classifier validate the physical significance of our feature engineering:

| Rank | Feature | Type | Importance | Physical Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Rotational speed [rpm]` | Telemetry | 20.10% | Primary kinematic cutting parameter |
| 2 | **`mechanical_power_kw`** | **Engineered** | **19.43%** | Workpiece drive load and motor stress |
| 3 | `Torque [Nm]` | Telemetry | 18.14% | Spindle resistance and cutting force |
| 4 | `Tool wear [min]` | Telemetry | 14.12% | Cumulative insert micro-fracture life |
| 5 | **`wear_load`** | **Engineered** | **11.59%** | Compound abrasion load |
| 6 | **`temperature_difference`** | **Engineered** | **9.38%** | Heat dissipation efficiency gradient |
| 7 | `Air temperature [K]` | Telemetry | 4.05% | Ambient baseline |
| 8 | `Process temperature [K]` | Telemetry | 2.45% | Cutting chamber temperature |
| 9 | `Type_M` | Encoded | 0.51% | Medium quality variant |
| 10 | `Type_H` | Encoded | 0.23% | High quality variant |

> Over **40.4%** of the model's total predictive power is directly driven by the three engineered physical features.

---

## 8. Practical Applications, Societal Impact & Limitations

### Practical Applications
- **Condition-Based Maintenance (CBM):** Transition from rigid calendar-based maintenance schedules to dynamic condition-informed inspections.
- **Tool Wear Management:** Alert operators before cutting tool micro-fractures compromise workpiece surface finish or cause insert breakage.
- **Drive Train Overload Mitigation:** Identify abnormal spindle torque-speed pairings before power failures or inverter trips occur.
- **Root Cause Explainability:** Provide maintenance technicians with immediate physical context (e.g. thermal dissipation bottlenecks vs mechanical strain) rather than opaque black-box alerts.

### Societal Impact
- **Resource Conservation:** Extends equipment lifespan and reduces premature scrappage of costly carbide cutting tools.
- **Energy Efficiency:** Preventing machines from running in degraded, high-friction states reduces excessive industrial power consumption.
- **Worker Safety:** Mitigates catastrophic mechanical failures (e.g. violent tool fracturing or motor seizure) that pose physical hazards on shop floors.

### Limitations
- **Synthetic Benchmark:** Trained on the AI4I 2020 synthetic dataset. While mathematically grounded in physical mechanics, synthetic distributions do not account for environmental noise, vibrational harmonics, sensor drift, or lubrication contamination present in real factories.
- **Educational Scope:** This project is an academic prototype. It is **not** validated for deployment in mission-critical industrial control or SCADA systems.
- **No Safety Replacement:** Predictive ML models should never replace hardwired safety interlocks, emergency stop switches, or certified human inspection protocols.

---

## 9. Repository Structure

```
/
├── README.md                          # Project documentation and empirical reports
├── .gitignore                         # Python and Node exclusions
├── .python-version                    # Python runtime pin (3.11.9)
├── render.yaml                        # Deployment configuration for Render web service
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI application and routing
│   │   ├── schemas.py                 # Pydantic v2 schemas and validation
│   │   ├── model_service.py           # Cached inference service & risk diagnostics
│   │   └── feature_engineering.py     # Sklearn-compatible feature transformer
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── download_data.py           # Canonical UCI dataset downloader
│   │   ├── train.py                   # Reproducible training & ablation pipeline
│   │   ├── evaluate.py                # Metric calculations and threshold tuner
│   │   └── verify_ml_integrity.py     # 12-point integrity test
│   ├── data/
│   │   └── raw/
│   │       └── ai4i2020.csv           # Canonical 10,000-record dataset
│   ├── artifacts/
│   │   ├── model.joblib               # Serialized sklearn pipeline
│   │   ├── metadata.json              # Model parameters, threshold, stats
│   │   └── metrics.json               # Ablation & test evaluation metrics
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_health.py             # Health and metadata endpoint tests
│   │   └── test_prediction.py         # Prediction and feature validation tests
│   └── requirements.txt               # Backend dependencies
├── notebooks/
│   └── predictive_maintenance.ipynb   # Comprehensive story-driven research notebook
└── frontend/                          # React + Vite + TypeScript web interface
```

---

## 10. Quickstart & Local Execution

### Backend Setup

```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# or: .venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt

# Download data and train model (if not already trained)
python ml/train.py

# Run tests
pytest tests/ -v

# Start FastAPI server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### API Endpoints
- `GET /health` or `GET /api/health`: Model status and active decision threshold.
- `GET /api/model-info`: Aggregated metadata, test metrics, feature importances, and dataset statistics.
- `GET /metadata`: Detailed model metadata and schema definitions.
- `GET /metrics`: Ablation results, test metrics, ROC and PR curve coordinates.
- `POST /predict` or `POST /api/predict`: Single machine inference with risk diagnostics and recommendations.
- `POST /predict/batch`: Multi-record batch inference.

---

## 11. Frontend Application

The web frontend is built using **React, Vite, TypeScript, and Tailwind CSS**, tailored for an engineering analytics workflow:
- **Clean 2-Column Desktop Layout:** Left column handles input parameters and presets; right column focuses on real-time failure probability, risk categorization, and prescriptive advice.
- **Physics-Informed Visualizations:** Global Random Forest Gini feature importances, live test metrics (Precision, Recall, F1, PR-AUC, ROC-AUC), and stepped architecture flow.
- **Calm, Serious Aesthetics:** Slate neutral palette, accessible contrast ratios, zero neon gradients or decorative fluff.

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```

The frontend will run at `http://localhost:5173`.

### Environment Variables
Configure in `frontend/.env` or deployment platform:
```bash
# Backend API Base URL
VITE_API_URL=http://localhost:8000
```

---

## 12. Testing & Verification

### Backend Tests (pytest)
```bash
# Run 9 backend unit & integration tests
pytest backend/tests -v
```

### End-to-End Tests (Playwright)
```bash
# Run Playwright E2E test suite
npx playwright test
```

### Production Build
```bash
# Build frontend for production
npm --prefix frontend run build
```

---

## 13. Cloud Deployment Guide

### Backend on Render
1. Connect this GitHub repository to Render.
2. The included `render.yaml` automatically sets up the Python web service:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt && python ml/train.py`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Configure environment variable `ALLOWED_ORIGINS` to include your Vercel frontend URL.

### Frontend on Vercel
1. Import the repository in Vercel.
2. Set **Root Directory** to `frontend`.
3. Set **Framework Preset** to `Vite`.
4. Add Environment Variable:
   - `VITE_API_URL`: `https://<your-render-backend-url>.onrender.com`
5. Deploy.

---

## 14. License

This project is licensed under the MIT License — educational and research use only.


