# 🏦 LoanSense AI — Loan Eligibility Prediction & Risk Scoring System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **production-grade, modular, explainable** Loan Eligibility Prediction System powered by Scikit-Learn's Logistic Regression with an interactive fintech analytics dashboard built in Streamlit.

---

## 📌 Project Overview

**LoanSense AI** is an end-to-end machine learning system that:

1. Ingests a real-world loan application dataset
2. Engineers predictive features (debt-burden proxy, etc.)
3. Trains a leak-free Logistic Regression pipeline with stratified cross-validation
4. Scores applicants on a **0–100 Eligibility Scale** mapped to 4 risk tiers
5. Provides **per-applicant explainability** using LR coefficient contributions
6. Exposes a professional **Streamlit fintech dashboard** with Calculator, Metrics, Insights, and Audit Log views

---

## 🎯 Problem Statement

Financial institutions must assess the creditworthiness of loan applicants quickly and consistently. Manual assessment is slow, subjective, and error-prone. This system provides a **data-driven decision-support tool** that:

- Standardises eligibility assessment
- Quantifies risk on an interpretable scale
- Explains *why* a score was assigned (not just what it is)
- Maintains a full audit trail of predictions

---

## 🏆 Objectives

| # | Objective |
|---|-----------|
| 1 | Build a reproducible, leak-free ML pipeline |
| 2 | Achieve strong discrimination (ROC-AUC ≥ 0.70) |
| 3 | Score applicants 0–100 (Eligibility Score) |
| 4 | Categorise risk into 4 actionable tiers |
| 5 | Explain per-applicant predictions |
| 6 | Deliver a professional dashboard |

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| File | `data/Loan-Approval-Prediction.csv` |
| Rows | ~3,050 applicants |
| Target | `Loan_Status` (Approved / Rejected) |
| Source | Synthetic financial dataset |

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Numerical | Applicant age |
| `Annual_Income` | Numerical | Annual gross income |
| `Loan_Amount` | Numerical | Requested loan amount |
| `Loan_Term` | Numerical | Loan tenure in months |
| `Credit_Score` | Numerical | Bureau credit score |
| `Existing_Loans` | Numerical | Number of active loans |
| `Debt_to_Income_Ratio` | Numerical | Monthly debt / income ratio |
| `Property_Value` | Numerical | Collateral property value |
| `Gender` | Categorical | Male / Female |
| `Education` | Categorical | Graduate / Not Graduate |
| `Employment_Type` | Categorical | Salaried / Self-Employed |
| `Loan_Purpose` | Categorical | Home / Personal / Business / Education / Vehicle |
| `Previous_Default` | Categorical | Yes / No |

**Dropped:** `Applicant_ID` — identifier, never used as a predictive feature.

---

## ⚙️ Feature Engineering

| Engineered Feature | Formula | Purpose |
|-------------------|---------|---------|
| `Loan_to_Income_Ratio` | `Loan_Amount / (Annual_Income + 1)` | Debt burden proxy; analogous to standard LTI ratio used by lenders |

> **Note:** `Loan_to_Income_Ratio` is a debt-burden proxy. The dataset does not contain actual monthly debt obligations, so this ratio approximates loan affordability relative to income.

All feature engineering is performed **inside the sklearn Pipeline** to prevent data leakage.

---

## 🤖 ML Methodology

### Pipeline Architecture

```
Raw CSV
  │
  ├─ [LoanFeatureEngineer]       stateless transformer: creates derived features
  │
  ├─ [ColumnTransformer]
  │     ├─ StandardScaler         → numerical features
  │     └─ OneHotEncoder          → categorical features (handle_unknown='ignore')
  │
  └─ [LogisticRegression]        class_weight='balanced', max_iter=1000
```

### Train/Test Split

```python
train_test_split(test_size=0.20, random_state=42, stratify=y)
```

- 80% training / 20% test
- Stratified to preserve class proportions
- **No preprocessing is applied to the full dataset before splitting** — complete leak prevention

### Cross-Validation

- Strategy: **Stratified 5-Fold** on training data only
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Reports mean ± standard deviation for each metric

---

## 📊 Logistic Regression — Why This Model?

| Property | Benefit |
|----------|---------|
| Interpretability | Coefficients directly quantify feature impact |
| Calibrated probabilities | `predict_proba` outputs usable as eligibility scores |
| `class_weight='balanced'` | Handles class imbalance without oversampling |
| Regularisation (L2) | Prevents overfitting |
| Low latency | Sub-millisecond inference suitable for real-time APIs |

---

## 🎯 Loan Eligibility Score Methodology

```python
probability_positive = model.predict_proba(X)[positive_class_index]

eligibility_score = int(
    round(max(0, min(probability_positive * 100, 100)))
)
```

The score is the model's approval probability scaled to **[0, 100]**.

> ⚠️ **IMPORTANT:** This is an **AI-Based Loan Eligibility Score**. It is:
> - ❌ NOT an official credit score
> - ❌ NOT a CIBIL score
> - ❌ NOT a calibrated probability of loan repayment
> - ❌ NOT a guarantee of loan approval

---

## 🏷️ Risk Categories

| Score | Risk Level | Eligibility |
|-------|-----------|-------------|
| 80–100 | 🟢 Low Risk | Highly Eligible |
| 60–79 | 🟡 Moderate Risk | Eligible |
| 40–59 | 🟠 Medium-High Risk | Review Required |
| 0–39 | 🔴 High Risk | Low Eligibility |

---

## 🧠 Explainability Methodology

```
contribution_i ≈ transformed_feature_value_i × logistic_regression_coefficient_i
```

For each applicant:
1. Transform raw input through the fitted preprocessing pipeline
2. Multiply each transformed feature value by its LR coefficient
3. Positive contributions → push toward Approved
4. Negative contributions → push toward Rejected
5. Rank and display top-5 positive and top-5 negative factors

Feature names are **dynamically recovered** from the `ColumnTransformer` — nothing is hard-coded.

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct classifications |
| **Precision** | Of predicted approvals, fraction that are truly approved |
| **Recall** | Of actual approvals, fraction correctly identified |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Model's ability to discriminate between classes |
| **Confusion Matrix** | TP / FP / TN / FN breakdown |
| **Classification Report** | Per-class precision, recall, F1 |

---

## 🖥️ Dashboard Features

### 🧮 Calculator
- Full applicant input form with dataset-derived ranges
- KPI cards: Score, Risk Level, Probability, Decision
- Plotly gauge meter (0–100)
- Feature contribution bar chart (positive vs negative)
- Risk tier reference table

### 📊 Model Metrics
- Test set: Accuracy, Precision, Recall, F1, ROC-AUC
- 5-Fold CV results table (mean ± std)
- Confusion matrix (static PNG + interactive Plotly)
- ROC curve (static PNG)

### 🔍 Data Insights
- Summary KPIs: total applicants, approval/rejection rates, averages
- Approval distribution donut chart
- Credit score histogram by status
- Income vs Loan Amount scatter plot
- Approval rate by employment type
- DTI box plot by status
- Raw data browser

### 📋 Audit Log
- Full prediction history for the session
- Score distribution chart
- Complete exportable CSV download
- Privacy notice

---

## 🏗️ Architecture

```
loan-eligibility-system/
│
├── data/
│   └── Loan-Approval-Prediction.csv       ← raw dataset
│
├── models/
│   ├── loan_logistic_regression.pkl       ← trained pipeline (joblib)
│   ├── metrics.json                       ← evaluation metrics
│   ├── confusion_matrix.png               ← saved plot
│   └── roc_curve.png                      ← saved plot
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   ← feature engineering, ColumnTransformer, data loading
│   ├── scoring.py         ← eligibility score formula, risk tier logic
│   ├── explainability.py  ← LR coefficient-based local explanations
│   ├── train.py           ← full training pipeline + CLI entry point
│   └── predict.py         ← prediction engine (Streamlit-independent)
│
├── app.py                 ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- `uv` package manager (recommended) **or** `pip`

### Option A — uv (recommended)

```bash
# Clone / navigate to the project directory
cd loan-eligibility-system

# Create virtual environment
uv venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Option B — pip

```bash
cd loan-eligibility-system
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Execution

### Step 1 — Train the model

```bash
# From the loan-eligibility-system directory
python src/train.py
```

This will:
- Load and validate the dataset
- Engineer features
- Split data (80/20 stratified)
- Run 5-fold cross-validation
- Train final model on full training set
- Evaluate on held-out test set
- Save `models/loan_logistic_regression.pkl`
- Save `models/metrics.json`, confusion matrix and ROC curve plots
- Run 3 scenario tests (strong / borderline / high-risk)

### Step 2 — Run the dashboard

```bash
streamlit run app.py
```

Open: **http://localhost:8501**

### Step 3 — CLI prediction demo

```bash
python src/predict.py
```

---

## ⚖️ Responsible AI

| Principle | Details |
|-----------|---------|
| **Decision Support** | This tool is an analytical aid only. All final lending decisions require human oversight. |
| **Score Transparency** | The Eligibility Score is derived from model probabilities, not an official credit metric. |
| **Bias Awareness** | Historical training data may contain systemic biases. Approval patterns should be audited for demographic fairness. |
| **Borderline Cases** | Human review strongly recommended for scores 40–60. |
| **Non-Discrimination** | Protected characteristics should never be the primary basis for loan rejection. |
| **Model Monitoring** | Re-evaluate model performance periodically. Concept drift may degrade accuracy over time. |
| **No Guarantee** | A high score does not guarantee loan approval. |

---

## ⚠️ Limitations

1. **Single-applicant dataset** — no joint applicant income (CoapplicantIncome) in this dataset
2. **Static model** — trained on a fixed snapshot; performance degrades as lending patterns shift
3. **Logistic Regression** — assumes linear decision boundary; may underfit complex interactions
4. **No real-time data** — credit scores and income are self-reported/historical
5. **Synthetic dataset** — real-world performance may differ
6. **Class balance** — balanced class weights address imbalance but may over-approve borderline cases

---

## 🔮 Future Improvements

| Enhancement | Description |
|-------------|-------------|
| **Ensemble Models** | XGBoost / LightGBM for non-linear capture |
| **SHAP Values** | Rigorous Shapley-value explainability |
| **Calibration** | Platt scaling / isotonic regression for better probability calibration |
| **API Layer** | FastAPI REST endpoint for integration |
| **MLflow Tracking** | Experiment tracking and model registry |
| **Fairness Metrics** | Demographic parity, equalized odds analysis |
| **Drift Detection** | Population Stability Index (PSI) monitoring |
| **Feature Store** | Centralised feature versioning |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ as a production-grade Data Science PBL project.*
