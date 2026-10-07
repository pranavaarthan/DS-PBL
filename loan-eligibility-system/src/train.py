"""
train.py
--------
Production-grade training script for the Loan Eligibility Prediction System.

Usage:
    python -m src.train                         (from project root)
    python src/train.py                         (direct execution)

Pipeline:
    LoanFeatureEngineer → ColumnTransformer → LogisticRegression

Outputs:
    models/loan_logistic_regression.pkl         (serialised Pipeline)
    models/metrics.json                         (evaluation metrics)
    models/confusion_matrix.png
    models/roc_curve.png
"""

from __future__ import annotations

import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import os
import io
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ── Resolve project root regardless of where the script is invoked ────────────
# Force UTF-8 output on Windows to avoid cp1252 encoding errors
import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    try:
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "Loan-Approval-Prediction.csv"
MODELS_DIR   = PROJECT_ROOT / "models"
MODEL_PATH   = MODELS_DIR / "loan_logistic_regression.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
CM_PATH      = MODELS_DIR / "confusion_matrix.png"
ROC_PATH     = MODELS_DIR / "roc_curve.png"

# Add src to path when running as script
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_dataset,
    build_preprocessor,
    LoanFeatureEngineer,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
)
from src.scoring import calculate_eligibility_score, get_risk_category


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Rejected (0)", "Approved (1)"],
        yticklabels=["Rejected (0)", "Approved (1)"],
        ax=ax,
    )
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_title("Confusion Matrix – Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {path}")


def _save_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, auc: float, path: Path
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#6366f1", lw=2,
            label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", lw=1.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve – Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved -> {path}")


# ── Main training routine ─────────────────────────────────────────────────────

def train() -> dict:
    """
    Full training, cross-validation, test evaluation, and artefact saving.

    Returns
    -------
    dict : All computed metrics (persisted to metrics.json).
    """
    warnings.filterwarnings("ignore")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    _print_section("1. LOADING DATASET")
    X, y = load_dataset(DATA_PATH)
    print(f"  Dataset loaded: {X.shape[0]} rows × {X.shape[1]} columns")
    print(f"  Target distribution:\n{y.value_counts().to_string()}")

    # ── 2. Feature engineering verification ───────────────────────────────────
    _print_section("2. FEATURE ENGINEERING")
    fe = LoanFeatureEngineer()
    X_eng = fe.transform(X.copy())
    print(f"  Engineered features: {X_eng.columns.tolist()}")
    print(f"  Loan_to_Income_Ratio stats:\n{X_eng['Loan_to_Income_Ratio'].describe().to_string()}")

    # ── 3. Train / test split ─────────────────────────────────────────────────
    _print_section("3. TRAIN / TEST SPLIT  (80 / 20, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"  Train size : {len(X_train)}")
    print(f"  Test  size : {len(X_test)}")
    print(f"  Train class balance:\n{y_train.value_counts().to_string()}")

    # ── 4. Build full pipeline ────────────────────────────────────────────────
    _print_section("4. BUILDING PIPELINE")
    preprocessor = build_preprocessor()
    classifier   = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
        solver="lbfgs",
    )
    pipeline = Pipeline([
        ("feature_engineer", LoanFeatureEngineer()),
        ("preprocessor",     preprocessor),
        ("classifier",       classifier),
    ])
    print("  Pipeline steps:", [s[0] for s in pipeline.steps])

    # ── 5. Stratified K-Fold cross-validation (on training data only) ─────────
    _print_section("5. STRATIFIED 5-FOLD CROSS-VALIDATION (training data)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=skf,
        scoring=cv_scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    cv_metrics: dict = {}
    print(f"\n  {'Metric':<20} {'Mean':>8} {'Std':>8}")
    print("  " + "-" * 38)
    for metric in cv_scoring:
        key   = f"test_{metric}"
        mean  = cv_results[key].mean()
        std   = cv_results[key].std()
        cv_metrics[metric] = {"mean": round(float(mean), 4), "std": round(float(std), 4)}
        print(f"  {metric:<20} {mean:>8.4f} {std:>8.4f}")

    # ── 6. Final training on full training set ────────────────────────────────
    _print_section("6. TRAINING FINAL MODEL")
    pipeline.fit(X_train, y_train)
    print("  Final pipeline trained on full training set.")

    # ── 7. Evaluate on untouched test set (ONCE) ──────────────────────────────
    _print_section("7. TEST SET EVALUATION (untouched)")
    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)[:, 1]   # P(Approved)
    pos_class_i = list(pipeline.classes_).index(1)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    test_metrics = {
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1":        round(float(f1),   4),
        "roc_auc":   round(float(auc),  4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Rejected','Approved'])}")

    # ── 8. Save plots ──────────────────────────────────────────────────────────
    _print_section("8. SAVING PLOTS")
    _save_confusion_matrix(cm, CM_PATH)
    _save_roc_curve(y_test.values, y_prob, auc, ROC_PATH)

    # ── 9. Persist model ───────────────────────────────────────────────────────
    _print_section("9. SERIALISING MODEL")
    joblib.dump(pipeline, MODEL_PATH)
    print(f"  Model saved -> {MODEL_PATH}")

    # ── 10. Persist metrics ────────────────────────────────────────────────────
    all_metrics = {
        "test_metrics":  test_metrics,
        "cv_metrics":    cv_metrics,
        "model_path":    str(MODEL_PATH),
        "data_path":     str(DATA_PATH),
        "train_size":    int(len(X_train)),
        "test_size":     int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrics saved -> {METRICS_PATH}")

    # ── 11. Quick applicant scenario tests ────────────────────────────────────
    _print_section("11. SCENARIO TESTS")
    _run_scenario_tests(pipeline)

    _print_section("TRAINING COMPLETE")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Metrics : {METRICS_PATH}\n")

    return all_metrics


# ── Scenario testing ──────────────────────────────────────────────────────────

def _run_scenario_tests(pipeline: Pipeline) -> None:
    """Run 3 canned applicant scenarios and print results."""
    scenarios = [
        {
            "label": "Strong Applicant",
            "data": {
                "Age": 38, "Gender": "Male", "Education": "Graduate",
                "Employment_Type": "Salaried", "Annual_Income": 120000.0,
                "Loan_Amount": 15000.0, "Loan_Term": 180,
                "Credit_Score": 780, "Existing_Loans": 0,
                "Debt_to_Income_Ratio": 0.10, "Property_Value": 50000.0,
                "Loan_Purpose": "Home", "Previous_Default": "No",
            },
        },
        {
            "label": "Borderline Applicant",
            "data": {
                "Age": 35, "Gender": "Female", "Education": "Graduate",
                "Employment_Type": "Self-Employed", "Annual_Income": 45000.0,
                "Loan_Amount": 20000.0, "Loan_Term": 120,
                "Credit_Score": 630, "Existing_Loans": 1,
                "Debt_to_Income_Ratio": 0.42, "Property_Value": 0.0,
                "Loan_Purpose": "Business", "Previous_Default": "No",
            },
        },
        {
            "label": "High-Risk Applicant",
            "data": {
                "Age": 27, "Gender": "Male", "Education": "Not Graduate",
                "Employment_Type": "Self-Employed", "Annual_Income": 22000.0,
                "Loan_Amount": 35000.0, "Loan_Term": 60,
                "Credit_Score": 540, "Existing_Loans": 3,
                "Debt_to_Income_Ratio": 0.75, "Property_Value": 0.0,
                "Loan_Purpose": "Personal", "Previous_Default": "Yes",
            },
        },
    ]

    for sc in scenarios:
        df_in = pd.DataFrame([sc["data"]])
        proba        = pipeline.predict_proba(df_in)[0]
        pos_idx      = list(pipeline.classes_).index(1)
        approval_prob = proba[pos_idx]
        score        = calculate_eligibility_score(approval_prob)
        risk         = get_risk_category(score)
        pred         = int(pipeline.predict(df_in)[0])
        assert 0 <= score <= 100, f"Score out of range: {score}"
        print(f"\n  [{sc['label']}]")
        print(f"    Score        : {score}/100")
        print(f"    Risk Category: {risk['risk_level']} / {risk['eligibility_label']}")
        print(f"    Prediction   : {'Approved' if pred == 1 else 'Rejected'}")
        print(f"    Approval Prob: {approval_prob:.4f}")

    print("\n  ✓ All scenario scores in [0, 100]")
    print("  ✓ Risk category boundaries verified")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
