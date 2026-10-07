"""
predict.py
----------
Standalone prediction engine — works independently of Streamlit.

Usage (programmatic):
    from src.predict import predict_loan_eligibility
    result = predict_loan_eligibility(applicant_data)

Usage (CLI):
    python src/predict.py
"""

from __future__ import annotations

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Resolve project root ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "loan_logistic_regression.pkl"

sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring       import calculate_eligibility_score, get_risk_category
from src.explainability import explain_prediction, prettify_feature_name


# ── Model cache (singleton) ───────────────────────────────────────────────────
_PIPELINE = None

def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}.\n"
                f"Run training first:  python src/train.py"
            )
        _PIPELINE = joblib.load(MODEL_PATH)
    return _PIPELINE


# ── Core prediction function ──────────────────────────────────────────────────

def predict_loan_eligibility(applicant_data: dict) -> dict:
    """
    Run the full loan eligibility prediction for a single applicant.

    Parameters
    ----------
    applicant_data : dict
        Keys must match the raw CSV columns (excluding Applicant_ID and
        Loan_Status). Example keys:
          Age, Gender, Education, Employment_Type, Annual_Income,
          Loan_Amount, Loan_Term, Credit_Score, Existing_Loans,
          Debt_to_Income_Ratio, Property_Value, Loan_Purpose, Previous_Default

    Returns
    -------
    dict
        {
          "prediction"           : int (1=Approved, 0=Rejected),
          "eligibility_score"    : int (0–100),
          "risk_category"        : dict (risk_level, eligibility_label, color, score_range),
          "approval_probability" : float,
          "rejection_probability": float,
          "positive_factors"     : list[(readable_name, contribution)],
          "negative_factors"     : list[(readable_name, contribution)],
          "decision"             : str,
        }
    """
    pipeline = _load_pipeline()

    # Convert to DataFrame
    applicant_df = pd.DataFrame([applicant_data])

    # ── Probabilities ──────────────────────────────────────────────────────────
    proba         = pipeline.predict_proba(applicant_df)[0]
    pos_idx       = list(pipeline.classes_).index(1)
    neg_idx       = 1 - pos_idx
    approval_prob = float(proba[pos_idx])
    reject_prob   = float(proba[neg_idx])

    # ── Prediction ─────────────────────────────────────────────────────────────
    prediction = int(pipeline.predict(applicant_df)[0])

    # ── Score and risk tier ────────────────────────────────────────────────────
    score    = calculate_eligibility_score(approval_prob)
    risk     = get_risk_category(score)

    # ── Explainability ─────────────────────────────────────────────────────────
    expl = explain_prediction(pipeline, applicant_df, top_n=5)

    # Prettify factor names
    positive_factors = [
        (prettify_feature_name(name), float(val))
        for name, val in expl["positive_factors"]
    ]
    negative_factors = [
        (prettify_feature_name(name), float(val))
        for name, val in expl["negative_factors"]
    ]

    decision = "✅ Approved" if prediction == 1 else "❌ Rejected"

    return {
        "prediction":            prediction,
        "eligibility_score":     score,
        "risk_category":         risk,
        "approval_probability":  round(approval_prob, 4),
        "rejection_probability": round(reject_prob, 4),
        "positive_factors":      positive_factors,
        "negative_factors":      negative_factors,
        "decision":              decision,
    }


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Loan Eligibility Prediction Engine ===\n")
    demo_applicants = [
        ("Strong Applicant", {
            "Age": 38, "Gender": "Male", "Education": "Graduate",
            "Employment_Type": "Salaried", "Annual_Income": 120000.0,
            "Loan_Amount": 15000.0, "Loan_Term": 180,
            "Credit_Score": 780, "Existing_Loans": 0,
            "Debt_to_Income_Ratio": 0.10, "Property_Value": 50000.0,
            "Loan_Purpose": "Home", "Previous_Default": "No",
        }),
        ("Borderline Applicant", {
            "Age": 35, "Gender": "Female", "Education": "Graduate",
            "Employment_Type": "Self-Employed", "Annual_Income": 45000.0,
            "Loan_Amount": 20000.0, "Loan_Term": 120,
            "Credit_Score": 630, "Existing_Loans": 1,
            "Debt_to_Income_Ratio": 0.42, "Property_Value": 0.0,
            "Loan_Purpose": "Business", "Previous_Default": "No",
        }),
        ("High-Risk Applicant", {
            "Age": 27, "Gender": "Male", "Education": "Not Graduate",
            "Employment_Type": "Self-Employed", "Annual_Income": 22000.0,
            "Loan_Amount": 35000.0, "Loan_Term": 60,
            "Credit_Score": 540, "Existing_Loans": 3,
            "Debt_to_Income_Ratio": 0.75, "Property_Value": 0.0,
            "Loan_Purpose": "Personal", "Previous_Default": "Yes",
        }),
    ]

    for label, data in demo_applicants:
        print(f"{'─'*50}")
        print(f"  {label}")
        print(f"{'─'*50}")
        result = predict_loan_eligibility(data)
        print(f"  Decision          : {result['decision']}")
        print(f"  Eligibility Score : {result['eligibility_score']}/100")
        print(f"  Risk Category     : {result['risk_category']['risk_level']} / {result['risk_category']['eligibility_label']}")
        print(f"  Approval Prob     : {result['approval_probability']:.4f}")
        print(f"  Rejection Prob    : {result['rejection_probability']:.4f}")
        print(f"  Top Positive Factors:")
        for name, val in result["positive_factors"]:
            print(f"    + {name}: {val:+.4f}")
        print(f"  Top Negative Factors:")
        for name, val in result["negative_factors"]:
            print(f"    - {name}: {val:+.4f}")
        print()
