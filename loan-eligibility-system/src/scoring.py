"""
scoring.py
----------
Eligibility score calculation and risk-tier classification.

IMPORTANT DISCLAIMER:
This is an AI-Based Loan Eligibility Score.
It is NOT an official credit score, a CIBIL score, a calibrated probability
of loan repayment, or a guarantee of loan approval.
"""

from __future__ import annotations

# ── Risk tier boundaries ──────────────────────────────────────────────────────

RISK_TIERS = [
    (80, 100, "Low Risk",         "Highly Eligible",   "#22c55e"),   # green
    (60,  79, "Moderate Risk",    "Eligible",          "#84cc16"),   # lime
    (40,  59, "Medium-High Risk", "Review Required",   "#f59e0b"),   # amber
    (0,   39, "High Risk",        "Low Eligibility",   "#ef4444"),   # red
]


def calculate_eligibility_score(probability_positive: float) -> int:
    """
    Convert model approval probability [0.0–1.0] to an integer score [0–100].

    Parameters
    ----------
    probability_positive : float
        P(Approved) returned by model.predict_proba.

    Returns
    -------
    int
        Score clamped to [0, 100].
    """
    raw_score = probability_positive * 100
    clamped   = max(0.0, min(raw_score, 100.0))
    return int(round(clamped))


def get_risk_category(score: int) -> dict:
    """
    Map an eligibility score [0–100] to a risk tier dictionary.

    Returns
    -------
    dict with keys: risk_level, eligibility_label, color, score_range
    """
    for low, high, risk_level, eligibility_label, color in RISK_TIERS:
        if low <= score <= high:
            return {
                "risk_level":        risk_level,
                "eligibility_label": eligibility_label,
                "color":             color,
                "score_range":       f"{low}–{high}",
            }
    # Fallback (should never be reached for valid scores)
    return {
        "risk_level":        "Unknown",
        "eligibility_label": "Undefined",
        "color":             "#6b7280",
        "score_range":       "N/A",
    }


def format_decision(prediction: int) -> str:
    """Return a human-readable decision label."""
    return "✅ Approved" if prediction == 1 else "❌ Rejected"
