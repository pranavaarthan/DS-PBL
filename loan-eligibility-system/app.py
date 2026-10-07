"""
app.py
------
Production-grade Fintech Loan Eligibility Dashboard.
Powered by Streamlit + Plotly.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

import streamlit as st

warnings.filterwarnings("ignore")

# ── Project root & imports ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH    = PROJECT_ROOT / "data" / "Loan-Approval-Prediction.csv"
MODEL_PATH   = PROJECT_ROOT / "models" / "loan_logistic_regression.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
CM_PATH      = PROJECT_ROOT / "models" / "confusion_matrix.png"
ROC_PATH     = PROJECT_ROOT / "models" / "roc_curve.png"

from src.predict        import predict_loan_eligibility
from src.preprocessing  import get_feature_ranges, NUMERICAL_COLS, CATEGORICAL_COLS
from src.scoring        import RISK_TIERS

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LoanSense AI · Eligibility Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  – premium dark fintech theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #13131f 50%, #0d0d18 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #0d0d20 100%);
    border-right: 1px solid #2d2d4e;
}
section[data-testid="stSidebar"] * {
    color: #c4c9e2 !important;
}
.sidebar-logo {
    text-align: center;
    padding: 1.5rem 0 1rem;
    font-size: 2.2rem;
}
.sidebar-brand {
    text-align: center;
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sidebar-tagline {
    text-align: center;
    font-size: 0.78rem;
    color: #6b7280 !important;
    margin-bottom: 1.5rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #1a1a3e 0%, #16213e 50%, #1a1a3e 100%);
    border: 1px solid #2d2d5e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
}
.page-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
}
.page-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1e1e3a, #1a1a30);
    border: 1px solid #2d2d5e;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #818cf8, #a78bfa, #c084fc);
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #e2e8f0;
    line-height: 1;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-sub {
    font-size: 0.82rem;
    color: #64748b;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #1e1e3a, #16213e);
    border: 1px solid #3730a3;
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 8px 32px rgba(99,102,241,0.2);
}
.result-approved {
    border-color: #16a34a;
    box-shadow: 0 8px 32px rgba(34,197,94,0.2);
}
.result-rejected {
    border-color: #dc2626;
    box-shadow: 0 8px 32px rgba(239,68,68,0.2);
}

/* ── Section headings ── */
.section-heading {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    padding: 0.6rem 0;
    margin: 1.5rem 0 1rem;
    border-bottom: 2px solid #2d2d5e;
}
.section-heading span {
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Factor rows ── */
.factor-positive {
    background: rgba(34,197,94,0.08);
    border-left: 4px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.35rem 0;
    font-size: 0.88rem;
    color: #86efac;
}
.factor-negative {
    background: rgba(239,68,68,0.08);
    border-left: 4px solid #ef4444;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.35rem 0;
    font-size: 0.88rem;
    color: #fca5a5;
}

/* ── Metric pill ── */
.metric-pill {
    display: inline-block;
    background: #1e1e3a;
    border: 1px solid #3730a3;
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.85rem;
    color: #a5b4fc;
    margin: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Disclaimer ── */
.disclaimer-box {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: #fbbf24;
    margin: 1rem 0;
}

/* ── Streamlit overrides ── */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
div[data-testid="metric-container"] {
    background: #1e1e3a;
    border: 1px solid #2d2d5e;
    border-radius: 12px;
    padding: 1rem;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}
div[data-testid="stDataFrame"] {
    background: #1e1e3a;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab-list"] {
    background: #1e1e3a;
    border-radius: 10px;
    padding: 0.3rem;
    gap: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}
.stExpander {
    background: #1e1e3a;
    border: 1px solid #2d2d5e;
    border-radius: 12px;
}
hr { border-color: #2d2d5e !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load the trained pipeline (cached across reruns)."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    """Load saved training/evaluation metrics."""
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    """Load the raw CSV for Data Insights view."""
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df["Loan_Status_Num"] = df["Loan_Status"].map({"Approved": 1, "Rejected": 0})
    return df


@st.cache_data
def get_ranges():
    if not DATA_PATH.exists():
        return {}
    return get_feature_ranges(DATA_PATH)


def kpi_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def section_heading(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-heading">{icon} <span>{title}</span></div>',
        unsafe_allow_html=True,
    )


def plotly_gauge(score: int, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": "AI Loan Eligibility Score", "font": {"size": 18, "color": "#e2e8f0", "family": "Inter"}},
        number={"suffix": "/100", "font": {"size": 42, "color": "#e2e8f0", "family": "JetBrains Mono"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#4b5563",
                "tickfont": {"color": "#94a3b8"},
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e1e3a",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 39],  "color": "rgba(239,68,68,0.15)"},
                {"range": [40, 59], "color": "rgba(245,158,11,0.15)"},
                {"range": [60, 79], "color": "rgba(132,204,22,0.15)"},
                {"range": [80, 100],"color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def factor_bar_chart(positive: list, negative: list) -> go.Figure:
    all_factors = []
    all_values  = []
    all_colors  = []

    for name, val in positive[:5]:
        all_factors.append(name)
        all_values.append(val)
        all_colors.append("#22c55e")

    for name, val in reversed(negative[:5]):
        all_factors.append(name)
        all_values.append(val)
        all_colors.append("#ef4444")

    fig = go.Figure(go.Bar(
        x=all_values,
        y=all_factors,
        orientation="h",
        marker_color=all_colors,
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title={"text": "Feature Contributions (Log-Odds)",
               "font": {"size": 15, "color": "#e2e8f0"}},
        xaxis={"gridcolor": "#2d2d5e", "color": "#94a3b8", "zeroline": True,
               "zerolinecolor": "#4b5563"},
        yaxis={"gridcolor": "#2d2d5e", "color": "#e2e8f0"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18,18,42,1)",
        margin=dict(l=10, r=20, t=50, b=20),
        height=380,
        font={"family": "Inter", "color": "#e2e8f0"},
        bargap=0.3,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏦</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-brand">LoanSense AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Loan Eligibility Intelligence</div>', unsafe_allow_html=True)

    st.markdown("---")

    view = st.radio(
        "Navigation",
        ["🧮 Calculator", "📊 Model Metrics", "🔍 Data Insights", "📋 Audit Log"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    model_loaded = MODEL_PATH.exists()
    if model_loaded:
        st.success("✅ Model loaded", icon="🤖")
    else:
        st.error("⚠️ Model not found.\nRun: `python src/train.py`")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#4b5563; text-align:center;'>
        LoanSense AI v1.0<br>
        Built with Scikit-Learn + Streamlit<br>
        <em>For decision support only.</em>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

if view == "🧮 Calculator":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🧮 Loan Eligibility Calculator</div>
        <div class="page-subtitle">
            Enter applicant details to compute the AI-Based Loan Eligibility Score and Risk Category.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="disclaimer-box">⚠️ <strong>DISCLAIMER:</strong> This tool produces an AI-Based Loan Eligibility Score. It is NOT an official credit score, a CIBIL score, or a guarantee of loan approval. Outputs are model-derived and should be used for decision support only. Human review is recommended for all borderline cases.</div>', unsafe_allow_html=True)

    ranges = get_ranges()

    # ── Input form ─────────────────────────────────────────────────────────────
    with st.form("applicant_form", clear_on_submit=False):
        section_heading("👤", "Personal Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            if "Age" in NUMERICAL_COLS:
                age = st.slider(
                    "Age",
                    min_value=ranges.get("Age", (18, 80, 35))[0],
                    max_value=ranges.get("Age", (18, 80, 35))[1],
                    value=ranges.get("Age", (18, 80, 35))[2],
                    key="age_slider",
                )
            else:
                st.info("ℹ️ Age (Not in model)")
                age = ranges.get("Age", (18, 80, 35))[2]
        with col2:
            if "Gender" in CATEGORICAL_COLS:
                gender = st.selectbox(
                    "Gender",
                    options=ranges.get("Gender", ["Male", "Female"]),
                    key="gender_sel",
                )
            else:
                st.info("ℹ️ Gender (Not in model)")
                gender = ranges.get("Gender", ["Male", "Female"])[0]
        with col3:
            if "Education" in CATEGORICAL_COLS:
                education = st.selectbox(
                    "Education",
                    options=ranges.get("Education", ["Graduate", "Not Graduate"]),
                    key="edu_sel",
                )
            else:
                st.info("ℹ️ Education (Not in model)")
                education = ranges.get("Education", ["Graduate", "Not Graduate"])[0]

        section_heading("💼", "Employment & Income")
        col4, col5, col6 = st.columns(3)
        with col4:
            if "Employment_Type" in CATEGORICAL_COLS:
                employment_type = st.selectbox(
                    "Employment Type",
                    options=ranges.get("Employment_Type", ["Salaried", "Self-Employed"]),
                    key="emp_sel",
                )
            else:
                st.info("ℹ️ Employment Type (Not in model)")
                employment_type = ranges.get("Employment_Type", ["Salaried", "Self-Employed"])[0]
        with col5:
            if "Annual_Income" in NUMERICAL_COLS:
                annual_income = st.number_input(
                    "Annual Income (₹)",
                    min_value=float(ranges.get("Annual_Income", (10000, 500000, 60000))[0]),
                    max_value=float(ranges.get("Annual_Income", (10000, 500000, 60000))[1]),
                    value=float(ranges.get("Annual_Income", (10000, 500000, 60000))[2]),
                    step=1000.0,
                    format="%.2f",
                    key="income_inp",
                )
            else:
                st.info("ℹ️ Annual Income (Not in model)")
                annual_income = float(ranges.get("Annual_Income", (10000, 500000, 60000))[2])
        with col6:
            if "Existing_Loans" in NUMERICAL_COLS:
                existing_loans = st.number_input(
                    "Existing Loans",
                    min_value=int(ranges.get("Existing_Loans", (0, 10, 1))[0]),
                    max_value=int(ranges.get("Existing_Loans", (0, 10, 1))[1]),
                    value=int(ranges.get("Existing_Loans", (0, 10, 1))[2]),
                    step=1,
                    key="exist_loans_inp",
                )
            else:
                st.info("ℹ️ Existing Loans (Not in model)")
                existing_loans = int(ranges.get("Existing_Loans", (0, 10, 1))[2])

        section_heading("💰", "Loan Details")
        col7, col8, col9 = st.columns(3)
        with col7:
            if "Loan_Amount" in NUMERICAL_COLS:
                loan_amount = st.number_input(
                    "Loan Amount (₹)",
                    min_value=float(ranges.get("Loan_Amount", (1000, 100000, 20000))[0]),
                    max_value=float(ranges.get("Loan_Amount", (1000, 100000, 20000))[1]),
                    value=float(ranges.get("Loan_Amount", (1000, 100000, 20000))[2]),
                    step=500.0,
                    format="%.2f",
                    key="loan_amt_inp",
                )
            else:
                st.info("ℹ️ Loan Amount (Not in model)")
                loan_amount = float(ranges.get("Loan_Amount", (1000, 100000, 20000))[2])
        with col8:
            if "Loan_Term" in NUMERICAL_COLS:
                loan_term = st.selectbox(
                    "Loan Term (months)",
                    options=sorted(set([12, 24, 36, 48, 60, 84, 120, 180, 240, 360])),
                    index=4,
                    key="loan_term_sel",
                )
            else:
                st.info("ℹ️ Loan Term (Not in model)")
                loan_term = 60
        with col9:
            if "Loan_Purpose" in CATEGORICAL_COLS:
                loan_purpose = st.selectbox(
                    "Loan Purpose",
                    options=ranges.get("Loan_Purpose", ["Home", "Personal", "Business", "Education", "Vehicle"]),
                    key="purpose_sel",
                )
            else:
                st.info("ℹ️ Loan Purpose (Not in model)")
                loan_purpose = ranges.get("Loan_Purpose", ["Home", "Personal", "Business", "Education", "Vehicle"])[0]

        section_heading("📈", "Credit & Risk Profile")
        col10, col11, col12 = st.columns(3)
        with col10:
            if "Credit_Score" in NUMERICAL_COLS:
                credit_score = st.slider(
                    "Credit Score",
                    min_value=int(ranges.get("Credit_Score", (300, 900, 650))[0]),
                    max_value=int(ranges.get("Credit_Score", (300, 900, 650))[1]),
                    value=int(ranges.get("Credit_Score", (300, 900, 650))[2]),
                    key="credit_slider",
                )
            else:
                st.info("ℹ️ Credit Score (Not in model)")
                credit_score = int(ranges.get("Credit_Score", (300, 900, 650))[2])
        with col11:
            if "Debt_to_Income_Ratio" in NUMERICAL_COLS:
                dti = st.slider(
                    "Debt-to-Income Ratio",
                    min_value=float(ranges.get("Debt_to_Income_Ratio", (0.0, 1.0, 0.3))[0]),
                    max_value=float(ranges.get("Debt_to_Income_Ratio", (0.0, 1.0, 0.3))[1]),
                    value=float(ranges.get("Debt_to_Income_Ratio", (0.0, 1.0, 0.3))[2]),
                    step=0.01,
                    format="%.2f",
                    key="dti_slider",
                )
            else:
                st.info("ℹ️ DTI Ratio (Not in model)")
                dti = float(ranges.get("Debt_to_Income_Ratio", (0.0, 1.0, 0.3))[2])
        with col12:
            if "Property_Value" in NUMERICAL_COLS:
                property_value = st.number_input(
                    "Property Value (₹, 0 if none)",
                    min_value=0.0,
                    max_value=float(ranges.get("Property_Value", (0, 500000, 0))[1]),
                    value=0.0,
                    step=1000.0,
                    format="%.2f",
                    key="prop_val_inp",
                )
            else:
                st.info("ℹ️ Property Value (Not in model)")
                property_value = 0.0

        col13, _ = st.columns([1, 2])
        with col13:
            if "Previous_Default" in CATEGORICAL_COLS:
                previous_default = st.selectbox(
                    "Previous Default",
                    options=ranges.get("Previous_Default", ["No", "Yes"]),
                    key="prev_def_sel",
                )
            else:
                st.info("ℹ️ Previous Default (Not in model)")
                previous_default = "No"

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Calculate Eligibility", use_container_width=True)

    # ── Process submission ────────────────────────────────────────────────────
    if submitted:
        if not MODEL_PATH.exists():
            st.error("❌ Model not found. Train the model first: `python src/train.py`")
        else:
            with st.spinner("Analysing applicant profile…"):
                applicant = {
                    "Age":                  age,
                    "Gender":               gender,
                    "Education":            education,
                    "Employment_Type":      employment_type,
                    "Annual_Income":        annual_income,
                    "Loan_Amount":          loan_amount,
                    "Loan_Term":            loan_term,
                    "Credit_Score":         credit_score,
                    "Existing_Loans":       existing_loans,
                    "Debt_to_Income_Ratio": dti,
                    "Property_Value":       property_value,
                    "Loan_Purpose":         loan_purpose,
                    "Previous_Default":     previous_default,
                }
                result = predict_loan_eligibility(applicant)
                st.session_state.last_result = result

                # ── Audit log entry ────────────────────────────────────────────
                st.session_state.audit_log.append({
                    "Timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Age":                  age,
                    "Gender":               gender,
                    "Education":            education,
                    "Employment_Type":      employment_type,
                    "Annual_Income":        annual_income,
                    "Loan_Amount":          loan_amount,
                    "Loan_Term":            loan_term,
                    "Credit_Score":         credit_score,
                    "Existing_Loans":       existing_loans,
                    "Debt_to_Income_Ratio": dti,
                    "Loan_Purpose":         loan_purpose,
                    "Previous_Default":     previous_default,
                    "Eligibility_Score":    result["eligibility_score"],
                    "Risk_Category":        result["risk_category"]["risk_level"],
                    "Prediction":           "Approved" if result["prediction"] == 1 else "Rejected",
                    "Approval_Probability": result["approval_probability"],
                })

    # ── Results display ───────────────────────────────────────────────────────
    if st.session_state.last_result:
        result = st.session_state.last_result
        score  = result["eligibility_score"]
        risk   = result["risk_category"]
        pred   = result["prediction"]
        color  = risk["color"]

        st.markdown("---")
        section_heading("🎯", "Eligibility Results")

        # KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card(
                "Eligibility Score",
                f"{score}/100",
                f"Range: {risk['score_range']}",
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card(
                "Risk Level",
                risk["risk_level"],
                risk["eligibility_label"],
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card(
                "Approval Probability",
                f"{result['approval_probability']*100:.1f}%",
                f"Rejection: {result['rejection_probability']*100:.1f}%",
            ), unsafe_allow_html=True)
        with c4:
            decision_val = "✅ APPROVED" if pred == 1 else "❌ REJECTED"
            st.markdown(kpi_card(
                "AI Decision",
                decision_val,
                "Model-derived – review required",
            ), unsafe_allow_html=True)

        # Gauge + Factors
        g_col, f_col = st.columns([1, 1])
        with g_col:
            st.plotly_chart(
                plotly_gauge(score, color),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with f_col:
            st.plotly_chart(
                factor_bar_chart(result["positive_factors"], result["negative_factors"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # Positive / Negative factor tables
        fa_col, fb_col = st.columns(2)
        with fa_col:
            section_heading("✅", "Top Positive Factors")
            for name, val in result["positive_factors"]:
                st.markdown(
                    f'<div class="factor-positive">⬆ <strong>{name}</strong> &nbsp;·&nbsp; {val:+.4f}</div>',
                    unsafe_allow_html=True,
                )
        with fb_col:
            section_heading("⚠️", "Top Negative Factors")
            for name, val in result["negative_factors"]:
                st.markdown(
                    f'<div class="factor-negative">⬇ <strong>{name}</strong> &nbsp;·&nbsp; {val:+.4f}</div>',
                    unsafe_allow_html=True,
                )

        # Risk tier reference
        with st.expander("📖 Risk Tier Reference Guide"):
            tier_data = {
                "Score Range": ["80–100", "60–79", "40–59", "0–39"],
                "Risk Level":  ["Low Risk", "Moderate Risk", "Medium-High Risk", "High Risk"],
                "Eligibility": ["Highly Eligible", "Eligible", "Review Required", "Low Eligibility"],
                "Colour":      ["🟢", "🟡", "🟠", "🔴"],
            }
            st.dataframe(
                pd.DataFrame(tier_data),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════

elif view == "📊 Model Metrics":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📊 Model Performance Metrics</div>
        <div class="page-subtitle">
            Logistic Regression · Stratified K-Fold CV · Held-Out Test Set Evaluation
        </div>
    </div>
    """, unsafe_allow_html=True)

    metrics = load_metrics()

    if metrics is None:
        st.warning("⚠️ No metrics found. Run training first: `python src/train.py`")
    else:
        tm = metrics["test_metrics"]
        cv = metrics["cv_metrics"]

        # ── Test metrics KPI row ───────────────────────────────────────────────
        section_heading("🎯", "Test Set Performance")
        cols = st.columns(5)
        metric_map = {
            "Accuracy":  ("accuracy",  "Overall correct predictions"),
            "Precision": ("precision", "Approved predictions correct"),
            "Recall":    ("recall",    "True approvals captured"),
            "F1 Score":  ("f1",        "Precision–Recall balance"),
            "ROC-AUC":   ("roc_auc",   "Discrimination ability"),
        }
        for col, (label, (key, sub)) in zip(cols, metric_map.items()):
            with col:
                st.metric(
                    label=label,
                    value=f"{tm[key]:.4f}",
                    help=sub,
                )

        st.markdown(f"""
        <div style='margin:1rem 0; color:#94a3b8; font-size:0.85rem;'>
        📦 Trained on <strong style='color:#a5b4fc;'>{metrics['train_size']}</strong> samples &nbsp;|&nbsp;
        🧪 Tested on <strong style='color:#a5b4fc;'>{metrics['test_size']}</strong> samples &nbsp;|&nbsp;
        📐 Feature count: <strong style='color:#a5b4fc;'>{metrics['feature_count']}</strong> raw features
        </div>
        """, unsafe_allow_html=True)

        # ── CV metrics table ───────────────────────────────────────────────────
        section_heading("🔄", "5-Fold Cross-Validation Results (Training Data)")
        cv_df = pd.DataFrame({
            "Metric":       list(cv.keys()),
            "Mean":         [f"{v['mean']:.4f}" for v in cv.values()],
            "Std Dev":      [f"{v['std']:.4f}"  for v in cv.values()],
        })
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

        # ── Confusion matrix + ROC ─────────────────────────────────────────────
        section_heading("📉", "Confusion Matrix & ROC Curve")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            if CM_PATH.exists():
                st.image(str(CM_PATH), use_container_width=True,
                         caption="Confusion Matrix – Test Set")
            else:
                st.info("Confusion matrix image not found. Re-run training.")
        with img_col2:
            if ROC_PATH.exists():
                st.image(str(ROC_PATH), use_container_width=True,
                         caption="ROC Curve – Test Set")
            else:
                st.info("ROC curve image not found. Re-run training.")

        # ── Confusion matrix interactive ───────────────────────────────────────
        with st.expander("🔢 Interactive Confusion Matrix"):
            cm = np.array(tm["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels={"x": "Predicted", "y": "Actual"},
                x=["Rejected (0)", "Approved (1)"],
                y=["Rejected (0)", "Approved (1)"],
                color_continuous_scale="Blues",
                title="Confusion Matrix",
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e1e3a",
                font={"color": "#e2e8f0", "family": "Inter"},
                title_font_size=16,
            )
            st.plotly_chart(fig_cm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif view == "🔍 Data Insights":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🔍 Data Insights</div>
        <div class="page-subtitle">
            Exploratory analysis of the Loan Approval dataset
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.error("Dataset not found.")
    else:
        total        = len(df)
        approved_n   = int((df["Loan_Status"] == "Approved").sum())
        rejected_n   = total - approved_n
        approval_pct = approved_n / total * 100
        avg_income   = df["Annual_Income"].mean()
        avg_credit   = df["Credit_Score"].mean()
        avg_loan     = df["Loan_Amount"].mean()

        # ── Summary KPIs ───────────────────────────────────────────────────────
        section_heading("📊", "Dataset Summary")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(kpi_card("Total Applicants", f"{total:,}", "records in dataset"), unsafe_allow_html=True)
        with k2:
            st.markdown(kpi_card("Approval Rate", f"{approval_pct:.1f}%", f"{approved_n:,} approved"), unsafe_allow_html=True)
        with k3:
            st.markdown(kpi_card("Rejection Rate", f"{100-approval_pct:.1f}%", f"{rejected_n:,} rejected"), unsafe_allow_html=True)
        with k4:
            st.markdown(kpi_card("Avg Annual Income", f"₹{avg_income:,.0f}", "across all applicants"), unsafe_allow_html=True)
        with k5:
            st.markdown(kpi_card("Avg Credit Score", f"{avg_credit:.0f}", "credit bureau score"), unsafe_allow_html=True)
        with k6:
            st.markdown(kpi_card("Avg Loan Amount", f"₹{avg_loan:,.0f}", "requested amount"), unsafe_allow_html=True)

        st.markdown("---")

        # ── Charts row 1 ──────────────────────────────────────────────────────
        c1, c2 = st.columns(2)

        with c1:
            # Approval distribution donut
            fig_donut = go.Figure(go.Pie(
                labels=["Approved", "Rejected"],
                values=[approved_n, rejected_n],
                hole=0.6,
                marker=dict(colors=["#22c55e", "#ef4444"]),
                textinfo="label+percent",
                textfont=dict(size=13, color="#e2e8f0"),
            ))
            fig_donut.update_layout(
                title={"text": "Loan Approval Distribution", "font": {"size": 16, "color": "#e2e8f0"}},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#94a3b8")),
                showlegend=True,
                height=350,
                annotations=[dict(text=f"{approval_pct:.0f}%", x=0.5, y=0.5,
                                  font_size=28, showarrow=False,
                                  font=dict(color="#e2e8f0", family="JetBrains Mono"))],
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        with c2:
            # Credit Score distribution by status
            fig_hist = px.histogram(
                df, x="Credit_Score", color="Loan_Status",
                nbins=40, barmode="overlay",
                color_discrete_map={"Approved": "#818cf8", "Rejected": "#ef4444"},
                opacity=0.75,
                title="Credit Score Distribution by Loan Status",
                labels={"Credit_Score": "Credit Score", "count": "Count"},
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e1e3a",
                font={"color": "#e2e8f0", "family": "Inter"},
                title_font_size=16,
                legend=dict(title="Status", font=dict(color="#94a3b8")),
                xaxis=dict(gridcolor="#2d2d5e"),
                yaxis=dict(gridcolor="#2d2d5e"),
                height=350,
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

        # ── Charts row 2 ──────────────────────────────────────────────────────
        c3, c4 = st.columns(2)

        with c3:
            # Income vs Loan Amount scatter
            sample = df.sample(min(600, len(df)), random_state=42)
            fig_sc = px.scatter(
                sample, x="Annual_Income", y="Loan_Amount",
                color="Loan_Status",
                color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
                opacity=0.6, size_max=8,
                title="Annual Income vs Loan Amount",
                labels={"Annual_Income": "Annual Income (₹)", "Loan_Amount": "Loan Amount (₹)"},
            )
            fig_sc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e1e3a",
                font={"color": "#e2e8f0", "family": "Inter"},
                title_font_size=16,
                xaxis=dict(gridcolor="#2d2d5e"),
                yaxis=dict(gridcolor="#2d2d5e"),
                legend=dict(font=dict(color="#94a3b8")),
                height=350,
            )
            st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

        with c4:
            # Approval rate by employment type
            emp_rates = (
                df.groupby("Employment_Type")["Loan_Status_Num"]
                .mean().reset_index()
                .rename(columns={"Loan_Status_Num": "Approval_Rate"})
            )
            emp_rates["Approval_Rate"] = emp_rates["Approval_Rate"] * 100
            fig_emp = px.bar(
                emp_rates, x="Employment_Type", y="Approval_Rate",
                color="Approval_Rate",
                color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                range_color=[0, 100],
                title="Approval Rate by Employment Type",
                labels={"Approval_Rate": "Approval Rate (%)"},
                text="Approval_Rate",
            )
            fig_emp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_emp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e1e3a",
                font={"color": "#e2e8f0", "family": "Inter"},
                title_font_size=16,
                xaxis=dict(gridcolor="#2d2d5e"),
                yaxis=dict(gridcolor="#2d2d5e"),
                coloraxis_showscale=False,
                height=350,
            )
            st.plotly_chart(fig_emp, use_container_width=True, config={"displayModeBar": False})

        # ── Charts row 3 ──────────────────────────────────────────────────────
        c5, c6 = st.columns(2)

        with c5:
            # Loan purpose breakdown
            purpose_cnt = df["Loan_Purpose"].value_counts().reset_index()
            purpose_cnt.columns = ["Loan_Purpose", "Count"]
            fig_pur = px.bar(
                purpose_cnt, x="Count", y="Loan_Purpose", orientation="h",
                color="Count", color_continuous_scale="Purples",
                title="Applications by Loan Purpose",
            )
            fig_pur.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e1e3a",
                font={"color": "#e2e8f0", "family": "Inter"},
                title_font_size=16,
                xaxis=dict(gridcolor="#2d2d5e"),
                yaxis=dict(gridcolor="#2d2d5e"),
                coloraxis_showscale=False,
                height=350,
            )
            st.plotly_chart(fig_pur, use_container_width=True, config={"displayModeBar": False})

        with c6:
            # DTI distribution
            fig_dti = px.box(
                df, x="Loan_Status", y="Debt_to_Income_Ratio",
                color="Loan_Status",
                color_discrete_map={"Approved": "#818cf8", "Rejected": "#f472b6"},
                title="Debt-to-Income Ratio by Loan Status",
                points="outliers",
            )
            fig_dti.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e1e3a",
                font={"color": "#e2e8f0", "family": "Inter"},
                title_font_size=16,
                xaxis=dict(gridcolor="#2d2d5e"),
                yaxis=dict(gridcolor="#2d2d5e"),
                legend=dict(font=dict(color="#94a3b8")),
                height=350,
            )
            st.plotly_chart(fig_dti, use_container_width=True, config={"displayModeBar": False})

        # ── Raw data table ─────────────────────────────────────────────────────
        with st.expander("🗂 Browse Raw Dataset"):
            st.dataframe(
                df.drop(columns=["Loan_Status_Num"], errors="ignore").head(200),
                use_container_width=True,
                height=350,
            )


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════

elif view == "📋 Audit Log":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📋 Prediction Audit Log</div>
        <div class="page-subtitle">
            Complete record of all predictions made in this session.
            No sensitive data is persisted beyond this browser session.
        </div>
    </div>
    """, unsafe_allow_html=True)

    log = st.session_state.audit_log

    if not log:
        st.info("No predictions yet. Use the 🧮 Calculator to run your first prediction.")
    else:
        log_df = pd.DataFrame(log)

        # ── Summary KPIs ───────────────────────────────────────────────────────
        n_pred      = len(log_df)
        n_approved  = int((log_df["Prediction"] == "Approved").sum())
        n_rejected  = n_pred - n_approved
        avg_score   = log_df["Eligibility_Score"].mean()

        section_heading("📊", "Session Summary")
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            st.markdown(kpi_card("Total Queries", str(n_pred), "this session"), unsafe_allow_html=True)
        with kc2:
            st.markdown(kpi_card("Approved", str(n_approved), f"{n_approved/n_pred*100:.0f}%"), unsafe_allow_html=True)
        with kc3:
            st.markdown(kpi_card("Rejected", str(n_rejected), f"{n_rejected/n_pred*100:.0f}%"), unsafe_allow_html=True)
        with kc4:
            st.markdown(kpi_card("Avg Score", f"{avg_score:.1f}/100", "session average"), unsafe_allow_html=True)

        # ── Score distribution (session) ────────────────────────────────────────
        if n_pred >= 2:
            section_heading("📈", "Score Distribution (Session)")
            fig_hist_s = px.histogram(
                log_df, x="Eligibility_Score", nbins=20,
                color="Prediction",
                color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
                barmode="overlay", opacity=0.8,
                title="Eligibility Score Distribution",
            )
            fig_hist_s.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e1e3a",
                font={"color": "#e2e8f0", "family": "Inter"},
                title_font_size=14,
                xaxis=dict(gridcolor="#2d2d5e"),
                yaxis=dict(gridcolor="#2d2d5e"),
                legend=dict(font=dict(color="#94a3b8")),
                height=280,
            )
            st.plotly_chart(fig_hist_s, use_container_width=True, config={"displayModeBar": False})

        # ── Full table ─────────────────────────────────────────────────────────
        section_heading("🗃", "Prediction History")
        display_cols = [
            "Timestamp", "Age", "Gender", "Annual_Income", "Loan_Amount",
            "Credit_Score", "Eligibility_Score", "Risk_Category",
            "Prediction", "Approval_Probability",
        ]
        st.dataframe(
            log_df[display_cols],
            use_container_width=True,
            height=350,
            hide_index=True,
        )

        # ── CSV download ───────────────────────────────────────────────────────
        csv_data = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Audit Log (CSV)",
            data=csv_data,
            file_name=f"loan_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ── Disclaimer ─────────────────────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer-box">
        ⚠️ <strong>Data Privacy Notice:</strong> Audit log records are stored in-session memory only
        and are cleared when the Streamlit app is restarted. Do not include PII in the applicant data
        unless your deployment is compliant with applicable data-protection regulations.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSIBLE AI FOOTER (visible on all pages)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
with st.expander("⚖️ Responsible AI & Compliance Notice"):
    st.markdown("""
    ### Responsible AI Statement

    **LoanSense AI** is a decision-support tool built on historical data and a Logistic Regression model.
    The following principles govern its use:

    | Principle | Guidance |
    |-----------|----------|
    | **Decision Support** | The model is an analytical aid, NOT an autonomous lender. Final lending decisions must involve human review. |
    | **Score Transparency** | The Eligibility Score is derived from model probabilities. It is NOT a CIBIL score or any official credit rating. |
    | **Bias Awareness** | The model may inherit biases present in historical training data. Approval patterns should be monitored for demographic disparities. |
    | **Borderline Cases** | Human review is strongly recommended for scores in the 40–60 range. |
    | **Non-Discrimination** | Protected characteristics (gender, age, etc.) should never be used as the primary basis for loan rejection. |
    | **Model Monitoring** | Model performance should be re-evaluated periodically as new data becomes available. Concept drift may degrade accuracy. |
    | **No Guarantee** | A high eligibility score does not guarantee loan approval. Final credit decisions involve additional factors not captured in this model. |
    | **Data Privacy** | Do not expose applicant PII in shared or public environments without proper data governance controls. |

    *This system is intended for educational and decision-support purposes only.*
    """)
