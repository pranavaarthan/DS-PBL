"""
preprocessing.py
----------------
Modular, leak-free feature engineering and preprocessing pipeline.

The actual dataset columns are:
  Applicant_ID, Age, Gender, Education, Employment_Type, Annual_Income,
  Loan_Amount, Loan_Term, Credit_Score, Existing_Loans, Debt_to_Income_Ratio,
  Property_Value, Loan_Purpose, Previous_Default, Loan_Status

Feature engineering performed:
  - TotalIncome          = Annual_Income   (single-applicant dataset; proxy kept for API compatibility)
  - Loan_to_Income_Ratio = Loan_Amount / (Annual_Income + 1)
  - Previous_Default     is binarised: Yes→1, No→0

All transformations are fitted ONLY on training data (leak-free).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# ── Column definitions ────────────────────────────────────────────────────────

DROP_COLS = ["Applicant_ID"]        # identifier – never used as a feature

TARGET_COL = "Loan_Status"

# Raw columns that exist in the CSV
NUMERICAL_COLS = [
    "Age",
    "Annual_Income",
    "Loan_Amount",
    "Loan_Term",
    "Credit_Score",
    "Existing_Loans",
    "Debt_to_Income_Ratio",
    "Property_Value",
    # Engineered
    "Loan_to_Income_Ratio",
]

CATEGORICAL_COLS = [
    "Gender",
    "Education",
    "Employment_Type",
    "Loan_Purpose",
    "Previous_Default",
]

# ── Feature-engineering transformer ──────────────────────────────────────────

class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateless transformer that creates derived features.
    Fitted inside the pipeline so it is safe to use in cross-validation.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "LoanFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Drop identifier columns if still present
        for col in DROP_COLS:
            if col in X.columns:
                X = X.drop(columns=[col])

        # Derived features
        # Loan_to_Income_Ratio: debt burden proxy
        X["Loan_to_Income_Ratio"] = X["Loan_Amount"] / (X["Annual_Income"] + 1)

        return X


# ── Build the sklearn Pipeline ────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Return a ColumnTransformer that:
      - StandardScales all numerical features
      - OneHotEncodes all categorical features (handle_unknown='ignore')
    """
    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    return preprocessor


def build_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Recover feature names after fitting the ColumnTransformer.
    Returns a list of strings matching the order of the transformed columns.
    """
    num_names = NUMERICAL_COLS.copy()
    cat_names = (
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_COLS)
        .tolist()
    )
    return num_names + cat_names


# ── Dataset loading & target encoding ────────────────────────────────────────

def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the CSV, apply target encoding (Approved→1, Rejected→0), and return
    (X, y) where X still contains raw columns (engineering happens in pipeline).
    """
    df = pd.read_csv(data_path)

    # Target encoding
    df[TARGET_COL] = df[TARGET_COL].map({"Approved": 1, "Rejected": 0})
    if df[TARGET_COL].isnull().any():
        # Gracefully handle Y/N format as well
        df[TARGET_COL] = df[TARGET_COL].fillna(
            df[TARGET_COL].map({"Y": 1, "N": 0})
        )

    y = df[TARGET_COL].astype(int)

    # Drop identifier and target
    X = df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")

    return X, y


def get_feature_ranges(data_path: Path) -> dict:
    """
    Return dataset-derived sensible ranges for the Streamlit input widgets.
    """
    df = pd.read_csv(data_path)

    ranges = {
        "Age":                  (int(df["Age"].min()),                 int(df["Age"].max()),                 int(df["Age"].median())),
        "Annual_Income":        (float(df["Annual_Income"].min()),      float(df["Annual_Income"].max()),      float(df["Annual_Income"].median())),
        "Loan_Amount":          (float(df["Loan_Amount"].min()),        float(df["Loan_Amount"].max()),        float(df["Loan_Amount"].median())),
        "Loan_Term":            (int(df["Loan_Term"].min()),            int(df["Loan_Term"].max()),            int(df["Loan_Term"].median())),
        "Credit_Score":         (int(df["Credit_Score"].min()),         int(df["Credit_Score"].max()),         int(df["Credit_Score"].median())),
        "Existing_Loans":       (int(df["Existing_Loans"].min()),       int(df["Existing_Loans"].max()),       int(df["Existing_Loans"].median())),
        "Debt_to_Income_Ratio": (float(df["Debt_to_Income_Ratio"].min()),float(df["Debt_to_Income_Ratio"].max()),float(df["Debt_to_Income_Ratio"].median())),
        "Property_Value":       (float(df["Property_Value"].min()),     float(df["Property_Value"].max()),     float(df["Property_Value"].median())),
        # Categoricals – unique values
        "Gender":               sorted(df["Gender"].dropna().unique().tolist()),
        "Education":            sorted(df["Education"].dropna().unique().tolist()),
        "Employment_Type":      sorted(df["Employment_Type"].dropna().unique().tolist()),
        "Loan_Purpose":         sorted(df["Loan_Purpose"].dropna().unique().tolist()),
        "Previous_Default":     sorted(df["Previous_Default"].dropna().unique().tolist()),
    }
    return ranges
