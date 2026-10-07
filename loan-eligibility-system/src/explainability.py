"""
explainability.py
-----------------
Local feature-contribution explainability using Logistic Regression coefficients.

Method:
  contribution_i ≈ transformed_feature_value_i × coefficient_i

This is a first-order linear approximation. It reflects the direction and
magnitude each feature pushes the log-odds toward approval.

Feature names are dynamically recovered from the fitted pipeline —
nothing is hard-coded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def get_feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """
    Recover ordered feature names from the fitted pipeline's preprocessor
    (ColumnTransformer step named 'preprocessor').

    Returns a flat list of feature names matching the transformed array columns.
    """
    preprocessor = pipeline.named_steps["preprocessor"]

    # transformers_ is a list of (name, fitted_transformer, columns) tuples
    transformer_dict = {name: (estimator, cols)
                        for name, estimator, cols in preprocessor.transformers_}

    # Numerical features (StandardScaler)
    num_estimator, num_cols = transformer_dict["num"]
    numerical_names: list[str] = list(num_cols)

    # Categorical features (OneHotEncoder)
    cat_estimator, cat_cols = transformer_dict["cat"]
    categorical_names: list[str] = (
        cat_estimator.get_feature_names_out(cat_cols).tolist()
    )

    return numerical_names + categorical_names


def explain_prediction(
    pipeline: Pipeline,
    applicant_df: pd.DataFrame,
    top_n: int = 5,
) -> dict:
    """
    Generate local feature contributions for a single applicant.

    Parameters
    ----------
    pipeline      : Fitted sklearn Pipeline with steps 'feature_engineer',
                    'preprocessor', 'classifier'.
    applicant_df  : Single-row DataFrame with raw applicant features.
    top_n         : Number of top positive/negative factors to return.

    Returns
    -------
    dict with keys:
        'positive_factors' : list of (feature_name, contribution) tuples
        'negative_factors' : list of (feature_name, contribution) tuples
        'all_contributions': pd.Series indexed by feature name
    """
    # 1. Apply feature engineering step
    fe_step = pipeline.named_steps["feature_engineer"]
    X_engineered = fe_step.transform(applicant_df)

    # 2. Apply preprocessor to get the numeric array
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_engineered)   # shape (1, n_features)

    # 3. Retrieve coefficients (binary classification: first row)
    classifier = pipeline.named_steps["classifier"]
    coef = classifier.coef_[0]   # shape (n_features,)

    # 4. Compute per-feature contributions
    feature_values = X_transformed[0]          # shape (n_features,)
    contributions  = feature_values * coef     # element-wise

    # 5. Build named series
    feature_names = get_feature_names_from_pipeline(pipeline)
    contrib_series = pd.Series(contributions, index=feature_names)

    # 6. Separate positive and negative
    positive = (
        contrib_series[contrib_series > 0]
        .sort_values(ascending=False)
        .head(top_n)
    )
    negative = (
        contrib_series[contrib_series < 0]
        .sort_values(ascending=True)
        .head(top_n)
    )

    return {
        "positive_factors":  list(zip(positive.index, positive.values)),
        "negative_factors":  list(zip(negative.index, negative.values)),
        "all_contributions": contrib_series,
    }


def prettify_feature_name(raw_name: str) -> str:
    """
    Convert a sklearn-style OHE feature name like 'Gender_Male' or a
    plain numerical name like 'Credit_Score' to a human-readable label.

    OneHotEncoder names follow the pattern:  ColumnName_Value
    e.g. 'Previous_Default_Yes' -> 'Previous Default: Yes'
         'Employment_Type_Self-Employed' -> 'Employment Type: Self-Employed'
         'Annual_Income' (numerical) -> 'Annual Income'
    """
    # Known categorical columns (to detect OHE names)
    _CAT_PREFIXES = [
        "Gender", "Education", "Employment_Type",
        "Loan_Purpose", "Previous_Default",
    ]

    name = raw_name.strip()

    for prefix in _CAT_PREFIXES:
        if name.startswith(prefix + "_"):
            value = name[len(prefix) + 1:]
            label_col = prefix.replace("_", " ")
            return f"{label_col}: {value}"

    # Numerical or engineered feature — just replace underscores with spaces
    return name.replace("_", " ").title()

