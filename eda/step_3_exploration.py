"""
============================================================
EDA Step 3 — Exploratory Data Analysis
============================================================

PURPOSE
-------
Once the data is clean, we explore it to find patterns BEFORE training a model:
- Summary statistics for every column
- Class balance of the prediction target
- Numeric distributions
- Correlation between numeric features
- IQR-based outlier detection

USED BY THE APP
---------------
app.py's `profile_data()` and `dataset_context_for_llm()` functions call this
analysis to populate the workspace dashboard and to feed the chat layer with
real numbers about the data.

PUBLIC API
----------
- profile_dataset(df, target_col=None) : full structural + statistical profile
- correlation_matrix(df)               : numeric-column correlations as DataFrame
- iqr_outliers(series)                 : count outliers via the IQR rule

REAL IMPLEMENTATION
-------------------
The web app's profiler logic; thin wrappers here so the step is callable
from the EDA pipeline as a standalone module.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


__all__ = ["profile_dataset", "correlation_matrix", "iqr_outliers"]


def profile_dataset(df: pd.DataFrame, target_col: str | None = None) -> dict:
    """Return a structured snapshot of the dataset for EDA reporting."""
    numeric     = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    missing     = df.isna().sum().sort_values(ascending=False)
    top_missing = missing[missing > 0].head(8).to_dict()

    profile: dict = {
        "rows":        len(df),
        "columns":     len(df.columns),
        "numeric":     numeric,
        "categorical": categorical,
        "missing":     top_missing,
        "target":      target_col,
    }
    if target_col and target_col in df.columns:
        profile["target_distribution"] = df[target_col].astype(str).value_counts().head(8).to_dict()
    if numeric:
        stats = df[numeric].describe().T
        profile["numeric_stats"] = stats[["mean", "std", "min", "max"]].round(3).to_dict("index")
    return profile


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix of numeric columns (rounded to 2dp)."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr().round(2)


def iqr_outliers(series: pd.Series) -> int:
    """Return the number of IQR-rule outliers in a numeric Series."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr    = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lo) | (series > hi)).sum())


if __name__ == "__main__":
    from data_loader import generate_synthetic_data, clean_data

    print("=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS DEMO")
    print("=" * 60)

    df    = generate_synthetic_data(n_rows=1000)
    clean = clean_data(df, target_col="sales_category")
    prof  = profile_dataset(clean, target_col="sales_category")

    print(f"\nRows × cols      : {prof['rows']:,} × {prof['columns']}")
    print(f"Numeric columns  : {len(prof['numeric'])}")
    print(f"Categorical cols : {len(prof['categorical'])}")
    print(f"Target dist      : {prof.get('target_distribution', {})}")
    print(f"Missing values   : {prof['missing'] or 'none'}")

    corr = correlation_matrix(clean)
    print(f"\nCorrelation matrix shape: {corr.shape}")

    numeric_cols = clean.select_dtypes(include=[np.number]).columns[:5]
    print("\nOutlier counts (IQR rule):")
    for c in numeric_cols:
        print(f"  {c:20} → {iqr_outliers(clean[c])} outliers")
