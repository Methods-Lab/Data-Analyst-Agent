"""
============================================================
EDA Step 4 — Feature Engineering & Preprocessing
============================================================

PURPOSE
-------
Transform the cleaned data into the exact shape sklearn needs:
- Date columns → year / month / day-of-week numeric features
- Categorical strings → integer codes (LabelEncoder)
- Target binning when no explicit target column exists
  (continuous "net_sales" → discrete "low" / "medium" / "high")
- Train / test split with stratification

USED BY THE APP
---------------
app.py's `ensure_target()` and the LabelEncoder + train_test_split
calls inside trainer.train() perform this step on every training run.

PUBLIC API
----------
- ensure_target(df, target_col)  : create / bin target column if needed
- prepare_xy(df, target_col)     : split DataFrame into (X, y) for sklearn
- stratified_split(X, y, test_size=0.2) : train/test split that preserves class ratios

REAL IMPLEMENTATION
-------------------
`ensure_target` is mirrored from app.py; `prepare_xy` and `stratified_split`
expose the train/test split logic from trainer.train() as reusable helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

__all__ = ["ensure_target", "prepare_xy", "stratified_split"]


def ensure_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Create a 3-class categorical target by binning a numeric column if needed."""
    working = df.copy()
    if target_col in working.columns and working[target_col].nunique(dropna=True) >= 2:
        return working

    base_name = target_col.replace("_tier", "")
    if base_name in working.columns and pd.api.types.is_numeric_dtype(working[base_name]):
        series = working[base_name].astype("float64")
        series = series.fillna(series.median())
    else:
        numeric_cols = [c for c in working.columns
                        if pd.api.types.is_numeric_dtype(working[c]) and working[c].dtype != bool]
        if numeric_cols:
            base_name = numeric_cols[0]
            series = working[base_name].astype("float64")
            series = series.fillna(series.median())
        else:
            working["text_signal"] = working.astype(str).agg(" ".join, axis=1).str.len()
            series = working["text_signal"].astype("float64")

    if series.nunique() < 3:
        working[target_col] = np.where(series >= series.median(), "high", "low")
        return working

    low_q  = series.quantile(0.33)
    high_q = series.quantile(0.67)
    if low_q == high_q:
        working[target_col] = np.where(series >= series.median(), "high", "low")
    else:
        working[target_col] = pd.cut(
            series, bins=[-np.inf, low_q, high_q, np.inf],
            labels=["low", "medium", "high"],
        ).astype(str)
    return working


def prepare_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Split DataFrame into feature matrix X, encoded target y, and class names."""
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not in DataFrame")
    X     = df.drop(columns=[target_col])
    le    = LabelEncoder()
    y_enc = le.fit_transform(df[target_col].astype(str))
    return X, y_enc, list(le.classes_)


def stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train/test split that preserves class proportions; falls back gracefully
    when a class has too few samples for stratification."""
    counts   = np.bincount(y)
    can_strat = counts.min() >= 2
    if not can_strat:
        keep = np.isin(y, np.where(counts >= 2)[0])
        X, y = X[keep], y[keep]
        counts = np.bincount(y)
        can_strat = counts.min() >= 2
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if can_strat else None,
    )


if __name__ == "__main__":
    from data_loader import generate_synthetic_data, clean_data

    print("=" * 60)
    print("STEP 4: PREPROCESSING DEMO")
    print("=" * 60)

    df    = generate_synthetic_data(n_rows=1000)
    clean = clean_data(df, target_col="sales_category")
    print(f"\nClean shape: {clean.shape}")

    X, y, classes = prepare_xy(clean, target_col="sales_category")
    print(f"Feature matrix: {X.shape}  |  target classes: {classes}")

    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"\nTrain: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")
    print(f"Train class balance: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test  class balance: {dict(zip(*np.unique(y_test,  return_counts=True)))}")
