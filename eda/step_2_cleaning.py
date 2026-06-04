"""
============================================================
EDA Step 2 — Data Cleaning
============================================================

PURPOSE
-------
Raw uploaded files are messy. Before any model can learn from the data
we must:
- Strip whitespace from column names
- Drop duplicate / Excel-artifact / constant / ID columns
- Normalise 22 common missing-value markers ("N/A", "--", "#DIV/0!", ...)
- Coerce string-numerics ("$1,200.50", "25%", "10 kg") to real numbers
- Cast booleans to int (numpy.quantile refuses bool subtract)
- Impute remaining NaN values (median for numeric, mode for categorical)
- Encode categoricals as integer codes for sklearn

USED BY THE APP
---------------
app.py calls `clean_data()` after `load_data()` and before model training.
Every uploaded file passes through this step.

PUBLIC API
----------
- clean_data(df, target_col) : returns a fully numeric, NaN-free DataFrame
                               ready for the preprocessing step

REAL IMPLEMENTATION
-------------------
`data_loader.clean_data()` — see source for the 10 ordered cleaning passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import clean_data

__all__ = ["clean_data"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from data_loader import generate_synthetic_data

    print("=" * 60)
    print("STEP 2: DATA CLEANING DEMO")
    print("=" * 60)

    df = generate_synthetic_data(n_rows=500)
    # Inject realistic dirt
    df.loc[df.sample(50, random_state=1).index, "unit_price"] = np.nan
    df["empty_col"]    = np.nan
    df["constant_col"] = "always_same"
    df["Unnamed: 0"]   = range(len(df))

    print(f"\nDirty input : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Missing    : {df.isna().sum().sum()} cells")
    print(f"  Junk cols  : empty_col, constant_col, Unnamed: 0")

    clean = clean_data(df, target_col="sales_category")
    print(f"\nCleaned out : {clean.shape[0]:,} rows × {clean.shape[1]} columns")
    print(f"  Missing    : {clean.isna().sum().sum()} cells")
    print(f"  Dropped    : {set(df.columns) - set(clean.columns)}")
