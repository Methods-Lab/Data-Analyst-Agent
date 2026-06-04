"""
============================================================
EDA Step 1 — Data Loading & First Look
============================================================

PURPOSE
-------
Every analysis begins by reading a file into memory and understanding
its basic shape: how many rows, how many columns, what data types,
and whether there are obvious gaps.

USED BY THE APP
---------------
- app.py train_clicked handler — when the user clicks "Train Agent"
  this module's `load_data()` reads the uploaded file.
- groq_cleaner.py — uses `generate_synthetic_data()` for demo mode.

PUBLIC API
----------
- load_data(source)              : read CSV / Excel from path or buffer
- generate_synthetic_data(n_rows): produce a realistic sales demo dataset
- infer_column_types(df)         : classify columns as numeric / categorical / datetime

REAL IMPLEMENTATION
-------------------
The underlying file-reading + synthetic data code lives in `data_loader.py`.
This module re-exports those functions so the EDA pipeline reads
top-to-bottom as a clean sequence of steps.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable so `data_loader` resolves cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import (
    load_data,
    generate_synthetic_data,
    infer_column_types,
)

__all__ = ["load_data", "generate_synthetic_data", "infer_column_types"]


# --------------------------------------------------------------------
# Standalone demo: `python -m eda.step_1_loading`
# --------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: DATA LOADING DEMO")
    print("=" * 60)

    df = generate_synthetic_data(n_rows=500)
    print(f"\nLoaded synthetic dataset")
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Column types   : {infer_column_types(df)}")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nDataset summary:")
    print(df.describe().round(2))
