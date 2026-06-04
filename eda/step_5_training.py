"""
============================================================
EDA Step 5 — Model Training
============================================================

PURPOSE
-------
Train two complementary tree-based models on the preprocessed data:

  1. DecisionTreeClassifier — produces human-readable IF-THEN rules,
     used by the chatbot's deductive reasoning engine.
  2. RandomForestClassifier — 200-tree ensemble, gives higher accuracy
     and works with SHAP TreeExplainer for explainability.

USED BY THE APP
---------------
app.py's `train_agent()` calls this module's `train()` function whenever
the user clicks the "Train Agent" button.

PUBLIC API
----------
- train(df, target_col, max_tree_depth=6, n_estimators=200, test_size=0.2)
    Trains both models, runs SHAP + 5-fold cross-validation, persists
    artifacts to disk, and returns a result dict with all metrics.

WHY TREE-BASED MODELS (not neural nets)
---------------------------------------
- Tabular data winners: trees consistently outperform deep learning on structured data
- Handle mixed types and missing values natively
- No GPU required — train in seconds on CPU
- Interpretable via SHAP — every prediction can be explained
- DecisionTree gives human-readable IF-THEN rules out of the box

REAL IMPLEMENTATION
-------------------
The full training routine lives in `trainer.train()`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer import train

__all__ = ["train"]


if __name__ == "__main__":
    from data_loader import generate_synthetic_data, clean_data

    print("=" * 60)
    print("STEP 5: MODEL TRAINING DEMO")
    print("=" * 60)

    df     = generate_synthetic_data(n_rows=1000)
    clean  = clean_data(df, target_col="sales_category")
    result = train(clean, target_col="sales_category", n_estimators=50)

    print("\nTraining complete.")
    print(f"  Decision Tree accuracy : {result['dt_accuracy']:.1%}")
    print(f"  Random Forest accuracy : {result['rf_accuracy']:.1%}")
    print(f"  5-fold CV accuracy     : {result.get('cv_mean_accuracy', 0):.1%} "
          f"± {result.get('cv_std', 0):.3f}")
    print(f"  Rules extracted        : {result['n_rules']}")
    print(f"  SHAP computed          : {result.get('shap_available', False)}")
    print(f"\nArtifacts saved to     : {result['rules_path']}")
    print(f"                         {result['importance_path']}")
