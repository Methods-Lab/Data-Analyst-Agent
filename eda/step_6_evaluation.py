"""
============================================================
EDA Step 6 — Model Evaluation
============================================================

PURPOSE
-------
Verify the trained model actually works on data it has never seen,
measure honest generalisation accuracy, and identify which features
mattered most:

- Test-set accuracy (raw score)
- 5-fold cross-validation (more honest generalisation estimate)
- Confusion matrix (where the model goes wrong)
- Classification report (precision / recall / F1 per class)
- Permutation importance (which features matter at test time)
- SHAP values (per-prediction explainability)

USED BY THE APP
---------------
The metrics surfaced here populate the workspace dashboard:
"Model accuracy", "CV accuracy", "Decision rules" cards, plus the
"Most impactful columns" chart.

PUBLIC API
----------
- load_models()        : returns the persisted DT + RF + LabelEncoder
- load_rules()         : returns the IF-THEN rule payload from rules.json
- load_importances()   : returns SHAP + permutation importance payload

REAL IMPLEMENTATION
-------------------
`trainer.py` performs the metric computation during train(); this module
exposes the loader functions so the app and downstream code can read the
persisted evaluation results without re-training.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer import load_models, load_rules, load_importances

__all__ = ["load_models", "load_rules", "load_importances"]


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6: MODEL EVALUATION DEMO")
    print("=" * 60)

    try:
        imp = load_importances()
        print("\nEvaluation summary loaded from disk:")
        print(f"  Random Forest accuracy : {imp.get('rf_accuracy', 0):.1%}")
        print(f"  5-fold CV accuracy     : {imp.get('cv_mean_accuracy', 0):.1%} "
              f"± {imp.get('cv_std', 0):.3f}")
        print(f"  Classes                : {imp.get('class_names', [])}")
        print(f"  Total features         : {len(imp.get('feature_names', []))}")

        importances = imp.get("shap_importances") or imp.get("importances", {})
        if importances:
            top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\n  Top 5 features by importance:")
            for i, (feat, score) in enumerate(top, 1):
                print(f"    {i}. {feat:25} → {score:.4f}")
    except FileNotFoundError:
        print("\nNo trained model found. Run step_5_training.py first:")
        print("  python -m eda.step_5_training")
