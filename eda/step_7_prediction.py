"""
============================================================
EDA Step 7 — Making Predictions
============================================================

PURPOSE
-------
Use the trained models to answer three kinds of business questions:

  1. CLASSIFICATION
     "Given these feature values, which class will this record belong to?"
     - Uses Random Forest + the saved IF-THEN rule base
     - `reasoner.deduce(facts)` performs forward chaining

  2. ABDUCTIVE REASONING
     "Why does outcome X happen? What features drive it?"
     - Uses permutation importance + Bayesian posterior probability
     - `reasoner.abduce(outcome)` + `abduction.BayesianAbducer`

  3. REGRESSION & FORECASTING
     "What numeric value will column Y take?"
     "What will column Y look like over the next 6 months?"
     - These on-demand models live in app.py (_run_regression / _run_forecast)
     - They use sklearn's RandomForestRegressor and LinearRegression

USED BY THE APP
---------------
- "Predict" pinned button → opens the dialog that calls all three flows
- "Why high?" pinned button → calls reasoner.abduce()
- Custom chat queries like "predict if quantity=5" → calls reasoner.deduce()

PUBLIC API
----------
- deduce(facts, top_k=3)              : forward-chain through saved rules
- abduce(outcome, top_k=5)            : backward-infer drivers of an outcome
- BayesianAbducer                     : Naive-Bayes posterior reasoner

REAL IMPLEMENTATION
-------------------
- `reasoner.py` — deduce + abduce + condition matching
- `abduction.py` — BayesianAbducer class
- `app._run_regression()` / `app._run_forecast()` — on-demand sklearn models
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoner import deduce, abduce, format_deduction_result, format_abduction_result
from abduction import BayesianAbducer, try_bayesian_abduce

__all__ = [
    "deduce", "abduce",
    "format_deduction_result", "format_abduction_result",
    "BayesianAbducer", "try_bayesian_abduce",
]


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 7: PREDICTION DEMO")
    print("=" * 60)

    try:
        # Deduction example: predict outcome from feature values
        facts   = {"quantity": 8, "discount_pct": 10, "unit_price": 25000}
        results = deduce(facts, top_k=2)
        print(f"\nDEDUCTION — predict from facts {facts}:")
        if results:
            for i, r in enumerate(results, 1):
                print(f"  #{i} → {r['prediction']!r} "
                      f"(confidence {r['confidence']:.0%}, match {r['match_score']:.0%})")
        else:
            print("  No matching rules.")

        # Abduction example: explain a class
        print(f"\nABDUCTION — what drives outcome 'high'?")
        result = abduce("high", top_k=5)
        for c in result["causes"][:5]:
            print(f"  - {c['feature']:25} importance={c['importance']:.4f}  {c['direction_hint']}")

    except FileNotFoundError as exc:
        print(f"\nMissing trained model: {exc}")
        print("Train the agent first:  python -m eda.step_5_training")
