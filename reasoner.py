"""
reasoner.py
-----------
Implements two reasoning modes:

1. **Deductive reasoning** (forward chaining)
   Given a set of user-supplied facts (feature → value pairs), traverse the
   saved IF-THEN rule base and return the best-matching predicted outcome
   together with the full matched rule chain.

2. **Abductive reasoning** (best-explanation inference)
   Given an observed outcome (e.g. "sales_category = high"), use the
   feature-importance ranking to list the top-N features most likely
   *responsible* for that outcome, and explain the direction of influence
   by querying the saved rules that predict that outcome.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

RULES_PATH = Path("rules.json")
IMPORTANCE_PATH = Path("feature_importances.pkl")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_rules() -> dict:
    if not RULES_PATH.exists():
        raise FileNotFoundError("No trained rules found. Please train the agent first.")
    with open(RULES_PATH) as f:
        return json.load(f)


def _load_importances() -> dict:
    if not IMPORTANCE_PATH.exists():
        raise FileNotFoundError("No importance data found. Please train the agent first.")
    with open(IMPORTANCE_PATH, "rb") as f:
        return pickle.load(f)


def _parse_condition(condition: str) -> tuple[str, str, float]:
    """
    Parse a rule condition string like 'discount_pct > 10.5000' into
    (feature_name, operator, threshold).
    """
    for op in [" <= ", " > "]:
        if op in condition:
            parts = condition.split(op)
            return parts[0].strip(), op.strip(), float(parts[1].strip())
    raise ValueError(f"Cannot parse condition: {condition}")


def _condition_matches(condition: str, facts: dict[str, Any]) -> bool:
    """
    Return True if the user-supplied facts satisfy a single rule condition.

    If the relevant feature is absent from facts, the condition is treated
    as a wildcard match (True) so partial-fact queries still work.
    """
    feat, op, threshold = _parse_condition(condition)
    if feat not in facts:
        return True  # unknown → optimistic match

    val = facts[feat]
    try:
        val = float(val)
    except (ValueError, TypeError):
        return True  # non-numeric → skip

    if op == "<=":
        return val <= threshold
    elif op == ">":
        return val > threshold
    return False


# ---------------------------------------------------------------------------
# Deductive reasoning: forward chaining
# ---------------------------------------------------------------------------

def deduce(facts: dict[str, Any], top_k: int = 3) -> list[dict]:
    """
    Forward-chain through the IF-THEN rule base to find predictions that
    match the supplied facts.

    Algorithm
    ---------
    1. For each rule, check how many of its conditions are *satisfied* by
       the supplied facts and how many are *unknown* (feature missing).
    2. Score = satisfied / total_conditions (penalised by unknown ratio).
    3. Return the top-k highest-scoring rules with their predictions.

    Parameters
    ----------
    facts : dict
        Mapping of feature_name → value (strings or numbers).
        Keys should match the feature names used during training.
    top_k : int
        Number of top-matching rules to return.

    Returns
    -------
    list[dict]
        Each dict contains:
        - ``prediction`` : predicted class label
        - ``confidence`` : tree leaf confidence
        - ``match_score`` : fraction of conditions satisfied
        - ``conditions`` : list of rule conditions
        - ``samples`` : number of training samples that reached this leaf
    """
    payload = _load_rules()
    rules: list[dict] = payload["rules"]
    results = []

    for rule in rules:
        conditions = rule["conditions"]
        if not conditions:
            continue

        satisfied = sum(1 for c in conditions if _condition_matches(c, facts))
        score = satisfied / len(conditions)

        results.append({
            "prediction": rule["prediction"],
            "confidence": rule["confidence"],
            "match_score": round(score, 4),
            "conditions": conditions,
            "samples": rule["samples"],
        })

    # Sort by match_score desc, then confidence desc
    results.sort(key=lambda r: (r["match_score"], r["confidence"]), reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Abductive reasoning: best-explanation inference
# ---------------------------------------------------------------------------

def abduce(observed_outcome: str, top_k: int = 5, min_importance: float = 0.0) -> dict:
    """
    Given an observed outcome, infer the most probable causes.

    Algorithm
    ---------
    1. Load permutation importances (global feature relevance).
    2. Filter rules that predict ``observed_outcome``; collect feature
       conditions from those rules to determine *directional* influence
       (e.g. "discount_pct > 15 → high sales").
    3. Rank features by importance score; annotate each with a natural-
       language explanation derived from the matching rule conditions.

    Parameters
    ----------
    observed_outcome : str
        The class label to explain (e.g. "high", "low", "medium").
    top_k : int
        Number of causal features to return.
    min_importance : float
        Minimum permutation importance to include a feature.

    Returns
    -------
    dict with keys:
        - ``observed_outcome`` : the queried label
        - ``causes`` : list of dicts (feature, importance, direction_hint)
        - ``supporting_rules`` : top rules predicting this outcome
    """
    imp_payload = _load_importances()
    importances: dict[str, float] = imp_payload["importances"]
    importances_std: dict[str, float] = imp_payload.get("importances_std", {})

    rules_payload = _load_rules()
    rules: list[dict] = rules_payload["rules"]

    # Find rules that predict the observed outcome
    matching_rules = [r for r in rules if r["prediction"].lower() == observed_outcome.lower()]

    # Build a map: feature → list of conditions from matching rules
    feature_conditions: dict[str, list[str]] = {}
    for rule in matching_rules:
        for cond in rule["conditions"]:
            try:
                feat, _, _ = _parse_condition(cond)
                feature_conditions.setdefault(feat, []).append(cond)
            except ValueError:
                pass

    # Rank features by importance
    ranked = sorted(
        [(feat, imp) for feat, imp in importances.items() if imp >= min_importance],
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    causes = []
    for feat, imp in ranked:
        # Derive a direction hint from matching-rule conditions
        hints = feature_conditions.get(feat, [])
        if hints:
            # Majority vote on direction
            gt_count = sum(1 for c in hints if " > " in c)
            le_count = len(hints) - gt_count
            if gt_count > le_count:
                direction = f"tends to be HIGH when {feat} is large"
            else:
                direction = f"tends to be HIGH when {feat} is small"
            example_condition = hints[0]
        else:
            direction = f"is a significant predictor (importance={imp:.4f})"
            example_condition = None

        causes.append({
            "feature": feat,
            "importance": round(imp, 6),
            "importance_std": round(importances_std.get(feat, 0.0), 6),
            "direction_hint": direction,
            "example_condition": example_condition,
        })

    # Return top-3 supporting rules for transparency
    matching_rules_sorted = sorted(matching_rules, key=lambda r: r["confidence"], reverse=True)

    return {
        "observed_outcome": observed_outcome,
        "causes": causes,
        "supporting_rules": matching_rules_sorted[:3],
        "total_matching_rules": len(matching_rules),
    }


# ---------------------------------------------------------------------------
# Natural-language formatters
# ---------------------------------------------------------------------------

def format_deduction_result(results: list[dict]) -> str:
    """Format deduction output as a readable string."""
    if not results:
        return "No matching rules found for the provided facts."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"**Result #{i}** — Prediction: `{r['prediction']}`")
        lines.append(f"  • Match score: {r['match_score']:.0%}  |  Confidence: {r['confidence']:.0%}  |  Training samples: {r['samples']}")
        lines.append("  • Matched conditions:")
        for c in r["conditions"]:
            lines.append(f"    – {c}")
        lines.append("")
    return "\n".join(lines)


def format_abduction_result(result: dict) -> str:
    """Format abduction output as a readable string."""
    lines = [
        f"**Observed outcome:** `{result['observed_outcome']}`",
        f"**Most probable causes** (ranked by feature importance):\n",
    ]
    for i, cause in enumerate(result["causes"], 1):
        lines.append(f"{i}. **{cause['feature']}**")
        lines.append(f"   Importance: {cause['importance']:.4f} ± {cause['importance_std']:.4f}")
        lines.append(f"   Explanation: {cause['direction_hint']}")
        if cause["example_condition"]:
            lines.append(f"   Example rule condition: `{cause['example_condition']}`")
        lines.append("")

    lines.append(f"*{result['total_matching_rules']} rules in the rule base predict this outcome.*")
    return "\n".join(lines)
