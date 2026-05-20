"""
abduction.py  (optional advanced module)
-----------------------------------------
Provides a more sophisticated causal-inference layer using a
Naive-Bayes–style probabilistic model to compute P(cause | effect).

This module is entirely *optional*. If ``pgmpy`` is not installed, the
reasoner.py fallback (importance-based abduction) is used instead.

Design
------
We model abduction as:

    P(feature=v | outcome=k)  ∝  P(outcome=k | feature=v) · P(feature=v)

For each feature we discretise values into bins, compute the conditional
probability table (CPT) from training data, and use Bayes' theorem to
rank feature values that most "explain" an observed outcome.

Usage
-----
    from abduction import BayesianAbducer
    ba = BayesianAbducer()
    ba.fit(df, target_col="sales_category")
    result = ba.explain(outcome="high", top_k=5)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class BayesianAbducer:
    """
    Naive-Bayes abductive reasoner.

    Attributes
    ----------
    cpt_ : dict[str, pd.DataFrame]
        Conditional probability table per feature column.
    prior_ : pd.Series
        Prior class probabilities P(outcome=k).
    target_col_ : str
        Name of the target column.
    feature_cols_ : list[str]
        Feature columns used during fitting.
    """

    def __init__(self, n_bins: int = 5, laplace_alpha: float = 1.0):
        """
        Parameters
        ----------
        n_bins : int
            Number of bins for discretising continuous features.
        laplace_alpha : float
            Laplace (add-alpha) smoothing for CPTs.
        """
        self.n_bins = n_bins
        self.laplace_alpha = laplace_alpha
        self.cpt_: dict[str, pd.DataFrame] = {}
        self.prior_: Optional[pd.Series] = None
        self.target_col_: Optional[str] = None
        self.feature_cols_: list[str] = []

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, target_col: str = "sales_category") -> "BayesianAbducer":
        """
        Fit the abducer from a (cleaned, encoded) DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame where all columns are numeric (post-cleaning).
        target_col : str
            Target class column.

        Returns
        -------
        self
        """
        self.target_col_ = target_col
        self.feature_cols_ = [c for c in df.columns if c != target_col]

        # Prior P(y = k)
        self.prior_ = df[target_col].value_counts(normalize=True)

        for col in self.feature_cols_:
            # Discretise numeric columns into bins
            series = df[col]
            try:
                binned, bin_edges = pd.cut(
                    series, bins=self.n_bins, retbins=True, labels=False, duplicates="drop"
                )
                bin_labels = [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)]
            except Exception:
                # Fallback: treat as categorical
                binned = series.astype(str)
                bin_labels = None

            # Build contingency table: rows = feature bins, cols = target classes
            contingency = pd.crosstab(binned, df[target_col])

            # Laplace smoothing
            contingency = contingency + self.laplace_alpha

            # Normalise per class column → P(feature=v | class=k)
            cpt = contingency.div(contingency.sum(axis=0), axis=1)

            if bin_labels is not None and len(bin_labels) == len(cpt):
                cpt.index = bin_labels

            self.cpt_[col] = cpt

        return self

    # ------------------------------------------------------------------
    def explain(self, outcome: str, top_k: int = 5) -> list[dict]:
        """
        Return the top-k (feature, value) pairs that best explain the
        observed outcome using Bayesian abduction.

        P(feature=v | outcome=k) scored as:
            score = P(outcome=k | feature=v) · P(feature=v)
                  ∝ CPT[v, k] · marginal[v]

        Parameters
        ----------
        outcome : str
            The observed class label to explain.
        top_k : int
            Number of explanations to return.

        Returns
        -------
        list[dict]
            Sorted by posterior score descending.
        """
        if self.prior_ is None:
            raise RuntimeError("BayesianAbducer has not been fitted. Call .fit() first.")

        if outcome not in self.prior_.index:
            available = list(self.prior_.index)
            raise ValueError(f"Outcome '{outcome}' not seen during training. Available: {available}")

        explanations = []

        for col, cpt in self.cpt_.items():
            if outcome not in cpt.columns:
                continue

            likelihood_col = cpt[outcome]  # P(feature=v | outcome=k), per bin

            for bin_label, likelihood in likelihood_col.items():
                # Marginal P(feature=v) = average likelihood across classes
                marginal = cpt.loc[bin_label].mean() if bin_label in cpt.index else 1.0
                score = float(likelihood) * float(marginal)

                explanations.append({
                    "feature": col,
                    "feature_value_range": str(bin_label),
                    "posterior_score": round(score, 6),
                    "likelihood": round(float(likelihood), 6),
                    "marginal": round(float(marginal), 6),
                })

        # Sort by posterior score
        explanations.sort(key=lambda x: x["posterior_score"], reverse=True)

        # Deduplicate: keep best bin per feature
        seen_features: set[str] = set()
        unique_explanations = []
        for exp in explanations:
            if exp["feature"] not in seen_features:
                seen_features.add(exp["feature"])
                unique_explanations.append(exp)
            if len(unique_explanations) >= top_k:
                break

        return unique_explanations


# ---------------------------------------------------------------------------
# Convenience wrapper that gracefully degrades
# ---------------------------------------------------------------------------

def try_bayesian_abduce(
    df: pd.DataFrame,
    outcome: str,
    target_col: str = "sales_category",
    top_k: int = 5,
) -> Optional[list[dict]]:
    """
    Attempt Bayesian abduction. Returns None if fitting fails so the caller
    can fall back to importance-based abduction.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned training DataFrame.
    outcome : str
        Class label to explain.
    target_col : str
        Target column name.
    top_k : int
        Number of explanations.

    Returns
    -------
    list[dict] | None
    """
    try:
        ba = BayesianAbducer()
        ba.fit(df, target_col=target_col)
        return ba.explain(outcome=outcome, top_k=top_k)
    except Exception as exc:
        print(f"[BayesianAbducer] Falling back to importance-based abduction. Reason: {exc}")
        return None
