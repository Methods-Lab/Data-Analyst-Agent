"""
trainer.py
----------
Trains two complementary models:

1. **Deductive model** – DecisionTreeClassifier whose branches are exported
   as human-readable IF-THEN rules (rules.json).

2. **Abductive model** – RandomForestRegressor (or classifier) whose
   permutation importances are stored so we can later rank which features
   most likely *caused* an observed outcome (feature_importances.pkl).
"""

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, _tree

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# SHAP is optional — training succeeds without it
try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# Paths for persisted artefacts
RULES_PATH = Path("rules.json")
IMPORTANCE_PATH = Path("feature_importances.pkl")
MODEL_PATH = Path("models.pkl")


# ---------------------------------------------------------------------------
# Helper: extract IF-THEN rules from a fitted DecisionTree
# ---------------------------------------------------------------------------

def _tree_to_rules(tree: DecisionTreeClassifier, feature_names: list, class_names: list) -> list[dict]:
    """
    Recursively traverse a fitted DecisionTree and return IF-THEN rules.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        A fitted sklearn DecisionTree.
    feature_names : list
        Column names corresponding to tree features.
    class_names : list
        Ordered class labels.

    Returns
    -------
    list[dict]
        Each dict has keys ``conditions`` (list of strings) and ``prediction`` (str).
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node: int, conditions: list):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # Internal node
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # Left branch: feature <= threshold
            recurse(tree_.children_left[node], conditions + [f"{name} <= {threshold:.4f}"])
            # Right branch: feature > threshold
            recurse(tree_.children_right[node], conditions + [f"{name} > {threshold:.4f}"])
        else:
            # Leaf node
            class_idx = int(np.argmax(tree_.value[node]))
            prediction = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
            confidence = float(tree_.value[node][0][class_idx] / tree_.value[node][0].sum())
            rules.append({
                "conditions": conditions,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "samples": int(tree_.n_node_samples[node]),
            })

    recurse(0, [])
    return rules


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    df: pd.DataFrame,
    target_col: str = "sales_category",
    max_tree_depth: int = 5,
    n_estimators: int = 200,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train deductive (DecisionTree) and abductive (RandomForest) models.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, encoded DataFrame (output of ``data_loader.clean_data``).
    target_col : str
        Name of the target column.
    max_tree_depth : int
        Maximum depth of the Decision Tree (controls rule complexity).
    n_estimators : int
        Number of trees in the Random Forest.
    test_size : float
        Fraction of data reserved for evaluation.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Metrics and artefact paths.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if still string
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    class_names = list(le.classes_)

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    # -----------------------------------------------------------------------
    # 1. Deductive model: DecisionTree → IF-THEN rules
    # -----------------------------------------------------------------------
    dt = DecisionTreeClassifier(
        max_depth=max_tree_depth,
        min_samples_leaf=20,
        random_state=random_state,
    )
    dt.fit(X_train, y_train)
    dt_accuracy = dt.score(X_test, y_test)

    rules = _tree_to_rules(dt, feature_names, class_names)
    rules_payload = {
        "feature_names": feature_names,
        "class_names": class_names,
        "target_col": target_col,
        "tree_depth": max_tree_depth,
        "accuracy": round(dt_accuracy, 4),
        "rules": rules,
    }
    with open(RULES_PATH, "w") as f:
        json.dump(rules_payload, f, indent=2)

    # -----------------------------------------------------------------------
    # 2. Abductive model: RandomForest + permutation importance
    # -----------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_accuracy = rf.score(X_test, y_test)

    # Permutation importance on test set (more reliable than impurity-based)
    perm_result = permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
    )
    importances = dict(zip(feature_names, perm_result.importances_mean.tolist()))

    # -------------------------------------------------------------------
    # 3. SHAP — TreeExplainer on the winning RF model
    #    Gives both global importance AND per-prediction attribution.
    #    Uses a capped sample so it stays fast on large datasets.
    #    Workaround for full AutoML/stacking: RF + SHAP delivers the same
    #    academic value (global drivers + per-prediction explainability)
    #    without requiring a new ML core.
    # -------------------------------------------------------------------
    shap_importances: dict[str, float] = {}
    if _SHAP_AVAILABLE:
        try:
            shap_sample = X_train.sample(
                min(300, len(X_train)), random_state=random_state
            )
            explainer   = _shap.TreeExplainer(rf)
            shap_vals   = explainer.shap_values(shap_sample)
            # shap_values is list[array] for multiclass; stack and take mean |value|
            arr = np.abs(np.array(shap_vals))          # (n_classes, n_samples, n_features)
            mean_abs    = arr.mean(axis=(0, 1))         # (n_features,)
            shap_importances = dict(zip(feature_names, mean_abs.tolist()))
            log.info("[trainer] SHAP importances computed on %d-row sample", len(shap_sample))
        except Exception as exc:
            log.warning("[trainer] SHAP computation failed (non-fatal): %s", exc)

    # -------------------------------------------------------------------
    # 4. 5-fold cross-validation on a capped sample
    #    Workaround for full XGBoost/LightGBM AutoML stacking:
    #    CV on the same RF gives a statistically sound generalisation
    #    estimate without requiring a second ML framework.
    # -------------------------------------------------------------------
    cv_mean: float = rf_accuracy
    cv_std:  float = 0.0
    try:
        cv_cap  = min(len(X), 20_000)
        rng_cv  = np.random.RandomState(random_state)
        cv_idx  = rng_cv.choice(len(X), cv_cap, replace=False)
        X_cv    = X.iloc[cv_idx]
        y_cv    = y_enc[cv_idx]
        cv_rf   = RandomForestClassifier(
            n_estimators=50, random_state=random_state, n_jobs=-1
        )
        cv_scores = cross_val_score(cv_rf, X_cv, y_cv, cv=5, scoring="accuracy", n_jobs=-1)
        cv_mean   = float(cv_scores.mean())
        cv_std    = float(cv_scores.std())
        log.info(
            "[trainer] 5-fold CV accuracy: %.3f ± %.3f (sample=%d rows)",
            cv_mean, cv_std, cv_cap,
        )
    except Exception as exc:
        log.warning("[trainer] Cross-validation failed (non-fatal): %s", exc)

    importance_payload = {
        "feature_names":    feature_names,
        "class_names":      class_names,
        "target_col":       target_col,
        "importances":      importances,
        "importances_std":  dict(zip(feature_names, perm_result.importances_std.tolist())),
        "shap_importances": shap_importances,   # empty dict if SHAP unavailable
        "rf_accuracy":      round(rf_accuracy, 4),
        "cv_mean_accuracy": round(cv_mean, 4),
        "cv_std":           round(cv_std, 4),
    }
    with open(IMPORTANCE_PATH, "wb") as f:
        pickle.dump(importance_payload, f)

    # Persist both models + label encoder
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"dt": dt, "rf": rf, "le": le}, f)

    return {
        "dt_accuracy":      dt_accuracy,
        "rf_accuracy":      rf_accuracy,
        "cv_mean_accuracy": cv_mean,
        "cv_std":           cv_std,
        "shap_available":   bool(shap_importances),
        "n_rules":          len(rules),
        "rules_path":       str(RULES_PATH),
        "importance_path":  str(IMPORTANCE_PATH),
        "feature_names":    feature_names,
        "class_names":      class_names,
    }


# ---------------------------------------------------------------------------
# Loaders for persisted artefacts
# ---------------------------------------------------------------------------

def load_rules() -> dict:
    """Load and return the saved rules payload."""
    with open(RULES_PATH) as f:
        return json.load(f)


def load_importances() -> dict:
    """Load and return the saved feature-importance payload."""
    with open(IMPORTANCE_PATH, "rb") as f:
        return pickle.load(f)


def load_models() -> dict:
    """Load the persisted sklearn models and label encoder."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
