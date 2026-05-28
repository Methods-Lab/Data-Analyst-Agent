"""
groq_cleaner.py
---------------
Large-file cleaning pipeline: "Groq advises, pandas executes."

Three techniques from the Scalability PDF:
  Technique 1 — Representative sampling  : fixed ~200-row payload to LLM
  Technique 2 — Chunked reading          : memory stays flat on 1M+ rows
  Technique 3 — Vectorised local apply   : pandas in C; data never leaves machine

Single public entry point:
    clean_large_file(path, api_key) -> (clean_df, rules, schema)

The LLM is called EXACTLY ONCE with a fixed-size payload regardless of file size.
LLM latency is therefore O(1) w.r.t. row count.

Disable this module without touching the ML core: simply do not call
clean_large_file() and fall back to data_loader.clean_data() instead.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match Scalability PDF reference design)
# ---------------------------------------------------------------------------

CHUNK_SIZE = 200_000   # rows per chunk for CSV streaming
SAMPLE_N   = 200       # max rows sent to the LLM
HEAD_N     = 50        # always include first N rows in sample
RANDOM_N   = 150       # random draw to fill the rest of the sample

# ---------------------------------------------------------------------------
# Groq client — imported at module level so tests can patch it cleanly
# ---------------------------------------------------------------------------

try:
    from groq import Groq as _GroqClient  # noqa: F401
    _GROQ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GroqClient = None  # type: ignore[assignment,misc]
    _GROQ_AVAILABLE = False


# ===========================================================================
# Technique 1 — Representative sampling (Scalability PDF §Technique 1)
# ===========================================================================

def _guess_target_col(df: pd.DataFrame) -> str | None:
    """
    Heuristic target-column guess used for stratified sampling BEFORE the
    Groq call (Scalability PDF §Stratified Sampling for Skewed Data).

    Priority order:
      1. Well-known target-like names
      2. Lowest-cardinality object column (least likely to be an ID)
    """
    known = ["sales_category", "target", "label", "class", "outcome",
             "status", "category", "churn", "result"]
    for c in known:
        if c in df.columns and df[c].nunique(dropna=True) >= 2:
            return c
    obj_cols = [
        c for c in df.columns
        if df[c].dtype == object and 2 <= df[c].nunique(dropna=True) <= 20
    ]
    return min(obj_cols, key=lambda c: df[c].nunique(), default=None)


def build_profile(
    path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    """
    Stream the file in chunks so memory stays flat on 1M+ rows.

    Returns
    -------
    sample : pd.DataFrame
        Head-50 rows + random-150-row draw, capped at SAMPLE_N = 200.
    meta : dict
        Per-column metadata: dtype, nulls_pct, nunique, 3 example values.
    n_rows : int
        Total row count of the full file.
    """
    path = Path(path)
    t0 = time.perf_counter()
    logger.info("[groq_cleaner] build_profile started: %s", path.name)

    is_excel = path.suffix.lower() in {".xlsx", ".xls"}

    if is_excel:
        # Excel has no chunk API; files are typically small enough for RAM
        df_full = pd.read_excel(path, engine="openpyxl")
        head   = df_full.head(HEAD_N)
        dtypes = df_full.dtypes.astype(str).to_dict()
        n_rows = len(df_full)
        # Stratified draw so rare classes are represented (Scalability PDF §Technique 1)
        target_guess = _guess_target_col(df_full)
        if target_guess and df_full[target_guess].nunique() > 1:
            classes   = df_full[target_guess].dropna().unique()
            per_class = max(1, RANDOM_N // len(classes))
            parts = [
                df_full[df_full[target_guess] == cls].sample(
                    min(len(df_full[df_full[target_guess] == cls]), per_class),
                    random_state=42,
                )
                for cls in classes
            ]
            rand_draw = pd.concat(parts).head(RANDOM_N)
        else:
            rand_draw = df_full.sample(min(RANDOM_N, len(df_full)), random_state=42)
        sample = pd.concat([head, rand_draw]).head(SAMPLE_N).reset_index(drop=True)
    else:
        # CSV: stream in chunks — Technique 2 applied at profile stage too
        head: pd.DataFrame | None = None
        dtypes: dict[str, str] = {}
        n_rows = 0
        reservoir: list[pd.DataFrame] = []

        def _open_reader(enc: str = "utf-8"):
            return pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False, encoding=enc)

        try:
            reader = _open_reader("utf-8")
            for chunk in reader:
                if head is None:
                    head   = chunk.head(HEAD_N)
                    dtypes = chunk.dtypes.astype(str).to_dict()
                n_rows += len(chunk)
                reservoir.append(chunk.sample(min(40, len(chunk)), random_state=42))
        except UnicodeDecodeError:
            logger.info("[groq_cleaner] UTF-8 failed, retrying with latin-1")
            head, dtypes, n_rows, reservoir = None, {}, 0, []
            for chunk in _open_reader("latin-1"):
                if head is None:
                    head   = chunk.head(HEAD_N)
                    dtypes = chunk.dtypes.astype(str).to_dict()
                n_rows += len(chunk)
                reservoir.append(chunk.sample(min(40, len(chunk)), random_state=42))

        raw_sample   = pd.concat([head, *reservoir]).reset_index(drop=True)
        # Stratified top-up: ensure rare classes appear in the sample.
        # Uses per-class slicing instead of groupby.apply to stay compatible
        # with pandas 3.x (which changed apply column-inclusion behaviour).
        target_guess = _guess_target_col(raw_sample)
        if target_guess and raw_sample[target_guess].nunique() > 1:
            classes     = raw_sample[target_guess].dropna().unique()
            per_class   = max(1, SAMPLE_N // len(classes))
            parts = [
                raw_sample[raw_sample[target_guess] == cls].sample(
                    min(len(raw_sample[raw_sample[target_guess] == cls]), per_class),
                    random_state=42,
                )
                for cls in classes
            ]
            sample = pd.concat(parts).head(SAMPLE_N).reset_index(drop=True)
        else:
            sample = raw_sample.head(SAMPLE_N).reset_index(drop=True)

    # Build compact per-column metadata — never the full data
    meta: dict[str, Any] = {
        col: {
            "dtype":     dtypes.get(col, str(sample[col].dtype)),
            "nulls_pct": round(float(sample[col].isna().mean()), 3),
            "nunique":   int(sample[col].nunique(dropna=True)),
            "examples":  sample[col].dropna().astype(str).head(3).tolist(),
        }
        for col in sample.columns
    }

    elapsed = time.perf_counter() - t0
    logger.info(
        "[groq_cleaner] build_profile done: %d rows total | %d cols | "
        "sample=%d rows | %.3fs",
        n_rows, len(sample.columns), len(sample), elapsed,
    )
    return sample, meta, n_rows


# ===========================================================================
# ONE Groq call → JSON rules (Scalability PDF §Technique 1, §Core Insight)
# ===========================================================================

def get_rules(
    sample:  pd.DataFrame,
    meta:    dict[str, Any],
    n_rows:  int,
    api_key: str | None = None,
    model:   str = "llama-3.3-70b-versatile",
) -> dict[str, Any]:
    """
    Send exactly ONE Groq call with a fixed-size payload.

    Returns a JSON instruction set:
        drop[]        — columns to remove
        rename{}      — old_name → new_name
        cast{}        — col → dtype string
        impute{}      — col → strategy ("median" | "mode" | "zero")
        target        — suggested prediction target (or null)

    Robustness chain (Scalability PDF §Robustness):
        valid JSON  →  use it
        malformed   →  re-prompt ONCE
        still bad   →  safe rule-based defaults (pipeline never stalls)
    """
    api_key = api_key or os.getenv("GROQ_API_KEY", "")

    if not _GROQ_AVAILABLE:
        logger.warning("[groq_cleaner] groq SDK not installed — safe defaults.")
        return _safe_defaults(sample)

    if not api_key:
        logger.warning("[groq_cleaner] No API key provided — safe defaults.")
        return _safe_defaults(sample)

    _groq_call_count = 0

    def _call(prompt_text: str) -> dict | None:
        nonlocal _groq_call_count
        _groq_call_count += 1
        t0 = time.perf_counter()
        logger.info(
            "[groq_cleaner] Groq call #%d | payload=%d chars",
            _groq_call_count, len(prompt_text),
        )
        try:
            client   = _GroqClient(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            elapsed = time.perf_counter() - t0
            raw = response.choices[0].message.content.strip()
            logger.info("[groq_cleaner] Groq responded in %.2fs", elapsed)
            return json.loads(raw)
        except Exception as exc:
            logger.error("[groq_cleaner] Groq call #%d failed: %s", _groq_call_count, exc)
            return None

    base_prompt = (
        f"You are a data-cleaning planner. Based ONLY on this sample "
        f"(full file has {n_rows:,} rows), return a JSON object with exactly "
        "these five keys:\n"
        "  drop   : list of column names to remove (IDs, constants, >70% missing)\n"
        "  rename : object mapping old_name to new_name\n"
        "  cast   : object mapping column to dtype ('float64','int64','str')\n"
        "  impute : object mapping column to strategy ('median','mode','zero')\n"
        "  target : string name of the most likely prediction target, or null\n\n"
        "Return valid JSON ONLY — no markdown, no explanation.\n\n"
        f"SAMPLE ({len(sample)} rows):\n{sample.to_csv(index=False)}\n\n"
        f"COLUMN METADATA:\n{json.dumps(meta, indent=2)}"
    )

    rules = _call(base_prompt)

    if rules is None or not _validate_rules(rules):
        logger.warning("[groq_cleaner] Response invalid — re-prompting once.")
        retry_prompt = (
            base_prompt
            + "\n\nPrevious response was invalid JSON. "
              "Return ONLY a JSON object with keys: drop, rename, cast, impute, target."
        )
        rules = _call(retry_prompt)

    if rules is None or not _validate_rules(rules):
        logger.warning(
            "[groq_cleaner] Re-prompt also invalid — using rule-based defaults. "
            "Total Groq calls: %d", _groq_call_count,
        )
        return _safe_defaults(sample)

    logger.info(
        "[groq_cleaner] Rules accepted | drop=%s | target=%s | total calls=%d",
        rules.get("drop", []), rules.get("target"), _groq_call_count,
    )
    return rules


def _validate_rules(obj: Any) -> bool:
    """Return True only when the object is a well-formed rules dict."""
    if not isinstance(obj, dict):
        return False
    required = {"drop", "rename", "cast", "impute", "target"}
    if not required.issubset(obj.keys()):
        return False
    if not isinstance(obj["drop"], list):
        return False
    if not isinstance(obj.get("rename", {}), dict):
        return False
    if not isinstance(obj.get("cast", {}), dict):
        return False
    if not isinstance(obj.get("impute", {}), dict):
        return False
    return True


def _safe_defaults(sample: pd.DataFrame) -> dict[str, Any]:
    """
    Rule-based fallback — mirrors data_loader.clean_data() heuristics so
    the pipeline never stalls when the LLM is unavailable or returns garbage.
    """
    drop:   list[str]       = []
    cast:   dict[str, str]  = {}
    impute: dict[str, str]  = {}

    for col in sample.columns:
        # Drop >70% missing
        if sample[col].isna().mean() > 0.7:
            drop.append(col)
            continue
        # Drop ID-like (>90% unique object column)
        if sample[col].dtype == object and sample[col].nunique() / max(len(sample), 1) > 0.9:
            drop.append(col)
            continue
        # Impute strategy
        if pd.api.types.is_numeric_dtype(sample[col]):
            impute[col] = "median"
        else:
            impute[col] = "mode"

    # Guess target: known names first, then lowest-cardinality object column
    known = ["sales_category", "target", "label", "class", "outcome", "status", "category"]
    target = next((c for c in known if c in sample.columns and c not in drop), None)
    if target is None:
        obj_cols = [c for c in sample.columns if sample[c].dtype == object and c not in drop]
        target = min(obj_cols, key=lambda c: sample[c].nunique(), default=None)

    logger.info("[groq_cleaner] safe_defaults: drop=%s | target=%s", drop, target)
    return {"drop": drop, "rename": {}, "cast": cast, "impute": impute, "target": target}


# ===========================================================================
# Technique 3 — Vectorised local apply (Scalability PDF §Technique 3)
# ===========================================================================

def apply_rules(
    path:  str | Path,
    rules: dict[str, Any],
) -> pd.DataFrame:
    """
    Apply the cleaning rules to the FULL dataset locally.

    Processing is chunked (CHUNK_SIZE rows at a time) and vectorised —
    no Python row-loops, no network calls. Bulk data never leaves the machine.

    Returns the fully cleaned DataFrame.
    """
    path       = Path(path)
    drop_cols  = [c for c in (rules.get("drop")   or []) if isinstance(c, str)]
    rename_map = dict(rules.get("rename") or {})
    cast_map   = dict(rules.get("cast")   or {})
    impute_map = dict(rules.get("impute") or {})

    is_excel = path.suffix.lower() in {".xlsx", ".xls"}
    out: list[pd.DataFrame] = []

    t0 = time.perf_counter()
    logger.info("[groq_cleaner] apply_rules started: %s", path.name)

    if is_excel:
        chunk = pd.read_excel(path, engine="openpyxl")
        out.append(_apply_to_chunk(chunk, drop_cols, rename_map, cast_map, impute_map))
    else:
        def _open(enc: str = "utf-8"):
            return pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False, encoding=enc)

        try:
            for chunk in _open("utf-8"):
                out.append(_apply_to_chunk(chunk, drop_cols, rename_map, cast_map, impute_map))
        except UnicodeDecodeError:
            out.clear()
            for chunk in _open("latin-1"):
                out.append(_apply_to_chunk(chunk, drop_cols, rename_map, cast_map, impute_map))

    clean_df = pd.concat(out, ignore_index=True)
    elapsed  = time.perf_counter() - t0
    logger.info(
        "[groq_cleaner] apply_rules done: %d rows × %d cols in %.3fs",
        len(clean_df), len(clean_df.columns), elapsed,
    )
    return clean_df


def _apply_to_chunk(
    chunk:      pd.DataFrame,
    drop_cols:  list[str],
    rename_map: dict[str, str],
    cast_map:   dict[str, str],
    impute_map: dict[str, str],
) -> pd.DataFrame:
    """Apply all rules to a single chunk — fully vectorised pandas, no row loops."""
    # 1. Drop
    cols_to_drop = [c for c in drop_cols if c in chunk.columns]
    if cols_to_drop:
        chunk = chunk.drop(columns=cols_to_drop)

    # 2. Rename
    if rename_map:
        chunk = chunk.rename(columns={k: v for k, v in rename_map.items() if k in chunk.columns})

    # 3. Cast (vectorised)
    for col, dtype_str in cast_map.items():
        if col not in chunk.columns:
            continue
        try:
            dtype_lower = str(dtype_str).lower()
            if "int" in dtype_lower:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            elif "float" in dtype_lower or "num" in dtype_lower:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            elif "str" in dtype_lower or "object" in dtype_lower:
                chunk[col] = chunk[col].astype(str)
        except Exception as exc:
            logger.debug("[groq_cleaner] cast %s → %s failed: %s", col, dtype_str, exc)

    # 4. Impute (vectorised — chunk-level stats, matching Scalability PDF reference)
    for col, strategy in impute_map.items():
        if col not in chunk.columns:
            continue
        if not chunk[col].isna().any():
            continue
        if strategy == "median":
            fill = (
                chunk[col].median()
                if pd.api.types.is_numeric_dtype(chunk[col])
                else (chunk[col].mode().iloc[0] if not chunk[col].mode().empty else "Unknown")
            )
        elif strategy == "zero":
            fill = 0
        else:  # "mode" or anything else
            fill = chunk[col].mode().iloc[0] if not chunk[col].mode().empty else "Unknown"
        chunk[col] = chunk[col].fillna(fill)

    return chunk


# ===========================================================================
# In-memory variant — for DataFrames already loaded (e.g. a specific Excel sheet)
# ===========================================================================

def clean_dataframe(
    df:      pd.DataFrame,
    api_key: str | None = None,
    model:   str = "llama-3.3-70b-versatile",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    Same pipeline as clean_large_file() but operates on an already-loaded
    DataFrame.  Used when a specific Excel sheet is selected by the user
    (Excel cannot be chunked, so we read it first and then clean it here).

    Returns (clean_df, rules, schema) — identical contract to clean_large_file().
    """
    t0     = time.perf_counter()
    n_rows = len(df)
    logger.info("[groq_cleaner] clean_dataframe: %d rows × %d cols", n_rows, len(df.columns))

    # ---- build profile directly from the DataFrame ----
    head   = df.head(HEAD_N)
    dtypes = df.dtypes.astype(str).to_dict()

    # Stratified draw (same logic as build_profile CSV path)
    target_guess = _guess_target_col(df)
    if target_guess and df[target_guess].nunique(dropna=True) > 1:
        classes   = df[target_guess].dropna().unique()
        per_class = max(1, RANDOM_N // len(classes))
        parts     = [
            df[df[target_guess] == cls].sample(
                min(len(df[df[target_guess] == cls]), per_class), random_state=42
            )
            for cls in classes
        ]
        rand_draw = pd.concat(parts).head(RANDOM_N)
    else:
        rand_draw = df.sample(min(RANDOM_N, n_rows), random_state=42)

    sample = pd.concat([head, rand_draw]).head(SAMPLE_N).reset_index(drop=True)

    meta: dict[str, Any] = {
        col: {
            "dtype":     dtypes.get(col, str(sample[col].dtype)),
            "nulls_pct": round(float(sample[col].isna().mean()), 3),
            "nunique":   int(sample[col].nunique(dropna=True)),
            "examples":  sample[col].dropna().astype(str).head(3).tolist(),
        }
        for col in sample.columns
    }
    logger.info(
        "[groq_cleaner] clean_dataframe profile: %d rows total | %d cols | sample=%d rows",
        n_rows, len(sample.columns), len(sample),
    )

    # ---- ONE Groq call ----
    rules = get_rules(sample, meta, n_rows, api_key=api_key, model=model)

    # ---- apply rules vectorised (single chunk since df is already in RAM) ----
    drop_cols  = [c for c in (rules.get("drop")   or []) if isinstance(c, str)]
    rename_map = dict(rules.get("rename") or {})
    cast_map   = dict(rules.get("cast")   or {})
    impute_map = dict(rules.get("impute") or {})

    t_apply  = time.perf_counter()
    clean_df = _apply_to_chunk(df.copy(), drop_cols, rename_map, cast_map, impute_map)
    schema   = _build_schema(clean_df, rules)

    logger.info(
        "[groq_cleaner] clean_dataframe done: %d rows × %d cols in %.3fs total",
        len(clean_df), len(clean_df.columns), time.perf_counter() - t0,
    )
    return clean_df, rules, schema


# ===========================================================================
# Orchestrator — file-path entry point
# ===========================================================================

def clean_large_file(
    path:    str | Path,
    api_key: str | None = None,
    model:   str = "llama-3.3-70b-versatile",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    Full pipeline: profile → ONE Groq call → vectorised local apply.

    Latency breakdown (Scalability PDF):
      • Sampling  : milliseconds (head + one random draw)
      • Groq call : ~1–2 s  — the ONLY network step, fixed cost, O(1) vs rows
      • Local apply: seconds on 1M rows with vectorised pandas

    Parameters
    ----------
    path : str | Path
        Path to the CSV or Excel file to clean.
    api_key : str | None
        Groq API key. Falls back to GROQ_API_KEY env var, then safe defaults.
    model : str
        Groq model ID.

    Returns
    -------
    clean_df : pd.DataFrame
        Cleaned DataFrame ready for the ML preprocessing stage (OBJ 4).
    rules : dict
        The JSON instruction set (from Groq or safe defaults).
    schema : dict
        Schema object compatible with the EDA stage (OBJ 3).
    """
    t_pipeline = time.perf_counter()
    logger.info("[groq_cleaner] clean_large_file: %s", Path(path).name)

    # Step 1 — profile (chunked, memory-flat)
    sample, meta, n_rows = build_profile(path)

    # Step 2 — ONE Groq call (fixed-size payload, O(1) latency)
    rules = get_rules(sample, meta, n_rows, api_key=api_key, model=model)

    # Step 3 — apply rules to full dataset locally (vectorised, chunked)
    clean_df = apply_rules(path, rules)

    # Step 4 — build schema for EDA / downstream stages
    schema = _build_schema(clean_df, rules)

    total = time.perf_counter() - t_pipeline
    logger.info(
        "[groq_cleaner] Pipeline complete: %d rows × %d cols in %.3fs total",
        len(clean_df), len(clean_df.columns), total,
    )
    return clean_df, rules, schema


def _build_schema(df: pd.DataFrame, rules: dict[str, Any]) -> dict[str, Any]:
    """Build a schema object that OBJ 3 (EDA) and later stages consume."""
    numeric     = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    missing_map = df.isna().sum()
    missing     = {k: int(v) for k, v in missing_map[missing_map > 0].items()}

    return {
        "columns":     list(df.columns),
        "numeric":     numeric,
        "categorical": categorical,
        "target":      rules.get("target"),
        "missing":     missing,
        "n_rows":      len(df),
        "n_cols":      len(df.columns),
        "flagged_issues": {
            "dropped": rules.get("drop", []),
            "renamed": rules.get("rename", {}),
            "cast":    rules.get("cast", {}),
            "imputed": rules.get("impute", {}),
        },
    }
