"""
test_scalability.py
-------------------
Phase 2 validation suite for the groq_cleaner scalability fix.

Tests
-----
1. SMALL  : 500-row CSV  → full pipeline, clean_df produced, schema correct
2. LARGE  : 1 000 000-row CSV → chunked reading, exactly 1 Groq call, timing
3. MESSY  : "$1,200.50" values, "25 years" age, 80% missing col, ID col
4. BAD-LLM: malformed JSON first → re-prompt once → valid; and both bad → fallback
5. DEPS   : every required library imports cleanly (versions reported)

Run with:
    python test_scalability.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Logging — show groq_cleaner's own log lines so the tester can see them
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Module under test
# --------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
import groq_cleaner


# ===========================================================================
# Helpers
# ===========================================================================

def _make_response(content: str) -> MagicMock:
    """Build a mock Groq completion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


VALID_RULES = {
    "drop":   ["id"],
    "rename": {},
    "cast":   {"revenue": "float64", "quantity": "int64"},
    "impute": {"revenue": "median", "region": "mode", "discount": "median"},
    "target": "status",
}
VALID_JSON = json.dumps(VALID_RULES)


def _write_csv(df: pd.DataFrame) -> Path:
    """Write DataFrame to a temp CSV and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return Path(tmp.name)


def _make_small_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "id":       np.arange(n),
        "revenue":  rng.normal(5000, 1500, n).round(2),
        "quantity": rng.integers(1, 20, n),
        "discount": rng.uniform(0, 30, n).round(1),
        "region":   rng.choice(["North", "South", "East", "West"], n),
        "status":   rng.choice(["completed", "pending", "cancelled"], n),
    })


def _make_large_df(n: int = 1_000_000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "id":       np.arange(n),
        "revenue":  rng.normal(5000, 1500, n).round(2),
        "quantity": rng.integers(1, 20, n),
        "discount": rng.uniform(0, 30, n).round(1),
        "region":   rng.choice(["North", "South", "East", "West"], n),
        "status":   rng.choice(["completed", "pending", "cancelled"], n),
        "score":    rng.uniform(0, 1, n).round(4),
    })
    # inject ~1% missing values
    df.loc[rng.choice(n, 10_000, replace=False), "discount"] = np.nan
    df.loc[rng.choice(n, 5_000, replace=False), "region"]   = np.nan
    return df


def _make_messy_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "cust_id":   [f"C{i:06d}" for i in range(n)],   # ID col → should drop
        "revenue":   [f"${rng.integers(100,9999):,}.{rng.integers(0,99):02d}"
                      for _ in range(n)],                # "$1,200.50" → cast float
        "age":       [f"{v} years" for v in rng.integers(18, 70, n)],  # "25 years" → cast int
        "score":     rng.uniform(0, 1, n).round(3),
        "useless":   [None] * n,                         # 100% missing → should drop
        "category":  rng.choice(["A", "B", "C"], n),
    })
    # Make "useless" truly 80%+ missing
    df.loc[rng.choice(n, int(n * 0.85), replace=False), "score"] = np.nan
    return df


# ===========================================================================
# Test Cases
# ===========================================================================

class TestDependencies(unittest.TestCase):
    """Test 5 — all required libraries import cleanly."""

    def test_imports(self):
        log.info("=== TEST: Dependencies ===")
        import pandas as _pd
        import numpy as _np
        import sklearn
        import groq as _groq
        import streamlit  # noqa: F401
        import openpyxl   # noqa: F401
        import faker      # noqa: F401

        for name, mod in [
            ("pandas",     _pd),
            ("numpy",      _np),
            ("scikit-learn", sklearn),
            ("groq",       _groq),
        ]:
            version = getattr(mod, "__version__", "?")
            log.info("  %-20s %s", name, version)

        log.info("  All required libraries: OK")


class TestBuildProfile(unittest.TestCase):
    """Validates build_profile: chunked read, sample size, metadata."""

    def setUp(self):
        self.small_path = _write_csv(_make_small_df(500))

    def tearDown(self):
        self.small_path.unlink(missing_ok=True)

    def test_sample_capped_at_200(self):
        log.info("=== TEST: build_profile sample <= 200 rows ===")
        sample, meta, n_rows = groq_cleaner.build_profile(self.small_path)
        self.assertLessEqual(len(sample), groq_cleaner.SAMPLE_N)
        self.assertEqual(n_rows, 500)
        log.info("  sample=%d rows, n_rows=%d  OK", len(sample), n_rows)

    def test_meta_has_required_keys(self):
        log.info("=== TEST: build_profile metadata keys ===")
        sample, meta, _ = groq_cleaner.build_profile(self.small_path)
        for col in sample.columns:
            self.assertIn("dtype",     meta[col])
            self.assertIn("nulls_pct", meta[col])
            self.assertIn("nunique",   meta[col])
            self.assertIn("examples",  meta[col])
        log.info("  Metadata keys: OK for %d columns", len(sample.columns))


class TestSmallPipeline(unittest.TestCase):
    """Test 1 — SMALL: 500-row CSV → clean_df produced, schema correct."""

    def setUp(self):
        self.path = _write_csv(_make_small_df(500))

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_full_pipeline_small(self):
        log.info("=== TEST: SMALL (500 rows) ===")

        with patch("groq_cleaner._GroqClient") as MockClient:
            mock_inst = MagicMock()
            MockClient.return_value = mock_inst
            mock_inst.chat.completions.create.return_value = _make_response(VALID_JSON)

            clean_df, rules, schema = groq_cleaner.clean_large_file(
                self.path, api_key="test-key"
            )

        # Exactly 1 Groq call
        self.assertEqual(mock_inst.chat.completions.create.call_count, 1,
                         "Expected exactly 1 Groq call")

        # clean_df is a proper DataFrame
        self.assertIsInstance(clean_df, pd.DataFrame)
        self.assertGreater(len(clean_df), 0)

        # "id" column was dropped per rules
        self.assertNotIn("id", clean_df.columns)

        # Schema has required keys
        for key in ("columns", "numeric", "categorical", "target", "n_rows", "n_cols"):
            self.assertIn(key, schema)

        self.assertEqual(schema["target"], "status")
        log.info("  clean_df: %d rows × %d cols | target=%s | Groq calls=1  OK",
                 len(clean_df), len(clean_df.columns), schema["target"])


class TestLargePipeline(unittest.TestCase):
    """Test 2 — LARGE: 1 000 000-row CSV → chunked, 1 Groq call, timing."""

    @classmethod
    def setUpClass(cls):
        log.info("=== TEST: LARGE (1 000 000 rows) — generating CSV... ===")
        t0 = time.perf_counter()
        cls.path = _write_csv(_make_large_df(1_000_000))
        log.info("  CSV written in %.1fs (%s)",
                 time.perf_counter() - t0,
                 cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.path.unlink(missing_ok=True)

    def test_exactly_one_groq_call(self):
        log.info("--- LARGE: asserting exactly 1 Groq call ---")

        with patch("groq_cleaner._GroqClient") as MockClient:
            mock_inst = MagicMock()
            MockClient.return_value = mock_inst
            mock_inst.chat.completions.create.return_value = _make_response(VALID_JSON)

            t0 = time.perf_counter()
            clean_df, rules, schema = groq_cleaner.clean_large_file(
                self.path, api_key="test-key"
            )
            elapsed = time.perf_counter() - t0

        call_count = mock_inst.chat.completions.create.call_count
        log.info("  Groq calls: %d  (expected 1)", call_count)
        self.assertEqual(call_count, 1, "Must make exactly 1 Groq call regardless of file size")

        log.info("  Total pipeline time: %.2fs", elapsed)
        log.info("  clean_df: %d rows × %d cols", len(clean_df), len(clean_df.columns))

        # Row count correct (1M rows minus none — rules drop "id" column only)
        self.assertEqual(len(clean_df), 1_000_000)

        # Pipeline should complete in reasonable time (< 60s)
        self.assertLess(elapsed, 60, "Pipeline took longer than 60s on 1M rows")
        log.info("  Timing < 60s: OK (%.2fs)", elapsed)

    def test_memory_bounded(self):
        log.info("--- LARGE: memory stays bounded (chunked reading) ---")

        with patch("groq_cleaner._GroqClient") as MockClient:
            mock_inst = MagicMock()
            MockClient.return_value = mock_inst
            mock_inst.chat.completions.create.return_value = _make_response(VALID_JSON)

            tracemalloc.start()
            groq_cleaner.clean_large_file(self.path, api_key="test-key")
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        log.info("  Peak memory during pipeline: %.1f MB", peak_mb)
        # Chunked reading should keep the cleaning stage well under 2 GB
        self.assertLess(peak_mb, 2048, f"Peak memory {peak_mb:.0f} MB exceeds 2 GB limit")
        log.info("  Memory bounded: OK (%.0f MB peak)", peak_mb)


class TestMessyData(unittest.TestCase):
    """Test 3 — MESSY: mixed types, high missing, ID column."""

    def setUp(self):
        self.path = _write_csv(_make_messy_df(500))

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_messy_with_groq_rules(self):
        log.info("=== TEST: MESSY data ===")

        messy_rules = {
            "drop":   ["cust_id", "useless"],  # ID + 100% missing
            "rename": {},
            "cast":   {"revenue": "float64", "age": "float64"},
            "impute": {"score": "median"},
            "target": "category",
        }

        with patch("groq_cleaner._GroqClient") as MockClient:
            mock_inst = MagicMock()
            MockClient.return_value = mock_inst
            mock_inst.chat.completions.create.return_value = _make_response(
                json.dumps(messy_rules)
            )
            clean_df, rules, schema = groq_cleaner.clean_large_file(
                self.path, api_key="test-key"
            )

        # ID column dropped
        self.assertNotIn("cust_id", clean_df.columns, "cust_id (ID) must be dropped")
        # Fully-missing column dropped
        self.assertNotIn("useless", clean_df.columns, "useless (100% missing) must be dropped")
        # Revenue cast to numeric (not strings like "$1,200.50")
        self.assertTrue(pd.api.types.is_numeric_dtype(clean_df["revenue"]),
                        "revenue must be cast to numeric")
        # No missing values in score after impute
        self.assertEqual(clean_df["score"].isna().sum(), 0,
                         "score must have no NaN after median impute")

        log.info("  Drop ID: OK | Drop missing: OK | Cast numeric: OK | Impute: OK")
        log.info("  clean_df: %d rows × %d cols", len(clean_df), len(clean_df.columns))

    def test_messy_fallback_to_safe_defaults(self):
        log.info("=== TEST: MESSY → safe defaults (no API key) ===")
        clean_df, rules, schema = groq_cleaner.clean_large_file(
            self.path, api_key=""
        )
        self.assertIsInstance(clean_df, pd.DataFrame)
        self.assertGreater(len(clean_df), 0)
        # useless column (100% missing) must still be dropped by safe defaults
        self.assertNotIn("useless", clean_df.columns,
                         "Safe defaults must drop 100%-missing column")
        log.info("  safe_defaults dropped 100%%-missing col: OK")


class TestBadLLMResponse(unittest.TestCase):
    """Test 4 — BAD-LLM: malformed JSON → retry once; both bad → safe defaults."""

    def setUp(self):
        self.path = _write_csv(_make_small_df(200))

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_retry_once_then_succeed(self):
        log.info("=== TEST: BAD-LLM (malformed first, valid second) ===")

        call_count = {"n": 0}

        def _side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_response("NOT VALID JSON {{{")  # malformed
            return _make_response(VALID_JSON)               # valid on retry

        with patch("groq_cleaner._GroqClient") as MockClient:
            mock_inst = MagicMock()
            MockClient.return_value = mock_inst
            mock_inst.chat.completions.create.side_effect = _side_effect

            clean_df, rules, schema = groq_cleaner.clean_large_file(
                self.path, api_key="test-key"
            )

        self.assertEqual(call_count["n"], 2, "Must retry exactly once (2 total calls)")
        self.assertIsInstance(clean_df, pd.DataFrame)
        self.assertGreater(len(clean_df), 0)
        log.info("  2 Groq calls (1 retry): OK | Pipeline completed: OK")

    def test_both_bad_falls_back_to_safe_defaults(self):
        log.info("=== TEST: BAD-LLM (both responses malformed → safe defaults) ===")

        with patch("groq_cleaner._GroqClient") as MockClient:
            mock_inst = MagicMock()
            MockClient.return_value = mock_inst
            mock_inst.chat.completions.create.return_value = _make_response(
                "this is not json at all"
            )

            clean_df, rules, schema = groq_cleaner.clean_large_file(
                self.path, api_key="test-key"
            )

        # 2 attempts made, then safe defaults
        self.assertEqual(mock_inst.chat.completions.create.call_count, 2,
                         "Must attempt exactly 2 calls before falling back")
        # Pipeline must still produce a valid DataFrame (never stalls)
        self.assertIsInstance(clean_df, pd.DataFrame)
        self.assertGreater(len(clean_df), 0)
        log.info("  2 Groq calls → safe defaults → pipeline completed: OK")

    def test_no_api_key_uses_safe_defaults_immediately(self):
        log.info("=== TEST: No API key → immediate safe defaults (0 Groq calls) ===")

        with patch("groq_cleaner._GroqClient") as MockClient:
            clean_df, rules, schema = groq_cleaner.clean_large_file(
                self.path, api_key=""
            )
            MockClient.assert_not_called()

        self.assertIsInstance(clean_df, pd.DataFrame)
        log.info("  0 Groq calls (no key): OK")


class TestValidateRules(unittest.TestCase):
    """Unit-test _validate_rules directly."""

    def test_valid(self):
        self.assertTrue(groq_cleaner._validate_rules(VALID_RULES))

    def test_missing_key(self):
        bad = {k: v for k, v in VALID_RULES.items() if k != "target"}
        self.assertFalse(groq_cleaner._validate_rules(bad))

    def test_wrong_type_drop(self):
        bad = dict(VALID_RULES, drop="not-a-list")
        self.assertFalse(groq_cleaner._validate_rules(bad))

    def test_not_a_dict(self):
        self.assertFalse(groq_cleaner._validate_rules("just a string"))


# ===========================================================================
# Main runner — prints a clear summary
# ===========================================================================

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  groq_cleaner Scalability Validation Suite")
    log.info("=" * 60)

    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    for cls in [
        TestDependencies,
        TestBuildProfile,
        TestSmallPipeline,
        TestLargePipeline,
        TestMessyData,
        TestBadLLMResponse,
        TestValidateRules,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    log.info("=" * 60)
    if result.wasSuccessful():
        log.info("  ALL TESTS PASSED (%d tests)", result.testsRun)
    else:
        log.info("  FAILURES: %d  ERRORS: %d  (of %d tests)",
                 len(result.failures), len(result.errors), result.testsRun)
        sys.exit(1)
    log.info("=" * 60)
