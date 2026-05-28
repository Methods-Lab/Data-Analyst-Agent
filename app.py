"""
Streamlit app for a data-trained analyst chatbot.

The app accepts structured data (CSV, Excel, JSON) and lightweight text data,
trains local sklearn models when a usable target exists, and answers questions
as a data analyst grounded in the active dataset. When no dataset is provided,
it behaves as a general analyst assistant and invites the user to attach data.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import traceback as _tb
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import data_loader as dl
import groq_cleaner as _gc
import reasoner
import trainer
from abduction import try_bayesian_abduce

# ---------------------------------------------------------------------------
# Groq API key — used in background for data structuring and suggestions.
# Set GROQ_API_KEY in Railway environment variables (or .env for local dev).
# ---------------------------------------------------------------------------
_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_GROQ_MODEL   = "llama-3.3-70b-versatile"

SAMPLE_ROWS = 10_000


st.set_page_config(
    page_title="Analyst ML Chatbot",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg:          #F7F3EC;
    --surface:     #FFFFFF;
    --surface-2:   #FBF8F2;
    --border:      #ECE4D6;
    --ink:         #2B2A26;
    --muted:       #6F6A60;
    --coral:       #E07856;
    --coral-dark:  #D86848;
    --coral-soft:  #FCE5DC;
    --sage:        #7FA98B;
    --sage-soft:   #E3EEDF;
    --honey:       #D9A85C;
    --honey-soft:  #F8EBCF;
    --shadow:      0 1px 2px rgba(43,42,38,.04), 0 8px 24px rgba(43,42,38,.06);
}

html, body, [class*="css"], .stApp, [data-testid="stMarkdownContainer"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--ink) !important;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%,  rgba(224,120,86,.07), transparent 40%),
        radial-gradient(circle at 95% 30%, rgba(127,169,139,.06), transparent 38%),
        var(--bg) !important;
}

.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1640px;
}

/* ===== Sidebar styled as the left "ribbon" ===== */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 18px;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: var(--ink) !important;
    font-weight: 700 !important;
    letter-spacing: -.01em !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption {
    color: var(--muted) !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    opacity: .8;
}

/* All text default to warm charcoal */
h1, h2, h3, h4, h5, h6, p, li, label, span, div {
    color: var(--ink);
}
.stMarkdown, .stCaption { color: var(--ink) !important; }
small, .stCaption { color: var(--muted) !important; }

/* ===== Header card (file info + status chips) ===== */
.app-head {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: var(--shadow);
}
.app-head-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
}
.app-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -.01em;
}
.status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.status-pill {
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 5px 11px;
    background: var(--surface-2);
    color: var(--muted);
    font-size: .76rem;
    font-weight: 500;
}

/* ===== Metric cards ===== */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 4px 0 16px;
}
.metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: var(--shadow);
    position: relative;
}
.metric .ico {
    width: 30px; height: 30px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700;
    margin-bottom: 8px;
}
.metric.coral   .ico { background: var(--coral-soft);  color: var(--coral); }
.metric.sage    .ico { background: var(--sage-soft);   color: #4F7E5C; }
.metric.honey   .ico { background: var(--honey-soft);  color: #8A6A2E; }
.metric.neutral .ico { background: #EDE7DD;            color: var(--muted); }
.metric .value {
    display: block;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -.02em;
}
.metric .label {
    display: block;
    margin-top: 2px;
    color: var(--muted);
    font-size: .68rem;
    text-transform: uppercase;
    letter-spacing: .06em;
    font-weight: 600;
}

/* ===== Chat panel ===== */
.chat-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 14px;
    min-height: 600px;
    box-shadow: var(--shadow);
}
.chat-header {
    display: flex; align-items: center; gap: 10px;
    padding: 4px 6px 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 10px;
}
.chat-avatar {
    width: 34px; height: 34px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--sage), var(--honey));
    color: white;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 14px;
    box-shadow: 0 4px 10px rgba(127,169,139,.3);
}
.chat-header .title { font-size: 13px; font-weight: 700; color: var(--ink); }
.chat-header .sub   { font-size: 11px; color: var(--muted); display: flex; align-items: center; gap: 4px; }
.chat-header .dot   { width: 6px; height: 6px; border-radius: 50%; background: var(--sage); }

/* ===== Chat messages ===== */
div[data-testid="stChatMessage"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 10px 13px !important;
    margin: 8px 0 !important;
    background: var(--surface) !important;
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] strong {
    color: var(--ink) !important;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--coral) !important;
    border-color: var(--coral-dark) !important;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) * {
    color: #FFFFFF !important;
}

/* ===== Chat input ===== */
div[data-testid="stChatInput"] {
    background: transparent !important;
}
div[data-testid="stChatInput"] textarea {
    color: var(--ink) !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
}

/* ===== Buttons ===== */
.stButton > button {
    border-radius: 11px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--ink);
    font-weight: 600;
    transition: transform 120ms ease, box-shadow 150ms ease, background 150ms ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    background: var(--coral-soft);
    border-color: #F2C8B7;
    color: var(--coral);
}
/* Primary buttons (sidebar Train Agent) — use first button in sidebar */
[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: linear-gradient(135deg, var(--coral), var(--coral-dark));
    color: white !important;
    border-color: var(--coral-dark);
    box-shadow: 0 6px 14px rgba(224,120,86,.28);
}
[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
    background: linear-gradient(135deg, #E58668, #DE7253);
    color: white !important;
}

/* Sliders */
[data-testid="stSlider"] [role="slider"] {
    background: white !important;
    border: 2px solid var(--coral) !important;
}
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, var(--coral), var(--honey)) !important;
}

/* Inputs */
input, textarea, [data-testid="stTextInput"] input {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    border-radius: 9px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface-2);
    border: 2px dashed #DCD2C2;
    border-radius: 14px;
    padding: 8px;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] button {
    background: var(--coral) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span { color: var(--muted) !important; }

/* Radio buttons */
[data-testid="stRadio"] label {
    color: var(--ink) !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--border);
}
div[data-testid="stAlert"][data-baseweb="notification"] {
    background: var(--sage-soft) !important;
    color: #2B2A26 !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: var(--muted) !important;
    border-bottom-color: var(--border) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--coral) !important;
    border-bottom-color: var(--coral) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: var(--shadow);
}
[data-testid="stExpander"] summary {
    color: var(--ink) !important;
    font-weight: 600;
}

/* Headings */
h1, h2, h3, h4 {
    letter-spacing: -.01em;
}

.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 10px 0 6px;
}

/* Pinned section heading */
h4 { font-size: .85rem !important; color: var(--muted) !important; text-transform: uppercase; letter-spacing: .08em; }

/* Pinned buttons styled as rounded pills with a colored dot before text */
.pin-row .stButton > button {
    border-radius: 999px !important;
    padding: 8px 16px !important;
    font-size: 12.5px !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    position: relative;
    padding-left: 26px !important;
    text-align: left;
}
.pin-row .stButton > button::before {
    content: "";
    position: absolute;
    left: 13px; top: 50%;
    transform: translateY(-50%);
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--coral);
}
.pin-row .stButton:nth-of-type(1) > button::before { background: var(--coral); }
.pin-row .stButton:nth-of-type(2) > button::before { background: var(--sage); }
.pin-row .stButton:nth-of-type(3) > button::before { background: var(--honey); }
.pin-row .stButton:nth-of-type(4) > button::before { background: var(--coral); }
.pin-row .stButton:nth-of-type(5) > button::before { background: var(--sage); }
.pin-row .stButton:nth-of-type(6) > button::before { background: var(--honey); }

/* Chat: AI message bubble = white card, user bubble = coral right-aligned */
div[data-testid="stChatMessageContent"] {
    color: var(--ink) !important;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
    text-align: left !important;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: var(--coral) !important;
    border-radius: 13px 13px 4px 13px !important;
    padding: 10px 13px !important;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] * {
    color: #FFFFFF !important;
}

/* Quick-chip suggestions below input */
.quick-chips {
    display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px;
}

/* Inline send button (➤) inside chat form */
[data-testid="stForm"] .stButton > button {
    background: var(--coral) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 3px 8px rgba(224,120,86,.3);
}
[data-testid="stForm"] .stButton > button:hover {
    background: var(--coral-dark) !important;
    color: white !important;
}

/* Hidden marker that flags the chat column */
.chat-card-marker { display: none; }

/* Right chat column gets a white card surface so it stands out */
[data-testid="stColumn"]:has(> div > .chat-card-marker),
[data-testid="column"]:has(> div > .chat-card-marker) {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 16px !important;
    box-shadow: var(--shadow);
}

@media (max-width: 900px) {
    .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "df_raw":          None,
        "df_clean":        None,
        "train_result":    None,
        "target_col":      None,
        "chat":            [],
        "text_corpus":     None,
        "text_vectors":    None,
        "text_vectorizer": None,
        "data_profile":    None,
        "source_filename": None,
        "source_sheet_count": 1,
        "trained_at":      None,
        "recent_files":    [],     # list of dicts: {name, size_kb, ext, trained}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_recent_file(name: str, size_bytes: int) -> None:
    """Track uploaded/trained files for the Recent Files sidebar list."""
    ext = name.split(".")[-1].lower() if "." in name else "csv"
    entry = {
        "name":    name,
        "size_kb": round(size_bytes / 1024, 1),
        "ext":     ext.upper()[:3],
        "trained": True,
    }
    existing = st.session_state.recent_files
    existing = [f for f in existing if f["name"] != name]
    st.session_state.recent_files = [entry] + existing[:4]   # keep last 5


def get_excel_sheets(uploaded_file) -> list[str]:
    """Return sheet names from an uploaded Excel file without consuming the stream."""
    try:
        xf = pd.ExcelFile(io.BytesIO(uploaded_file.getvalue()), engine="openpyxl")
        return xf.sheet_names
    except Exception:
        return []


def read_excel_sheet(source, sheet_name) -> pd.DataFrame:
    """Read a specific sheet from an Excel path or uploaded file."""
    if isinstance(source, (str, Path)):
        return pd.read_excel(source, sheet_name=sheet_name, engine="openpyxl")
    return pd.read_excel(io.BytesIO(source.getvalue()), sheet_name=sheet_name, engine="openpyxl")


def read_local_file(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".csv"}:
        return dl.load_data(path)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict):
            return pd.json_normalize(raw)
    if suffix == ".txt":
        return text_to_frame(path.read_text(encoding="utf-8", errors="ignore"))
    return dl.load_data(path)


def text_to_frame(text: str) -> pd.DataFrame:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if chunk.strip()]
    if not chunks:
        chunks = ["No readable text was provided."]
    return pd.DataFrame(
        {
            "document_id": range(1, len(chunks) + 1),
            "content": chunks,
            "char_count": [len(chunk) for chunk in chunks],
            "word_count": [len(chunk.split()) for chunk in chunks],
        }
    )


def detect_target(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested

    names = {c.lower(): c for c in df.columns}
    for candidate in ["sales_category", "target", "label", "class", "outcome", "status", "category"]:
        if candidate in names:
            return names[candidate]

    for candidate in ["net_sales", "sales", "revenue", "amount", "profit", "total"]:
        if candidate in names and pd.api.types.is_numeric_dtype(df[names[candidate]]):
            return f"{names[candidate]}_tier"

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return f"{numeric_cols[0]}_tier"

    object_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= min(20, max(2, len(df) // 3))]
    if object_cols:
        return object_cols[0]

    return "analyst_cluster"


def ensure_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    working = df.copy()
    if target_col in working.columns and working[target_col].nunique(dropna=True) >= 2:
        return working

    base_name = target_col.replace("_tier", "")
    if base_name in working.columns and pd.api.types.is_numeric_dtype(working[base_name]):
        series = working[base_name].fillna(working[base_name].median())
    else:
        numeric_cols = [c for c in working.columns if pd.api.types.is_numeric_dtype(working[c])]
        if numeric_cols:
            base_name = numeric_cols[0]
            series = working[base_name].fillna(working[base_name].median())
        else:
            working["text_signal"] = working.astype(str).agg(" ".join, axis=1).str.len()
            base_name = "text_signal"
            series = working[base_name]

    low_q = series.quantile(0.33)
    high_q = series.quantile(0.67)
    if low_q == high_q:
        working[target_col] = np.where(series >= series.median(), "high", "low")
    else:
        working[target_col] = pd.cut(
            series,
            bins=[-np.inf, low_q, high_q, np.inf],
            labels=["low", "medium", "high"],
        ).astype(str)
    return working


def build_text_index(df: pd.DataFrame) -> tuple[list[str] | None, TfidfVectorizer | None, Any | None]:
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not text_cols:
        return None, None, None

    rows = df[text_cols].fillna("").astype(str).agg(" | ".join, axis=1).tolist()
    rows = [row for row in rows if row.strip()]
    if not rows:
        return None, None, None

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2))
        vectors = vectorizer.fit_transform(rows)
        return rows, vectorizer, vectors
    except ValueError:
        return None, None, None


def profile_data(df: pd.DataFrame, target_col: str | None) -> dict[str, Any]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    missing = df.isna().sum().sort_values(ascending=False)
    top_missing = missing[missing > 0].head(8).to_dict()

    profile: dict[str, Any] = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric": numeric,
        "categorical": categorical,
        "missing": top_missing,
        "target": target_col,
    }

    if target_col and target_col in df.columns:
        profile["target_distribution"] = df[target_col].astype(str).value_counts().head(8).to_dict()

    if numeric:
        stats = df[numeric].describe().T
        profile["numeric_stats"] = stats[["mean", "std", "min", "max"]].round(3).to_dict("index")

    return profile


def parse_facts(prompt: str, feature_names: list[str]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for feature in feature_names:
        pattern = rf"{re.escape(feature)}\s*(?:=|is|:)\s*([A-Za-z0-9_.-]+)"
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            value = match.group(1)
            try:
                facts[feature] = float(value)
            except ValueError:
                facts[feature] = value
    return facts


def compact_table(items: dict[str, Any], limit: int = 8) -> str:
    if not items:
        return "None detected."
    lines = []
    for key, value in list(items.items())[:limit]:
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)



def dataset_context_for_llm(prompt: str) -> str:
    """
    Build a rich, data-grounded context string for Groq so it can answer
    ANY question about the uploaded dataset — not just keyword-matched ones.
    Includes actual row samples, per-column breakdowns, real percentages,
    and feature importance rankings.
    """
    df     = st.session_state.df_raw
    result = st.session_state.train_result or {}
    target = st.session_state.target_col

    if df is None:
        return "No dataset loaded yet."

    lines: list[str] = []

    # File identity
    fname = st.session_state.get("source_filename") or "uploaded file"
    lines.append(f"File: {fname}")
    lines.append(f"Size: {len(df):,} rows × {len(df.columns)} columns")
    lines.append(f"All columns: {list(df.columns)}")

    # Per-column breakdown with real counts and percentages
    lines.append("\nColumn details:")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].describe()
            lines.append(
                f"  '{col}' (numeric) — "
                f"min={s['min']:.2f}, max={s['max']:.2f}, "
                f"avg={s['mean']:.2f}, {df[col].nunique()} unique values"
            )
        else:
            vc  = df[col].value_counts()
            pct = (vc / len(df) * 100).round(1)
            top = {str(k): f"{v:,} rows ({pct[k]}%)" for k, v in vc.head(6).items()}
            lines.append(f"  '{col}' (text/category) — {df[col].nunique()} unique, top values: {top}")

    # Target distribution with percentages
    if target and target in df.columns:
        vc  = df[target].value_counts()
        pct = (vc / len(df) * 100).round(1)
        dist = {str(k): f"{v:,} ({pct[k]}%)" for k, v in vc.items()}
        lines.append(f"\nPrediction target '{target}': {dist}")

    # Most impactful columns ranked (SHAP or permutation importance)
    if Path("feature_importances.pkl").exists():
        try:
            imp_data = trainer.load_importances()
            imp      = imp_data.get("shap_importances") or imp_data.get("importances", {})
            if imp:
                ranked   = sorted(imp.items(), key=lambda x: x[1], reverse=True)
                top_cols = [f"#{i+1}: {c}" for i, (c, _) in enumerate(ranked[:8])]
                lines.append(f"\nMost impactful columns (ranked): {', '.join(top_cols)}")
        except Exception:
            pass

    # Model accuracy
    if result:
        lines.append(f"Prediction accuracy: {result.get('rf_accuracy', 0):.1%}")
        cv = result.get("cv_mean_accuracy")
        if cv:
            lines.append(f"Cross-validation accuracy: {cv:.1%}")

    # Missing values
    missing = df.isna().sum()
    missing_d = {c: int(v) for c, v in missing[missing > 0].items()}
    lines.append(f"Missing values: {missing_d if missing_d else 'none'}")

    # Actual data sample so Groq can see real values
    sample_n = min(20, len(df))
    lines.append(f"\nSample data ({sample_n} rows):\n{df.head(sample_n).to_csv(index=False)}")

    return "\n".join(lines)


def _normalize_bullets(text: str) -> str:
    """
    Force bullets onto separate lines, in case the LLM concatenates them
    inline like 'X. - Y. - Z.' instead of newline-separated.
    """
    if not text:
        return text
    lines  = text.split("\n")
    output = []
    for line in lines:
        stripped = line.strip()
        # Inline-bullet detection: contains ' - ' at least twice but isn't already a bullet line
        if stripped.count(" - ") >= 2 and not stripped.startswith("-"):
            parts = re.split(r"\s+-\s+", stripped)
            head  = parts[0].strip()
            if head:
                output.append(head)
                output.append("")
            for p in parts[1:]:
                p = p.strip().rstrip(".") + "."
                if p and p != ".":
                    output.append(f"- {p}")
        else:
            output.append(line)
    return "\n".join(output)


def call_external_ai(prompt: str, local_ml_answer: str) -> str | None:
    """
    Groq acts as the PRIMARY responder, answering the user's EXACT question
    using the actual uploaded data as context.  Returns None on failure so
    the local ML answer is shown as fallback.
    """
    if not _GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        has_data = st.session_state.df_raw is not None

        if has_data:
            data_context = dataset_context_for_llm(prompt)
            system_content = (
                "You are an AI data analyst with FULL access to the user's uploaded dataset. "
                "Answer ANY question about the data.\n\n"
                "OUTPUT FORMAT — this is critical, follow EXACTLY:\n\n"
                "For data-analysis questions, write the response as MARKDOWN with this exact structure:\n\n"
                "**One bold headline sentence here.**\n\n"
                "- First bullet on its own line\n"
                "- Second bullet on its own line\n"
                "- Third bullet on its own line\n"
                "- Fourth bullet on its own line\n\n"
                "Rules:\n"
                "1. Put a REAL NEWLINE between each bullet — NEVER put bullets on the same line.\n"
                "2. Put a BLANK LINE between the headline and the first bullet.\n"
                "3. Each bullet starts with '- ' (dash space) and is max 15 words.\n"
                "4. Each bullet must contain a specific number, column name, or percentage from the data.\n"
                "5. Total 3 to 4 bullets only.\n\n"
                "For conversational messages (hi, thanks, how are you): one natural sentence, no bullets.\n"
                "For simple lookups (file name, row count): one clear sentence, no bullets.\n\n"
                "Content: use real column names + real numbers from the data. Plain English. "
                "Maximum 140 words total."
            )
            user_content = (
                f"User's question: \"{prompt}\"\n\n"
                f"Full dataset (use this to answer):\n{data_context}"
            )
        else:
            system_content = (
                "You are a friendly AI data analyst. No dataset is uploaded yet. "
                "Reply conversationally in 1-2 sentences and invite the user to upload a file."
            )
            user_content = f"User said: \"{prompt}\""

        client = Groq(api_key=_GROQ_API_KEY)
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        return _normalize_bullets(raw)
    except Exception:
        return None


def answer_without_data(prompt: str) -> str:
    return (
        "I can help as a data analyst right now, but I will become much sharper after you upload "
        "CSV, Excel, JSON, or text data. Ask me for analysis plans, KPI definitions, chart ideas, "
        "data cleaning steps, experiment design, forecasting strategy, or sales diagnosis. Once "
        "data is loaded, I will train local machine-learning models and answer from the evidence."
    )


def answer_with_text_retrieval(prompt: str) -> str | None:
    rows = st.session_state.text_corpus
    vectorizer = st.session_state.text_vectorizer
    vectors = st.session_state.text_vectors
    if rows is None or vectorizer is None or vectors is None:
        return None

    query_vec = vectorizer.transform([prompt])
    scores = cosine_similarity(query_vec, vectors).ravel()
    best_idx = scores.argsort()[::-1][:4]
    if len(best_idx) == 0 or scores[best_idx[0]] < 0.04:
        return None

    evidence = "\n".join(
        f"- Match {i + 1} ({scores[idx]:.0%} relevance): {rows[idx][:420]}"
        for i, idx in enumerate(best_idx)
    )
    return (
        "I found the closest evidence in your text data:\n\n"
        f"{evidence}\n\n"
        "Analyst read: the strongest answer should be based on the first match, with the later "
        "matches used as supporting context. If you ask a more specific follow-up, I can narrow it."
    )


def answer_with_data(prompt: str) -> str:
    df = st.session_state.df_raw
    profile = st.session_state.data_profile or {}
    train_result = st.session_state.train_result
    target_col = st.session_state.target_col
    low_prompt = prompt.lower()

    if df is None:
        return answer_without_data(prompt)

    retrieval = answer_with_text_retrieval(prompt)
    if retrieval and any(word in low_prompt for word in ["document", "text", "say", "mention", "find", "what is"]):
        return retrieval

    if any(word in low_prompt for word in ["overview", "summary", "describe", "profile", "dataset"]):
        return (
            f"Dataset overview:\n\n"
            f"- Rows: {profile.get('rows', len(df)):,}\n"
            f"- Columns: {profile.get('columns', len(df.columns)):,}\n"
            f"- Numeric columns: {len(profile.get('numeric', []))}\n"
            f"- Categorical/text columns: {len(profile.get('categorical', []))}\n"
            f"- Active target: {target_col or 'not selected'}\n\n"
            f"Target distribution:\n{compact_table(profile.get('target_distribution', {}))}\n\n"
            f"Missing values:\n{compact_table(profile.get('missing', {}))}"
        )

    if any(word in low_prompt for word in ["missing", "null", "clean", "quality"]):
        return (
            "Data quality scan:\n\n"
            f"{compact_table(profile.get('missing', {}), limit=10)}\n\n"
            "Recommended next actions: impute numeric gaps with median values, fill categorical gaps "
            "with the most common label or 'Unknown', and review high-cardinality ID columns before training."
        )

    if any(word in low_prompt for word in ["important", "importance", "driver", "drivers", "influence"]):
        if train_result and Path("feature_importances.pkl").exists():
            imp_data = trainer.load_importances()
            ranked = sorted(imp_data["importances"].items(), key=lambda item: item[1], reverse=True)[:8]
            lines = [f"- {feature}: {score:.4f}" for feature, score in ranked]
            return (
                "Top model drivers from permutation importance:\n\n"
                + "\n".join(lines)
                + "\n\nThese are the fields that most changed model accuracy when shuffled, so they are "
                "the best first suspects for business investigation."
            )
        return "I need a trained model before I can rank drivers. Load data and click Train agent."

    if any(word in low_prompt for word in ["why", "cause", "explain"]) and train_result:
        classes = train_result["class_names"]
        observed = next((c for c in classes if c.lower() in low_prompt), classes[-1])
        try:
            imp_result = reasoner.abduce(observed, top_k=5)
            bayes = None
            if st.session_state.df_clean is not None:
                bayes = try_bayesian_abduce(st.session_state.df_clean, observed, target_col, top_k=3)

            lines = [
                f"Best explanation for outcome '{observed}':",
                "",
                *[
                    f"- {c['feature']}: importance {c['importance']:.4f}. {c['direction_hint']}."
                    for c in imp_result["causes"]
                ],
            ]
            if bayes:
                lines.extend(["", "Bayesian posterior hints:"])
                lines.extend(
                    f"- {b['feature']} around {b['feature_value_range']}: score {b['posterior_score']:.4f}"
                    for b in bayes
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"I tried to explain that outcome, but the model could not complete abduction: {exc}"

    if any(word in low_prompt for word in ["predict", "classify", "if "]) and train_result:
        facts = parse_facts(prompt, train_result["feature_names"])
        if not facts:
            example_features = ", ".join(train_result["feature_names"][:5])
            return (
                "I can predict from feature facts. Try a message like "
                f"`predict if {example_features.split(', ')[0]}=10 and {example_features.split(', ')[-1]}=3`. "
                "Use exact column names from your data."
            )
        results = reasoner.deduce(facts, top_k=3)
        if not results:
            return "No rule matched those facts. Try fewer fields or values closer to the training data."
        top = results[0]
        return (
            f"Prediction: {top['prediction']} with {top['confidence']:.0%} rule confidence "
            f"and {top['match_score']:.0%} fact match.\n\n"
            "Supporting rule conditions:\n"
            + "\n".join(f"- {condition}" for condition in top["conditions"][:8])
        )

    numeric_cols = profile.get("numeric", [])
    mentioned_numeric = [col for col in numeric_cols if col.lower() in low_prompt]
    if mentioned_numeric:
        col = mentioned_numeric[0]
        stats = df[col].describe()
        return (
            f"Column read for '{col}':\n\n"
            f"- Mean: {stats['mean']:.3f}\n"
            f"- Median: {stats['50%']:.3f}\n"
            f"- Minimum: {stats['min']:.3f}\n"
            f"- Maximum: {stats['max']:.3f}\n\n"
            "This is a quick statistical answer. Ask for drivers or causes if you want model reasoning."
        )

    return (
        "Here is the analyst read from the active data: start with the dataset profile, inspect missing "
        "values, then use the trained model drivers to focus the business story. I can answer specific "
        "questions like `summarize this dataset`, `what are the top drivers`, `why high`, "
        "`predict if feature=value`, or `show missing values`."
    )


# ML responses that mean "I have nothing specific to say"
_ML_GENERIC = (
    "Here is the analyst read from the active data",
    "I can help as a data analyst right now",
)

# Sage-coloured AI ANALYSIS label rendered inline inside the chat bubble
_AI_LABEL = (
    '<div style="display:inline-flex;align-items:center;gap:5px;'
    'font-size:10px;color:#7FA98B;font-weight:700;text-transform:uppercase;'
    'letter-spacing:.10em;margin-bottom:6px;">✦ AI Analysis</div>'
)
_AI_EXPLANATION_LABEL = (
    '<div style="display:inline-flex;align-items:center;gap:5px;'
    'font-size:10px;color:#7FA98B;font-weight:700;text-transform:uppercase;'
    'letter-spacing:.10em;margin:10px 0 6px;">✦ AI Explanation</div>'
)

def generate_response(prompt: str) -> str:
    """
    Produce a chat-bubble-ready response:
      - structured ML queries: ML result + thin divider + 'AI Explanation' label + Groq
      - open-ended: just sage 'AI Analysis' label + Groq's direct answer
      - Groq unavailable: ML answer only
    """
    ml_answer   = answer_with_data(prompt)
    groq_answer = call_external_ai(prompt, ml_answer)

    ml_useful = not any(ml_answer.startswith(p) for p in _ML_GENERIC)

    if groq_answer:
        if ml_useful:
            return (
                f"{ml_answer}\n\n"
                f"{_AI_EXPLANATION_LABEL}\n"
                f"{groq_answer}"
            )
        else:
            return f"{_AI_LABEL}\n\n{groq_answer}"

    return ml_answer


def train_agent(df: pd.DataFrame, target_request: str, tree_depth: int, n_estimators: int) -> None:
    # Strip Excel artifact columns before anything else
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if df.empty or len(df.columns) == 0:
        raise ValueError("The uploaded file has no usable columns after removing unnamed/empty columns.")

    target_col = detect_target(df, target_request.strip() or None)
    model_df = ensure_target(df, target_col)
    clean_df = dl.clean_data(model_df, target_col=target_col)

    if clean_df[target_col].nunique(dropna=True) < 2:
        raise ValueError("The target has fewer than two classes after cleaning.")

    result = trainer.train(
        clean_df,
        target_col=target_col,
        max_tree_depth=tree_depth,
        n_estimators=n_estimators,
    )

    corpus, vectorizer, vectors = build_text_index(df)

    st.session_state.df_raw = model_df
    st.session_state.df_clean = clean_df
    st.session_state.train_result = result
    st.session_state.target_col = target_col
    st.session_state.text_corpus = corpus
    st.session_state.text_vectorizer = vectorizer
    st.session_state.text_vectors = vectors
    st.session_state.data_profile = profile_data(model_df, target_col)

    st.session_state.chat.append(
        {
            "role": "assistant",
            "content": (
                f"Data loaded and model trained. I learned {len(model_df):,} rows, "
                f"{len(model_df.columns):,} columns, and I am using `{target_col}` as the outcome. "
                "Ask me for a summary, drivers, causes, predictions, missing values, or document evidence."
            ),
        }
    )


def render_metrics() -> None:
    profile = st.session_state.data_profile or {}
    result  = st.session_state.train_result or {}
    cv_mean = result.get("cv_mean_accuracy", result.get("rf_accuracy", 0))
    cv_std  = result.get("cv_std", 0.0)
    cv_label = (
        f"{cv_mean:.0%} ±{cv_std:.2f}"
        if cv_std > 0
        else f"{cv_mean:.0%}"
    )
    shap_badge = "SHAP ✓" if result.get("shap_available") else "Perm. Imp."
    st.markdown(
        f"""
<div class="metric-grid">
    <div class="metric coral"><div class="ico">↗</div><span class="value">{profile.get('rows', 0):,}</span><span class="label">Rows Learned</span></div>
    <div class="metric sage"><div class="ico">◆</div><span class="value">{profile.get('columns', 0):,}</span><span class="label">Signals</span></div>
    <div class="metric honey"><div class="ico">✦</div><span class="value">{cv_label}</span><span class="label">CV Accuracy</span></div>
    <div class="metric neutral"><div class="ico">∎</div><span class="value">{result.get('n_rules', 0):,}</span><span class="label">Rules · {shap_badge}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Visualization dialog — purely ML-library driven (plotly), no AI
# ---------------------------------------------------------------------------

@st.dialog("Visualize your data", width="large")
def visualize_dialog() -> None:
    """Render correlation-based graphs on numeric columns using plotly."""
    import plotly.express as px

    df = st.session_state.df_raw
    if df is None:
        st.warning("Train the agent on a file first.")
        return

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.warning("Need at least 2 numeric columns to visualize correlations.")
        return

    # Brand palette so charts match the rest of the UI
    coral_scale = [
        [0.00, "#7FA98B"],   # sage  (strong negative)
        [0.25, "#C8DEC6"],   # sage soft
        [0.50, "#FBF8F2"],   # cream (no correlation)
        [0.75, "#F8E1D2"],   # coral soft
        [1.00, "#E07856"],   # coral (strong positive)
    ]

    graph_type = st.radio(
        "Choose graph type",
        ["Correlation Heatmap", "Scatter Plot (top correlation)", "Distribution Histogram"],
        horizontal=True,
    )

    fig_layout = dict(
        font=dict(family="Inter, sans-serif", color="#2B2A26", size=12),
        paper_bgcolor="#FBF8F2",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=50, b=40),
    )

    if graph_type == "Correlation Heatmap":
        corr = numeric_df.corr().round(2)
        fig  = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=coral_scale,
            zmin=-1, zmax=1,
            title="Correlation between numeric columns",
        )
        fig.update_layout(**fig_layout)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Values near **+1** (coral) mean two columns move together; "
            "values near **−1** (sage) mean they move opposite; "
            "values near **0** mean little relationship."
        )

    elif graph_type == "Scatter Plot (top correlation)":
        corr = numeric_df.corr().abs()
        np.fill_diagonal(corr.values, 0)
        # Most correlated pair
        max_val   = corr.values.max()
        x_col, y_col = corr.stack().idxmax()
        signed    = numeric_df.corr().loc[x_col, y_col]
        fig = px.scatter(
            numeric_df, x=x_col, y=y_col,
            title=f"{x_col} vs {y_col} — correlation {signed:+.2f}",
            opacity=0.55,
        )
        fig.update_traces(marker=dict(color="#E07856", size=8,
                                       line=dict(color="#D86848", width=0.6)))
        fig.update_layout(**fig_layout)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"These two columns are the **most strongly related** in your data "
            f"(absolute correlation = {max_val:.2f})."
        )

    else:  # Distribution Histogram
        col = st.selectbox("Choose a numeric column", numeric_df.columns)
        fig = px.histogram(
            numeric_df, x=col, nbins=30,
            title=f"Distribution of {col}",
            color_discrete_sequence=["#7FA98B"],
        )
        fig.update_layout(**fig_layout)
        st.plotly_chart(fig, use_container_width=True)
        s = numeric_df[col].describe()
        st.caption(
            f"**Average:** {s['mean']:.2f}  ·  "
            f"**Median:** {s['50%']:.2f}  ·  "
            f"**Min:** {s['min']:.2f}  ·  "
            f"**Max:** {s['max']:.2f}"
        )


init_state()


with st.sidebar:
    # Brand block — coral/honey gradient "DA" logo + title + tagline
    st.markdown(
        """
<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <div style="width:36px;height:36px;border-radius:10px;
                background:linear-gradient(135deg,#E07856,#D9A85C);
                color:white;display:flex;align-items:center;justify-content:center;
                font-weight:800;font-size:14px;letter-spacing:.02em;
                box-shadow:0 4px 10px rgba(224,120,86,.25);">DA</div>
    <div>
        <div style="font-size:14px;font-weight:700;color:#2B2A26;letter-spacing:-.01em;">Data Analyst Agent</div>
        <div style="font-size:10.5px;color:#6F6A60;margin-top:1px;">Offline ML · AI-guided</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.divider()

    source = st.radio(
        "Data source",
        ["Upload file", "Local file path", "Generate demo data"],
        index=0,
    )

    uploaded      = None
    local_path    = ""
    selected_sheet: str | None = None
    rows          = SAMPLE_ROWS

    if source == "Upload file":
        uploaded = st.file_uploader(
            "CSV or Excel (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"],
        )
        if uploaded is not None:
            _is_excel = uploaded.name.lower().endswith((".xlsx", ".xls"))
            if _is_excel:
                _sheets = get_excel_sheets(uploaded)
                if len(_sheets) > 1:
                    selected_sheet = st.selectbox(
                        f"Sheet to train on ({len(_sheets)} sheets found)",
                        _sheets,
                    )
                    # Show a quick overview of every sheet's column count
                    _sheet_info = []
                    for _sn in _sheets:
                        try:
                            _peek = pd.read_excel(
                                io.BytesIO(uploaded.getvalue()),
                                sheet_name=_sn, engine="openpyxl", nrows=1,
                            )
                            _sheet_info.append(f"**{_sn}** — {len(_peek.columns)} cols")
                        except Exception:
                            _sheet_info.append(f"**{_sn}** — unreadable")
                    st.caption("  ·  ".join(_sheet_info))
                elif _sheets:
                    selected_sheet = _sheets[0]
            st.success(f"Ready: `{uploaded.name}`")

    elif source == "Local file path":
        local_path = st.text_input("Full file path (.csv, .xlsx, .xls)", value="")
        if local_path.strip():
            _lp = Path(local_path.strip())
            if _lp.exists() and _lp.suffix.lower() in (".xlsx", ".xls"):
                try:
                    _sheets = pd.ExcelFile(_lp, engine="openpyxl").sheet_names
                    if len(_sheets) > 1:
                        selected_sheet = st.selectbox(
                            f"Sheet ({len(_sheets)} found)", _sheets
                        )
                    elif _sheets:
                        selected_sheet = _sheets[0]
                except Exception:
                    pass

    else:  # Generate demo data
        rows = st.slider("Demo rows", 1_000, 50_000, SAMPLE_ROWS, step=1_000)

    # Recent Files section (mirrors the mockup)
    if st.session_state.recent_files:
        st.divider()
        st.markdown(
            "<div style='font-size:10px;font-weight:700;letter-spacing:.12em;"
            "text-transform:uppercase;color:#6F6A60;margin-bottom:8px;'>Recent Files</div>",
            unsafe_allow_html=True,
        )
        for _f in st.session_state.recent_files[:3]:
            _is_active = (_f["name"] == st.session_state.get("source_filename"))
            _bg        = "#FCE5DC" if _is_active else "#FBF8F2"
            _bd        = "#F2C8B7" if _is_active else "#ECE4D6"
            _ic_bg     = "#FFFFFF" if _is_active else "#F8EBCF"
            _ic_clr    = "#E07856" if _is_active else "#8A6A2E"
            _trained_lbl = " · trained" if _f.get("trained") else ""
            st.markdown(
                f"""
<div style="display:flex;align-items:center;gap:10px;padding:8px 10px;
            background:{_bg};border:1px solid {_bd};border-radius:10px;margin-bottom:5px;">
    <div style="width:28px;height:28px;border-radius:7px;background:{_ic_bg};
                color:{_ic_clr};display:flex;align-items:center;justify-content:center;
                font-size:10px;font-weight:700;">{_f['ext']}</div>
    <div style="flex:1;min-width:0;">
        <div style="font-size:11.5px;font-weight:600;color:#2B2A26;
                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{_f['name']}</div>
        <div style="font-size:10px;color:#6F6A60;">{_f['size_kb']} KB{_trained_lbl}</div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Training")
    target_request = st.text_input(
        "Target column (blank = auto-detect)",
        value=st.session_state.target_col or "",
    )
    tree_depth   = st.slider("Reasoning depth", 2, 12, 6)
    n_estimators = st.slider("Forest trees", 50, 500, 150, step=50)

    train_clicked = st.button("Train agent", use_container_width=True)
    reset_clicked = st.button("Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.chat = []
        st.rerun()

    if st.session_state.train_result:
        st.divider()
        st.markdown("### Active model")
        st.write(f"Target: `{st.session_state.target_col}`")
        st.write(f"Random Forest: `{st.session_state.train_result['rf_accuracy']:.1%}`")
        st.write(f"Decision Tree: `{st.session_state.train_result['dt_accuracy']:.1%}`")
        _cv = st.session_state.train_result.get("cv_mean_accuracy")
        _cvs = st.session_state.train_result.get("cv_std", 0.0)
        if _cv:
            st.write(f"5-fold CV: `{_cv:.1%} ±{_cvs:.3f}`")
        if st.session_state.train_result.get("shap_available"):
            st.caption("Importance: SHAP ✓")


# ---------------------------------------------------------------------------
# Train button handler
# ---------------------------------------------------------------------------
if train_clicked:
    try:
        with st.spinner("Structuring data and training offline ML model..."):

            if source == "Upload file":
                if uploaded is None:
                    st.error("Upload a file first.")
                    st.stop()

                st.session_state.source_filename = uploaded.name
                add_recent_file(uploaded.name, len(uploaded.getvalue()))

                _is_excel = uploaded.name.lower().endswith((".xlsx", ".xls"))
                if _is_excel:
                    _sheet = selected_sheet if selected_sheet is not None else 0
                    _raw = read_excel_sheet(uploaded, _sheet)
                    raw_df, _rules, _ = _gc.clean_dataframe(_raw, api_key=_GROQ_API_KEY)
                else:
                    _tmp_path = tempfile.mktemp(suffix=".csv")
                    try:
                        with open(_tmp_path, "wb") as _f:
                            _f.write(uploaded.getvalue())
                        raw_df, _rules, _ = _gc.clean_large_file(
                            _tmp_path, api_key=_GROQ_API_KEY
                        )
                    finally:
                        Path(_tmp_path).unlink(missing_ok=True)

                target_request = _rules.get("target") or target_request

            elif source == "Local file path":
                if not local_path.strip():
                    st.error("Enter a file path first.")
                    st.stop()
                _lp = Path(local_path.strip())
                if not _lp.exists():
                    st.error(f"File not found: {_lp}")
                    st.stop()

                st.session_state.source_filename = _lp.name
                try:
                    add_recent_file(_lp.name, _lp.stat().st_size)
                except Exception:
                    pass

                if _lp.suffix.lower() in (".xlsx", ".xls"):
                    _sheet = selected_sheet if selected_sheet is not None else 0
                    _raw = read_excel_sheet(_lp, _sheet)
                    raw_df, _rules, _ = _gc.clean_dataframe(_raw, api_key=_GROQ_API_KEY)
                else:
                    raw_df, _rules, _ = _gc.clean_large_file(
                        str(_lp), api_key=_GROQ_API_KEY
                    )
                target_request = _rules.get("target") or target_request

            else:  # Generate demo data
                st.session_state.source_filename = "synthetic_sales_data.csv"
                raw_df = dl.generate_synthetic_data(n_rows=rows)

            train_agent(raw_df, target_request, tree_depth, n_estimators)

        st.success("Agent trained and ready to chat.")
        st.rerun()

    except Exception as exc:
        st.error(f"Training failed: {exc}")
        with st.expander("Error detail"):
            st.code(_tb.format_exc())


# ---------------------------------------------------------------------------
# Status header
# ---------------------------------------------------------------------------
trained     = st.session_state.train_result is not None
file_title  = st.session_state.get("source_filename") or "No file loaded yet"
target_text = st.session_state.target_col or "auto-detect"
rows_text   = (
    f"{st.session_state.data_profile['rows']:,} rows"
    if st.session_state.data_profile
    else "no data"
)
cols_text   = (
    f"{st.session_state.data_profile['columns']} columns"
    if st.session_state.data_profile
    else "0 columns"
)
status_chip = (
    '<div class="status-pill" style="background:#E3EEDF;color:#4F7E5C;border-color:#C8DEC6;">● Model ready</div>'
    if trained
    else '<div class="status-pill">Ready — upload a file</div>'
)

if not st.session_state.chat:
    st.session_state.chat = [
        {
            "role": "assistant",
            "content": (
                "Hi! Upload or generate data from the sidebar, train me, then ask anything "
                "about your data — patterns, percentages, predictions, or just say hi."
            ),
        }
    ]

# ============================================================================
# Two-column layout: workspace (left) + chat panel (right)
# ============================================================================
workspace_col, chat_col = st.columns([1.7, 1], gap="medium")

# ----------------------------------------------------------------------------
# WORKSPACE (left)
# ----------------------------------------------------------------------------
with workspace_col:

    # Header card
    st.markdown(
        f"""
<section class="app-head">
    <div class="app-head-row">
        <div>
            <div class="app-title">{file_title}</div>
            <div style="font-size:11px;color:#6F6A60;margin-top:2px;">
                {rows_text} · {cols_text} · target: {target_text}
            </div>
        </div>
        <div class="status-row">
            {status_chip}
            <div class="status-pill">Offline ML</div>
        </div>
    </div>
</section>
""",
        unsafe_allow_html=True,
    )

    # Metrics — always shown (placeholders when empty)
    profile = st.session_state.data_profile or {}
    result  = st.session_state.train_result or {}
    cv_mean = result.get("cv_mean_accuracy", result.get("rf_accuracy", 0))
    cv_std  = result.get("cv_std", 0.0)
    cv_label = (f"{cv_mean:.0%}" if cv_mean else "—")
    rows_val = f"{profile.get('rows', 0):,}" if profile else "—"
    cols_val = f"{profile.get('columns', 0):,}" if profile else "—"
    rules_val = f"{result.get('n_rules', 0):,}" if result else "—"
    shap_badge = "SHAP ✓" if result.get("shap_available") else "Importance"
    st.markdown(
        f"""
<div class="metric-grid">
    <div class="metric coral"><div class="ico">↗</div><span class="value">{rows_val}</span><span class="label">Rows Learned</span></div>
    <div class="metric sage"><div class="ico">◆</div><span class="value">{cols_val}</span><span class="label">Signals</span></div>
    <div class="metric honey"><div class="ico">✦</div><span class="value">{cv_label}</span><span class="label">CV Accuracy</span></div>
    <div class="metric neutral"><div class="ico">∎</div><span class="value">{rules_val}</span><span class="label">Rules · {shap_badge}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Most impactful columns chart — or empty-state placeholder
    if st.session_state.df_raw is not None and Path("feature_importances.pkl").exists():
        try:
            import plotly.express as _px
            imp_payload = trainer.load_importances()
            shap_imp   = imp_payload.get("shap_importances", {})
            perm_imp   = imp_payload.get("importances", {})
            imp_source = shap_imp if shap_imp else perm_imp
            imp_label  = "SHAP" if shap_imp else "Permutation importance"
            if imp_source:
                ranked = sorted(imp_source.items(), key=lambda x: x[1], reverse=True)[:8]
                df_imp = pd.DataFrame(ranked, columns=["column", "score"])
                fig = _px.bar(
                    df_imp.iloc[::-1], x="score", y="column", orientation="h",
                    color="score",
                    color_continuous_scale=[[0, "#F8EBCF"], [0.5, "#D9A85C"], [1, "#E07856"]],
                    title="Most impactful columns",
                )
                fig.update_layout(
                    font=dict(family="Inter, sans-serif", color="#2B2A26", size=12),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    margin=dict(l=10, r=10, t=50, b=20),
                    coloraxis_showscale=False,
                    xaxis_title="", yaxis_title="",
                    height=320,
                )
                fig.update_xaxes(showgrid=True, gridcolor="#ECE4D6")
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Ranking method: **{imp_label}** — what drives the predictions most.")
        except Exception:
            pass
    else:
        st.markdown(
            """
<div style="background:#FFFFFF;border:1px solid #ECE4D6;border-radius:16px;
            padding:46px 24px;text-align:center;box-shadow:0 1px 2px rgba(43,42,38,.04),0 8px 24px rgba(43,42,38,.06);">
    <div style="width:56px;height:56px;border-radius:14px;margin:0 auto 14px;
                background:linear-gradient(135deg,#FCE5DC,#F8EBCF);
                display:flex;align-items:center;justify-content:center;
                font-size:24px;color:#E07856;">◆</div>
    <div style="font-size:15px;font-weight:700;color:#2B2A26;margin-bottom:5px;">
        Your insights will appear here
    </div>
    <div style="font-size:12px;color:#6F6A60;max-width:380px;margin:0 auto;line-height:1.5;">
        Upload a CSV or Excel file from the sidebar, click <b>Train agent</b>,
        and you'll see the most impactful columns, charts, and predictions appear in this space.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Pinned action row — styled as rounded pills with colored dots
    st.markdown("#### Pinned")
    st.markdown('<div class="pin-row">', unsafe_allow_html=True)
    pin_prompts = ["Summarize", "Top drivers", "Missing values", "Why high?", "Predict", "Visualize"]
    pin_map = {
        "Summarize":      "Summarize this dataset",
        "Top drivers":    "What are the top drivers?",
        "Missing values": "Show missing values",
        "Why high?":      "Why is the high outcome happening?",
        "Predict":        "Predict if quantity=5 and discount_pct=10",
    }
    pin_cols = st.columns(len(pin_prompts))
    for idx, label in enumerate(pin_prompts):
        with pin_cols[idx]:
            if st.button(label, key=f"pin_{idx}", use_container_width=True):
                if label == "Visualize":
                    visualize_dialog()
                else:
                    pinned_prompt = pin_map[label]
                    st.session_state.chat.append({"role": "user", "content": pinned_prompt})
                    st.session_state.chat.append({"role": "assistant", "content": generate_response(pinned_prompt)})
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Data & model details collapsed below
    if st.session_state.df_raw is not None:
        with st.expander("Data and model details", expanded=False):
            tabs = st.tabs(["Preview", "Profile"])
            with tabs[0]:
                st.dataframe(st.session_state.df_raw.head(30), use_container_width=True, height=320)
            with tabs[1]:
                st.json(st.session_state.data_profile or {})

# ----------------------------------------------------------------------------
# CHAT PANEL (right) — full vertical column with input at its bottom
# ----------------------------------------------------------------------------
with chat_col:
    # Hidden marker — CSS uses :has() to style only the column containing it
    st.markdown('<div class="chat-card-marker"></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="chat-header">
    <div class="chat-avatar">AI</div>
    <div>
        <div class="title">AI Assistant</div>
        <div class="sub"><span class="dot"></span>Reading your data</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    chat_container = st.container(height=560, border=False)
    with chat_container:
        for message in st.session_state.chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

    # Auto-scroll all scrollable Streamlit containers to their bottom
    # so the newest chat message is always in view.
    import streamlit.components.v1 as _components
    _components.html(
        """
<script>
  const scrollAll = () => {
    const doc = window.parent.document;
    doc.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]').forEach(el => {
      if (el.scrollHeight > el.clientHeight + 10) {
        el.scrollTop = el.scrollHeight;
      }
    });
  };
  setTimeout(scrollAll, 100);
  setTimeout(scrollAll, 350);
</script>
""",
        height=0,
    )

    # Input INSIDE the chat column (form-based so it lives in the column)
    with st.form("chat_form", clear_on_submit=True, border=False):
        col_in, col_send = st.columns([8, 1], gap="small")
        with col_in:
            _typed = st.text_input(
                "Message", label_visibility="collapsed",
                placeholder="Ask anything about your data…",
            )
        with col_send:
            _send = st.form_submit_button("➤", use_container_width=True)

    if _send and _typed.strip():
        st.session_state.chat.append({"role": "user", "content": _typed.strip()})
        st.session_state.chat.append({"role": "assistant", "content": generate_response(_typed.strip())})
        st.rerun()

    # Quick-chip suggestions — small buttons that auto-send when clicked
    _quick = ["Show top columns", "What's the average?", "Explain the target"]
    qcols  = st.columns(len(_quick))
    for _i, _q in enumerate(_quick):
        with qcols[_i]:
            if st.button(_q, key=f"qc_{_i}", use_container_width=True):
                st.session_state.chat.append({"role": "user", "content": _q})
                st.session_state.chat.append({"role": "assistant", "content": generate_response(_q)})
                st.rerun()
