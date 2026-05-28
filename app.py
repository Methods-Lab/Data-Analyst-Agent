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
    --ink: #ffffff;
    --muted: #cbd8e8;
    --panel: rgba(7, 13, 25, 0.92);
    --panel-2: rgba(17, 31, 50, 0.92);
    --line: rgba(175, 211, 238, 0.34);
    --cyan: #49d6d0;
    --green: #79e68f;
    --amber: #f6c66a;
    --rose: #ff7c9b;
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(circle at 12% 10%, rgba(73, 214, 208, 0.24), transparent 30%),
        radial-gradient(circle at 78% 0%, rgba(246, 198, 106, 0.14), transparent 28%),
        linear-gradient(125deg, #07111f 0%, #0d1d32 46%, #132923 100%);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,0.65), transparent 78%);
}

.main .block-container {
    padding-top: 1.2rem;
    max-width: 1420px;
}

[data-testid="stSidebar"] {
    background: rgba(5, 12, 24, 0.9);
    border-right: 1px solid var(--line);
}

h1, h2, h3 {
    letter-spacing: 0;
}

.app-head {
    position: relative;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 14px 16px;
    background: rgba(5, 12, 22, 0.84);
    box-shadow: 0 14px 38px rgba(0, 0, 0, 0.24);
    animation: liftIn 520ms ease-out both;
}

.app-head-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
}

.app-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #ffffff;
}

.status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 0;
}

.status-pill {
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 8px 12px;
    background: rgba(255,255,255,0.06);
    color: var(--muted);
    font-size: 0.84rem;
    backdrop-filter: blur(12px);
}

.panel {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 18px;
    background: var(--panel);
    box-shadow: 0 18px 46px rgba(0, 0, 0, 0.25);
    animation: liftIn 520ms ease-out both;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 18px 0;
}

.metric {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 15px;
    background: var(--panel-2);
}

.metric .value {
    display: block;
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--ink);
}

.metric .label {
    display: block;
    margin-top: 5px;
    color: var(--muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.chat-wrap {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 16px;
    background: rgba(4, 10, 20, 0.9);
    min-height: 620px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
}

.insight {
    border-left: 3px solid var(--cyan);
    background: rgba(73,214,208,0.09);
    padding: 12px 14px;
    border-radius: 8px;
    margin: 10px 0;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0 4px;
}

.chip {
    border: 1px solid rgba(73,214,208,0.28);
    border-radius: 999px;
    padding: 7px 10px;
    color: #d7fffb;
    background: rgba(73,214,208,0.08);
    font-size: 0.82rem;
}

.stButton > button {
    border-radius: 8px;
    border: 1px solid rgba(73,214,208,0.34);
    background: linear-gradient(135deg, #49d6d0, #79e68f);
    color: #03111a;
    font-weight: 800;
    transition: transform 160ms ease, box-shadow 160ms ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(73, 214, 208, 0.18);
}

div[data-testid="stChatMessage"] {
    border: 1px solid rgba(175, 211, 238, 0.28);
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    background: rgba(17, 31, 50, 0.94);
    color: #ffffff !important;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div {
    color: #ffffff !important;
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(33, 74, 96, 0.94);
    border-color: rgba(73, 214, 208, 0.42);
}

div[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    background: rgba(10, 20, 34, 0.96) !important;
    border: 1px solid rgba(73, 214, 208, 0.38) !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #b9c9d8 !important;
}

div[data-testid="stAlert"] {
    color: #ffffff;
}

div[data-testid="stChatMessage"] {
    animation: liftIn 240ms ease-out both;
}

@keyframes liftIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 900px) {
    .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .app-head { padding: 12px; }
}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "df_raw":        None,
        "df_clean":      None,
        "train_result":  None,
        "target_col":    None,
        "chat":          [],
        "text_corpus":   None,
        "text_vectors":  None,
        "text_vectorizer": None,
        "data_profile":  None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
    profile = st.session_state.data_profile or {}
    result = st.session_state.train_result or {}
    context = [
        f"Rows: {profile.get('rows', 0)}",
        f"Columns: {profile.get('columns', 0)}",
        f"Target: {st.session_state.target_col or 'none'}",
        f"Numeric columns: {', '.join(profile.get('numeric', [])[:30])}",
        f"Categorical/text columns: {', '.join(profile.get('categorical', [])[:30])}",
        f"Target distribution: {profile.get('target_distribution', {})}",
        f"Missing values: {profile.get('missing', {})}",
    ]

    if result:
        context.append(f"Random Forest accuracy: {result.get('rf_accuracy', 0):.3f}")
        context.append(f"Decision Tree accuracy: {result.get('dt_accuracy', 0):.3f}")

    if Path("feature_importances.pkl").exists():
        try:
            imp = trainer.load_importances()["importances"]
            ranked = sorted(imp.items(), key=lambda item: item[1], reverse=True)[:12]
            context.append(f"Top feature importances: {ranked}")
        except Exception:
            pass

    retrieval = answer_with_text_retrieval(prompt)
    if retrieval:
        context.append(f"Relevant text evidence:\n{retrieval}")

    if st.session_state.df_raw is not None:
        preview = st.session_state.df_raw.head(8).to_dict("records")
        context.append(f"Data preview: {preview}")

    return "\n".join(context)


def call_external_ai(prompt: str, local_answer: str) -> str | None:
    """
    Plain-language bullet-point explanation grounded in the ACTUAL DATA,
    written for a non-technical reader. Always additive — never replaces the ML output.
    Returns None silently on failure.
    """
    if not _GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=_GROQ_API_KEY)
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are explaining a dataset to someone with no technical background — "
                        "think of them as a business owner or manager who just wants to understand "
                        "their data in plain English.\n\n"
                        "Rules you must follow:\n"
                        "• Always respond in 4 to 6 bullet points (use the • symbol).\n"
                        "• Use simple, everyday language — no jargon, no technical terms.\n"
                        "• Each bullet must say something directly about the ACTUAL DATA "
                        "(columns, values, patterns, trends) — not about models or algorithms.\n"
                        "• Make each point practical and actionable: what does this mean for the business?\n"
                        "• Keep each bullet to one clear sentence.\n"
                        "• Do NOT use words like: model, algorithm, ML, training, accuracy, feature, "
                        "vector, classifier, hyperparameter, or any other technical term."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"The user asked: \"{prompt}\"\n\n"
                        f"Here is the actual data information:\n{dataset_context_for_llm(prompt)}\n\n"
                        "Now write 4 to 6 plain-English bullet points explaining what this data "
                        "shows and what it means, based directly on the data above."
                    ),
                },
            ],
            temperature=0.5,
            max_tokens=320,
        )
        return response.choices[0].message.content.strip()
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


def generate_response(prompt: str) -> str:
    local_answer = answer_with_data(prompt)
    ai_explanation = call_external_ai(prompt, local_answer)
    if ai_explanation:
        return (
            f"{local_answer}\n\n"
            "---\n"
            "**AI Explanation**\n\n"
            f"{ai_explanation}"
        )
    return local_answer


def train_agent(df: pd.DataFrame, target_request: str, tree_depth: int, n_estimators: int) -> None:
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
    <div class="metric"><span class="value">{profile.get('rows', 0):,}</span><span class="label">Rows Learned</span></div>
    <div class="metric"><span class="value">{profile.get('columns', 0):,}</span><span class="label">Signals</span></div>
    <div class="metric"><span class="value">{cv_label}</span><span class="label">5-Fold CV Accuracy</span></div>
    <div class="metric"><span class="value">{result.get('n_rules', 0):,}</span><span class="label">Rules · {shap_badge}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


init_state()


with st.sidebar:
    st.markdown("## Data Analyst Agent")
    st.caption("Offline ML · AI-structured pipeline")
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

                _is_excel = uploaded.name.lower().endswith((".xlsx", ".xls"))
                if _is_excel:
                    # Read the chosen sheet → clean in-memory (Excel can't be streamed)
                    _sheet = selected_sheet if selected_sheet is not None else 0
                    _raw = read_excel_sheet(uploaded, _sheet)
                    raw_df, _rules, _ = _gc.clean_dataframe(_raw, api_key=_GROQ_API_KEY)
                else:
                    # CSV → write to temp, use chunked streaming reader
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
status      = "Model trained" if trained else "Ready — upload a file to begin"
target_text = st.session_state.target_col or "auto-detected after training"
data_text   = (
    f"{st.session_state.data_profile['rows']:,} rows"
    if st.session_state.data_profile
    else "no data loaded"
)

st.markdown(
    f"""
<section class="app-head">
    <div class="app-head-row">
        <div class="app-title">Data Analyst Agent</div>
        <div class="status-row">
            <div class="status-pill">{status}</div>
            <div class="status-pill">Target: {target_text}</div>
            <div class="status-pill">{data_text}</div>
            <div class="status-pill">Offline ML</div>
        </div>
    </div>
</section>
""",
    unsafe_allow_html=True,
)


if trained:
    render_metrics()


st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if not st.session_state.chat:
    st.session_state.chat = [
        {
            "role": "assistant",
            "content": (
                "Hi. Upload or generate data from the sidebar, train me, then ask anything. "
                "You can also chat now if you only need analyst guidance."
            ),
        }
    ]

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("#### Pinned")
pin_prompts = [
    "Summarize",
    "Top drivers",
    "Missing values",
    "Why high?",
    "Predict",
]
pin_cols = st.columns(len(pin_prompts))
pin_map = {
    "Summarize": "Summarize this dataset",
    "Top drivers": "What are the top drivers?",
    "Missing values": "Show missing values",
    "Why high?": "Why is the high outcome happening?",
    "Predict": "Predict if quantity=5 and discount_pct=10",
}
for idx, label in enumerate(pin_prompts):
    with pin_cols[idx]:
        if st.button(label, key=f"pin_{idx}", use_container_width=True):
            pinned_prompt = pin_map[label]
            st.session_state.chat.append({"role": "user", "content": pinned_prompt})
            st.session_state.chat.append({"role": "assistant", "content": generate_response(pinned_prompt)})
            st.rerun()

prompt = st.chat_input("Message the analyst chatbot...")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.session_state.chat.append({"role": "assistant", "content": generate_response(prompt)})
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.df_raw is not None:
    with st.expander("Data and model details", expanded=False):
        tabs = st.tabs(["Preview", "Drivers", "Profile"])
        with tabs[0]:
            st.dataframe(st.session_state.df_raw.head(30), use_container_width=True, height=360)
        with tabs[1]:
            if st.session_state.train_result and Path("feature_importances.pkl").exists():
                imp_payload = trainer.load_importances()
                # Prefer SHAP importances (more accurate); fall back to permutation importance
                shap_imp = imp_payload.get("shap_importances", {})
                perm_imp = imp_payload.get("importances", {})
                imp_source = shap_imp if shap_imp else perm_imp
                imp_label  = "SHAP Mean |value|" if shap_imp else "Permutation Importance"
                ranked_imp = pd.DataFrame(
                    sorted(imp_source.items(), key=lambda item: item[1], reverse=True)[:15],
                    columns=["feature", "importance"],
                )
                st.caption(f"Driver ranking method: **{imp_label}**")
                st.bar_chart(ranked_imp.set_index("feature"))
                # Also show CV accuracy note if available
                cv_mean = st.session_state.train_result.get("cv_mean_accuracy")
                cv_std  = st.session_state.train_result.get("cv_std", 0.0)
                if cv_mean and cv_std > 0:
                    st.caption(
                        f"5-fold cross-validation accuracy: **{cv_mean:.1%} ± {cv_std:.3f}** "
                        "(generalisation estimate, more reliable than single train/test split)"
                    )
            else:
                st.info("Train the model to see drivers.")
        with tabs[2]:
            st.json(st.session_state.data_profile or {})
