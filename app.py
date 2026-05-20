"""
Streamlit app for a data-trained analyst chatbot.

The app accepts structured data (CSV, Excel, JSON) and lightweight text data,
trains local sklearn models when a usable target exists, and answers questions
as a data analyst grounded in the active dataset. When no dataset is provided,
it behaves as a general analyst assistant and invites the user to attach data.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import data_loader as dl
import reasoner
import trainer
from abduction import try_bayesian_abduce


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
        "df_raw": None,
        "df_clean": None,
        "train_result": None,
        "target_col": None,
        "chat": [],
        "text_corpus": None,
        "text_vectors": None,
        "text_vectorizer": None,
        "data_profile": None,
        "api_provider": "Offline ML",
        "api_key": "",
        "api_model": "",
        "api_base_url": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def read_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if name.endswith(".json"):
        raw = json.load(uploaded_file)
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict):
            try:
                return pd.json_normalize(raw)
            except Exception:
                return pd.DataFrame([raw])
    if name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        return text_to_frame(text)
    return pd.read_csv(uploaded_file)


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


def provider_settings(provider: str, custom_base_url: str, custom_model: str) -> tuple[str, str]:
    presets = {
        "Groq": ("https://api.groq.com/openai/v1/chat/completions", "llama-3.1-8b-instant"),
        "xAI Grok": ("https://api.x.ai/v1/chat/completions", "grok-2-latest"),
        "OpenRouter": ("https://openrouter.ai/api/v1/chat/completions", "meta-llama/llama-3.1-8b-instruct:free"),
        "Custom": (custom_base_url.strip(), custom_model.strip()),
    }
    return presets.get(provider, ("", ""))


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
    provider = st.session_state.get("api_provider", "Offline ML")
    if provider == "Offline ML":
        return None

    api_key = st.session_state.get("api_key") or os.getenv("AI_API_KEY", "")
    base_url, model = provider_settings(
        provider,
        st.session_state.get("api_base_url", ""),
        st.session_state.get("api_model", ""),
    )
    model = st.session_state.get("api_model") or model

    if not api_key:
        return (
            f"{provider} is selected, but no API key is set. Add the key in the sidebar "
            "or set AI_API_KEY, then ask again.\n\nLocal ML answer:\n"
            f"{local_answer}"
        )

    if not base_url or not model:
        return "The selected API provider is missing a base URL or model name."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior data analyst chatbot. Answer conversationally and directly. "
                "Use the provided dataset context, local ML result, and evidence. If the data does "
                "not support a claim, say so. Keep answers useful, not generic."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{prompt}\n\n"
                f"Dataset context:\n{dataset_context_for_llm(prompt)}\n\n"
                f"Local ML/statistical answer to improve:\n{local_answer}"
            ),
        },
    ]

    try:
        response = requests.post(
            base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Analyst ML Chatbot",
            },
            json={"model": model, "messages": messages, "temperature": 0.35, "max_tokens": 900},
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return (
            f"The {provider} API call failed: {exc}\n\n"
            f"Local ML answer:\n{local_answer}"
        )


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
    external_answer = call_external_ai(prompt, local_answer)
    return external_answer or local_answer


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
    result = st.session_state.train_result or {}
    st.markdown(
        f"""
<div class="metric-grid">
    <div class="metric"><span class="value">{profile.get('rows', 0):,}</span><span class="label">Rows Learned</span></div>
    <div class="metric"><span class="value">{profile.get('columns', 0):,}</span><span class="label">Signals</span></div>
    <div class="metric"><span class="value">{result.get('rf_accuracy', 0):.0%}</span><span class="label">Model Accuracy</span></div>
    <div class="metric"><span class="value">{result.get('n_rules', 0):,}</span><span class="label">Reasoning Rules</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


init_state()


with st.sidebar:
    st.markdown("## Analyst ML Chatbot")
    st.divider()

    source = st.radio(
        "Data source",
        ["Upload data", "Paste text", "Generate sample sales data", "Use local file path"],
        index=2,
    )

    uploaded = None
    pasted_text = ""
    local_path = ""
    rows = SAMPLE_ROWS

    if source == "Upload data":
        uploaded = st.file_uploader("CSV, Excel, JSON, or TXT", type=["csv", "xlsx", "xls", "json", "txt"])
        if uploaded is not None:
            st.info(f"Selected `{uploaded.name}`. Click Train agent to learn it.")
    elif source == "Paste text":
        pasted_text = st.text_area("Paste notes, documents, or raw text", height=180)
    elif source == "Generate sample sales data":
        rows = st.slider("Rows", 1_000, 50_000, SAMPLE_ROWS, step=1_000)
    else:
        local_path = st.text_input("File path", value="")

    st.divider()
    st.markdown("### AI provider")
    provider = st.selectbox(
        "Chat engine",
        ["Offline ML", "Groq", "xAI Grok", "OpenRouter", "Custom"],
        index=["Offline ML", "Groq", "xAI Grok", "OpenRouter", "Custom"].index(
            st.session_state.get("api_provider", "Offline ML")
        ),
    )
    st.session_state.api_provider = provider

    if provider != "Offline ML":
        default_base, default_model = provider_settings(
            provider,
            st.session_state.get("api_base_url", ""),
            st.session_state.get("api_model", ""),
        )
        st.session_state.api_key = st.text_input(
            "API key",
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="Paste provider key",
        )
        st.session_state.api_model = st.text_input("Model", value=st.session_state.get("api_model") or default_model)
        if provider == "Custom":
            st.session_state.api_base_url = st.text_input(
                "Chat completions URL",
                value=st.session_state.get("api_base_url") or default_base,
                placeholder="https://api.example.com/v1/chat/completions",
            )
        else:
            st.caption(default_base)

    st.divider()
    st.markdown("### Training")
    target_request = st.text_input("Target column or outcome", value=st.session_state.target_col or "")
    tree_depth = st.slider("Reasoning depth", 2, 12, 6)
    n_estimators = st.slider("Forest trees", 50, 500, 250, step=50)

    train_clicked = st.button("Train agent", use_container_width=True)
    reset_clicked = st.button("Reset chat", use_container_width=True)

    if reset_clicked:
        st.session_state.chat = []
        st.rerun()

    if st.session_state.train_result:
        st.divider()
        st.markdown("### Active model")
        st.write(f"Target: `{st.session_state.target_col}`")
        st.write(f"Random Forest: `{st.session_state.train_result['rf_accuracy']:.1%}`")
        st.write(f"Decision Tree: `{st.session_state.train_result['dt_accuracy']:.1%}`")


if train_clicked:
    try:
        with st.spinner("Reading data and fitting the local ML agent..."):
            if source == "Upload data":
                if uploaded is None:
                    st.error("Upload a file first.")
                    st.stop()
                raw_df = read_uploaded_file(uploaded)
            elif source == "Paste text":
                if not pasted_text.strip():
                    st.error("Paste some text first.")
                    st.stop()
                raw_df = text_to_frame(pasted_text)
            elif source == "Generate sample sales data":
                raw_df = dl.generate_synthetic_data(n_rows=rows)
            else:
                if not local_path.strip():
                    st.error("Enter a local file path first.")
                    st.stop()
                raw_df = read_local_file(local_path.strip())

            train_agent(raw_df, target_request, tree_depth, n_estimators)
        st.success("Agent trained and ready to chat.")
        st.rerun()
    except Exception as exc:
        st.error(f"Training failed: {exc}")


trained = st.session_state.train_result is not None
status = "Model trained" if trained else "No data required to start"
target_text = st.session_state.target_col or "auto-selected after training"
data_text = (
    f"{st.session_state.data_profile['rows']:,} rows active"
    if st.session_state.data_profile
    else "upload, paste, or generate data"
)

st.markdown(
    f"""
<section class="app-head">
    <div class="app-head-row">
        <div class="app-title">Analyst Chatbot</div>
        <div class="status-row">
            <div class="status-pill">{status}</div>
            <div class="status-pill">Target: {target_text}</div>
            <div class="status-pill">{data_text}</div>
            <div class="status-pill">{st.session_state.get("api_provider", "Offline ML")}</div>
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
                imp = trainer.load_importances()["importances"]
                ranked_imp = pd.DataFrame(
                    sorted(imp.items(), key=lambda item: item[1], reverse=True)[:15],
                    columns=["feature", "importance"],
                )
                st.bar_chart(ranked_imp.set_index("feature"))
            else:
                st.info("Train the model to see drivers.")
        with tabs[2]:
            st.json(st.session_state.data_profile or {})
