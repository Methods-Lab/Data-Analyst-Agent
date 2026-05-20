# Analyst ML Chatbot

An interactive Streamlit chatbot that can train on user-provided data and answer
as a data analyst. It supports CSV, Excel, JSON, pasted text, TXT files, local
file paths, and a built-in synthetic sales dataset.

## What It Does

- Trains local sklearn models from the active dataset.
- Auto-detects or creates a target/outcome column when one is not provided.
- Extracts Decision Tree rules for explainable prediction.
- Uses Random Forest permutation importance to identify key drivers.
- Builds a TF-IDF text retrieval index for document-like data.
- Answers natural-language questions in the chat interface.
- Optional OpenAI-compatible API mode for Groq, xAI Grok, OpenRouter, or a custom chat-completions endpoint.

Example questions:

```text
Summarize this dataset
What are the top drivers?
Show missing values
Why high?
Predict if quantity=5 and discount_pct=10
What does the document say about refunds?
```

## Run

```bash
cd C:\Users\Glow Computers\Downloads\sales_agent
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Files

```text
app.py                  Streamlit chatbot UI and analyst logic
data_loader.py          Data loading, cleaning, and sample sales data
trainer.py              Decision Tree and Random Forest training
reasoner.py             Deduction and abduction reasoning
abduction.py            Optional Bayesian-style explanation layer
requirements.txt        Python dependencies
```

Runtime artifacts:

```text
rules.json
feature_importances.pkl
models.pkl
```

## Notes

This is a local machine-learning analyst. It does not call an external LLM by
default, so its offline answers are grounded in the uploaded data, model rules,
feature importance, simple statistics, and text retrieval.

To use an external chat model, choose a provider in the sidebar and paste that
provider's API key. The app sends the model only the current question plus a
compact dataset summary, model drivers, and relevant text evidence.
