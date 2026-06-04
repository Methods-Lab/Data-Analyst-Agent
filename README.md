# Data Analyst Agent

A conversational machine-learning analytics application. Upload any CSV or
Excel file, train a model with one click, then chat with it in plain English.
Predictions, charts, and insights are produced by real scikit-learn models
running locally.

**Live demo:** https://data-analyst-agent-production-0f30.up.railway.app
**Repository:** https://github.com/Methods-Lab/Data-Analyst-Agent

---

## Folder structure

```
sales-agent/
│
├── app.py                          # Streamlit web app — entry point
├── groq_cleaner.py                 # AI-assisted structure detection
├── data_loader.py                  # File reading + cleaning implementation
├── trainer.py                      # ML training + SHAP + CV implementation
├── reasoner.py                     # Deductive + abductive reasoning
├── abduction.py                    # Naive-Bayes abducer
│
├── eda/                            # ←  THE 7-STEP ML PIPELINE
│   ├── __init__.py
│   ├── step_1_loading.py           # data ingestion + first look
│   ├── step_2_cleaning.py          # cleaning + NA handling
│   ├── step_3_exploration.py       # profile, distributions, correlations
│   ├── step_4_preprocessing.py     # encoding + split + target binning
│   ├── step_5_training.py          # DT + RF training
│   ├── step_6_evaluation.py        # accuracy + CV + SHAP loaders
│   ├── step_7_prediction.py        # deduce + abduce + forecast
│   ├── notebooks/                  # visual walkthrough of each step
│   │   ├── step_1_data_loading.ipynb
│   │   ├── step_2_data_cleaning.ipynb
│   │   ├── step_3_exploratory_analysis.ipynb
│   │   ├── step_4_preprocessing.ipynb
│   │   ├── step_5_model_training.ipynb
│   │   ├── step_6_model_evaluation.ipynb
│   │   └── step_7_predictions.ipynb
│   └── README.md
│
├── artifacts/                      # generated at runtime (model weights, rules)
│   ├── models.pkl
│   ├── rules.json
│   └── feature_importances.pkl
│
├── tests/
│   └── test_scalability.py         # 15-test pytest suite
│
├── docs/
│   ├── project_report.html         # printable project report
│   └── ui_mockup.html              # UI design reference
│
├── Procfile + railway.json         # deployment config
├── requirements.txt
└── README.md                       # this file
```

The `eda/` modules are **imported by `app.py` at runtime** — they are the
actual entry points the web app calls for each pipeline step. The notebooks
under `eda/notebooks/` use the same modules so the walkthrough matches the
running app exactly.

---

## The 7 ML pipeline steps

| Step | Module | What it does | Maps to teacher's step |
|------|--------|--------------|------------------------|
| 1 | `eda.step_1_loading` | Read CSV / Excel, inspect shape & types | 2. Data Collection |
| 2 | `eda.step_2_cleaning` | Strip whitespace, drop ID / constant cols, normalise NA markers, coerce string-numerics, impute medians/modes, encode categoricals | 5. Preprocessing |
| 3 | `eda.step_3_exploration` | Profile, distributions, correlations, IQR outliers, class balance | 3. EDA + 4. Problem Finding |
| 4 | `eda.step_4_preprocessing` | Date splitting, label encoding, target binning (low/medium/high), stratified train/test split | 6. Feature Selection + 7. Engineering + 8. Class Imbalance |
| 5 | `eda.step_5_training` | Train `DecisionTreeClassifier` + `RandomForestClassifier`, extract IF-THEN rules, persist | 9. Model Building |
| 6 | `eda.step_6_evaluation` | Test accuracy + 5-fold CV + confusion matrix + classification report + SHAP | 10. Evaluation + 11. Conclusion |
| 7 | `eda.step_7_prediction` | Forward chaining (deduce), backward chaining (abduce), Bayesian posterior | 12. Deployment / live use |

Run any step standalone:
```bash
python -m eda.step_1_loading
python -m eda.step_5_training
```

---

## Models used and why

| Model | Library | Role |
|---|---|---|
| `DecisionTreeClassifier` | scikit-learn | Produces human-readable IF-THEN rules saved to `artifacts/rules.json`. Drives the deductive reasoning engine. |
| `RandomForestClassifier` | scikit-learn | 200-tree ensemble — higher accuracy + works with SHAP TreeExplainer. |
| `RandomForestRegressor` | scikit-learn | Predicts numeric values on demand when user wants regression. |
| `LinearRegression` | scikit-learn | Fits a linear trend on monthly-aggregated time-series data for forecasts. |
| `TfidfVectorizer + cosine_similarity` | scikit-learn | Text retrieval for document-style data. |

**Why tree-based models, not neural networks?**
- Tree ensembles consistently outperform deep learning on tabular data
- Handle mixed types and missing values natively, no scaling needed
- Train in seconds on CPU — no GPU required
- Interpretable via SHAP — every prediction can be explained
- DecisionTree gives human-readable IF-THEN rules out of the box

---

## Accuracy & error

On the synthetic sales dataset (10,000 rows, target = `sales_category`):

| Metric | Value |
|---|---|
| Train accuracy | ~99–100% |
| Test accuracy | ~99% |
| 5-fold cross-validation accuracy | ~99% ± 0.5% |
| Train-test gap | < 2% (no overfitting) |

The high score is because `sales_category` is derived directly from
`net_sales` in the synthetic data. On real-world business data the same
pipeline typically lands at 70–92% accuracy.

See `eda/notebooks/step_6_model_evaluation.ipynb` for the full evaluation
report including confusion matrix and per-class precision/recall.

---

## How features were selected

Feature selection is **automatic and rule-based** during cleaning (Step 2):

1. Drop columns with >60% missing values
2. Drop ID-like columns (>90% unique values)
3. Drop constant columns (one unique value)
4. Drop Excel artifact columns (`Unnamed: 0`, etc.)
5. Keep everything else and let the model decide

After training, the surviving features are ranked by:
- **SHAP values** (`shap.TreeExplainer` on the RandomForest)
- **Permutation importance** (`sklearn.inspection.permutation_importance`)

The top-ranked features appear in the "Most impactful columns" chart in the
live app and in `eda/notebooks/step_6_model_evaluation.ipynb`.

---

## Tech stack

| Layer | Library | Version |
|---|---|---|
| Web UI | Streamlit | 1.57 |
| Data | pandas | 3.0 |
| Numerics | numpy | 2.4 |
| ML core | scikit-learn | 1.8 |
| Explainability | SHAP | 0.49 |
| Visualization | plotly | 6.7 |
| Excel I/O | openpyxl | 3.1 |
| Synthetic data | faker | 40 |
| Conversational AI | groq | 1.4 |
| Deployment | Railway | — |

---

## Running locally

```bash
git clone https://github.com/Methods-Lab/Data-Analyst-Agent.git
cd Data-Analyst-Agent
pip install -r requirements.txt

# Optional: enable the AI chat layer
export GROQ_API_KEY=your_key_here

streamlit run app.py
```

Open http://localhost:8501.

## Running the EDA notebooks

```bash
pip install jupyter
jupyter notebook eda/notebooks/
```

Open any step (1 → 7) and click **Run All**. Each notebook uses the same
`eda.step_N_*` modules the live app uses, so the walkthrough matches the
app's actual behaviour.

## Running the test suite

```bash
python tests/test_scalability.py
```

Validates large-file handling, edge cases (booleans, NA markers, duplicate
columns, mixed encodings), and confirms the 1M-row pipeline runs in seconds
with bounded memory.
