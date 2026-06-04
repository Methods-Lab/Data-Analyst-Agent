# EDA Pipeline — 7 Steps

This folder holds the project's complete machine-learning pipeline split into
seven importable, runnable Python modules.

**These modules are imported and used by `app.py` at runtime.** They are not demos —
they are the actual entry points the live app calls.

## What's in here

```
eda/
├── __init__.py                 # exports all step symbols
├── step_1_loading.py           # data ingestion + first look
├── step_2_cleaning.py          # cleaning + NA handling
├── step_3_exploration.py       # profile, correlations, outliers
├── step_4_preprocessing.py     # encoding + split + target binning
├── step_5_training.py          # Decision Tree + Random Forest training
├── step_6_evaluation.py        # accuracy + CV + SHAP loaders
├── step_7_prediction.py        # deduce + abduce + Bayesian
├── notebooks/                  # visual walkthrough of each step
│   ├── step_1_data_loading.ipynb
│   ├── step_2_data_cleaning.ipynb
│   ├── step_3_exploratory_analysis.ipynb
│   ├── step_4_preprocessing.ipynb
│   ├── step_5_model_training.ipynb
│   ├── step_6_model_evaluation.ipynb
│   └── step_7_predictions.ipynb
└── README.md
```

## Run a step standalone

```bash
python -m eda.step_1_loading       # data ingestion demo
python -m eda.step_2_cleaning      # cleaning demo with dirty injection
python -m eda.step_3_exploration   # profiling + correlations
python -m eda.step_4_preprocessing # train/test split demo
python -m eda.step_5_training      # actually trains models
python -m eda.step_6_evaluation    # loads + prints saved metrics
python -m eda.step_7_prediction    # makes example predictions
```

Each prints a short report showing exactly what happened in that step.

## Run the visual notebooks

```bash
pip install jupyter
jupyter notebook eda/notebooks/
```

Open any step and click **Run All**. The notebooks use the same step modules
the app uses, so what you see in the notebook is what the app actually does.

## How this maps to the teacher's ML lifecycle

| Teacher's step | Our module |
|---|---|
| 1. Problem Understanding | (introduction — explained in main README) |
| 2. Data Collection | Step 1 (load_data / generate_synthetic_data) |
| 3. Data Understanding — EDA | Step 3 (profile_dataset, correlation_matrix, iqr_outliers) |
| 4. Data Understanding — Problem Finding | Step 3 + Step 2 (missing-value + outlier scan) |
| 5. Data Preprocessing | Step 2 (clean_data) |
| 6. Feature Selection | Step 2 + Step 4 (rule-based selection during cleaning + post-hoc SHAP) |
| 7. Feature Engineering | Step 4 (date splitting, target binning, encoding) |
| 8. Handling Class Imbalance | Step 4 (stratified split with safe fallback) |
| 9. Model Building | Step 5 (DecisionTree + RandomForest) |
| 10. Model Evaluation | Step 6 (test accuracy + 5-fold CV + confusion matrix + classification report) |
| 11. Final Conclusion | Step 6 — printed summary |
| 12. Deployment | `app.py` (Streamlit) — the live web frontend |

## Why this structure

- **One folder = one pipeline**. Open `eda/` and you see the entire ML workflow.
- **The app uses these files**. Notebooks aren't fake — they import the same code.
- **Each step is standalone**. You can `python -m eda.step_2_cleaning` to verify any step in isolation.
- **Notebooks for the teacher**. Visual walkthrough with markdown + plots for evaluation.
