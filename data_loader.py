"""
data_loader.py
--------------
Handles loading, cleaning, and preprocessing of tabular data (Excel/CSV).
Supports uploaded files or auto-generated synthetic Pakistan sales data.
"""

import io
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Synthetic data generator (Faker-based) – used as fallback
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic Pakistan-themed e-commerce sales dataset.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate (default 10,000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame with 17 feature columns and a target column.
    """
    try:
        from faker import Faker
    except ImportError:
        raise ImportError("Install Faker: pip install faker")

    rng = np.random.default_rng(seed)
    fake = Faker("en_PK")
    Faker.seed(seed)

    cities = [
        "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad",
        "Multan", "Peshawar", "Quetta", "Sialkot", "Hyderabad",
        "Gujranwala", "Abbottabad", "Bahawalpur", "Sargodha", "Sukkur",
    ]
    regions = {
        "Karachi": "Sindh", "Hyderabad": "Sindh", "Sukkur": "Sindh",
        "Lahore": "Punjab", "Faisalabad": "Punjab", "Multan": "Punjab",
        "Rawalpindi": "Punjab", "Gujranwala": "Punjab", "Sialkot": "Punjab",
        "Bahawalpur": "Punjab", "Sargodha": "Punjab",
        "Islamabad": "Federal",
        "Peshawar": "KPK", "Abbottabad": "KPK",
        "Quetta": "Balochistan",
    }
    categories = [
        "Electronics", "Clothing", "Home & Kitchen", "Books",
        "Sports", "Beauty", "Grocery", "Toys", "Automotive", "Health",
    ]
    payment_methods = ["Cash on Delivery", "Credit Card", "Debit Card", "EasyPaisa", "JazzCash"]
    seasons = ["Spring", "Summer", "Autumn", "Winter"]

    order_ids = [f"ORD-{i:06d}" for i in range(1, n_rows + 1)]
    dates = pd.date_range("2020-01-01", "2023-12-31", periods=n_rows)
    customer_cities = rng.choice(cities, size=n_rows)
    product_categories = rng.choice(categories, size=n_rows)

    # Price logic: Electronics expensive, Grocery cheap
    base_price = {
        "Electronics": 35_000, "Clothing": 3_500, "Home & Kitchen": 8_000,
        "Books": 800, "Sports": 5_000, "Beauty": 2_500,
        "Grocery": 1_200, "Toys": 2_000, "Automotive": 15_000, "Health": 3_000,
    }
    unit_price = np.array([base_price[c] for c in product_categories], dtype=float)
    unit_price *= rng.uniform(0.7, 1.5, size=n_rows)

    quantity = rng.integers(1, 11, size=n_rows)
    discount_pct = rng.uniform(0, 30, size=n_rows).round(2)
    gross_sales = unit_price * quantity
    discount_amount = gross_sales * discount_pct / 100
    net_sales = (gross_sales - discount_amount).round(2)

    # City-tier affects shipping
    tier1 = {"Karachi", "Lahore", "Islamabad"}
    shipping_cost = np.where(
        np.isin(customer_cities, list(tier1)),
        rng.uniform(100, 300, size=n_rows),
        rng.uniform(200, 500, size=n_rows),
    ).round(2)

    customer_rating = rng.uniform(1, 5, size=n_rows).round(1)
    return_flag = rng.choice([0, 1], size=n_rows, p=[0.88, 0.12])
    repeat_customer = rng.choice([0, 1], size=n_rows, p=[0.55, 0.45])
    delivery_days = rng.integers(1, 15, size=n_rows)
    month = dates.month
    season = pd.cut(
        month, bins=[0, 3, 6, 9, 12],
        labels=["Winter", "Spring", "Summer", "Autumn"],
    )

    df = pd.DataFrame({
        "order_id": order_ids,
        "date": dates,
        "customer_city": customer_cities,
        "region": [regions[c] for c in customer_cities],
        "product_category": product_categories,
        "unit_price": unit_price.round(2),
        "quantity": quantity,
        "discount_pct": discount_pct,
        "discount_amount": discount_amount.round(2),
        "gross_sales": gross_sales.round(2),
        "net_sales": net_sales,
        "shipping_cost": shipping_cost,
        "payment_method": rng.choice(payment_methods, size=n_rows),
        "customer_rating": customer_rating,
        "return_flag": return_flag,
        "repeat_customer": repeat_customer,
        "delivery_days": delivery_days,
        "season": season.astype(str),
    })

    # Target: sales_category
    low_q = df["net_sales"].quantile(0.33)
    high_q = df["net_sales"].quantile(0.67)
    df["sales_category"] = pd.cut(
        df["net_sales"],
        bins=[-np.inf, low_q, high_q, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)

    return df


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_data(source) -> pd.DataFrame:
    """
    Load data from a file path, URL, or uploaded file-like object.

    Parameters
    ----------
    source : str | Path | BytesIO
        - A local file path (.xlsx, .csv)
        - A direct-download URL
        - A Streamlit UploadedFile / BytesIO object

    Returns
    -------
    pd.DataFrame
        Raw (uncleaned) DataFrame.
    """
    if source is None:
        raise ValueError("No data source provided.")

    # Streamlit UploadedFile or BytesIO
    if hasattr(source, "read"):
        name = getattr(source, "name", "upload.csv")
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(source, engine="openpyxl")
        return pd.read_csv(source)

    # URL
    if isinstance(source, str) and source.startswith("http"):
        import urllib.request
        with urllib.request.urlopen(source) as resp:
            raw = io.BytesIO(resp.read())
        if source.endswith(".xlsx") or source.endswith(".xls"):
            return pd.read_excel(raw, engine="openpyxl")
        return pd.read_csv(raw)

    # Local path
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Cleaning & preprocessing
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame, target_col: str = "sales_category") -> pd.DataFrame:
    """
    Clean and preprocess a DataFrame for training.

    Steps
    -----
    1. Drop columns with >60 % missing values.
    2. Impute numeric NaNs with median; categorical NaNs with mode.
    3. Parse date columns and extract year/month/day_of_week features.
    4. Create ``sales_category`` target if a numeric ``net_sales``/``sales``
       column exists and target is absent.
    5. Encode categoricals as integer codes (stored in mapping dict).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.
    target_col : str
        Name of the target column (default ``sales_category``).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for ML.
    """
    df = df.copy()

    # Drop near-empty columns
    threshold = 0.6 * len(df)
    df = df.dropna(axis=1, thresh=int(len(df) - threshold))

    # Drop ID-like columns (unique ratio > 90 %)
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() / len(df) > 0.9:
            df = df.drop(columns=[col])

    # Parse date columns
    for col in df.select_dtypes(include="object").columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    df[col + "_year"] = parsed.dt.year
                    df[col + "_month"] = parsed.dt.month
                    df[col + "_dow"] = parsed.dt.dayofweek
                    df = df.drop(columns=[col])
            except Exception:
                pass

    for col in df.select_dtypes(include=["datetime64"]):
        df[col + "_year"] = df[col].dt.year
        df[col + "_month"] = df[col].dt.month
        df[col + "_dow"] = df[col].dt.dayofweek
        df = df.drop(columns=[col])

    # Create target if absent
    if target_col not in df.columns:
        numeric_candidates = [c for c in ["net_sales", "sales", "revenue", "amount"] if c in df.columns]
        if numeric_candidates:
            sales_col = numeric_candidates[0]
            low_q = df[sales_col].quantile(0.33)
            high_q = df[sales_col].quantile(0.67)
            df[target_col] = pd.cut(
                df[sales_col],
                bins=[-np.inf, low_q, high_q, np.inf],
                labels=["low", "medium", "high"],
            ).astype(str)

    # Impute
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

    # Encode categoricals
    for col in df.select_dtypes(include="object").columns:
        if col == target_col:
            continue
        df[col] = df[col].astype("category").cat.codes

    return df


def infer_column_types(df: pd.DataFrame) -> dict:
    """
    Return a summary dict of column types in the raw DataFrame.

    Returns
    -------
    dict with keys ``numeric``, ``categorical``, ``datetime``.
    """
    summary = {"numeric": [], "categorical": [], "datetime": []}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            summary["datetime"].append(col)
        else:
            try:
                pd.to_datetime(df[col], infer_datetime_format=True, errors="raise")
                summary["datetime"].append(col)
            except Exception:
                summary["categorical"].append(col)
    return summary
