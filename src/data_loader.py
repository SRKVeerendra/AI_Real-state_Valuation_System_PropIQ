"""
src/data_loader.py
==================
Multi-Dataset Loader & Architectural Merging Strategy
======================================================

ARCHITECTURAL DECISION:
------------------------
These 8 datasets CANNOT be directly merged on a single shared key.
Instead, we use a HIERARCHICAL ENRICHMENT strategy:

PRIMARY DATASET  : House_Price_India.csv  (14,620 rows, richest feature set – property + amenities + geo)
SECONDARY        : Housing.csv / house_prices.csv  (21,613 rows, King County WA – used to validate / stack)
MACRO ENRICHMENT : real_year, real_index, nominal_year, nominal_index  (global price trends → macro features)
INDEX ENRICHMENT : housing_price_index_2010-11_100.csv  (quarterly city-level HPI → inflation adjustment)

Why NOT a direct merge?
  • real_*/nominal_* are country-level time-series with no property IDs.
  • housing_price_index has city aggregates (7 cities, 11 quarters) with no property IDs.
  • Housing.csv / house_prices.csv are King County WA (US), India CSV is India → no zip/lat overlap.
  
Strategy:
  1. Build two independent property-level datasets (India + King County).
  2. Enrich both with scalar macro features derived from the time-series and HPI datasets.
  3. Optionally train a stacked ensemble across both enriched sets with a 'market_region' flag.
  4. Primary model targets: House_Price_India (default), with cross-dataset validation.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


# ─────────────────────────────────────────────────────────
# 1.  Load the Primary India Dataset
# ─────────────────────────────────────────────────────────
def load_india_dataset() -> pd.DataFrame:
    path = DATA_DIR / "House_Price_India.csv"
    df = pd.read_csv(path, encoding="utf-8")

    # Standardise column names → snake_case
    df.rename(columns={
        "id": "id",
        "Date": "date_serial",
        "number of bedrooms": "bedrooms",
        "number of bathrooms": "bathrooms",
        "living area": "sqft_living",
        "lot area": "sqft_lot",
        "number of floors": "floors",
        "waterfront present": "waterfront",
        "number of views": "view",
        "condition of the house": "condition",
        "grade of the house": "grade",
        "Area of the house(excluding basement)": "sqft_above",
        "Area of the basement": "sqft_basement",
        "Built Year": "yr_built",
        "Renovation Year": "yr_renovated",
        "Postal Code": "zipcode",
        "Lattitude": "lat",
        "Longitude": "long",
        "living_area_renov": "sqft_living15",
        "lot_area_renov": "sqft_lot15",
        "Number of schools nearby": "schools_nearby",
        "Distance from the airport": "airport_distance_km",
        "Price": "price",
    }, inplace=True)

    # Convert Excel date serial → proper date (Date col is Excel ordinal)
    df["date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["date_serial"], unit="D")
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df.drop(columns=["date_serial"], inplace=True)

    df["market_region"] = "india"
    return df


# ─────────────────────────────────────────────────────────
# 2.  Load the King County (US) Dataset
# ─────────────────────────────────────────────────────────
def load_kc_dataset() -> pd.DataFrame:
    """Prefer House_Price_India for training; KC used for transfer/validation."""
    path = DATA_DIR / "house_prices.csv"
    df = pd.read_csv(path, encoding="utf-8")

    # Normalise waterfront & condition to numeric
    waterfront_map = {"N": 0, "Y": 1}
    condition_map = {"Poor": 1, "Fair": 2, "Average": 3, "Good": 4, "Very Good": 5}
    df["waterfront"] = df["waterfront"].map(waterfront_map).fillna(0).astype(int)
    df["condition"] = df["condition"].map(condition_map).fillna(3).astype(int)

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month

    # Add stub columns that India has but KC doesn't
    df["schools_nearby"] = 2          # median imputation placeholder
    df["airport_distance_km"] = 35    # King County median placeholder
    df["market_region"] = "us_kc"
    return df


# ─────────────────────────────────────────────────────────
# 3.  Load Global Macro Time-Series & Extract Scalar Features
# ─────────────────────────────────────────────────────────
def load_macro_features(sale_year: int = 2016) -> dict:
    """
    Derive scalar macro-economic features for a given sale year.
    Returns a dict of features that can be broadcast onto a DataFrame.
    """
    macro = {}

    for name, fname, enc in [
        ("real_price_yoy",   "real_year.csv",    "latin-1"),
        ("real_price_idx",   "real_index.csv",   "latin-1"),
        ("nom_price_yoy",    "nominal_year.csv",  "latin-1"),
        ("nom_price_idx",    "nominal_index.csv", "latin-1"),
    ]:
        df = pd.read_csv(DATA_DIR / fname, encoding=enc)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df_year = df[df["date"].dt.year == sale_year].copy()

        # Use world aggregate (country_code == "5R" = Advanced economies)
        subset = df_year[df_year["country_code"] == "5R"]["price"].dropna()
        macro[name] = float(subset.mean()) if not subset.empty else np.nan

    return macro


def load_hpi_latest() -> float:
    """Return the most-recent HPI value (last column of last available row)."""
    df = pd.read_csv(DATA_DIR / "housing_price_index_2010-11_100.csv", encoding="utf-8")
    df.columns = df.columns.str.strip()
    # All India row
    row = df[df["Particulars"].str.strip() == "All India"]
    if row.empty:
        return 200.0  # fallback
    numeric_cols = [c for c in df.columns if c != "Particulars"]
    return float(row.iloc[0][numeric_cols].dropna().iloc[-1])


# ─────────────────────────────────────────────────────────
# 4.  Master Loader – returns enriched primary DataFrame
# ─────────────────────────────────────────────────────────
def load_enriched_primary() -> pd.DataFrame:
    """
    Load House_Price_India, enrich with macro & HPI scalars.
    This is the main function called by the training pipeline.
    """
    df = load_india_dataset()

    macro = load_macro_features(sale_year=int(df["sale_year"].mode()[0]))
    for k, v in macro.items():
        df[k] = v  # broadcast scalar to all rows

    df["latest_hpi"] = load_hpi_latest()

    print(f"[data_loader] Enriched dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    df = load_enriched_primary()
    print(df.dtypes)
    print(df.head(3))
