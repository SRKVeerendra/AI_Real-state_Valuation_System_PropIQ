"""
src/cleaning.py
===============
Production-Grade Data Cleaning Pipeline
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression


def variance_inflation_factor(X: np.ndarray, idx: int) -> float:
    """
    Pure scikit-learn VIF computation.
    VIF_i = 1 / (1 - R²) where R² is from regressing feature i on all others.
    """
    y = X[:, idx]
    X_other = np.delete(X, idx, axis=1)
    r2 = LinearRegression().fit(X_other, y).score(X_other, y)
    if r2 >= 1.0:
        return np.inf
    return 1.0 / (1.0 - r2)

# ─────────────────────────────────────────────────────────
# STEP 1: Structural Standardisation
# ─────────────────────────────────────────────────────────
def standardise_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Strip whitespace from string columns
    - Enforce correct dtypes
    - Drop pure-duplicate rows
    - Reset index
    """
    # Strip column name whitespace
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Strip whitespace in all object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Dtype enforcement map (post rename)
    dtype_map = {
        "bedrooms":           "int32",
        "bathrooms":          "float32",
        "sqft_living":        "float32",
        "sqft_lot":           "float32",
        "floors":             "float32",
        "waterfront":         "int8",
        "view":               "int8",
        "condition":          "int8",
        "grade":              "int8",
        "sqft_above":         "float32",
        "sqft_basement":      "float32",
        "yr_built":           "int32",
        "yr_renovated":       "int32",
        "zipcode":            "int32",
        "lat":                "float32",
        "long":               "float32",
        "sqft_living15":      "float32",
        "sqft_lot15":         "float32",
        "schools_nearby":     "int8",
        "airport_distance_km":"float32",
        "price":              "float64",
        "sale_year":          "int32",
        "sale_month":         "int8",
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    # Drop exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[cleaning] Duplicates removed: {before - len(df)}")

    return df


# ─────────────────────────────────────────────────────────
# STEP 2: Advanced Missing Value Treatment
# ─────────────────────────────────────────────────────────
def treat_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiered imputation strategy:
      - Boolean / flag columns → mode
      - Ordinal / count columns → zipcode-group median
      - Continuous columns → KNN Imputer (k=5)
    """
    flag_cols    = ["waterfront"]
    ordinal_cols = ["bedrooms", "bathrooms", "floors", "view",
                    "condition", "grade", "schools_nearby"]
    group_key    = "zipcode"

    # Boolean flags → mode
    for col in flag_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Ordinal / count → group median (fallback global median)
    for col in ordinal_cols:
        if col in df.columns and df[col].isnull().any():
            if group_key in df.columns:
                df[col] = df.groupby(group_key)[col].transform(
                    lambda x: x.fillna(x.median())
                )
            df[col] = df[col].fillna(df[col].median())

    # Continuous spatial/area → KNN Imputer
    knn_cols = [c for c in ["sqft_living", "sqft_lot", "sqft_above",
                             "sqft_basement", "sqft_living15", "sqft_lot15",
                             "lat", "long", "airport_distance_km",
                             "yr_built", "yr_renovated"]
                if c in df.columns and df[c].isnull().any()]

    if knn_cols:
        knn_data = df[knn_cols].copy()
        imputer  = KNNImputer(n_neighbors=5, weights="distance")
        df[knn_cols] = imputer.fit_transform(knn_data)
        print(f"[cleaning] KNN-imputed columns: {knn_cols}")

    # Target: drop rows where price is null (can't train without label)
    before = len(df)
    df.dropna(subset=["price"], inplace=True)
    print(f"[cleaning] Rows dropped (null price): {before - len(df)}")

    return df


# ─────────────────────────────────────────────────────────
# STEP 3: Multivariate Outlier Eradication (Isolation Forest)
# ─────────────────────────────────────────────────────────
def remove_outliers(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    """
    Uses Isolation Forest on numeric features + hard IQR guard on price.
    contamination=0.03 → flags ~3% of rows as outliers.
    """
    # --- Hard price IQR filter first ---
    Q1, Q3 = df["price"].quantile(0.01), df["price"].quantile(0.99)
    iqr_lo, iqr_hi = Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1)
    before = len(df)
    df = df[(df["price"] >= iqr_lo) & (df["price"] <= iqr_hi)].copy()
    print(f"[cleaning] IQR price filter removed: {before - len(df)}")

    # --- Isolation Forest on numeric feature matrix ---
    feature_cols = [c for c in [
        "sqft_living", "sqft_lot", "bedrooms", "bathrooms", "floors",
        "grade", "condition", "sqft_above", "sqft_basement",
        "yr_built", "lat", "long"
    ] if c in df.columns]

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    outlier_labels = iso.fit_predict(df[feature_cols])
    before = len(df)
    df = df[outlier_labels == 1].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"[cleaning] Isolation Forest removed: {before - len(df)} outliers")

    return df


# ─────────────────────────────────────────────────────────
# STEP 4: Multicollinearity Cleanse via VIF
# ─────────────────────────────────────────────────────────
def vif_cleanse(df: pd.DataFrame,
                numeric_features: list,
                threshold: float = 10.0) -> pd.DataFrame:
    """
    Iteratively drops the feature with highest VIF > threshold
    until all remaining features are below threshold.
    Returns df with offending columns dropped.
    """
    features = [f for f in numeric_features if f in df.columns]
    vif_data = df[features].dropna()

    while True:
        vif_series = pd.Series(
            [variance_inflation_factor(vif_data.values.astype(float), i)
             for i in range(vif_data.shape[1])],
            index=vif_data.columns
        )
        max_vif = vif_series.max()
        if max_vif <= threshold:
            break
        drop_col = vif_series.idxmax()
        print(f"[cleaning] VIF drop: {drop_col} (VIF={max_vif:.2f})")
        vif_data.drop(columns=[drop_col], inplace=True)

    # Drop from main DataFrame
    dropped = set(features) - set(vif_data.columns)
    df.drop(columns=list(dropped), inplace=True, errors="ignore")
    print(f"[cleaning] VIF cleanse complete. Dropped: {dropped or 'none'}")
    return df


# ─────────────────────────────────────────────────────────
# Master Cleaning Orchestrator
# ─────────────────────────────────────────────────────────
def run_cleaning_pipeline(df: pd.DataFrame,
                           apply_vif: bool = True) -> pd.DataFrame:
    """
    Runs all 4 cleaning steps in sequence and returns a clean DataFrame.
    """
    print("\n" + "="*55)
    print("  CLEANING PIPELINE START")
    print(f"  Input rows: {len(df):,}")
    print("="*55)

    df = standardise_structure(df)
    df = treat_missing_values(df)
    df = remove_outliers(df, contamination=0.03)

    if apply_vif:
        vif_candidates = [
            "sqft_living", "sqft_above", "sqft_basement",
            "sqft_living15", "sqft_lot", "sqft_lot15",
            "bedrooms", "bathrooms", "floors",
            "lat", "long"
        ]
        df = vif_cleanse(df, vif_candidates, threshold=10.0)

    print(f"\n[cleaning] ✓ Clean dataset: {len(df):,} rows × {df.shape[1]} cols")
    print("="*55 + "\n")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from src.data_loader import load_enriched_primary
    raw = load_enriched_primary()
    clean = run_cleaning_pipeline(raw)
    print(clean.describe().T)
