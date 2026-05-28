"""
src/features.py
===============
Advanced Feature Engineering & Scikit-learn Preprocessing Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
try:
    import category_encoders as ce
    HAS_CE = True
except ImportError:
    HAS_CE = False


class SimpleTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target encoder fallback (no category_encoders dependency)."""
    def __init__(self, cols=None, smoothing=10, min_samples_leaf=20):
        self.cols = cols or []
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y=None):
        if y is None:
            return self
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.global_mean_ = float(np.mean(y))
        self.mapping_ = {}
        y_arr = np.array(y)
        for col in self.cols:
            if col not in df.columns:
                continue
            tmp = pd.DataFrame({"col": df[col].values, "y": y_arr})
            agg = tmp.groupby("col")["y"].agg(["mean", "count"])
            smoothed = (agg["count"] * agg["mean"] + self.smoothing * self.global_mean_) / (agg["count"] + self.smoothing)
            self.mapping_[col] = smoothed.to_dict()
        return self

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in self.cols:
            if col in df.columns and col in self.mapping_:
                df[col] = df[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return df



# ─────────────────────────────────────────────────────────
# STEP 1: Deep Feature Creation (Custom Transformer)
# ─────────────────────────────────────────────────────────
class DeepFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Generates 8+ advanced cross-dataset features that materially improve R².
    All features are mathematically motivated and interpretable.
    """

    def fit(self, X, y=None):
        # Store zipcode-level medians for Market_Demand_Index
        df = pd.DataFrame(X.copy()) if not isinstance(X, pd.DataFrame) else X.copy()
        if "zipcode" in df.columns and "price" in df.columns:
            self.zip_price_median_ = df.groupby("zipcode")["price"].median().to_dict()
        else:
            self.zip_price_median_ = {}

        if "zipcode" in df.columns and "sqft_living" in df.columns:
            self.zip_sqft_median_ = df.groupby("zipcode")["sqft_living"].median().to_dict()
        else:
            self.zip_sqft_median_ = {}

        self.global_price_median_ = df["price"].median() if "price" in df.columns else 1e6
        self.global_sqft_median_  = df["sqft_living"].median() if "sqft_living" in df.columns else 1500
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X.copy()) if not isinstance(X, pd.DataFrame) else X.copy()

        # ── Feature 1: Property_Age_At_Sale ──────────────────────────────────
        # Age of house when sold – older homes typically depreciate
        if "sale_year" in df.columns and "yr_built" in df.columns:
            df["property_age"] = (df["sale_year"] - df["yr_built"]).clip(0)

        # ── Feature 2: Has_Been_Renovated ────────────────────────────────────
        if "yr_renovated" in df.columns:
            df["has_renovated"] = (df["yr_renovated"] > 0).astype(int)
            if "sale_year" in df.columns:
                df["years_since_renovation"] = np.where(
                    df["yr_renovated"] > 0,
                    df["sale_year"] - df["yr_renovated"],
                    df.get("property_age", 0)
                ).clip(0)

        # ── Feature 3: Basement_Ratio ─────────────────────────────────────────
        # Proportion of total area that is basement (below-grade space discount)
        if "sqft_basement" in df.columns and "sqft_living" in df.columns:
            denom = df["sqft_living"].replace(0, np.nan)
            df["basement_ratio"] = (df["sqft_basement"] / denom).fillna(0).clip(0, 1)

        # ── Feature 4: Market_Demand_Index ───────────────────────────────────
        # Zip-level median price relative to global median → local heat signal
        if "zipcode" in df.columns:
            zip_med = df["zipcode"].map(self.zip_price_median_).fillna(self.global_price_median_)
            df["market_demand_index"] = zip_med / self.global_price_median_

        # ── Feature 5: Price_Per_SqFt_Zip_Ratio ─────────────────────────────
        # Local $/sqft premium – captures neighborhood pricing power
        if "zipcode" in df.columns and "sqft_living" in df.columns:
            zip_sqft = df["zipcode"].map(self.zip_sqft_median_).fillna(self.global_sqft_median_)
            df["size_vs_neighborhood"] = df["sqft_living"] / zip_sqft.replace(0, np.nan).fillna(1)

        # ── Feature 6: School_Accessibility_Score ────────────────────────────
        # For India dataset: schools_nearby / log(airport_distance + 1)
        if "schools_nearby" in df.columns and "airport_distance_km" in df.columns:
            df["school_accessibility"] = (
                df["schools_nearby"] / np.log1p(df["airport_distance_km"])
            )

        # ── Feature 7: Total_Interior_Area ───────────────────────────────────
        # All enclosed living space including basement
        if "sqft_above" in df.columns and "sqft_basement" in df.columns:
            df["total_interior_area"] = df["sqft_above"] + df["sqft_basement"]

        # ── Feature 8: Grade_x_Condition ─────────────────────────────────────
        # Interaction of structural quality × physical state → premium signal
        if "grade" in df.columns and "condition" in df.columns:
            df["grade_condition_score"] = df["grade"] * df["condition"]

        # ── Feature 9: Waterfront_Premium_Flag ───────────────────────────────
        # Binary × view interaction (waterfront with views commands premium)
        if "waterfront" in df.columns and "view" in df.columns:
            df["waterfront_view_premium"] = df["waterfront"] * (df["view"] > 0).astype(int)

        # ── Feature 10: Sale_Season ──────────────────────────────────────────
        # Real estate has strong seasonality
        if "sale_month" in df.columns:
            df["sale_season"] = df["sale_month"].map(
                {12: "winter", 1: "winter", 2: "winter",
                 3: "spring", 4: "spring", 5: "spring",
                 6: "summer", 7: "summer", 8: "summer",
                 9: "fall",   10: "fall",  11: "fall"}
            ).fillna("spring")

        return df


# ─────────────────────────────────────────────────────────
# STEP 2: Define Column Groups After Feature Engineering
# ─────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "sqft_above", "sqft_basement",
    "yr_built", "yr_renovated", "lat", "long",
    "sqft_living15", "sqft_lot15",
    "schools_nearby", "airport_distance_km",
    "property_age", "has_renovated", "years_since_renovation",
    "basement_ratio", "market_demand_index", "size_vs_neighborhood",
    "school_accessibility", "total_interior_area",
    "grade_condition_score", "waterfront_view_premium",
    "real_price_yoy", "real_price_idx", "nom_price_yoy", "nom_price_idx",
    "latest_hpi"
]

ORDINAL_FEATURES = ["waterfront", "view", "condition", "grade", "sale_month",
                    "sale_year", "schools_nearby"]

CATEGORICAL_FEATURES = ["sale_season"]

TARGET_ENCODE_FEATURES = ["zipcode"]   # high-cardinality → target encode


def get_feature_columns(df: pd.DataFrame) -> dict:
    """Returns column lists intersected with what's actually in df."""
    present = set(df.columns) - {"price", "id", "date", "market_region",
                                  "country_code", "country"}
    return {
        "numeric":          [c for c in NUMERIC_FEATURES    if c in present],
        "ordinal":          [c for c in ORDINAL_FEATURES     if c in present],
        "categorical":      [c for c in CATEGORICAL_FEATURES if c in present],
        "target_encode":    [c for c in TARGET_ENCODE_FEATURES if c in present],
    }


# ─────────────────────────────────────────────────────────
# STEP 3: Build the Full Scikit-learn Pipeline
# ─────────────────────────────────────────────────────────
def build_preprocessing_pipeline(df_train: pd.DataFrame,
                                  y_train: pd.Series) -> ColumnTransformer:
    """
    Returns a fitted ColumnTransformer that handles:
      - Target Encoding for high-cardinality zipcode
      - OrdinalEncoder for low-cardinality ordinals
      - RobustScaler for all continuous numeric features
      - OneHot for season (low cardinality categorical)
    """
    col_groups = get_feature_columns(df_train)

    # Target encoder (uses y to encode)
    TE_CLASS = ce.TargetEncoder if HAS_CE else SimpleTargetEncoder
    te = TE_CLASS(cols=col_groups["target_encode"], smoothing=10, min_samples_leaf=20)

    # We build a two-stage approach:
    # Stage 1: DeepFeatureEngineer (applied before ColumnTransformer)
    # Stage 2: ColumnTransformer for scaling/encoding

    numeric_pipe = Pipeline([
        ("scaler", RobustScaler())
    ])

    ordinal_pipe = Pipeline([
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    from sklearn.preprocessing import OneHotEncoder
    ohe_pipe = Pipeline([
        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if col_groups["numeric"]:
        transformers.append(("num", numeric_pipe, col_groups["numeric"]))
    if col_groups["ordinal"]:
        transformers.append(("ord", ordinal_pipe, col_groups["ordinal"]))
    if col_groups["categorical"]:
        transformers.append(("cat", ohe_pipe, col_groups["categorical"]))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    return ct, te, col_groups


def prepare_features(df: pd.DataFrame, y: pd.Series = None,
                     engineer: DeepFeatureEngineer = None,
                     ct: ColumnTransformer = None,
                     te=None,
                     col_groups: dict = None,
                     fit: bool = False):
    """
    End-to-end feature preparation.
    If fit=True → fits all transformers.
    Returns (X_transformed, engineer, ct, te, col_groups)
    """
    if engineer is None:
        engineer = DeepFeatureEngineer()

    if fit:
        df_eng = engineer.fit_transform(df, y)
    else:
        df_eng = engineer.transform(df)

    if fit:
        ct, te, col_groups = build_preprocessing_pipeline(df_eng, y)

        # Target-encode the zipcode
        te_cols = col_groups["target_encode"]
        if te_cols:
            df_eng[te_cols] = te.fit_transform(df_eng[te_cols], y)

        X = ct.fit_transform(df_eng)
    else:
        te_cols = col_groups["target_encode"] if col_groups else []
        if te_cols:
            df_eng[te_cols] = te.transform(df_eng[te_cols])
        X = ct.transform(df_eng)

    return X, engineer, ct, te, col_groups


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from src.data_loader import load_enriched_primary
    from src.cleaning import run_cleaning_pipeline
    df = load_enriched_primary()
    df = run_cleaning_pipeline(df, apply_vif=False)
    y = df["price"]
    df_feat = df.drop(columns=["price"])
    eng = DeepFeatureEngineer().fit(df, y)
    df_out = eng.transform(df_feat)
    print("Engineered columns:", [c for c in df_out.columns if c not in df.columns])
