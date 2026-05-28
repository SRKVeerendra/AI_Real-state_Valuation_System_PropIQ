"""
src/train.py
============
Competitive Modeling, Optuna Hyperparameter Tuning & Serialisation
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_enriched_primary
from src.cleaning import run_cleaning_pipeline
from src.features import DeepFeatureEngineer, prepare_features, get_feature_columns

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
N_FOLDS     = 3
OPTUNA_TRIALS = 10   # increase to 150+ for production


# ─────────────────────────────────────────────────────────
# Objective functions for Optuna
# ─────────────────────────────────────────────────────────
def xgb_objective(trial, X, y):
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 200, 600),
        "max_depth":          trial.suggest_int("max_depth", 4, 8),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
        "random_state":       RANDOM_SEED,
        "n_jobs":             -1,
        "tree_method":        "hist",
    }
    kf = KFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        model = xgb.XGBRegressor(**params)
        model.fit(X[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X[val_idx], y.iloc[val_idx])],
                  verbose=False)
        scores.append(r2_score(y.iloc[val_idx], model.predict(X[val_idx])))
    return np.mean(scores)


def lgb_objective(trial, X, y):
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 200, 600),
        "num_leaves":         trial.suggest_int("num_leaves", 20, 100),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples":  trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state":       RANDOM_SEED,
        "n_jobs":             -1,
        "verbose":            -1,
    }
    kf = KFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        model = lgb.LGBMRegressor(**params)
        model.fit(X[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X[val_idx], y.iloc[val_idx])])
        scores.append(r2_score(y.iloc[val_idx], model.predict(X[val_idx])))
    return np.mean(scores)


def cat_objective(trial, X, y):
    params = {
        "iterations":         trial.suggest_int("iterations", 200, 500),
        "depth":              trial.suggest_int("depth", 4, 8),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
        "bagging_temperature":trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "random_seed":        RANDOM_SEED,
        "verbose":            0,
    }
    kf = KFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        model = CatBoostRegressor(**params)
        model.fit(X[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X[val_idx], y.iloc[val_idx])],
                  verbose=False)
        scores.append(r2_score(y.iloc[val_idx], model.predict(X[val_idx])))
    return np.mean(scores)


# ─────────────────────────────────────────────────────────
# Tune a single model
# ─────────────────────────────────────────────────────────
def tune_model(name: str, objective_fn, X, y, n_trials: int) -> dict:
    print(f"\n  ▶ Tuning {name} ({n_trials} trials)…")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda trial: objective_fn(trial, X, y),
                   n_trials=n_trials, show_progress_bar=False)
    print(f"    Best {name} R²: {study.best_value:.4f}")
    return study.best_params


# ─────────────────────────────────────────────────────────
# 5-Fold Cross-Validation Evaluation
# ─────────────────────────────────────────────────────────
def cross_validate_model(model, X, y) -> dict:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    r2_scores, mae_scores, rmse_scores = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        model.fit(X[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X[val_idx])
        r2_scores.append(r2_score(y.iloc[val_idx], preds))
        mae_scores.append(mean_absolute_error(y.iloc[val_idx], preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))

    return {
        "r2_mean":   np.mean(r2_scores),
        "r2_std":    np.std(r2_scores),
        "mae_mean":  np.mean(mae_scores),
        "rmse_mean": np.mean(rmse_scores),
        "r2_folds":  r2_scores,
    }


# ─────────────────────────────────────────────────────────
# Full Training Orchestrator
# ─────────────────────────────────────────────────────────
def train():
    print("\n" + "="*60)
    print("  AI REAL ESTATE VALUATION — TRAINING PIPELINE")
    print("="*60)

    # ── Load & Clean ──────────────────────────────────────
    print("\n[1/6] Loading data…")
    df = load_enriched_primary()
    df = run_cleaning_pipeline(df, apply_vif=False)

    y = df["price"].reset_index(drop=True)
    df_feat = df.drop(columns=["price", "id", "date", "market_region"],
                      errors="ignore").reset_index(drop=True)

    # ── Feature Engineering ───────────────────────────────
    print("[2/6] Feature engineering…")
    X, engineer, ct, te, col_groups = prepare_features(
        df_feat, y=y, fit=True
    )
    print(f"  Feature matrix: {X.shape}")

    # ── Hyperparameter Tuning ─────────────────────────────
    print("\n[3/6] Hyperparameter optimisation (Optuna)…")
    xgb_params  = tune_model("XGBoost",  xgb_objective,  X, y, OPTUNA_TRIALS)
    lgb_params  = tune_model("LightGBM", lgb_objective,  X, y, OPTUNA_TRIALS)
    cat_params  = tune_model("CatBoost", cat_objective,  X, y, OPTUNA_TRIALS)

    # Add fixed params
    xgb_params.update({"random_state": RANDOM_SEED, "n_jobs": -1, "tree_method": "hist"})
    lgb_params.update({"random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1})
    cat_params.update({"random_seed": RANDOM_SEED, "verbose": 0})

    # ── 5-Fold CV Evaluation ──────────────────────────────
    print("\n[4/6] 5-Fold Cross-Validation…")
    models = {
        "XGBoost":  xgb.XGBRegressor(**xgb_params),
        "LightGBM": lgb.LGBMRegressor(**lgb_params),
        "CatBoost": CatBoostRegressor(**cat_params),
    }

    results = {}
    for name, model in models.items():
        cv_result = cross_validate_model(model, X, y)
        results[name] = cv_result
        print(f"\n  {name}:")
        print(f"    R²  = {cv_result['r2_mean']:.4f} ± {cv_result['r2_std']:.4f}")
        print(f"    MAE = {cv_result['mae_mean']:,.0f}")
        print(f"    RMSE= {cv_result['rmse_mean']:,.0f}")

    # ── Select Winner ─────────────────────────────────────
    print("\n[5/6] Selecting best model…")
    best_name = max(results, key=lambda n: results[n]["r2_mean"])
    best_cv   = results[best_name]
    print(f"  🏆 Winner: {best_name}  (R² = {best_cv['r2_mean']:.4f})")

    # Re-train winner on FULL dataset
    winner_model = models[best_name]
    winner_model.fit(X, y)

    # ── Extract Feature Importance ────────────────────────
    try:
        feat_names = (
            list(col_groups.get("numeric", [])) +
            list(col_groups.get("ordinal", [])) +
            [f"cat_{i}" for i in range(len(col_groups.get("categorical", [])))]
        )
        if hasattr(winner_model, "feature_importances_"):
            imp = winner_model.feature_importances_
            fi  = pd.Series(imp[:len(feat_names)], index=feat_names[:len(imp)])
        else:
            fi = pd.Series(dtype=float)
        fi.sort_values(ascending=False, inplace=True)
    except Exception:
        fi = pd.Series(dtype=float)

    # ── Serialise Everything ──────────────────────────────
    print("\n[6/6] Serialising artefacts…")

    joblib.dump(winner_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(engineer,     MODELS_DIR / "feature_engineer.joblib")
    joblib.dump(ct,           MODELS_DIR / "column_transformer.joblib")
    joblib.dump(te,           MODELS_DIR / "target_encoder.joblib")
    joblib.dump(col_groups,   MODELS_DIR / "col_groups.joblib")
    joblib.dump(fi,           MODELS_DIR / "feature_importance.joblib")

    metadata = {
        "best_model":    best_name,
        "r2_mean":       float(best_cv["r2_mean"]),
        "r2_std":        float(best_cv["r2_std"]),
        "mae_mean":      float(best_cv["mae_mean"]),
        "rmse_mean":     float(best_cv["rmse_mean"]),
        "r2_folds":      [float(x) for x in best_cv["r2_folds"]],
        "all_results":   {
            k: {kk: float(vv) if not isinstance(vv, list) else [float(x) for x in vv]
                for kk, vv in v.items()}
            for k, v in results.items()
        },
        "n_train_rows":  int(len(y)),
        "n_features":    int(X.shape[1]),
        "xgb_params":    xgb_params,
        "lgb_params":    lgb_params,
        "cat_params":    cat_params,
    }
    with open(MODELS_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✓ All artefacts saved to: {MODELS_DIR}")
    print("="*60)
    return metadata


if __name__ == "__main__":
    meta = train()
    print(f"\nFinal R² = {meta['r2_mean']:.4f}")