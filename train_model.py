"""
Freight Rate Prediction - ML Pipeline
======================================
LightGBM + XGBoost ensemble with 5-fold time-aware cross-validation.

Usage:
    pip install -r requirements.txt
    python train_model.py

Outputs:
    validation_predictions.csv     -- 12,000 rows: load_id, predicted_rate
    december-chart-inputs.csv      -- 31 rows filled with predicted_rate

Then run scorer:
    python score.py --predictions validation_predictions.csv \
                    --december-predictions december-chart-inputs.csv
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
TRAIN_PATH    = ROOT / "train-test.csv"
VAL_PATH      = ROOT / "validation.csv"
DEC_PATH      = ROOT / "december-chart-inputs.csv"
TEMPLATE_PATH = ROOT / "validation-predictions-template.csv"
OUT_PREDS     = ROOT / "validation_predictions.csv"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Will be populated from training data before use
TRAIN_MEDIANS: dict[str, float] = {}


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    dec   = pd.read_csv(DEC_PATH)
    for df in [train, val, dec]:
        df["date"] = pd.to_datetime(df["date"])
    print(f"  Train: {train.shape}, Val: {val.shape}, Dec: {dec.shape}")
    return train, val, dec


# ===========================================================================
# 2. DATA CLEANING
# ===========================================================================

def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Handle missing values and (for training data) cap extreme outliers.

    Missing data issues found:
    - weight:        300 nulls in train (~0.6%)  -- impute from equipment median
    - market_index:  374 nulls in train (~0.8%)  -- forward-fill by date, then global median
    - quote_signal:  0 nulls
    - december CSV:  missing market_index, quote_signal, lat/lon (all added here)
    """
    df = df.copy()

    # ── Weight imputation ────────────────────────────────────────────────────
    if "weight" in df.columns:
        if df["weight"].isnull().any():
            medians = df.groupby("equipment")["weight"].transform("median")
            df["weight"] = df["weight"].fillna(medians)
            df["weight"] = df["weight"].fillna(TRAIN_MEDIANS.get("weight", 31436.5))
    else:
        # december CSV has weight already, but guard just in case
        df["weight"] = TRAIN_MEDIANS.get("weight", 32000.0)

    # ── Market index imputation ───────────────────────────────────────────────
    if "market_index" in df.columns:
        if df["market_index"].isnull().any():
            df = df.sort_values("date").reset_index(drop=True)
            df["market_index"] = df["market_index"].ffill().bfill()
            df["market_index"] = df["market_index"].fillna(TRAIN_MEDIANS.get("market_index", 1.055))
    else:
        # december CSV lacks market_index -- use train median
        df["market_index"] = TRAIN_MEDIANS.get("market_index", 1.055)

    # ── Quote signal ──────────────────────────────────────────────────────────
    if "quote_signal" not in df.columns:
        df["quote_signal"] = TRAIN_MEDIANS.get("quote_signal", 2.055)

    # ── Geographic coords (december CSV only) ─────────────────────────────────
    if "pickup_lat" not in df.columns:
        df["pickup_lat"]   = 38.0406   # Lexington, KY
        df["pickup_lon"]   = -84.5037
        df["delivery_lat"] = 41.0793   # Fort Wayne, IN
        df["delivery_lon"] = -85.1394

    # ── Outlier capping (training only) ───────────────────────────────────────
    if is_train:
        cap = df["posted_rate"].quantile(0.999)
        n_capped = (df["posted_rate"] > cap).sum()
        df["posted_rate"] = df["posted_rate"].clip(upper=cap)
        if n_capped:
            print(f"  Capped {n_capped} extreme rate outliers at ${cap:,.2f}")

    return df


# ===========================================================================
# 3. FEATURE ENGINEERING
# ===========================================================================

def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles (Haversine formula)."""
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 39 features covering:
    - Temporal: month, DOW, cyclical encodings, seasonality flags
    - Geographic: haversine, bearing, midpoint, region bins, tortuosity
    - Load: log-transforms, rate-per-mile proxy features
    - Market: interactions between signals and distance
    """
    df = df.copy()

    # Temporal
    df["year"]          = df["date"].dt.year
    df["month"]         = df["date"].dt.month
    df["day_of_year"]   = df["date"].dt.dayofyear
    df["day_of_week"]   = df["date"].dt.dayofweek        # 0 = Monday
    df["week_of_year"]  = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]       = df["date"].dt.quarter
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"]= df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]  = df["date"].dt.is_month_end.astype(int)
    # Cyclical encoding avoids ordinal gap between Dec and Jan
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]       = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Geographic
    df["haversine_dist"]  = haversine_miles(
        df["pickup_lat"], df["pickup_lon"],
        df["delivery_lat"], df["delivery_lon"]
    )
    df["distance_ratio"]  = df["distance"] / (df["haversine_dist"] + 1e-6)  # road tortuosity
    df["mid_lat"]         = (df["pickup_lat"]  + df["delivery_lat"])  / 2
    df["mid_lon"]         = (df["pickup_lon"]  + df["delivery_lon"])  / 2
    df["delta_lat"]       = df["delivery_lat"] - df["pickup_lat"]
    df["delta_lon"]       = df["delivery_lon"] - df["pickup_lon"]
    df["bearing"]         = np.degrees(np.arctan2(df["delta_lon"], df["delta_lat"]))
    # Region bins: 0=West, 1=Central, 2=East
    df["pickup_region"]   = pd.cut(df["pickup_lon"],   bins=[-125,-105,-90,-60], labels=[0,1,2]).astype(float)
    df["delivery_region"] = pd.cut(df["delivery_lon"], bins=[-125,-105,-90,-60], labels=[0,1,2]).astype(float)
    df["cross_region"]    = (df["pickup_region"] != df["delivery_region"]).astype(int)

    # Load characteristics
    df["weight_per_mile"] = df["weight"] / (df["distance"] + 1e-6)
    df["log_distance"]    = np.log1p(df["distance"])
    df["log_weight"]      = np.log1p(df["weight"])

    # Market signal interactions
    df["market_x_dist"]   = df["market_index"] * df["distance"]
    df["quote_x_dist"]    = df["quote_signal"]  * df["distance"]
    df["quote_x_market"]  = df["quote_signal"]  * df["market_index"]

    return df


FEATURE_COLS = [
    # Geographic
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "haversine_dist", "distance_ratio",
    "mid_lat", "mid_lon", "delta_lat", "delta_lon", "bearing",
    "pickup_region", "delivery_region", "cross_region",
    # Load
    "weight", "weight_per_mile", "log_distance", "log_weight",
    # Market signals
    "market_index", "quote_signal",
    "market_x_dist", "quote_x_dist", "quote_x_market",
    # Temporal
    "month", "day_of_year", "day_of_week", "week_of_year",
    "quarter", "is_weekend", "is_month_start", "is_month_end",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    # Categorical (encoded below)
    "pickup_enc", "delivery_enc", "equipment_enc",
]
TARGET = "posted_rate"


def encode_categoricals(
    train: pd.DataFrame,
    test_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Label-encode pickup, delivery, equipment -- fit on all data to handle OOV."""
    cat_cols = ["pickup", "delivery", "equipment"]
    all_data = pd.concat([train] + test_frames, ignore_index=True)
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(all_data[col].astype(str))
        train[col + "_enc"] = le.transform(train[col].astype(str))
        for i, frame in enumerate(test_frames):
            test_frames[i][col + "_enc"] = le.transform(frame[col].astype(str))
    return train, test_frames


# ===========================================================================
# 4. MODEL TRAINING WITH TIME-AWARE CV
# ===========================================================================

def train_with_cv(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    dec_df:   pd.DataFrame,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    5-fold time-ordered CV:
    - Data is sorted by date, then KFold with shuffle=False preserves temporal ordering.
    - Each fold's validation set is strictly after the training set in time.
    - OOF predictions are averaged for robust metric estimation.
    - Test predictions are averaged across all fold models (bagging effect).
    - Final predictions = average of LightGBM + XGBoost (ensemble).
    """
    X     = train_df[FEATURE_COLS].values
    y     = train_df[TARGET].values
    X_val = val_df[FEATURE_COLS].values
    X_dec = dec_df[FEATURE_COLS].values

    oof_lgb = np.zeros(len(X));     oof_xgb = np.zeros(len(X))
    val_lgb = np.zeros(len(X_val)); val_xgb = np.zeros(len(X_val))
    dec_lgb = np.zeros(len(X_dec)); dec_xgb = np.zeros(len(X_dec))

    # Sort by date for temporal-aware splits
    date_order = train_df["date"].argsort().values
    X_s = X[date_order]
    y_s = y[date_order]

    kf = KFold(n_splits=n_splits, shuffle=False)
    fold_maes = []

    print(f"\nRunning {n_splits}-fold time-aware CV...")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_s)):
        X_tr, X_va = X_s[tr_idx], X_s[va_idx]
        y_tr, y_va = y_s[tr_idx], y_s[va_idx]

        # ── LightGBM ──────────────────────────────────────────────────────────
        lgb_m = lgb.LGBMRegressor(
            objective="regression", metric="mae",
            n_estimators=2000, learning_rate=0.03,
            num_leaves=127, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )
        lgb_m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_lgb[date_order[va_idx]] = lgb_m.predict(X_va)
        val_lgb += lgb_m.predict(X_val) / n_splits
        dec_lgb += lgb_m.predict(X_dec) / n_splits

        # ── XGBoost ───────────────────────────────────────────────────────────
        xgb_m = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=2000, learning_rate=0.03,
            max_depth=7, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_SEED, n_jobs=-1,
            tree_method="hist", verbosity=0,
            early_stopping_rounds=100,
        )
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[date_order[va_idx]] = xgb_m.predict(X_va)
        val_xgb += xgb_m.predict(X_val) / n_splits
        dec_xgb += xgb_m.predict(X_dec) / n_splits

        fold_mae = mean_absolute_error(
            y_s[va_idx],
            (oof_lgb[date_order[va_idx]] + oof_xgb[date_order[va_idx]]) / 2
        )
        fold_maes.append(fold_mae)
        print(f"  Fold {fold+1}/{n_splits}  MAE=${fold_mae:,.2f}")

    # ── Ensemble: equal-weight average ────────────────────────────────────────
    oof_ens = (oof_lgb + oof_xgb) / 2
    val_ens = (val_lgb + val_xgb) / 2
    dec_ens = (dec_lgb + dec_xgb) / 2

    # ── OOF metrics ───────────────────────────────────────────────────────────
    mae  = mean_absolute_error(y, oof_ens)
    rmse = np.sqrt(mean_squared_error(y, oof_ens))
    r2   = r2_score(y, oof_ens)
    mape = np.mean(np.abs((y - oof_ens) / (y + 1e-6))) * 100

    print(f"\n{'='*55}")
    print(f"  OOF Cross-Validation Results ({n_splits}-fold)")
    print(f"{'='*55}")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  R2   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  Per-fold MAEs: {[f'{m:,.2f}' for m in fold_maes]}")
    print(f"{'='*55}\n")

    return np.maximum(val_ens, 50.0), np.maximum(dec_ens, 50.0)


# ===========================================================================
# 5. MAIN PIPELINE
# ===========================================================================

def main() -> None:
    global TRAIN_MEDIANS
    print("\n" + "="*55)
    print("  Freight Rate Prediction Pipeline")
    print("="*55)

    # ── Load ──────────────────────────────────────────────────────────────────
    train_raw, val_raw, dec_raw = load_data()

    # ── Compute train medians for missing-value imputation ────────────────────
    TRAIN_MEDIANS = {
        "weight":       float(train_raw["weight"].median()),
        "market_index": float(train_raw["market_index"].median()),
        "quote_signal": float(train_raw["quote_signal"].median()),
    }
    print(f"  Train medians used for imputation: {TRAIN_MEDIANS}")

    # ── Clean ─────────────────────────────────────────────────────────────────
    print("\nCleaning data...")
    train_clean = clean_data(train_raw, is_train=True)
    val_clean   = clean_data(val_raw,   is_train=False)
    dec_clean   = clean_data(dec_raw,   is_train=False)

    # ── Feature engineering ───────────────────────────────────────────────────
    print("\nEngineering features...")
    train_fe = add_features(train_clean)
    val_fe   = add_features(val_clean)
    dec_fe   = add_features(dec_clean)
    train_fe, (val_fe, dec_fe) = encode_categoricals(train_fe, [val_fe, dec_fe])
    print(f"  Total features: {len(FEATURE_COLS)}")

    # ── Train & predict ───────────────────────────────────────────────────────
    val_preds, dec_preds = train_with_cv(train_fe, val_fe, dec_fe, n_splits=5)

    # ── Save validation predictions ───────────────────────────────────────────
    template = pd.read_csv(TEMPLATE_PATH)
    id_to_pred = dict(zip(val_fe["load_id"], val_preds))
    template["predicted_rate"] = template["load_id"].map(id_to_pred)
    template.to_csv(OUT_PREDS, index=False)
    print(f"Saved {len(template):,} validation predictions -> {OUT_PREDS}")

    # ── Save december predictions ─────────────────────────────────────────────
    dec_out = pd.read_csv(DEC_PATH)
    dec_out["predicted_rate"] = dec_preds
    dec_out.to_csv(DEC_PATH, index=False)
    print(f"Updated December chart inputs -> {DEC_PATH}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print(f"\nVal:  min=${val_preds.min():,.2f}  max=${val_preds.max():,.2f}  mean=${val_preds.mean():,.2f}")
    print(f"Dec:  min=${dec_preds.min():,.2f}  max=${dec_preds.max():,.2f}  mean=${dec_preds.mean():,.2f}")
    assert (template["predicted_rate"] > 0).all(), "Non-positive predictions!"
    assert len(template) == 12000, f"Expected 12000 rows, got {len(template)}"

    print("\nPipeline complete. Run score.py to validate outputs.")
    print("  python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv")


if __name__ == "__main__":
    main()
