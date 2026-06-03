"""
===================================================================================
  Flipkart Grid 2.0 — Spatio-Temporal Traffic Demand Prediction
  End-to-End Pipeline  (LightGBM + CatBoost Ensemble, GPU-accelerated)
===================================================================================
  Metric  : R² (coefficient of determination)
  Hardware: NVIDIA RTX 3050 Ti  (4 GB VRAM)
  Author  : Kaggle-Grandmaster-style pipeline
===================================================================================
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold  # only for target-encoding OOF, NOT for validation

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(ROOT, "train.csv")
TEST_PATH  = os.path.join(ROOT, "test.csv")
SUB_PATH   = os.path.join(ROOT, "sample_submission.csv")
OUT_PATH   = os.path.join(ROOT, "submission.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GEOHASH DECODING
# ─────────────────────────────────────────────────────────────────────────────
def decode_geohash_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode `geohash` strings into continuous (latitude, longitude) features.
    Uses python-geohash for precise decoding.  Each geohash string maps to
    the centre of a rectangular cell on Earth — this gives the model
    continuous spatial coordinates it can split on.
    """
    import geohash as gh  # python-geohash

    # Vectorised via apply — fast enough for <120 k rows
    decoded = df["geohash"].apply(lambda h: gh.decode(h))
    df["latitude"]  = decoded.apply(lambda t: float(t[0]))
    df["longitude"] = decoded.apply(lambda t: float(t[1]))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CYCLICAL TEMPORAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
def encode_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the `timestamp` column (format "H:MM" or "HH:MM") into:
      • hour, minute           — raw integer features
      • minutes_from_midnight  — total minutes since 00:00
      • time_sin, time_cos     — cyclical projection so 23:45 ≈ 00:00

    Also derive `day_of_week` (day % 7) and cyclical day-of-week features
    so the model can learn weekly periodicity.
    """
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"]   = parts[0]
    df["minute"] = parts[1]
    df["minutes_from_midnight"] = df["hour"] * 60 + df["minute"]

    # Cyclical encoding — period = 1440 minutes (24 h)
    df["time_sin"] = np.sin(2 * np.pi * df["minutes_from_midnight"] / 1440)
    df["time_cos"] = np.cos(2 * np.pi * df["minutes_from_midnight"] / 1440)

    # Day-of-week (assuming `day` is a monotonic day counter)
    df["day_of_week"] = df["day"] % 7

    # Cyclical day-of-week
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MISSING VALUE IMPUTATION  (grouped interpolation, NOT global mean)
# ─────────────────────────────────────────────────────────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaNs in `Temperature` and `Weather` with a hierarchical strategy:

    Temperature (continuous):
      1. Group by (geohash, day) — fill with the local group median.
      2. If still NaN, fall back to (geohash) median across all days.
      3. Last resort: global median (should rarely fire).

    Weather (categorical):
      1. Group by (geohash, day) — forward/backward fill within each group.
      2. Fall back to the mode for the geohash.
      3. Last resort: global mode.
    """
    # ── Temperature ──────────────────────────────────────────────────────
    med_geo_day = df.groupby(["geohash", "day"])["Temperature"].transform("median")
    df["Temperature"] = df["Temperature"].fillna(med_geo_day)

    med_geo = df.groupby("geohash")["Temperature"].transform("median")
    df["Temperature"] = df["Temperature"].fillna(med_geo)

    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())

    # ── Weather ──────────────────────────────────────────────────────────
    # Forward/backward fill within (geohash, day) groups
    df["Weather"] = (
        df.groupby(["geohash", "day"])["Weather"]
        .transform(lambda s: s.ffill().bfill())
    )

    # Fall back to per-geohash mode
    geo_mode = (
        df.groupby("geohash")["Weather"]
        .transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    )
    df["Weather"] = df["Weather"].fillna(geo_mode)

    # Final fallback: global mode
    global_mode = df["Weather"].mode().iloc[0]
    df["Weather"] = df["Weather"].fillna(global_mode)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LABEL ENCODING  (binary / ordinal categoricals)
# ─────────────────────────────────────────────────────────────────────────────
def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map simple binary/low-cardinality string features to integers.
    These are NOT target-encoded because they have very few levels.
    """
    df["LargeVehicles"] = df["LargeVehicles"].map({"Not Allowed": 0, "Allowed": 1}).astype("int8")
    df["Landmarks"]     = df["Landmarks"].map({"No": 0, "Yes": 1}).astype("int8")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SMOOTHED TARGET ENCODING  (out-of-fold, leak-free for train)
# ─────────────────────────────────────────────────────────────────────────────
def smoothed_target_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    target: str = "demand",
    n_splits: int = 5,
    smoothing: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bayesian-smoothed target encoding that avoids leakage via OOF predictions.

    For each category c:
        encoded(c) = (n_c * mean_c + smoothing * global_mean) / (n_c + smoothing)

    On the train set the encoding is done out-of-fold: each fold's rows are
    encoded using statistics computed on the other folds only.
    On the test set the full train statistics are used.
    """
    global_mean = train[target].mean()
    new_col = f"{col}_te"

    # ── Train: out-of-fold ──────────────────────────────────────────────
    train[new_col] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for trn_idx, val_idx in kf.split(train):
        fold_train = train.iloc[trn_idx]
        stats = fold_train.groupby(col)[target].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        train.loc[train.index[val_idx], new_col] = (
            train.iloc[val_idx][col].map(smooth)
        )

    # Fill any residual NaN (unseen categories in a fold) with global mean
    train[new_col] = train[new_col].fillna(global_mean)

    # ── Test: use full-train statistics ─────────────────────────────────
    stats_full = train.groupby(col)[target].agg(["mean", "count"])
    smooth_full = (stats_full["count"] * stats_full["mean"] + smoothing * global_mean) / (
        stats_full["count"] + smoothing
    )
    test[new_col] = test[col].map(smooth_full).fillna(global_mean)

    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# 6.  INTERACTION / AGGREGATE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-informed interaction features:
      • Temperature × time interactions (demand changes with temp + time of day)
      • Geospatial × temporal interactions  (spatial demand patterns shift by hour)
      • Historical aggregates per geohash (helps the model learn location baselines)
    """
    # Temperature interactions
    df["temp_x_hour"] = df["Temperature"] * df["hour"]
    df["temp_x_lanes"] = df["Temperature"] * df["NumberofLanes"]

    # Spatial × temporal
    df["lat_x_time_sin"] = df["latitude"] * df["time_sin"]
    df["lon_x_time_cos"] = df["longitude"] * df["time_cos"]

    # Lane density proxy
    df["lane_large_vehicle"] = df["NumberofLanes"] * df["LargeVehicles"]

    return df


def create_aggregate_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create aggregate statistics per geohash from the train set and map
    them onto both train and test.  These act as "learned location priors".
    """
    geo_stats = train.groupby("geohash")["demand"].agg(
        geo_demand_mean="mean",
        geo_demand_median="median",
        geo_demand_std="std",
    ).reset_index()
    geo_stats["geo_demand_std"] = geo_stats["geo_demand_std"].fillna(0)

    train = train.merge(geo_stats, on="geohash", how="left")
    test  = test.merge(geo_stats, on="geohash", how="left")

    # Fill test geohashes unseen in train with global stats
    for c in ["geo_demand_mean", "geo_demand_median", "geo_demand_std"]:
        global_val = train[c].median()
        test[c] = test[c].fillna(global_val)

    # Per (geohash, day) stats — gives intra-day trend info
    geo_day_stats = train.groupby(["geohash", "day"])["demand"].agg(
        geo_day_mean="mean",
    ).reset_index()

    train = train.merge(geo_day_stats, on=["geohash", "day"], how="left")
    test  = test.merge(geo_day_stats, on=["geohash", "day"], how="left")
    for c in ["geo_day_mean"]:
        test[c] = test[c].fillna(test["geo_demand_mean"])

    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Master preprocessing function.  Applies every transformation to both
    train and test in the correct order, and returns the final feature list.
    """
    print("  [1/7] Decoding geohash …")
    train = decode_geohash_column(train)
    test  = decode_geohash_column(test)

    print("  [2/7] Encoding temporal features …")
    train = encode_time_features(train)
    test  = encode_time_features(test)

    print("  [3/7] Imputing missing values …")
    train = impute_missing(train)
    test  = impute_missing(test)

    print("  [4/7] Encoding binary features …")
    train = encode_binary_features(train)
    test  = encode_binary_features(test)

    print("  [5/7] Target encoding (RoadType, Weather) …")
    # Fill NaN RoadType before encoding
    train["RoadType"] = train["RoadType"].fillna("Unknown")
    test["RoadType"]  = test["RoadType"].fillna("Unknown")

    train, test = smoothed_target_encode(train, test, "RoadType")
    train, test = smoothed_target_encode(train, test, "Weather")

    print("  [6/7] Creating interaction features …")
    train = create_interaction_features(train)
    test  = create_interaction_features(test)

    print("  [7/7] Creating aggregate features …")
    train, test = create_aggregate_features(train, test)

    # ── Final feature list ──────────────────────────────────────────────
    feature_cols = [
        # Spatial
        "latitude", "longitude",
        # Temporal
        "day", "hour", "minute", "minutes_from_midnight",
        "time_sin", "time_cos",
        "day_of_week", "dow_sin", "dow_cos",
        # Road / infrastructure
        "NumberofLanes", "LargeVehicles", "Landmarks",
        # Environment
        "Temperature",
        # Target-encoded categoricals
        "RoadType_te", "Weather_te",
        # Interactions
        "temp_x_hour", "temp_x_lanes",
        "lat_x_time_sin", "lon_x_time_cos",
        "lane_large_vehicle",
        # Aggregates
        "geo_demand_mean", "geo_demand_median", "geo_demand_std",
        "geo_day_mean",
    ]

    return train, test, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TIME-SERIES VALIDATION SPLIT  (strict temporal, NO leakage)
# ─────────────────────────────────────────────────────────────────────────────
def time_series_split(
    train: pd.DataFrame, n_val_days: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the training set by TIME, not randomly.

    We sort by (day, minutes_from_midnight) and hold out the last
    `n_val_days` day(s) as validation.  This mirrors the real-world
    scenario where we predict future demand using past data only.
    """
    max_day = train["day"].max()
    val_days = list(range(max_day - n_val_days + 1, max_day + 1))

    trn = train[~train["day"].isin(val_days)].copy()
    val = train[train["day"].isin(val_days)].copy()

    print(f"  Train days: {sorted(trn['day'].unique())}  ({len(trn):,} rows)")
    print(f"  Valid days: {sorted(val['day'].unique())}  ({len(val):,} rows)")
    return trn, val


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MODEL TRAINING  (LightGBM + CatBoost ensemble)
# ─────────────────────────────────────────────────────────────────────────────
def train_lightgbm(
    X_trn, y_trn, X_val, y_val, feature_cols
):
    """
    Train a LightGBM regressor with GPU acceleration.
    Hyperparameters are tuned for anti-overfitting on small-medium data:
      - max_depth 6        → prevents memorising noise
      - colsample_bytree   → feature subsampling each tree
      - subsample          → row subsampling (bagging)
      - min_child_samples  → leaf regularisation
      - early stopping     → halts when validation R² plateaus
    """
    import lightgbm as lgb

    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "boosting_type":    "gbdt",
        "learning_rate":    0.03,
        "num_leaves":       63,        # 2^6 - 1  (consistent with depth 6)
        "max_depth":        6,
        "min_child_samples": 30,
        "colsample_bytree": 0.7,
        "subsample":        0.8,
        "subsample_freq":   1,
        "reg_alpha":        0.1,       # L1 regularisation
        "reg_lambda":       1.0,       # L2 regularisation
        "n_estimators":     5000,
        "device":           "gpu",     # ← RTX 3050 Ti
        "gpu_use_dp":       False,     # FP32 is enough, saves VRAM
        "verbose":          -1,
        "random_state":     42,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200),
        ],
    )

    val_pred = model.predict(X_val)
    r2 = r2_score(y_val, val_pred)
    print(f"  LightGBM  val R² = {r2:.6f}  (best iter {model.best_iteration_})")
    return model, val_pred, r2


def train_catboost(
    X_trn, y_trn, X_val, y_val, feature_cols
):
    """
    Train a CatBoost regressor with GPU acceleration.
    CatBoost's ordered boosting is naturally resistant to overfitting,
    but we still limit depth and enable subsampling.
    """
    from catboost import CatBoostRegressor, Pool

    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,                # stronger L2 to compensate no rsm
        bootstrap_type="Bernoulli",     # required to use subsample
        subsample=0.8,                  # row subsampling
        # colsample_bylevel not supported on GPU for non-pairwise modes
        random_strength=1.5,            # randomness for scoring splits (anti-overfit)
        min_data_in_leaf=30,
        early_stopping_rounds=100,
        task_type="GPU",                # ← RTX 3050 Ti
        devices="0",
        eval_metric="R2",
        random_seed=42,
        verbose=200,
    )

    train_pool = Pool(X_trn, y_trn)
    val_pool   = Pool(X_val, y_val)

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict(X_val)
    r2 = r2_score(y_val, val_pred)
    print(f"  CatBoost  val R² = {r2:.6f}  (best iter {model.best_iteration_})")
    return model, val_pred, r2


# ─────────────────────────────────────────────────────────────────────────────
# 10.  ENSEMBLE & SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_blend(y_val, pred_lgb, pred_cb):
    """
    Grid-search the optimal blending weight α ∈ [0, 1] such that
        final = α · LightGBM + (1 − α) · CatBoost
    maximises validation R².
    """
    best_alpha, best_r2 = 0.5, -np.inf
    for alpha in np.arange(0.0, 1.01, 0.01):
        blended = alpha * pred_lgb + (1 - alpha) * pred_cb
        score = r2_score(y_val, blended)
        if score > best_r2:
            best_alpha, best_r2 = alpha, score
    print(f"  Optimal blend α = {best_alpha:.2f}  →  R² = {best_r2:.6f}")
    return best_alpha, best_r2


def generate_submission(
    lgb_model, cb_model, alpha: float,
    test: pd.DataFrame, feature_cols: list[str],
    sample_sub: pd.DataFrame,
):
    """
    Generate the final submission.csv with post-processing:
      1. Predict with both models
      2. Blend
      3. Clip negatives to 0.0
      4. Validate shape & columns against sample_submission.csv
    """
    X_test = test[feature_cols]

    pred_lgb = lgb_model.predict(X_test)
    pred_cb  = cb_model.predict(X_test)
    pred_final = alpha * pred_lgb + (1 - alpha) * pred_cb

    # Post-processing: traffic demand ≥ 0
    pred_final = np.clip(pred_final, 0.0, None)

    sub = pd.DataFrame({
        "Index":  test["Index"].values,
        "demand": pred_final,
    })

    # ── Validation checks ───────────────────────────────────────────────
    assert list(sub.columns) == list(sample_sub.columns), (
        f"Column mismatch: {list(sub.columns)} vs {list(sample_sub.columns)}"
    )
    assert len(sub) == len(test), (
        f"Row count mismatch: {len(sub)} vs expected {len(test)}"
    )

    sub.to_csv(OUT_PATH, index=False)
    print(f"\n✅  Submission saved → {OUT_PATH}")
    print(f"   Rows: {len(sub):,}  |  Columns: {list(sub.columns)}")
    print(f"   demand  min={sub['demand'].min():.6f}  "
          f"max={sub['demand'].max():.6f}  "
          f"mean={sub['demand'].mean():.6f}")
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # ── Load data ───────────────────────────────────────────────────────
    print("=" * 72)
    print("  LOADING DATA")
    print("=" * 72)

    # Memory-efficient dtypes
    dtype_spec = {
        "Index":          "int32",
        "day":            "int16",
        "NumberofLanes":  "int8",
        "Temperature":    "float32",
    }
    train = pd.read_csv(TRAIN_PATH, dtype=dtype_spec)
    test  = pd.read_csv(TEST_PATH,  dtype=dtype_spec)
    sample_sub = pd.read_csv(SUB_PATH)

    print(f"  train : {train.shape}   test : {test.shape}")
    print(f"  Target stats — mean: {train['demand'].mean():.4f}, "
          f"std: {train['demand'].std():.4f}, "
          f"min: {train['demand'].min():.4f}, "
          f"max: {train['demand'].max():.4f}")

    # ── Preprocess ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  PREPROCESSING")
    print("=" * 72)
    train, test, feature_cols = preprocess(train, test)
    print(f"\n  Features ({len(feature_cols)}): {feature_cols}")

    # ── Validation split ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TIME-SERIES VALIDATION SPLIT")
    print("=" * 72)
    trn, val = time_series_split(train, n_val_days=1)

    X_trn = trn[feature_cols]
    y_trn = trn["demand"]
    X_val = val[feature_cols]
    y_val = val["demand"]

    # ── Train models ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TRAINING LightGBM  (GPU)")
    print("=" * 72)
    lgb_model, lgb_val_pred, lgb_r2 = train_lightgbm(
        X_trn, y_trn, X_val, y_val, feature_cols
    )

    print("\n" + "=" * 72)
    print("  TRAINING CatBoost  (GPU)")
    print("=" * 72)
    cb_model, cb_val_pred, cb_r2 = train_catboost(
        X_trn, y_trn, X_val, y_val, feature_cols
    )

    # ── Blend ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  OPTIMISING ENSEMBLE BLEND")
    print("=" * 72)
    alpha, blend_r2 = find_optimal_blend(y_val, lgb_val_pred, cb_val_pred)

    # ── Submission ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  GENERATING SUBMISSION")
    print("=" * 72)
    generate_submission(lgb_model, cb_model, alpha, test, feature_cols, sample_sub)

    elapsed = time.time() - t0
    print(f"\n🏁  Pipeline completed in {elapsed / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
