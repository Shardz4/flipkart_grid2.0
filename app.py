"""
===================================================================================
  Flipkart Grid 2.0 — Spatio-Temporal Traffic Demand Prediction
  End-to-End Improved Pipeline (LightGBM + CatBoost Ensemble, GPU-accelerated)
===================================================================================
  Metric  : R² (coefficient of determination)
  Hardware: NVIDIA RTX 3050 Ti (4 GB VRAM)
  Author  : Kaggle Grandmaster & Expert Data Scientist
===================================================================================
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import lightgbm as lgb
from catboost import CatBoostRegressor
import geohash as gh

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(ROOT, "train.csv")
TEST_PATH  = os.path.join(ROOT, "test.csv")
SUB_PATH   = os.path.join(ROOT, "sample_submission.csv")
OUT_PATH   = os.path.join(ROOT, "submission.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 1. GEOHASH DECODING
# ─────────────────────────────────────────────────────────────────────────────
def decode_geohash_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode `geohash` strings into continuous (latitude, longitude) features.
    """
    decoded = df["geohash"].apply(lambda h: gh.decode(h))
    df["latitude"]  = decoded.apply(lambda t: float(t[0]))
    df["longitude"] = decoded.apply(lambda t: float(t[1]))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CYCLICAL TEMPORAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
def encode_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the `timestamp` column (format "H:MM" or "HH:MM") into cyclical sin/cos features.
    """
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"]   = parts[0]
    df["minute"] = parts[1]
    df["minutes_from_midnight"] = df["hour"] * 60 + df["minute"]

    # Cyclical encoding (24 hour period)
    df["time_sin"] = np.sin(2 * np.pi * df["minutes_from_midnight"] / 1440)
    df["time_cos"] = np.cos(2 * np.pi * df["minutes_from_midnight"] / 1440)

    # Day-of-week feature
    df["day_of_week"] = df["day"] % 7
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. MISSING VALUE IMPUTATION & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_base(train: pd.DataFrame, test: pd.DataFrame):
    """
    Preprocess common features, imputing missing values and converting categorical fields.
    """
    # Decoding geohashes
    train = decode_geohash_column(train)
    test  = decode_geohash_column(test)

    # Temporal features
    train = encode_time_features(train)
    test  = encode_time_features(test)

    # Missing values imputation for Temperature
    temp_median = train["Temperature"].median()
    train["Temperature"] = train["Temperature"].fillna(temp_median)
    test["Temperature"]  = test["Temperature"].fillna(temp_median)

    # Missing values imputation for Weather
    train["Weather"] = train["Weather"].fillna("Sunny")
    test["Weather"]  = test["Weather"].fillna("Sunny")

    # Binary features mapping
    for df in [train, test]:
        df["LargeVehicles"] = df["LargeVehicles"].map({"Not Allowed": 0, "Allowed": 1}).fillna(0).astype("int8")
        df["Landmarks"]     = df["Landmarks"].map({"No": 0, "Yes": 1}).fillna(0).astype("int8")
        df["RoadType"]      = df["RoadType"].fillna("Unknown")

    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    print("=" * 72)
    print("  LOADING DATA")
    print("=" * 72)

    # Load datasets
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SUB_PATH)

    print(f"  train : {train.shape}   test : {test.shape}")

    # Base Preprocessing
    train, test = preprocess_base(train, test)

    # Create Geohash prefix features
    train['geohash_4'] = train['geohash'].str[:4]
    train['geohash_5'] = train['geohash'].str[:5]
    test['geohash_4']  = test['geohash'].str[:4]
    test['geohash_5']  = test['geohash'].str[:5]

    # ─────────────────────────────────────────────────────────────────────────
    # 5. GEOGRAPHIC MORNING MEAN FEATURE (0:00 to 2:00)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CREATING MORNING MEAN FEATURES")
    print("=" * 72)
    
    # Morning data from train.csv (both Day 48 and Day 49 morning exist in train)
    morning_data = train[train['minutes_from_midnight'] <= 120]
    geo_morning_stats = morning_data.groupby(['day', 'geohash'])['demand'].mean().reset_index().rename(columns={'demand': 'geo_morning_mean'})

    train = train.merge(geo_morning_stats, on=['day', 'geohash'], how='left')
    test  = test.merge(geo_morning_stats, on=['day', 'geohash'], how='left')

    # Fill missing morning mean using the global morning mean of that day
    global_morning_mean = morning_data.groupby('day')['demand'].mean().to_dict()
    
    train['geo_morning_mean'] = train.apply(
        lambda r: r['geo_morning_mean'] if not pd.isna(r['geo_morning_mean']) else global_morning_mean.get(r['day'], 0.08), 
        axis=1
    )
    test['geo_morning_mean'] = test.apply(
        lambda r: r['geo_morning_mean'] if not pd.isna(r['geo_morning_mean']) else global_morning_mean.get(r['day'], 0.08), 
        axis=1
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. TIME-SERIES VALIDATION SPLIT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TIME-SERIES VALIDATION SPLIT")
    print("=" * 72)
    
    trn_val = train[train["day"] == 48].copy()
    val_val = train[train["day"] == 49].copy()

    # Target encode for the validation run
    def target_encode(train_df, val_df, col, target='demand', smoothing=20):
        global_mean = train_df[target].mean()
        stats = train_df.groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        train_df[f'{col}_te'] = train_df[col].map(smooth).fillna(global_mean)
        val_df[f'{col}_te'] = val_df[col].map(smooth).fillna(global_mean)

    for col in ['RoadType', 'Weather', 'geohash_4', 'geohash_5']:
        target_encode(trn_val, val_val, col)

    # Features list
    features = [
        "latitude", "longitude",
        "hour", "minute", "minutes_from_midnight",
        "time_sin", "time_cos",
        "NumberofLanes", "LargeVehicles", "Landmarks",
        "Temperature",
        "RoadType_te", "Weather_te",
        "geohash_4_te", "geohash_5_te",
        "geo_morning_mean"
    ]

    X_trn = trn_val[features]
    y_trn = trn_val["demand"]
    X_val = val_val[features]
    y_val = val_val["demand"]

    # Fit validation LightGBM to find optimal iterations
    lgb_val = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=63,
        colsample_bytree=0.7,
        subsample=0.8,
        subsample_freq=1,
        device='gpu',
        random_state=42,
        verbose=-1
    )
    lgb_val.fit(
        X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    best_iter_lgb = lgb_val.best_iteration_
    print(f"  LightGBM best iteration: {best_iter_lgb} (Val R2 = {r2_score(y_val, lgb_val.predict(X_val)):.6f})")

    # Fit validation CatBoost to find optimal iterations
    cb_val = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_strength=1.5,
        min_data_in_leaf=30,
        early_stopping_rounds=50,
        task_type="GPU",
        devices="0",
        random_seed=42,
        verbose=0
    )
    cb_val.fit(X_trn, y_trn, eval_set=(X_val, y_val), use_best_model=True)
    best_iter_cb = cb_val.get_best_iteration()
    print(f"  CatBoost best iteration: {best_iter_cb} (Val R2 = {r2_score(y_val, cb_val.predict(X_val)):.6f})")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. FINAL MODEL RETRAINING ON ENTIRE DATASET
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RETRAINING FINAL MODELS ON ENTIRE DATASET")
    print("=" * 72)

    # Compute target encodings on the entire training set (Day 48 + Day 49 morning)
    def target_encode_full(train_df, test_df, col, target='demand', smoothing=20):
        global_mean = train_df[target].mean()
        stats = train_df.groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        train_df[f'{col}_te'] = train_df[col].map(smooth).fillna(global_mean)
        test_df[f'{col}_te'] = test_df[col].map(smooth).fillna(global_mean)

    for col in ['RoadType', 'Weather', 'geohash_4', 'geohash_5']:
        target_encode_full(train, test, col)

    X_train_full = train[features]
    y_train_full = train["demand"]
    X_test = test[features]

    # Retrain final LightGBM model
    lgb_final = lgb.LGBMRegressor(
        n_estimators=best_iter_lgb,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=63,
        colsample_bytree=0.7,
        subsample=0.8,
        subsample_freq=1,
        device='gpu',
        random_state=42,
        verbose=-1
    )
    lgb_final.fit(X_train_full, y_train_full)

    # Retrain final CatBoost model
    cb_final = CatBoostRegressor(
        iterations=best_iter_cb,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_strength=1.5,
        min_data_in_leaf=30,
        task_type="GPU",
        devices="0",
        random_seed=42,
        verbose=0
    )
    cb_final.fit(X_train_full, y_train_full)

    # ─────────────────────────────────────────────────────────────────────────
    # 8. PREDICTIONS AND BLENDING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  GENERATING PREDICTIONS")
    print("=" * 72)

    pred_lgb = lgb_final.predict(X_test)
    pred_cb  = cb_final.predict(X_test)

    # Final Ensemble Blend (90% LightGBM + 10% CatBoost)
    pred_final = 0.90 * pred_lgb + 0.10 * pred_cb
    pred_final = np.clip(pred_final, 0.0, None)

    # ─────────────────────────────────────────────────────────────────────────
    # 9. VALIDATION AND SUBMISSION
    # ─────────────────────────────────────────────────────────────────────────
    sub = pd.DataFrame({
        "Index":  test["Index"].values,
        "demand": pred_final,
    })

    # Validate shape and column names
    assert list(sub.columns) == list(sample_sub.columns), (
        f"Column mismatch: {list(sub.columns)} vs {list(sample_sub.columns)}"
    )
    assert len(sub) == len(test), (
        f"Row count mismatch: {len(sub)} vs expected {len(test)}"
    )

    sub.to_csv(OUT_PATH, index=False)
    print(f"\n[OK]  Submission saved -> {OUT_PATH}")
    print(f"   Rows: {len(sub):,}  |  Columns: {list(sub.columns)}")
    print(f"   demand  min={sub['demand'].min():.6f}  "
          f"max={sub['demand'].max():.6f}  "
          f"mean={sub['demand'].mean():.6f}")

    elapsed = time.time() - t0
    print(f"\n[DONE]  Pipeline completed in {elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
