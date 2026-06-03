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
# 4. HIERARCHICAL BIAS CORRECTION
# ─────────────────────────────────────────────────────────────────────────────
def get_hierarchical_bias(val_m: pd.DataFrame, geohashes: list) -> pd.Series:
    """
    Computes geohash-specific bias correction from morning data (val_m)
    with hierarchical fallbacks for geohashes missing from the morning data.
    """
    val_m = val_m.copy()
    val_m['bias'] = val_m['demand'] - val_m['pred_base_B']
    
    bias_geo = val_m.groupby('geohash')['bias'].mean()
    
    val_m['geohash_5'] = val_m['geohash'].str[:5]
    bias_g5 = val_m.groupby('geohash_5')['bias'].mean()
    
    val_m['geohash_4'] = val_m['geohash'].str[:4]
    bias_g4 = val_m.groupby('geohash_4')['bias'].mean()
    
    global_bias = val_m['bias'].mean()
    
    lookup = {}
    for g in geohashes:
        if g in bias_geo:
            lookup[g] = bias_geo[g]
        elif g[:5] in bias_g5:
            lookup[g] = bias_g5[g[:5]]
        elif g[:4] in bias_g4:
            lookup[g] = bias_g4[g[:4]]
        else:
            lookup[g] = global_bias
            
    return pd.Series(lookup)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
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

    # Split train set into Day 48 (base training) and Day 49 (morning overlap)
    trn_48 = train[train["day"] == 48].copy()
    val    = train[train["day"] == 49].copy()

    # ─────────────────────────────────────────────────────────────────────────
    # 6. LAG FEATURE GENERATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CREATING PIVOT-BASED LAG FEATURES")
    print("=" * 72)
    
    piv_48 = trn_48.pivot(index='geohash', columns='minutes_from_midnight', values='demand')
    all_geohashes = list(set(train['geohash'].unique()).union(test['geohash'].unique()))
    all_minutes = list(range(0, 1440, 15))
    piv_48 = piv_48.reindex(index=all_geohashes, columns=all_minutes)
    
    # Interpolate time-wise and fill any completely unseen geohashes
    piv_48_filled = piv_48.interpolate(method='linear', axis=1, limit_direction='both')
    global_mean_48 = trn_48['demand'].mean()
    piv_48_filled = piv_48_filled.fillna(global_mean_48)

    def add_lags(df):
        lags = {}
        for offset in [-60, -45, -30, -15, 0, 15, 30, 45, 60]:
            col_name = f"lag_1d_{offset}m"
            geohashes = df['geohash'].values
            minutes = df['minutes_from_midnight'].values
            shifted_minutes = (minutes + offset) % 1440
            
            row_idx = piv_48_filled.index.get_indexer(geohashes)
            col_idx = piv_48_filled.columns.get_indexer(shifted_minutes)
            values = piv_48_filled.values[row_idx, col_idx]
            lags[col_name] = values
            
        lag_df = pd.DataFrame(lags, index=df.index)
        df = pd.concat([df, lag_df], axis=1)
        
        # Add summary statistics
        lag_cols = [f"lag_1d_{offset}m" for offset in [-60, -45, -30, -15, 0, 15, 30, 45, 60]]
        df['lag_1d_mean'] = df[lag_cols].mean(axis=1)
        df['lag_1d_std'] = df[lag_cols].std(axis=1)
        df['lag_1d_min'] = df[lag_cols].min(axis=1)
        df['lag_1d_max'] = df[lag_cols].max(axis=1)
        return df

    val  = add_lags(val)
    test = add_lags(test)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. SMOOTHED TARGET ENCODING (ROAD TYPE & WEATHER)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TARGET ENCODING CATEGORICALS")
    print("=" * 72)
    
    def target_encode(train_df, val_df, test_df, col, target='demand', smoothing=20):
        global_mean = train_df[target].mean()
        stats = train_df.groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        train_df[f'{col}_te'] = train_df[col].map(smooth).fillna(global_mean)
        val_df[f'{col}_te'] = val_df[col].map(smooth).fillna(global_mean)
        test_df[f'{col}_te'] = test_df[col].map(smooth).fillna(global_mean)

    for col in ['RoadType', 'Weather']:
        target_encode(trn_48, val, test, col)

    # Define Feature Sets
    lag_cols = [f"lag_1d_{offset}m" for offset in [-60, -45, -30, -15, 0, 15, 30, 45, 60]]
    
    features_A = [
        "latitude", "longitude",
        "hour", "minute", "minutes_from_midnight",
        "time_sin", "time_cos",
        "NumberofLanes", "LargeVehicles", "Landmarks",
        "Temperature",
        "RoadType_te", "Weather_te",
        "lag_1d_mean", "lag_1d_std", "lag_1d_min", "lag_1d_max"
    ] + lag_cols

    features_B = [
        "latitude", "longitude",
        "hour", "minute", "minutes_from_midnight",
        "time_sin", "time_cos",
        "NumberofLanes", "LargeVehicles", "Landmarks",
        "Temperature",
        "RoadType_te", "Weather_te"
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # 8. TRAINING MODEL A (DAY 49 MORNING + LAG FEATURES)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TRAINING MODEL A (DAY 49 MORNING)")
    print("=" * 72)

    cb_A = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_strength=1.5,
        min_data_in_leaf=10,
        task_type="GPU",
        devices="0",
        random_seed=42,
        verbose=0
    )
    cb_A.fit(val[features_A], val["demand"])

    lgb_A = lgb.LGBMRegressor(
        n_estimators=800,
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
    lgb_A.fit(val[features_A], val["demand"])

    # ─────────────────────────────────────────────────────────────────────────
    # 9. TRAINING MODEL B (DAY 48 + HIERARCHICAL BIAS CORRECTION)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TRAINING MODEL B (DAY 48)")
    print("=" * 72)

    cb_B = CatBoostRegressor(
        iterations=1200,
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
    cb_B.fit(trn_48[features_B], trn_48["demand"])

    lgb_B = lgb.LGBMRegressor(
        n_estimators=1200,
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
    lgb_B.fit(trn_48[features_B], trn_48["demand"])

    # ─────────────────────────────────────────────────────────────────────────
    # 10. COMPUTE HIERARCHICAL BIAS MAP
    # ─────────────────────────────────────────────────────────────────────────
    # Predict base B on val
    pred_base_B_cb = cb_B.predict(val[features_B])
    pred_base_B_lgb = lgb_B.predict(val[features_B])
    val['pred_base_B'] = 0.5 * pred_base_B_cb + 0.5 * pred_base_B_lgb

    bias_map = get_hierarchical_bias(val, test['geohash'].unique())

    # ─────────────────────────────────────────────────────────────────────────
    # 11. GENERATE FINAL PREDICTIONS ON TEST SET
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  GENERATING PREDICTIONS")
    print("=" * 72)

    # Predict Model A
    pred_cb_A = cb_A.predict(test[features_A])
    pred_lgb_A = lgb_A.predict(test[features_A])
    pred_A = 0.5 * pred_cb_A + 0.5 * pred_lgb_A

    # Predict Model B and Apply Bias Correction
    pred_base_B_cb_test = cb_B.predict(test[features_B])
    pred_base_B_lgb_test = lgb_B.predict(test[features_B])
    pred_base_B_test = 0.5 * pred_base_B_cb_test + 0.5 * pred_base_B_lgb_test
    
    pred_B = pred_base_B_test + test['geohash'].map(bias_map).fillna(0.0)
    pred_B = np.clip(pred_B, 0.0, None)

    # Final Ensemble Blend (0.60 Model A + 0.40 Model B)
    pred_final = 0.60 * pred_A + 0.40 * pred_B
    pred_final = np.clip(pred_final, 0.0, None)

    # ─────────────────────────────────────────────────────────────────────────
    # 12. GENERATE AND VALIDATE SUBMISSION FILE
    # ─────────────────────────────────────────────────────────────────────────
    sub = pd.DataFrame({
        "Index":  test["Index"].values,
        "demand": pred_final,
    })

    # Assertions
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
