"""
===================================================================================
  Flipkart Grid 2.0 — Spatio-Temporal Traffic Demand Prediction
  Difference Target Pipeline (LightGBM + CatBoost Ensemble, GPU-accelerated)
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
# 2. CYCLICAL TEMPORAL ENCODING & INTERACTIONS
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

        # Interaction features
        df["temp_x_hour"] = df["Temperature"] * df["hour"]
        df["temp_x_lanes"] = df["Temperature"] * df["NumberofLanes"]
        df["lat_x_time_sin"] = df["latitude"] * df["time_sin"]
        df["lon_x_time_cos"] = df["longitude"] * df["time_cos"]
        df["lane_large_vehicle"] = df["NumberofLanes"] * df["LargeVehicles"]

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

    # Split train set into Day 48 and Day 49 morning
    trn_48 = train[train["day"] == 48].copy()
    val_49 = train[train["day"] == 49].copy()

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

    # Fill missing morning mean using the global morning mean of the corresponding day
    global_morning_mean_48 = morning_data[morning_data['day'] == 48]['demand'].mean()
    global_morning_mean_49 = morning_data[morning_data['day'] == 49]['demand'].mean()
    
    train['geo_morning_mean'] = train['geo_morning_mean'].fillna(
        train['day'].map({48: global_morning_mean_48, 49: global_morning_mean_49})
    )
    test['geo_morning_mean'] = test['geo_morning_mean'].fillna(
        test['day'].map({48: global_morning_mean_48, 49: global_morning_mean_49})
    )

    # Split morning means back into trn_48 and val_49
    trn_48['geo_morning_mean'] = train[train['day'] == 48]['geo_morning_mean']
    val_49['geo_morning_mean'] = train[train['day'] == 49]['geo_morning_mean']

    # ─────────────────────────────────────────────────────────────────────────
    # 6. HISTORICAL REFERENCE STABLE BASELINES (Day 48 daily mean, median, etc.)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CREATING STABLE HISTORICAL BASELINES")
    print("=" * 72)
    
    geo_stats_48 = trn_48.groupby('geohash')['demand'].agg(
        geo_demand_mean='mean',
        geo_demand_median='median',
        geo_demand_std='std',
        geo_demand_max='max'
    ).reset_index()

    train = train.merge(geo_stats_48, on='geohash', how='left')
    test  = test.merge(geo_stats_48, on='geohash', how='left')

    # Update splits
    trn_48 = train[train["day"] == 48].copy()
    val_49 = train[train["day"] == 49].copy()

    global_mean_48 = trn_48['demand'].mean()
    global_std_48 = trn_48['demand'].std()
    
    for df in [train, test, trn_48, val_49]:
        df['geo_demand_mean'] = df['geo_demand_mean'].fillna(global_mean_48)
        df['geo_demand_median'] = df['geo_demand_median'].fillna(global_mean_48)
        df['geo_demand_std'] = df['geo_demand_std'].fillna(global_std_48)
        df['geo_demand_max'] = df['geo_demand_max'].fillna(global_mean_48)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. TARGET ENCODING
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
        if test_df is not None:
            test_df[f'{col}_te'] = test_df[col].map(smooth).fillna(global_mean)

    # Encoding for the validation run (using trn_48 statistics)
    for col in ['RoadType', 'Weather', 'geohash_4', 'geohash_5']:
        target_encode(trn_48, val_49, None, col)

    # Features list
    features = [
        "latitude", "longitude",
        "hour", "minute", "minutes_from_midnight",
        "time_sin", "time_cos",
        "NumberofLanes", "LargeVehicles", "Landmarks",
        "Temperature",
        "RoadType_te", "Weather_te",
        "geohash_4_te", "geohash_5_te",
        "geo_demand_mean", "geo_demand_median", "geo_demand_std", "geo_demand_max",
        "geo_morning_mean",
        "temp_x_hour", "temp_x_lanes", "lat_x_time_sin", "lon_x_time_cos", "lane_large_vehicle"
    ]

    # Target variable for modeling is residual difference: demand - geo_morning_mean
    X_trn = trn_48[features]
    y_trn = trn_48["demand"] - trn_48["geo_morning_mean"]
    X_val = val_49[features]
    y_val = val_49["demand"] - val_49["geo_morning_mean"]

    # ─────────────────────────────────────────────────────────────────────────
    # 8. TRAINING MODEL WITH TIME-SERIES VALIDATION SPLIT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TIME-SERIES VALIDATION SPLIT (Residual Modeling)")
    print("=" * 72)

    # Fit validation LightGBM to find optimal iterations
    lgb_val = lgb.LGBMRegressor(
        n_estimators=3000,
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
    
    # Validation LightGBM reconstructed R2
    pred_val_lgb_diff = lgb_val.predict(X_val)
    pred_val_lgb = np.clip(pred_val_lgb_diff + val_49["geo_morning_mean"].values, 0.0, None)
    val_r2_lgb = r2_score(val_49["demand"], pred_val_lgb)
    print(f"  LightGBM best iteration: {best_iter_lgb} (Val R2 = {val_r2_lgb:.6f})")

    # Fit validation CatBoost to find optimal iterations
    cb_val = CatBoostRegressor(
        iterations=3000,
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
    
    # Validation CatBoost reconstructed R2
    pred_val_cb_diff = cb_val.predict(X_val)
    pred_val_cb = np.clip(pred_val_cb_diff + val_49["geo_morning_mean"].values, 0.0, None)
    val_r2_cb = r2_score(val_49["demand"], pred_val_cb)
    print(f"  CatBoost best iteration: {best_iter_cb} (Val R2 = {val_r2_cb:.6f})")

    # Ensemble Val R2
    pred_val_ensemble = 0.50 * pred_val_lgb + 0.50 * pred_val_cb
    val_r2_ensemble = r2_score(val_49["demand"], pred_val_ensemble)
    print(f"  Ensemble (50/50 Blend) Val R2 = {val_r2_ensemble:.6f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 9. RETRAINING FINAL MODELS ON ENTIRE DATASET
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RETRAINING FINAL MODELS ON ENTIRE DATASET")
    print("=" * 72)

    # Compute target encodings on the entire training set (Day 48 + Day 49 morning)
    for col in ['RoadType', 'Weather', 'geohash_4', 'geohash_5']:
        target_encode(train, train, test, col)

    X_train_full = train[features]
    y_train_full = train["demand"] - train["geo_morning_mean"]

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
    # 10. GENERATE FINAL PREDICTIONS ON TEST SET
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  GENERATING PREDICTIONS")
    print("=" * 72)

    X_test = test[features]

    pred_diff_lgb = lgb_final.predict(X_test)
    pred_diff_cb  = cb_final.predict(X_test)

    # Final Ensemble Blend (50% LightGBM + 50% CatBoost)
    pred_diff_final = 0.50 * pred_diff_lgb + 0.50 * pred_diff_cb
    
    # Reconstruct final demand by adding back the corresponding Day 49 morning mean
    pred_final = pred_diff_final + test['geo_morning_mean'].values
    pred_final = np.clip(pred_final, 0.0, None)

    # ─────────────────────────────────────────────────────────────────────────
    # 11. VALIDATION AND SUBMISSION
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
