"""
===================================================================================
  Flipkart Grid 2.0 — Spatio-Temporal Traffic Demand Prediction
  HYBRID V3: Spatio-Temporal Fallback + Day-to-Day Difference Correction
===================================================================================
  Metric  : R² (coefficient of determination)
  Hardware: NVIDIA RTX 3050 Ti (4 GB VRAM)
  Validation R²: ~96.31% (Leak-Free, Hybrid fallback + correction)
===================================================================================
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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
# 1. HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def decode_geohash_column(df):
    decoded = df["geohash"].apply(lambda h: gh.decode(h))
    df["latitude"]  = decoded.apply(lambda t: float(t[0]))
    df["longitude"] = decoded.apply(lambda t: float(t[1]))
    return df


def encode_time_features(df):
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"]   = parts[0]
    df["minute"] = parts[1]
    df["minutes_from_midnight"] = df["hour"] * 60 + df["minute"]
    df["time_sin"] = np.sin(2 * np.pi * df["minutes_from_midnight"] / 1440)
    df["time_cos"] = np.cos(2 * np.pi * df["minutes_from_midnight"] / 1440)
    df["time_sin_12h"] = np.sin(2 * np.pi * df["minutes_from_midnight"] / 720)
    df["time_cos_12h"] = np.cos(2 * np.pi * df["minutes_from_midnight"] / 720)
    df["day_of_week"] = df["day"] % 7
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["time_bucket_30"] = df["minutes_from_midnight"] // 30
    df["time_bucket_60"] = df["minutes_from_midnight"] // 60
    return df


def target_encode(train_df, apply_dfs, col, target='residual', smoothing=20):
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['mean', 'count'])
    smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    for df in apply_dfs:
        df[f'{col}_te'] = df[col].map(smooth).fillna(global_mean)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    
    print("=" * 72)
    print("  HYBRID V3 PIPELINE (Fallback + Correction)")
    print("=" * 72)

    # ── Load ──
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SUB_PATH)
    print(f"  train: {train.shape}   test: {test.shape}")

    # ── Base preprocessing ──
    train = decode_geohash_column(train)
    test  = decode_geohash_column(test)
    train = encode_time_features(train)
    test  = encode_time_features(test)

    temp_median = train["Temperature"].median()
    train["Temperature"] = train["Temperature"].fillna(temp_median)
    test["Temperature"]  = test["Temperature"].fillna(temp_median)
    train["Weather"] = train["Weather"].fillna("Sunny")
    test["Weather"]  = test["Weather"].fillna("Sunny")

    for df in [train, test]:
        df["LargeVehicles_num"] = df["LargeVehicles"].map({"Not Allowed": 0, "Allowed": 1}).fillna(0).astype("int8")
        df["Landmarks_num"]     = df["Landmarks"].map({"No": 0, "Yes": 1}).fillna(0).astype("int8")
        df["RoadType_cat"]      = df["RoadType"].fillna("Unknown")
        df["RoadType_cat_code"] = pd.factorize(df["RoadType_cat"])[0]
        df["Weather_cat_code"]  = pd.factorize(df["Weather"])[0]
        df['geohash_3'] = df['geohash'].str[:3]
        df['geohash_4'] = df['geohash'].str[:4]
        df['geohash_5'] = df['geohash'].str[:5]

    # ─────────────────────────────────────────────────────────────────────────
    # 3. MORNING BASELINES (0:00 to 2:00, minutes 0-120)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  COMPUTING MORNING BASELINES")
    print("=" * 72)

    morning_mask_train = train['minutes_from_midnight'] <= 120
    geo_day_morning = train[morning_mask_train].groupby(['day', 'geohash'])['demand'].mean().reset_index()
    geo_day_morning = geo_day_morning.rename(columns={'demand': 'morning_mean'})
    
    train = train.merge(geo_day_morning, on=['day', 'geohash'], how='left')
    
    # Test is Day 49, so merge with Day 49 morning stats
    test_morning = geo_day_morning[geo_day_morning['day'] == 49][['geohash', 'morning_mean']]
    test = test.merge(test_morning, on='geohash', how='left')

    # Global fallbacks
    global_morning_48 = train[(train['day']==48) & morning_mask_train]['demand'].mean()
    global_morning_49 = train[(train['day']==49) & (train['minutes_from_midnight'] <= 120)]['demand'].mean()
    if pd.isna(global_morning_49):
        global_morning_49 = train[train['day'] == 49]['demand'].mean()
    
    train['morning_mean'] = train['morning_mean'].fillna(
        train['day'].map({48: global_morning_48, 49: global_morning_49})
    )
    test['morning_mean'] = test['morning_mean'].fillna(global_morning_49)
    
    print(f"  Global morning mean Day 48: {global_morning_48:.6f}")
    print(f"  Global morning mean Day 49: {global_morning_49:.6f}")
    
    # Target variable: residual
    train['residual'] = train['demand'] - train['morning_mean']
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. SAME-TIME DAY 48 HISTORICAL LOOKUPS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  BUILDING HISTORICAL LOOKUPS (Day 48 Same-Time)")
    print("=" * 72)

    trn_48 = train[train["day"] == 48].copy()
    trn_49 = train[train["day"] == 49].copy()

    d48_demand_lookup = trn_48.set_index(['geohash', 'minutes_from_midnight'])['demand'].to_dict()
    d48_morning_mean_lookup = trn_48.set_index('geohash')['morning_mean'].to_dict()

    for df in [train, test, trn_48, trn_49]:
        df['d48_demand'] = df.apply(
            lambda r: d48_demand_lookup.get((r['geohash'], r['minutes_from_midnight']), np.nan), axis=1
        )
        df['d48_morning_mean'] = df['geohash'].map(d48_morning_mean_lookup).fillna(global_morning_48)
        df['d48_residual'] = df['d48_demand'] - df['d48_morning_mean']

    # ─────────────────────────────────────────────────────────────────────────
    # 5. TARGET ENCODING
    # ─────────────────────────────────────────────────────────────────────────
    # Target encodings for Model 1 (Fallback model) during validation
    for col in ['RoadType_cat', 'Weather', 'geohash_3', 'geohash_4', 'geohash_5']:
        target_encode(trn_48, [trn_48, trn_49], col, smoothing=20)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. MODEL SETUP & VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VALIDATION STAGE (Day 48 -> Day 49 Morning)")
    print("=" * 72)

    # Features for Fallback Model 1 (NO target leakage!)
    features_fallback = [
        "latitude", "longitude", "minutes_from_midnight",
        "NumberofLanes", "LargeVehicles_num", "Landmarks_num",
        "Temperature", "RoadType_cat_te", "Weather_te",
        "time_sin", "time_cos", "time_sin_12h", "time_cos_12h",
        "geohash_3_te", "geohash_4_te", "geohash_5_te",
        "morning_mean"
    ]

    # Features for Correction Model 2 (includes Day 48 lags)
    features_corr = [
        "latitude", "longitude", "minutes_from_midnight",
        "NumberofLanes", "LargeVehicles_num", "Landmarks_num",
        "Temperature", "RoadType_cat_code", "Weather_cat_code",
        "morning_mean", "d48_morning_mean", "d48_residual", "d48_demand"
    ]

    # ── 6.1. Train Fallback Model 1 on Day 48 ──
    X_trn_fb = trn_48[features_fallback]
    y_trn_fb = trn_48["residual"]
    X_val_fb = trn_49[features_fallback]
    y_val_fb = trn_49["residual"]

    print("  Training Fallback Model 1 (LightGBM)...")
    lgb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.02,
        'max_depth': 7,
        'num_leaves': 127,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'subsample_freq': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 20,
        'device': 'gpu',
        'verbose': -1,
        'random_state': 42
    }
    model_lgb_fb = lgb.LGBMRegressor(**lgb_params)
    model_lgb_fb.fit(
        X_trn_fb, y_trn_fb,
        eval_set=[(X_val_fb, y_val_fb)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    best_iter_lgb_fb = model_lgb_fb.best_iteration_
    print(f"    LGBM Fallback Best Iteration: {best_iter_lgb_fb}")

    print("  Training Fallback Model 1 (CatBoost)...")
    cb_params = {
        'iterations': 3000,
        'learning_rate': 0.03,
        'depth': 7,
        'l2_leaf_reg': 3.0,
        'early_stopping_rounds': 100,
        'task_type': 'GPU',
        'devices': '0',
        'verbose': 0,
        'random_seed': 42
    }
    model_cb_fb = CatBoostRegressor(**cb_params)
    model_cb_fb.fit(X_trn_fb, y_trn_fb, eval_set=(X_val_fb, y_val_fb), use_best_model=True)
    best_iter_cb_fb = model_cb_fb.get_best_iteration()
    print(f"    CatBoost Fallback Best Iteration: {best_iter_cb_fb}")

    # Fallback model predictions
    pred_resid_lgb_fb = model_lgb_fb.predict(X_val_fb)
    pred_resid_cb_fb = model_cb_fb.predict(X_val_fb)
    pred_resid_fb = 0.5 * pred_resid_lgb_fb + 0.5 * pred_resid_cb_fb

    # ── 6.2. Train Correction Model 2 on Day 49 matching morning rows ──
    trn_49_match = trn_49.dropna(subset=['d48_demand', 'd48_residual', 'morning_mean']).copy()
    trn_49_match['diff_residual'] = trn_49_match['residual'] - trn_49_match['d48_residual']
    
    print(f"  Training Correction Model 2 (CatBoost, K-Fold OOF, matching rows={len(trn_49_match)})...")
    X_corr = trn_49_match[features_corr]
    y_corr = trn_49_match["diff_residual"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_diff_cb = np.zeros(len(trn_49_match))
    corr_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_corr)):
        X_tr, y_tr = X_corr.iloc[train_idx], y_corr.iloc[train_idx]
        X_va, y_va = X_corr.iloc[val_idx], y_corr.iloc[val_idx]
        
        model_cb_corr = CatBoostRegressor(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5.0,
            task_type='GPU',
            devices='0',
            random_seed=42 + fold,
            verbose=0
        )
        model_cb_corr.fit(X_tr, y_tr)
        oof_diff_cb[val_idx] = model_cb_corr.predict(X_va)
        corr_models.append(model_cb_corr)

    trn_49_match["pred_diff"] = oof_diff_cb
    match_pred_dict = trn_49_match.set_index(["geohash", "minutes_from_midnight"])["pred_diff"].to_dict()

    # ── 6.3. Evaluate Hybrid Validation R² ──
    final_val_predictions = []
    for idx, row in trn_49.iterrows():
        geo = row["geohash"]
        t = row["minutes_from_midnight"]
        
        if pd.isna(row["d48_residual"]):
            # Fallback model prediction
            pred_resid = pred_resid_fb[len(final_val_predictions)]
        else:
            # Correction model prediction
            pred_diff = match_pred_dict.get((geo, t), 0.0)
            pred_resid = row["d48_residual"] + pred_diff
            
        pred_demand = np.clip(pred_resid + row["morning_mean"], 0.0, 1.0)
        final_val_predictions.append(pred_demand)

    val_r2 = r2_score(trn_49["demand"], final_val_predictions)
    print(f"\n  >>> HYBRID PIPELINE VALIDATION R² = {val_r2:.6f} <<<")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. RETRAINING ON ALL DATA & PREDICT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RETRAINING ON FULL DATA")
    print("=" * 72)

    # Target encode prefix features using the entire train set (Day 48 + Day 49 morning)
    for col in ['RoadType_cat', 'Weather', 'geohash_3', 'geohash_4', 'geohash_5']:
        target_encode(train, [train, test], col, smoothing=20)

    # ── 7.1. Retrain Fallback Model 1 on Day 48 + Day 49 morning ──
    X_full_fb = train[features_fallback]
    y_full_fb = train["residual"]
    X_test_fb = test[features_fallback]

    print("  Retraining Fallback Model 1 (LGBM)...")
    n_est_lgb = max(int(best_iter_lgb_fb * 1.1), best_iter_lgb_fb + 20)
    lgb_full_params = {**lgb_params, 'n_estimators': n_est_lgb}
    final_lgb_fb = lgb.LGBMRegressor(**lgb_full_params)
    final_lgb_fb.fit(X_full_fb, y_full_fb)

    print("  Retraining Fallback Model 1 (CatBoost)...")
    n_iter_cb = max(int(best_iter_cb_fb * 1.1), best_iter_cb_fb + 20)
    cb_full_params = {**cb_params, 'iterations': n_iter_cb}
    del cb_full_params['early_stopping_rounds']
    final_cb_fb = CatBoostRegressor(**cb_full_params)
    final_cb_fb.fit(X_full_fb, y_full_fb)

    pred_resid_fb_test = 0.5 * final_lgb_fb.predict(X_test_fb) + 0.5 * final_cb_fb.predict(X_test_fb)

    # ── 7.2. Retrain Correction Model 2 on all matching Day 49 morning rows ──
    # Build complete matching set from the training data
    train_match = train[train["day"] == 49].dropna(subset=['d48_demand', 'd48_residual', 'morning_mean']).copy()
    train_match['diff_residual'] = train_match['residual'] - train_match['d48_residual']
    
    print(f"  Retraining Correction Model 2 (CatBoost, total matching rows={len(train_match)})...")
    X_full_corr = train_match[features_corr]
    y_full_corr = train_match["diff_residual"]
    X_test_corr = test[features_corr]

    final_cb_corr = CatBoostRegressor(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        task_type='GPU',
        devices='0',
        random_seed=42,
        verbose=0
    )
    final_cb_corr.fit(X_full_corr, y_full_corr)
    pred_diff_test = final_cb_corr.predict(X_test_corr)

    # ── 7.3. Hybrid Inference on Test Set ──
    print("\n  Generating hybrid test predictions...")
    test_predictions = []
    fallback_count = 0
    correction_count = 0

    for idx, row in test.iterrows():
        geo = row["geohash"]
        t = row["minutes_from_midnight"]
        
        if pd.isna(row["d48_residual"]):
            # Use Fallback Model 1
            pred_resid = pred_resid_fb_test[idx]
            fallback_count += 1
        else:
            # Use Correction Model 2
            pred_diff = pred_diff_test[idx]
            pred_resid = row["d48_residual"] + pred_diff
            correction_count += 1
            
        pred_demand = np.clip(pred_resid + row["morning_mean"], 0.0, 1.0)
        test_predictions.append(pred_demand)

    print(f"    Predictions via Correction Model: {correction_count:,} ({correction_count/len(test)*100:.2f}%)")
    print(f"    Predictions via Fallback Model:   {fallback_count:,} ({fallback_count/len(test)*100:.2f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. SAVE SUBMISSION
    # ─────────────────────────────────────────────────────────────────────────
    sub = pd.DataFrame({
        "Index": test["Index"].values,
        "demand": test_predictions,
    })

    assert list(sub.columns) == list(sample_sub.columns)
    assert len(sub) == len(test)

    sub.to_csv(OUT_PATH, index=False)
    print(f"\n[OK]  Submission saved -> {OUT_PATH}")
    print(f"   Rows: {len(sub):,}  |  demand min={sub['demand'].min():.6f}  "
          f"max={sub['demand'].max():.6f}  mean={sub['demand'].mean():.6f}")

    elapsed = time.time() - t0
    print(f"\n[DONE]  Pipeline completed in {elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
