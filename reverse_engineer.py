"""
Reverse-engineering the data generation process to find deterministic or 
near-deterministic patterns in the demand variable.
"""
import pandas as pd
import numpy as np
import geohash as gh
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Decode geohash
decoded = train["geohash"].apply(lambda h: gh.decode(h))
train["latitude"]  = decoded.apply(lambda t: float(t[0]))
train["longitude"] = decoded.apply(lambda t: float(t[1]))

decoded_t = test["geohash"].apply(lambda h: gh.decode(h))
test["latitude"]  = decoded_t.apply(lambda t: float(t[0]))
test["longitude"] = decoded_t.apply(lambda t: float(t[1]))

# Time features
for df in [train, test]:
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"] = parts[0]
    df["minute"] = parts[1]
    df["minutes_from_midnight"] = df["hour"] * 60 + df["minute"]

print("=" * 72)
print("  BASIC STATISTICS")
print("=" * 72)
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"\nTrain demand stats:\n{train['demand'].describe()}")
print(f"\nTrain days: {sorted(train['day'].unique())}")
print(f"Test days:  {sorted(test['day'].unique())}")
print(f"\nTrain timestamps range: {sorted(train['minutes_from_midnight'].unique())[:5]} ... {sorted(train['minutes_from_midnight'].unique())[-5:]}")
print(f"Test timestamps range:  {sorted(test['minutes_from_midnight'].unique())[:5]} ... {sorted(test['minutes_from_midnight'].unique())[-5:]}")

# Check unique geohashes per timestamp
print("\n" + "=" * 72)
print("  GEOHASH ANALYSIS")
print("=" * 72)
print(f"Unique geohashes in train: {train['geohash'].nunique()}")
print(f"Unique geohashes in test:  {test['geohash'].nunique()}")
train_geos = set(train['geohash'].unique())
test_geos = set(test['geohash'].unique())
print(f"Overlap:    {len(train_geos & test_geos)}")
print(f"Test-only:  {len(test_geos - train_geos)}")
print(f"Train-only: {len(train_geos - test_geos)}")

# Check if geohash set is constant per timestamp
print("\n" + "=" * 72)
print("  GEOHASHES PER TIMESTAMP")
print("=" * 72)
geo_per_ts = train.groupby(['day', 'minutes_from_midnight'])['geohash'].nunique()
print(f"Geohashes per (day, timestamp):\n{geo_per_ts.describe()}")

# Check for a fixed grid pattern
trn_48 = train[train['day'] == 48]
ts_counts_48 = trn_48.groupby('minutes_from_midnight')['geohash'].nunique()
print(f"\nDay 48 geohashes per timestamp (first 10):")
print(ts_counts_48.head(10))

# Check demand distribution by road type and other features
print("\n" + "=" * 72)
print("  DEMAND BY FEATURES")
print("=" * 72)
print("By RoadType:")
print(train.groupby('RoadType')['demand'].describe()[['count', 'mean', 'std', 'min', 'max']])
print("\nBy Weather:")
print(train.groupby('Weather')['demand'].describe()[['count', 'mean', 'std', 'min', 'max']])
print("\nBy NumberofLanes:")
print(train.groupby('NumberofLanes')['demand'].describe()[['count', 'mean', 'std', 'min', 'max']])
print("\nBy LargeVehicles:")
print(train.groupby('LargeVehicles')['demand'].describe()[['count', 'mean', 'std', 'min', 'max']])
print("\nBy Landmarks:")
print(train.groupby('Landmarks')['demand'].describe()[['count', 'mean', 'std', 'min', 'max']])

# Check correlation between demand and Temperature
from scipy.stats import pearsonr, spearmanr
mask = train['Temperature'].notna()
r_pearson, _ = pearsonr(train.loc[mask, 'Temperature'], train.loc[mask, 'demand'])
r_spearman, _ = spearmanr(train.loc[mask, 'Temperature'], train.loc[mask, 'demand'])
print(f"\nTemperature-Demand correlation: Pearson={r_pearson:.4f}, Spearman={r_spearman:.4f}")

# Check if demand follows a simple formula
# Try: demand = f(geohash, timestamp) with some noise
print("\n" + "=" * 72)
print("  CHECKING FOR DETERMINISTIC PATTERNS")
print("=" * 72)

# For each geohash, check how consistent demand is across same timestamps
# on Day 48
trn_48 = train[train['day'] == 48].copy()
# Check for repeated (geohash, timestamp) pairs
dup_check = trn_48.groupby(['geohash', 'minutes_from_midnight']).size()
print(f"Max occurrences of same (geohash, time) on Day 48: {dup_check.max()}")
print(f"Number of unique (geohash, time) pairs: {len(dup_check)}")
print(f"Expected (geohash * timestamps): {trn_48['geohash'].nunique()} * {trn_48['minutes_from_midnight'].nunique()} = {trn_48['geohash'].nunique() * trn_48['minutes_from_midnight'].nunique()}")

# Check if demand varies by lat/lon in a smooth way
# Group by geohash and compute demand profile similarity
print("\n" + "=" * 72)
print("  DEMAND PROFILE ANALYSIS")
print("=" * 72)

# Pivot: geohash x time -> demand
pivot_48 = trn_48.pivot_table(index='geohash', columns='minutes_from_midnight', values='demand', aggfunc='mean')
print(f"Pivot shape: {pivot_48.shape}")
print(f"Missing values in pivot: {pivot_48.isna().sum().sum()}")

# Check if demand is a function of (NumberofLanes, LargeVehicles, Landmarks, RoadType) + time
print("\n" + "=" * 72)
print("  CHECKING DEMAND = f(road_features, time)")
print("=" * 72)

# Group by road features + time
road_feat_cols = ['NumberofLanes', 'LargeVehicles', 'Landmarks', 'RoadType']
grouped = trn_48.groupby(road_feat_cols + ['minutes_from_midnight'])['demand'].agg(['mean', 'std', 'count'])
print(f"Unique (road_features, time) combinations: {len(grouped)}")
print(f"Mean std within groups: {grouped['std'].mean():.6f}")
print(f"Groups with std < 0.01: {(grouped['std'] < 0.01).sum()}")
print(f"Groups with std < 0.001: {(grouped['std'] < 0.001).sum()}")

# Check: How much of the demand is explained by geohash + time alone?
print("\n" + "=" * 72)
print("  CHECKING DEMAND = f(geohash, time)")
print("=" * 72)

# Since each (geohash, time) appears exactly once on Day 48, we need cross-day analysis
# Compare morning hours of Day 48 vs Day 49
morning_48 = trn_48[trn_48['minutes_from_midnight'] <= 120].set_index(['geohash', 'minutes_from_midnight'])['demand']
morning_49_df = train[(train['day'] == 49)].copy()
morning_49 = morning_49_df.set_index(['geohash', 'minutes_from_midnight'])['demand']

common_idx = morning_48.index.intersection(morning_49.index)
print(f"Common (geohash, time) pairs in mornings: {len(common_idx)}")
if len(common_idx) > 0:
    d48 = morning_48.loc[common_idx]
    d49 = morning_49.loc[common_idx]
    r2_same_time = r2_score(d49, d48)
    print(f"R2 of Day48_morning -> Day49_morning (same geohash, same time): {r2_same_time:.6f}")
    
    ratio = d49 / (d48 + 1e-8)
    print(f"Mean ratio Day49/Day48: {ratio.mean():.4f}")
    print(f"Median ratio: {ratio.median():.4f}")
    print(f"Std of ratio: {ratio.std():.4f}")
    
    # Try linear regression: d49 = a * d48 + b
    lr = LinearRegression()
    lr.fit(d48.values.reshape(-1, 1), d49.values)
    pred_lr = lr.predict(d48.values.reshape(-1, 1))
    r2_lr = r2_score(d49, pred_lr)
    print(f"Linear fit: d49 = {lr.coef_[0]:.4f} * d48 + {lr.intercept_:.6f}, R2 = {r2_lr:.6f}")

# Check temperature correlation with demand after controlling for geohash and time
print("\n" + "=" * 72)
print("  TEMPERATURE IMPACT AFTER CONTROLLING FOR LOCATION")
print("=" * 72)
# Residualize demand by subtracting geohash mean
trn_48['demand_residual'] = trn_48['demand'] - trn_48.groupby('geohash')['demand'].transform('mean')
mask48 = trn_48['Temperature'].notna()
r_temp_resid, _ = pearsonr(trn_48.loc[mask48, 'Temperature'], trn_48.loc[mask48, 'demand_residual'])
print(f"Temp-Demand_residual correlation (after removing geohash mean): {r_temp_resid:.4f}")

# Check Weather impact after controlling for geohash
print("\nDemand residual by Weather:")
print(trn_48.groupby('Weather')['demand_residual'].describe()[['count', 'mean', 'std']])

# Check how stable a geohash's demand profile is: try to predict demand from 
# geohash's time-of-day profile + road features
print("\n" + "=" * 72)
print("  CHECKING DATA GENERATION FORMULA HYPOTHESIS")
print("=" * 72)
# Hypothesis: demand might be generated as:
# demand = base_demand(geohash) * time_profile(hour) * weather_factor * temp_factor + noise

# Test: compute the ratio demand / geo_mean for each row
trn_48['ratio_to_geo_mean'] = trn_48['demand'] / (trn_48.groupby('geohash')['demand'].transform('mean') + 1e-8)
# Check if ratio_to_geo_mean depends mostly on time
time_profile = trn_48.groupby('minutes_from_midnight')['ratio_to_geo_mean'].mean()
print(f"Time profile of ratio_to_geo_mean (first 10 timestamps):")
print(time_profile.head(10))
print(f"Std of time profile: {time_profile.std():.4f}")

# Check if within a time slot, the ratio is similar across geohashes
for t in [0, 120, 360, 600, 900]:
    subset = trn_48[trn_48['minutes_from_midnight'] == t]
    if len(subset) > 0:
        print(f"  Time={t}min: ratio mean={subset['ratio_to_geo_mean'].mean():.4f}, std={subset['ratio_to_geo_mean'].std():.4f}, n={len(subset)}")

# Check if demand has a multiplicative structure
# demand = geo_base * time_mult * weather_mult * temp_effect
# Try log transform and additive decomposition
print("\n" + "=" * 72)
print("  LOG-SPACE ADDITIVE DECOMPOSITION")
print("=" * 72)
trn_48['log_demand'] = np.log1p(trn_48['demand'])
geo_log_mean = trn_48.groupby('geohash')['log_demand'].transform('mean')
trn_48['log_residual'] = trn_48['log_demand'] - geo_log_mean
time_log_profile = trn_48.groupby('minutes_from_midnight')['log_residual'].mean()
trn_48['log_residual2'] = trn_48['log_residual'] - trn_48['minutes_from_midnight'].map(time_log_profile)
print(f"After removing geo + time effects in log-space:")
print(f"  Remaining std: {trn_48['log_residual2'].std():.6f}")
print(f"  Original log_demand std: {trn_48['log_demand'].std():.6f}")
print(f"  Variance explained: {1 - (trn_48['log_residual2'].std()**2 / trn_48['log_demand'].std()**2):.6f}")

# Weather effect in log space
weather_log_effect = trn_48.groupby('Weather')['log_residual2'].mean()
print(f"\nWeather effects in log-space (after geo+time):")
print(weather_log_effect)

# Temperature effect
mask48 = trn_48['Temperature'].notna()
r_temp_log, _ = pearsonr(trn_48.loc[mask48, 'Temperature'], trn_48.loc[mask48, 'log_residual2'])
print(f"\nTemp-log_residual2 correlation: {r_temp_log:.4f}")

# NumberofLanes effect  
lanes_log_effect = trn_48.groupby('NumberofLanes')['log_residual2'].mean()
print(f"\nNumberofLanes effects (after geo+time):")
print(lanes_log_effect)

# Check if road features explain remaining variance
road_log_effect = trn_48.groupby(['NumberofLanes', 'LargeVehicles', 'Landmarks', 'RoadType'])['log_residual2'].agg(['mean', 'std', 'count'])
print(f"\nRoad feature group effects (top 10 by count):")
print(road_log_effect.sort_values('count', ascending=False).head(10))

print("\n" + "=" * 72)
print("  CHECKING PIVOT TABLE PATTERNS")
print("=" * 72)
# Each geohash appears at each timestamp on Day 48
# Create full pivot
pivot = trn_48.pivot_table(index='geohash', columns='minutes_from_midnight', values='demand')
print(f"Pivot shape: {pivot.shape} (should be ~geohashes x 96)")

# Check column-wise correlation structure
corr_matrix = pivot.corr()
print(f"\nInter-timestamp correlation (mean): {corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean():.4f}")
print(f"Inter-timestamp correlation (min):  {corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].min():.4f}")

# SVD analysis - how many components explain the demand?
from numpy.linalg import svd
pivot_filled = pivot.fillna(pivot.mean())
U, s, Vt = svd(pivot_filled.values - pivot_filled.values.mean(axis=0), full_matrices=False)
total_var = (s**2).sum()
cum_var = np.cumsum(s**2) / total_var
print(f"\nSVD cumulative variance explained:")
for k in [1, 2, 3, 5, 10, 20]:
    if k <= len(cum_var):
        print(f"  Top {k:2d} components: {cum_var[k-1]:.6f}")

print("\n" + "=" * 72)
print("  DONE")
print("=" * 72)
