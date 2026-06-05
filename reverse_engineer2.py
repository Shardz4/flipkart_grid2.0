"""
Phase 2: Deeper reverse engineering - find the data generation formula.
"""
import pandas as pd
import numpy as np
import geohash as gh
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Decode geohash
for df in [train, test]:
    decoded = df["geohash"].apply(lambda h: gh.decode(h))
    df["latitude"]  = decoded.apply(lambda t: float(t[0]))
    df["longitude"] = decoded.apply(lambda t: float(t[1]))
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"] = parts[0]
    df["minute"] = parts[1]
    df["minutes_from_midnight"] = df["hour"] * 60 + df["minute"]
    df["LargeVehicles_num"] = df["LargeVehicles"].map({"Not Allowed": 0, "Allowed": 1}).fillna(0)
    df["Landmarks_num"] = df["Landmarks"].map({"No": 0, "Yes": 1}).fillna(0)

trn_48 = train[train['day'] == 48].copy()
trn_49 = train[train['day'] == 49].copy()

print("=" * 72)
print("  INVESTIGATING SAME-TIME CROSS-DAY PREDICTION")
print("=" * 72)

# Build Day 48 lookup: (geohash, minutes) -> demand
d48_lookup = trn_48.set_index(['geohash', 'minutes_from_midnight'])['demand'].to_dict()

# For Day 49 morning, look up same (geo, time) from Day 48
trn_49['d48_same_time'] = trn_49.apply(
    lambda r: d48_lookup.get((r['geohash'], r['minutes_from_midnight']), np.nan), axis=1
)
mask = trn_49['d48_same_time'].notna()
print(f"Day 49 morning rows with Day 48 match: {mask.sum()} / {len(trn_49)}")

# R2 of just using Day 48 same-time demand
r2_raw = r2_score(trn_49.loc[mask, 'demand'], trn_49.loc[mask, 'd48_same_time'])
print(f"R2 (raw same-time): {r2_raw:.6f}")

# Linear fit
lr = LinearRegression()
X = trn_49.loc[mask, 'd48_same_time'].values.reshape(-1, 1)
y = trn_49.loc[mask, 'demand'].values
lr.fit(X, y)
r2_lr = r2_score(y, lr.predict(X))
print(f"Linear fit: d49 = {lr.coef_[0]:.6f} * d48 + {lr.intercept_:.6f}, R2 = {r2_lr:.6f}")

# Check the residuals from linear fit
resid = y - lr.predict(X)
print(f"Residual std: {resid.std():.6f}")
print(f"Residual mean: {resid.mean():.6f}")

# Check if residuals correlate with any feature
trn_49_m = trn_49.loc[mask].copy()
trn_49_m['resid'] = resid
for col in ['Temperature', 'minutes_from_midnight', 'latitude', 'longitude', 'NumberofLanes', 'LargeVehicles_num', 'Landmarks_num']:
    if trn_49_m[col].notna().sum() > 0:
        from scipy.stats import pearsonr
        m2 = trn_49_m[col].notna()
        r, _ = pearsonr(trn_49_m.loc[m2, col], trn_49_m.loc[m2, 'resid'])
        print(f"  {col} vs residual: r={r:.4f}")

# Check if the ratio d49/d48 is consistent per geohash (multiplicative model)
print("\n" + "=" * 72)
print("  GEOHASH-LEVEL DAY SCALING")
print("=" * 72)
trn_49_m['ratio'] = trn_49_m['demand'] / (trn_49_m['d48_same_time'] + 1e-10)

# Per-geohash ratio statistics
geo_ratios = trn_49_m.groupby('geohash')['ratio'].agg(['mean', 'std', 'count'])
print(f"Geohash ratio stats:")
print(f"  Mean of mean ratios: {geo_ratios['mean'].mean():.4f}")
print(f"  Std of mean ratios: {geo_ratios['mean'].std():.4f}")
print(f"  Mean of std ratios: {geo_ratios['std'].mean():.4f}")

# Check if there's a global time-dependent scaling
print("\n" + "=" * 72)
print("  TIME-DEPENDENT SCALING FROM DAY 48 TO DAY 49")
print("=" * 72)
time_scale = trn_49_m.groupby('minutes_from_midnight').agg(
    mean_ratio=('ratio', 'mean'),
    median_ratio=('ratio', 'median'),
    count=('ratio', 'count')
)
print(time_scale)

# Try: predict d49 = d48 * geo_scale_factor
# Where geo_scale_factor is the mean ratio for each geohash
geo_mean_ratio = trn_49_m.groupby('geohash')['ratio'].mean()
trn_49_m['pred_scaled'] = trn_49_m.apply(
    lambda r: r['d48_same_time'] * geo_mean_ratio.get(r['geohash'], 1.0), axis=1
)
r2_scaled = r2_score(trn_49_m['demand'], trn_49_m['pred_scaled'])
print(f"\nR2 with geohash-specific scaling: {r2_scaled:.6f}")

# ─────────────────────────────────────────────────────────────────
# KEY EXPERIMENT: How well can we predict Day 49 afternoon from Day 48?
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SIMULATING TEST SET PREDICTION")
print("=" * 72)

# The test set is Day 49, 2:15 to 13:45 (minutes 135 to 825)
# Let's hold out Day 48 afternoon (same time range) and train on the rest

# Split Day 48 into "train_sim" (morning) and "test_sim" (afternoon = same as test times)
test_minutes = sorted(test['minutes_from_midnight'].unique())
d48_test_sim = trn_48[trn_48['minutes_from_midnight'].isin(test_minutes)].copy()
d48_train_sim = trn_48[~trn_48['minutes_from_midnight'].isin(test_minutes)].copy()

print(f"Day 48 train sim: {len(d48_train_sim)} rows (morning+evening)")
print(f"Day 48 test sim:  {len(d48_test_sim)} rows (afternoon)")

# Approach 1: Global mean per geohash
geo_mean_morning = d48_train_sim.groupby('geohash')['demand'].mean()
d48_test_sim['pred_geo_mean'] = d48_test_sim['geohash'].map(geo_mean_morning)
mask_ts = d48_test_sim['pred_geo_mean'].notna()
r2_geo_mean = r2_score(d48_test_sim.loc[mask_ts, 'demand'], d48_test_sim.loc[mask_ts, 'pred_geo_mean'])
print(f"  Approach 1 (geo_mean from morning): R2 = {r2_geo_mean:.6f}")

# Check the number of non-overlapping geohashes
print(f"\n  Overlap geohashes (train_sim has {d48_train_sim['geohash'].nunique()}, test_sim has {d48_test_sim['geohash'].nunique()})")

# ─────────────────────────────────────────────────────────────────
# ADVANCED: Investigating the structure of demand more carefully  
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  INVESTIGATING DEMAND STRUCTURE BY ROAD TYPE")
print("=" * 72)

# Each RoadType has very different demand levels
# Is the time profile similar within each RoadType?
for rt in ['Residential', 'Street', 'Highway']:
    subset = trn_48[trn_48['RoadType'] == rt]
    time_profile = subset.groupby('minutes_from_midnight')['demand'].agg(['mean', 'std'])
    print(f"\n  {rt} time profile (every 2 hours):")
    for t in range(0, 1440, 120):
        row = time_profile.loc[t] if t in time_profile.index else None
        if row is not None:
            print(f"    {t//60:02d}:{t%60:02d} - mean={row['mean']:.6f}, std={row['std']:.6f}")

# ─────────────────────────────────────────────────────────────────
# DEMAND PREDICTION FORMULA INVESTIGATION
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  DEMAND FORMULA INVESTIGATION")
print("=" * 72)

# Check: is demand = base_level(geohash) * time_curve(RoadType, timestamp)?
# Compute base_level as mean demand per geohash
trn_48['geo_base'] = trn_48.groupby('geohash')['demand'].transform('mean')
trn_48['normalized'] = trn_48['demand'] / (trn_48['geo_base'] + 1e-10)

# Compute time curve per RoadType
for rt in ['Residential', 'Street', 'Highway']:
    subset = trn_48[trn_48['RoadType'] == rt]
    time_curve = subset.groupby('minutes_from_midnight')['normalized'].agg(['mean', 'std'])
    print(f"\n{rt} normalized time profile (mean, std):")
    print(f"  Mean std across timestamps: {time_curve['std'].mean():.6f}")
    # Reconstruct prediction
    time_mean_lookup = time_curve['mean'].to_dict()
    subset = subset.copy()
    subset['pred_formula'] = subset['geo_base'] * subset['minutes_from_midnight'].map(time_mean_lookup)
    r2_formula = r2_score(subset['demand'], subset['pred_formula'])
    print(f"  R2 (geo_base * time_curve): {r2_formula:.6f}")

# Overall R2 with this approach
time_norm_by_rt = trn_48.groupby(['RoadType', 'minutes_from_midnight'])['normalized'].mean()
trn_48['pred_formula_all'] = trn_48.apply(
    lambda r: r['geo_base'] * time_norm_by_rt.get((r['RoadType'], r['minutes_from_midnight']), 1.0),
    axis=1
)
r2_formula_all = r2_score(trn_48['demand'], trn_48['pred_formula_all'])
print(f"\nOverall R2 (geo_base * time_curve_by_roadtype): {r2_formula_all:.6f}")

# Check: is there a geohash-specific time curve?
# This would be overfitting on Day 48, but let's check the structure
geo_time_mean = trn_48.groupby(['geohash', 'minutes_from_midnight'])['demand'].mean()
trn_48['pred_geotime'] = trn_48.apply(
    lambda r: geo_time_mean.get((r['geohash'], r['minutes_from_midnight']), r['geo_base']),
    axis=1
)
r2_geotime = r2_score(trn_48['demand'], trn_48['pred_geotime'])
print(f"R2 (exact geo+time lookup on Day 48): {r2_geotime:.6f}")

# ─────────────────────────────────────────────────────────────────
# CHECK: Do nearby geohashes have similar demand profiles?
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  GEOGRAPHIC PROXIMITY IN DEMAND PROFILES")
print("=" * 72)

# Get geohash info
geo_info = trn_48.groupby('geohash').agg(
    lat=('latitude', 'first'),
    lon=('longitude', 'first'),
    mean_demand=('demand', 'mean'),
    road_type=('RoadType', 'first')
).reset_index()

# Compute pairwise distances between a sample of geohashes
from scipy.spatial.distance import cdist
coords = geo_info[['lat', 'lon']].values
sample_idx = np.random.choice(len(coords), min(200, len(coords)), replace=False)
sample_coords = coords[sample_idx]
sample_demands = geo_info.iloc[sample_idx]['mean_demand'].values

dists = cdist(sample_coords, sample_coords)
demand_diffs = np.abs(sample_demands[:, None] - sample_demands[None, :])

# Correlation between distance and demand difference
mask_upper = np.triu_indices(len(sample_coords), k=1)
dist_corr = np.corrcoef(dists[mask_upper], demand_diffs[mask_upper])[0, 1]
print(f"Correlation between geographic distance and demand difference: {dist_corr:.4f}")

# ─────────────────────────────────────────────────────────────────
# ADVANCED: Check if Day 49 = Day 48 + some systematic shift
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  DAY-TO-DAY SHIFT ANALYSIS")
print("=" * 72)

# For each geohash that appears in both day 48 and day 49 morning
common_geos_morning = set(trn_48[trn_48['minutes_from_midnight'] <= 120]['geohash'].unique()) & set(trn_49['geohash'].unique())
print(f"Common geohashes in morning: {len(common_geos_morning)}")

# Compute the average demand ratio d49/d48 per geohash for morning hours
ratios_per_geo = {}
for geo in common_geos_morning:
    d48_morning = trn_48[(trn_48['geohash'] == geo) & (trn_48['minutes_from_midnight'] <= 120)]
    d49_morning = trn_49[trn_49['geohash'] == geo]
    
    # Match by time
    merged = pd.merge(
        d48_morning[['minutes_from_midnight', 'demand']],
        d49_morning[['minutes_from_midnight', 'demand']],
        on='minutes_from_midnight',
        suffixes=('_48', '_49')
    )
    if len(merged) > 0 and merged['demand_48'].sum() > 0:
        ratio = merged['demand_49'].sum() / merged['demand_48'].sum()
        ratios_per_geo[geo] = ratio

ratios_series = pd.Series(ratios_per_geo)
print(f"Ratio statistics:")
print(f"  Mean: {ratios_series.mean():.4f}")
print(f"  Median: {ratios_series.median():.4f}")
print(f"  Std: {ratios_series.std():.4f}")
print(f"  Min: {ratios_series.min():.4f}")
print(f"  Max: {ratios_series.max():.4f}")
print(f"  25%: {ratios_series.quantile(0.25):.4f}")
print(f"  75%: {ratios_series.quantile(0.75):.4f}")

# Check if ratio correlates with any geohash property
geo_ratio_df = pd.DataFrame({'geohash': list(ratios_per_geo.keys()), 'ratio': list(ratios_per_geo.values())})
geo_ratio_df = geo_ratio_df.merge(geo_info, on='geohash', how='left')
for col in ['lat', 'lon', 'mean_demand']:
    r, _ = pearsonr(geo_ratio_df[col], geo_ratio_df['ratio'])
    print(f"  {col} vs ratio: r={r:.4f}")

# Distribution of ratio by road_type
print("\nRatio by road type:")
print(geo_ratio_df.groupby('road_type')['ratio'].describe()[['count', 'mean', 'std', 'min', 'max']])

# ─────────────────────────────────────────────────────────────────
# CRITICAL: Check if there are exact duplicate rows or near-duplicates
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  CHECKING FOR EXACT FEATURE DUPLICATES")
print("=" * 72)

# Are there rows with identical features but different demands?
feat_cols = ['geohash', 'day', 'timestamp', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks']
dup_groups = train.groupby(feat_cols).agg(
    demand_std=('demand', 'std'),
    demand_count=('demand', 'count'),
    demand_min=('demand', 'min'),
    demand_max=('demand', 'max')
)
multi = dup_groups[dup_groups['demand_count'] > 1]
print(f"Groups with >1 row (same features except temp/weather): {len(multi)}")

# Same check without weather and temperature
feat_cols2 = ['geohash', 'day', 'minutes_from_midnight']
dup_groups2 = train.groupby(feat_cols2).size()
print(f"Max rows per (geohash, day, time): {dup_groups2.max()}")
print(f"Groups with >1 row: {(dup_groups2 > 1).sum()}")

# ─────────────────────────────────────────────────────────────────
# CHECK: Does demand have a finite set of possible values?
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  CHECKING DEMAND VALUE DISTRIBUTION")
print("=" * 72)
print(f"Unique demand values: {train['demand'].nunique()}")
print(f"Total rows: {len(train)}")

# Check if demand values can be expressed as rational fractions
# or if they have a specific decimal precision
demand_vals = train['demand'].values
# Check number of significant digits
for d in demand_vals[:10]:
    print(f"  {d:.15f}")

# Check if demand * N gives integer values for some N
for N in [100, 1000, 10000, 100000]:
    is_int = np.all(np.abs(demand_vals * N - np.round(demand_vals * N)) < 1e-6)
    print(f"  demand * {N} = integer? {is_int}")

# Check if demand follows a specific distribution
from scipy.stats import kstest, norm, lognorm
# Per RoadType
for rt in ['Residential', 'Street', 'Highway']:
    subset = train[train['RoadType'] == rt]['demand']
    print(f"\n  {rt}: mean={subset.mean():.6f}, std={subset.std():.6f}, skew={subset.skew():.4f}, kurt={subset.kurtosis():.4f}")

print("\n" + "=" * 72)
print("  DONE WITH PHASE 2 ANALYSIS")
print("=" * 72)
