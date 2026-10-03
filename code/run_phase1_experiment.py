"""
run_phase1_experiment.py
=========================
Master runner for Phase 1: corrected, fully-validated 18-feature primary analysis.

Produces every per-condition result needed for condition_metrics_18_features.csv,
full_baseline_f1.csv, and the downstream tables/figure/docs (built by a separate
finalize_phase1.py that consumes this script's raw outputs).

Structure (see audit_report.md / sampled_feature_builder.py for the corrected
pipeline this implements):
  1. Load the Phase 1 train/test raw (pre-lag) ping sets.
  2. Build the FULL (rate=100%) training and test feature matrices once.
  3. Build vessel-type labels (VC/FD) and fixed Isolation Forest ground truth
     (AD pooled, ADT per-type) from those full matrices.
  4. Compute majority-class baselines.
  5. For each seed (0-4): fit the "full-data" Random Forest for each task
     (identical training data every seed at rate=1.0; only RF's own random_state
     varies) -> full_baseline_f1 rows.
  6. For each strategy x rate<1.0 x seed: build the corrected sampled feature
     matrix ONCE (shared across all 4 tasks, since sampling operates on pings,
     not task labels), then fit + evaluate all 4 tasks from it.
  7. Retention is computed per-seed against that SAME seed's full_data_f1 (Part 5
     of the reviewer request) -- never against a cross-seed average.

Any condition that cannot be trained (empty or single-class y_train) is recorded
with an explicit failure reason, never silently skipped.
"""

import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision')
import sampled_feature_builder as sfb

OUT = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase1_18feature_corrected'
FEATURE_COLS = sfb.FEATURE_COLS_18
STRATEGIES = ['random', 'stratified', 'spatial', 'importance', 'adaptive', 'temporal']
RATES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2, 3, 4]
TASKS = ['VC', 'FD', 'AD', 'ADT']
IF_CONTAMINATION = 0.05
MIN_GROUP_SIZE = 20

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ---------------------------------------------------------------------------
# 1. Load Phase 1 raw pre-lag ping sets
# ---------------------------------------------------------------------------
log("Loading train/test raw pre-lag ping sets...")
train_raw = pd.read_parquet(f'{OUT}/train_raw_prelag.parquet')
test_raw = pd.read_parquet(f'{OUT}/test_raw_prelag.parquet')
n_train_vessels_full = train_raw['mmsi'].nunique()
n_test_vessels_full = test_raw['mmsi'].nunique()
total_train_pings = len(train_raw)
log(f"train_raw: {total_train_pings:,} pings, {n_train_vessels_full:,} vessels")
log(f"test_raw: {len(test_raw):,} pings, {n_test_vessels_full:,} vessels")

# Sanity: no vessel overlap between train/test
overlap = set(train_raw['mmsi'].unique()) & set(test_raw['mmsi'].unique())
assert len(overlap) == 0, f"FATAL: {len(overlap)} vessels appear in both train and test"
log("Confirmed: zero vessel overlap between train and test.")

# ---------------------------------------------------------------------------
# 2. Build FULL (rate=100%) training and test feature matrices
# ---------------------------------------------------------------------------
log("Building FULL training feature matrix (rate=100%)...")
train_agg_full = sfb.build_sampled_features('random', train_raw, 1.0, seed=0, feature_cols=FEATURE_COLS)
log(f"  -> {len(train_agg_full):,} valid training vessels (complete 18-feature vector)")

log("Building FULL test feature matrix (test set is always used in full)...")
test_agg_full = sfb.build_sampled_features('random', test_raw, 1.0, seed=0, feature_cols=FEATURE_COLS)
log(f"  -> {len(test_agg_full):,} valid test vessels (complete 18-feature vector)")

# ---------------------------------------------------------------------------
# 3a. Vessel-type labels for VC / FD (mode of per-ping vessel_type_group, on RAW data)
# ---------------------------------------------------------------------------
log("Building vessel-type labels (VC/FD)...")

def vessel_type_mode_labels(raw_df):
    return (raw_df.groupby('mmsi')['vessel_type_group']
            .agg(lambda x: x.mode()[0]).reset_index()
            .rename(columns={'vessel_type_group': 'type_group'}))

train_type_labels = vessel_type_mode_labels(train_raw)
test_type_labels = vessel_type_mode_labels(test_raw)

TYPE_ORDER = sorted(train_type_labels['type_group'].unique())
TYPE_TO_INT = {t: i for i, t in enumerate(TYPE_ORDER)}
train_type_labels['label'] = train_type_labels['type_group'].map(TYPE_TO_INT)
test_type_labels['label'] = test_type_labels['type_group'].map(TYPE_TO_INT)
train_type_labels['is_fishing'] = (train_type_labels['type_group'] == 'fishing').astype(int)
test_type_labels['is_fishing'] = (test_type_labels['type_group'] == 'fishing').astype(int)

type_dummies_train = pd.get_dummies(train_type_labels['type_group'], prefix='type').astype(int)
TYPE_COLS = list(type_dummies_train.columns)
type_dummies_train['mmsi'] = train_type_labels['mmsi'].values

def add_type_features(agg_df, type_group_labels):
    dummies = pd.get_dummies(type_group_labels['type_group'], prefix='type').astype(int)
    for c in TYPE_COLS:
        if c not in dummies.columns:
            dummies[c] = 0
    dummies = dummies[TYPE_COLS]
    dummies['mmsi'] = type_group_labels['mmsi'].values
    return agg_df.merge(dummies, on='mmsi', how='left').fillna(0)

log(f"  Vessel type groups found: {TYPE_ORDER}")

# ---------------------------------------------------------------------------
# 3b. Fixed Isolation Forest ground truth: AD (pooled) and ADT (per type)
# ---------------------------------------------------------------------------
log("Building AD (pooled) Isolation Forest ground truth...")
scaler_ad = StandardScaler()
X_train_ad_scaled = scaler_ad.fit_transform(train_agg_full[FEATURE_COLS].values)
X_test_ad_scaled = scaler_ad.transform(test_agg_full[FEATURE_COLS].values)
iso_ad = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION, random_state=42, n_jobs=-1)
iso_ad.fit(X_train_ad_scaled)
y_train_ad_full = pd.Series((iso_ad.predict(X_train_ad_scaled) == -1).astype(int),
                            index=train_agg_full['mmsi'].values, name='is_anomaly')
y_test_ad_full = pd.Series((iso_ad.predict(X_test_ad_scaled) == -1).astype(int),
                           index=test_agg_full['mmsi'].values, name='is_anomaly')
log(f"  AD train anomalies: {y_train_ad_full.sum()}/{len(y_train_ad_full)} | "
    f"test anomalies: {y_test_ad_full.sum()}/{len(y_test_ad_full)}")

log("Building ADT (per vessel type) Isolation Forest ground truth...")
train_agg_typed = train_agg_full.merge(train_type_labels[['mmsi', 'type_group']], on='mmsi')
test_agg_typed = test_agg_full.merge(test_type_labels[['mmsi', 'type_group']], on='mmsi')

adt_train_parts, adt_test_parts = [], []
adt_group_report = []
for vtype, group in train_agg_typed.groupby('type_group'):
    n = len(group)
    test_group = test_agg_typed[test_agg_typed['type_group'] == vtype]
    if n < MIN_GROUP_SIZE:
        adt_train_parts.append(pd.Series(0, index=group['mmsi'].values))
        adt_test_parts.append(pd.Series(0, index=test_group['mmsi'].values))
        adt_group_report.append((vtype, n, len(test_group), 'SKIPPED_TOO_FEW -> labeled normal'))
        continue
    sc = StandardScaler()
    X_g = sc.fit_transform(group[FEATURE_COLS].values)
    iso_g = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION, random_state=42, n_jobs=-1)
    iso_g.fit(X_g)
    y_g = (iso_g.predict(X_g) == -1).astype(int)
    adt_train_parts.append(pd.Series(y_g, index=group['mmsi'].values))
    if len(test_group) > 0:
        X_gt = sc.transform(test_group[FEATURE_COLS].values)
        y_gt = (iso_g.predict(X_gt) == -1).astype(int)
        adt_test_parts.append(pd.Series(y_gt, index=test_group['mmsi'].values))
    adt_group_report.append((vtype, n, len(test_group), f'fit OK, {y_g.sum()} anomalies'))

y_train_adt_full = pd.concat(adt_train_parts).rename('is_anomaly')
y_test_adt_full = pd.concat(adt_test_parts).rename('is_anomaly') if adt_test_parts else pd.Series(dtype=int)
for row in adt_group_report:
    log(f"    IF[{row[0]}] train_n={row[1]} test_n={row[2]} -> {row[3]}")
log(f"  ADT train anomalies: {y_train_adt_full.sum()}/{len(y_train_adt_full)} | "
    f"test anomalies: {y_test_adt_full.sum()}/{len(y_test_adt_full)}")

with open(f'{OUT}/adt_group_report.json', 'w') as f:
    json.dump(adt_group_report, f, indent=2)

# Fixed evaluation matrices (never resampled -- test set always used in full)
X_test_vc = test_agg_full.merge(test_type_labels[['mmsi', 'label']], on='mmsi')
X_test_fd = test_agg_full.merge(test_type_labels[['mmsi', 'is_fishing']], on='mmsi')
test_agg_adt_full = add_type_features(test_agg_full, test_type_labels)
X_test_adt_full = test_agg_adt_full[test_agg_adt_full['mmsi'].isin(y_test_adt_full.index)].copy()

# ---------------------------------------------------------------------------
# 4. Majority-class baselines (constant prediction = training-set majority class,
#    scored against the fixed test labels for that task)
# ---------------------------------------------------------------------------
log("Computing majority-class baselines...")
majority_info = {}

def majority_baseline(y_train, y_test, average):
    maj_class = pd.Series(y_train).mode()[0]
    y_pred = np.full(len(y_test), maj_class)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
    return maj_class, f1

y_train_vc = train_type_labels.set_index('mmsi').loc[train_agg_full['mmsi'], 'label'].values
y_test_vc = X_test_vc['label'].values
maj_vc, f1_maj_vc = majority_baseline(y_train_vc, y_test_vc, 'macro')
majority_info['VC'] = {'majority_class': int(maj_vc), 'majority_class_label': TYPE_ORDER[maj_vc], 'majority_class_f1': f1_maj_vc}

y_train_fd = train_type_labels.set_index('mmsi').loc[train_agg_full['mmsi'], 'is_fishing'].values
y_test_fd = X_test_fd['is_fishing'].values
maj_fd, f1_maj_fd = majority_baseline(y_train_fd, y_test_fd, 'binary')
majority_info['FD'] = {'majority_class': int(maj_fd), 'majority_class_f1': f1_maj_fd}

maj_ad, f1_maj_ad = majority_baseline(y_train_ad_full.values, y_test_ad_full.values, 'binary')
majority_info['AD'] = {'majority_class': int(maj_ad), 'majority_class_f1': f1_maj_ad}

maj_adt, f1_maj_adt = majority_baseline(y_train_adt_full.values, y_test_adt_full.values, 'binary')
majority_info['ADT'] = {'majority_class': int(maj_adt), 'majority_class_f1': f1_maj_adt}

log(f"  Majority-class F1: VC={f1_maj_vc:.4f} FD={f1_maj_fd:.4f} AD={f1_maj_ad:.4f} ADT={f1_maj_adt:.4f}")
with open(f'{OUT}/majority_baseline_info.json', 'w') as f:
    json.dump(majority_info, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# Helper: fit+evaluate one task given a training feature matrix
# ---------------------------------------------------------------------------
def fit_eval_task(task, agg_df, seed):
    """Returns (f1, n_pos_train, n_neg_train, status, reason)."""
    if task == 'VC':
        merged = agg_df.merge(train_type_labels[['mmsi', 'label']], on='mmsi')
        X_train = merged[FEATURE_COLS].values
        y_train = merged['label'].values
        X_eval = X_test_vc[FEATURE_COLS].values
        y_eval = y_test_vc
        average = 'macro'
    elif task == 'FD':
        merged = agg_df.merge(train_type_labels[['mmsi', 'is_fishing']], on='mmsi')
        X_train = merged[FEATURE_COLS].values
        y_train = merged['is_fishing'].values
        X_eval = X_test_fd[FEATURE_COLS].values
        y_eval = y_test_fd
        average = 'binary'
    elif task == 'AD':
        avail = agg_df[agg_df['mmsi'].isin(y_train_ad_full.index)]
        y_train = y_train_ad_full.loc[avail['mmsi']].values
        X_train = scaler_ad.transform(avail[FEATURE_COLS].values)
        X_eval = X_test_ad_scaled
        y_eval = y_test_ad_full.values
        average = 'binary'
    elif task == 'ADT':
        typed = add_type_features(agg_df, train_type_labels[train_type_labels['mmsi'].isin(agg_df['mmsi'])])
        typed = typed[typed['mmsi'].isin(y_train_adt_full.index)]
        y_train = y_train_adt_full.loc[typed['mmsi']].values
        X_train = typed[FEATURE_COLS + TYPE_COLS].values
        X_eval = X_test_adt_full[FEATURE_COLS + TYPE_COLS].values
        y_eval = y_test_adt_full.loc[X_test_adt_full['mmsi']].values
        average = 'binary'
    else:
        raise ValueError(task)

    n_train = len(y_train)
    if n_train == 0:
        return None, 0, 0, 'FAIL', 'zero training vessels available for this task'
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        n_pos = int((y_train == 1).sum()) if 1 in unique_classes else 0
        n_neg = n_train - n_pos
        return None, n_pos, n_neg, 'FAIL', f'only one class present in training labels ({unique_classes.tolist()})'

    if average == 'macro':
        n_pos, n_neg = -1, -1  # not binary; positive/negative counts not meaningful for 7-class VC
    else:
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_eval)
    f1 = f1_score(y_eval, y_pred, average=average, zero_division=0)
    return f1, n_pos, n_neg, 'OK', ''

# ---------------------------------------------------------------------------
# 5. Full-data baseline F1 per seed per task (rate = 100%)
# ---------------------------------------------------------------------------
log("Computing full-data baseline F1 per seed per task...")
full_baseline_rows = []
full_f1_by_task_seed = {}  # (task, seed) -> f1
for seed in SEEDS:
    for task in TASKS:
        f1, n_pos, n_neg, status, reason = fit_eval_task(task, train_agg_full, seed)
        full_f1_by_task_seed[(task, seed)] = f1
        if task == 'VC':
            test_vessels, test_pos, test_prev = len(y_test_vc), -1, -1
        elif task == 'FD':
            test_vessels, test_pos, test_prev = len(y_test_fd), int(y_test_fd.sum()), float(y_test_fd.mean())
        elif task == 'AD':
            test_vessels, test_pos, test_prev = len(y_test_ad_full), int(y_test_ad_full.sum()), float(y_test_ad_full.mean())
        else:
            yt = y_test_adt_full.loc[X_test_adt_full['mmsi']].values
            test_vessels, test_pos, test_prev = len(yt), int(yt.sum()), float(yt.mean())
        full_baseline_rows.append({
            'task': task, 'seed': seed, 'full_data_f1': f1, 'status': status, 'fail_reason': reason,
            'test_vessels': test_vessels, 'test_positive_count': test_pos,
            'test_negative_count': (test_vessels - test_pos) if test_pos >= 0 else -1,
            'test_positive_prevalence': test_prev,
            'majority_class': majority_info[task]['majority_class'],
            'majority_class_f1': majority_info[task]['majority_class_f1'],
        })
        log(f"  seed={seed} task={task}: full_data_f1={f1}")

full_baseline_df = pd.DataFrame(full_baseline_rows)
full_baseline_df.to_csv(f'{OUT}/full_baseline_f1.csv', index=False)
log(f"Saved full_baseline_f1.csv ({len(full_baseline_df)} rows)")

# ---------------------------------------------------------------------------
# 6. Full corrected grid: strategy x rate x seed -> build once, evaluate all 4 tasks
# ---------------------------------------------------------------------------
log("Precomputing Adaptive selection-time deltas (full training stream, no row drop)...")
adaptive_selection_deltas = sfb.compute_selection_deltas(train_raw)

condition_rows = []
n_conditions_total = len(STRATEGIES) * len(RATES) * len(SEEDS)
cond_i = 0

for strategy in STRATEGIES:
    for rate in RATES:
        for seed in SEEDS:
            cond_i += 1
            requested_pings = int(round(total_train_pings * rate))
            unique_vessels_before = n_train_vessels_full

            if rate >= 1.0:
                agg = train_agg_full
                n_retained_pings = total_train_pings
                unique_vessels_with_ping = n_train_vessels_full
            else:
                agg, n_retained_pings = sfb.build_sampled_features(
                    strategy, train_raw, rate, seed, feature_cols=FEATURE_COLS,
                    precomputed_selection_deltas=adaptive_selection_deltas if strategy == 'adaptive' else None,
                    return_ping_count=True,
                )
                # vessels with >=1 sampled ping, BEFORE the lag-derived feature dropna
                if strategy == 'adaptive':
                    sel_raw_for_count = sfb.select_adaptive(adaptive_selection_deltas, rate, seed)
                elif strategy == 'temporal':
                    N = sfb.find_temporal_window(train_raw[sfb.RAW_COLS], rate)
                    sel_raw_for_count = sfb.select_temporal(train_raw[sfb.RAW_COLS], N)
                else:
                    sel_raw_for_count = sfb.STRATEGY_SELECTORS[strategy](train_raw[sfb.RAW_COLS], rate, seed)
                unique_vessels_with_ping = sel_raw_for_count['mmsi'].nunique()

            valid_vessels = len(agg)
            actual_ping_retention_pct = 100.0 * n_retained_pings / total_train_pings

            for task in TASKS:
                f1, n_pos, n_neg, status, reason = fit_eval_task(task, agg, seed)
                full_f1_same_seed = full_f1_by_task_seed[(task, seed)]
                if status == 'OK' and full_f1_same_seed is not None and full_f1_same_seed > 0:
                    retention_pct = 100.0 * f1 / full_f1_same_seed
                else:
                    retention_pct = None

                condition_rows.append({
                    'task': task, 'strategy': strategy, 'target_rate': rate, 'seed': seed,
                    'requested_training_pings': requested_pings,
                    'retained_training_pings': n_retained_pings,
                    'actual_ping_retention_percent': actual_ping_retention_pct,
                    'unique_vessels_before_sampling': unique_vessels_before,
                    'unique_vessels_with_at_least_one_sampled_ping': unique_vessels_with_ping,
                    'unique_vessels_with_valid_final_feature_vector': valid_vessels,
                    'training_positive_count': n_pos,
                    'training_negative_count': n_neg,
                    'sampled_f1': f1,
                    'full_f1_same_seed': full_f1_same_seed,
                    'retention_percent': retention_pct,
                    'status': status,
                    'fail_reason': reason,
                })

            if cond_i % 10 == 0 or cond_i == n_conditions_total:
                log(f"  progress: {cond_i}/{n_conditions_total} (strategy={strategy}, rate={rate}, seed={seed})")

condition_df = pd.DataFrame(condition_rows)
condition_df.to_csv(f'{OUT}/condition_metrics_18_features.csv', index=False)
log(f"Saved condition_metrics_18_features.csv ({len(condition_df)} rows)")

log("PHASE 1 EXPERIMENT RUN COMPLETE.")
