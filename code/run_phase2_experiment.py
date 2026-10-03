"""
run_phase2_experiment.py
==========================
Phase 2: 16-feature timing ablation. Freezes and reuses the Phase 1 train/test
split exactly (checksums verified before anything runs). Never touches any Phase 1
output file. Produces two separate AD/ADT analyses (16F_RELABELED and
16F_FIXED_18F_LABELS) plus the shared VC/FD grid, all at 16 features
(the two ping-timing features, median_ping_gap and std_ping_gap, removed).
"""

import sys
import time
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision')
import sampled_feature_builder as sfb

P1 = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase1_18feature_corrected'
OUT = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase2_16feature_ablation'

FEATURE_COLS_16 = sfb.FEATURE_COLS_16
FEATURE_COLS_18 = sfb.FEATURE_COLS_18
assert len(FEATURE_COLS_16) == 16, f"expected 16 features, got {len(FEATURE_COLS_16)}"
assert 'median_ping_gap' not in FEATURE_COLS_16 and 'std_ping_gap' not in FEATURE_COLS_16
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
# STEP 1: Freeze and verify Phase 1 split (STOP if checksums differ)
# ---------------------------------------------------------------------------
EXPECTED_TRAIN_HASH = '5824aa4da1d63de8c128f83839f819b5b9aa1d32e4dc99ae9429ffc465b6b87d'
EXPECTED_TEST_HASH = '4b6b47051bbd0916f331e2a1b162701d52731a0ef32d5917e19ad25b89d274b2'

train_raw = pd.read_parquet(f'{P1}/train_raw_prelag.parquet')
test_raw = pd.read_parquet(f'{P1}/test_raw_prelag.parquet')

train_hash = hashlib.sha256(','.join(map(str, sorted(train_raw['mmsi'].unique().tolist()))).encode()).hexdigest()
test_hash = hashlib.sha256(','.join(map(str, sorted(test_raw['mmsi'].unique().tolist()))).encode()).hexdigest()

log(f"Train checksum: {train_hash} (expected {EXPECTED_TRAIN_HASH}) match={train_hash==EXPECTED_TRAIN_HASH}")
log(f"Test checksum:  {test_hash} (expected {EXPECTED_TEST_HASH}) match={test_hash==EXPECTED_TEST_HASH}")
if train_hash != EXPECTED_TRAIN_HASH or test_hash != EXPECTED_TEST_HASH:
    log("FATAL: checksum mismatch. STOPPING per instructions.")
    sys.exit(1)
log(f"Confirmed: {train_raw['mmsi'].nunique()} train vessels, {test_raw['mmsi'].nunique()} test vessels -- Phase 1 split verified.")

overlap = set(train_raw['mmsi'].unique()) & set(test_raw['mmsi'].unique())
assert len(overlap) == 0
total_train_pings = len(train_raw)

# ---------------------------------------------------------------------------
# STEP 2: Build FULL (100%) 16-feature training/test matrices
# ---------------------------------------------------------------------------
log("Building FULL 16-feature training feature matrix...")
train_agg_full_16 = sfb.build_sampled_features('random', train_raw, 1.0, seed=0, feature_cols=FEATURE_COLS_16)
log(f"  -> {len(train_agg_full_16)} valid training vessels (16-feature)")
test_agg_full_16 = sfb.build_sampled_features('random', test_raw, 1.0, seed=0, feature_cols=FEATURE_COLS_16)
log(f"  -> {len(test_agg_full_16)} valid test vessels (16-feature)")

# Also rebuild the FULL 18-feature matrices, deterministically, ONLY to source the
# fixed Phase 1 AD/ADT labels for 16F_FIXED_18F_LABELS (NOT reusing a cached file --
# this recomputation is byte-for-byte identical to what run_phase1_experiment.py did).
log("Rebuilding FULL 18-feature matrices (to source fixed Phase 1 AD/ADT labels for 16F_FIXED_18F_LABELS)...")
train_agg_full_18 = sfb.build_sampled_features('random', train_raw, 1.0, seed=0, feature_cols=FEATURE_COLS_18)
test_agg_full_18 = sfb.build_sampled_features('random', test_raw, 1.0, seed=0, feature_cols=FEATURE_COLS_18)

n_train_vessels_full = train_raw['mmsi'].nunique()

# ---------------------------------------------------------------------------
# STEP 3: Vessel-type labels (identical source/logic to Phase 1 -- VC/FD labels
# do not depend on the feature set)
# ---------------------------------------------------------------------------
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

def add_type_features(agg_df, type_group_labels):
    dummies = pd.get_dummies(type_group_labels['type_group'], prefix='type').astype(int)
    for c in TYPE_COLS:
        if c not in dummies.columns:
            dummies[c] = 0
    dummies = dummies[TYPE_COLS]
    dummies['mmsi'] = type_group_labels['mmsi'].values
    return agg_df.merge(dummies, on='mmsi', how='left').fillna(0)

# ---------------------------------------------------------------------------
# STEP 4a: 16F_RELABELED ground truth (AD pooled, ADT per-type) -- fit fresh
# Isolation Forests using ONLY the 16 features
# ---------------------------------------------------------------------------
log("Building 16F_RELABELED AD (pooled) ground truth...")
scaler_ad_16 = StandardScaler()
X_tr_16 = scaler_ad_16.fit_transform(train_agg_full_16[FEATURE_COLS_16].values)
X_te_16 = scaler_ad_16.transform(test_agg_full_16[FEATURE_COLS_16].values)
iso_ad_16 = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION, random_state=42, n_jobs=-1)
iso_ad_16.fit(X_tr_16)
y_train_ad_16 = pd.Series((iso_ad_16.predict(X_tr_16) == -1).astype(int), index=train_agg_full_16['mmsi'].values, name='is_anomaly')
y_test_ad_16 = pd.Series((iso_ad_16.predict(X_te_16) == -1).astype(int), index=test_agg_full_16['mmsi'].values, name='is_anomaly')
log(f"  16F_RELABELED AD train anomalies: {y_train_ad_16.sum()}/{len(y_train_ad_16)} | test: {y_test_ad_16.sum()}/{len(y_test_ad_16)}")

log("Building 16F_RELABELED ADT (per vessel type) ground truth...")
train_typed_16 = train_agg_full_16.merge(train_type_labels[['mmsi', 'type_group']], on='mmsi')
test_typed_16 = test_agg_full_16.merge(test_type_labels[['mmsi', 'type_group']], on='mmsi')
adt16_train_parts, adt16_test_parts = [], []
for vtype, group in train_typed_16.groupby('type_group'):
    tgroup = test_typed_16[test_typed_16['type_group'] == vtype]
    if len(group) < MIN_GROUP_SIZE:
        adt16_train_parts.append(pd.Series(0, index=group['mmsi'].values))
        adt16_test_parts.append(pd.Series(0, index=tgroup['mmsi'].values))
        continue
    sc = StandardScaler()
    Xg = sc.fit_transform(group[FEATURE_COLS_16].values)
    iso_g = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION, random_state=42, n_jobs=-1)
    iso_g.fit(Xg)
    y_g = (iso_g.predict(Xg) == -1).astype(int)
    adt16_train_parts.append(pd.Series(y_g, index=group['mmsi'].values))
    if len(tgroup) > 0:
        Xgt = sc.transform(tgroup[FEATURE_COLS_16].values)
        adt16_test_parts.append(pd.Series((iso_g.predict(Xgt) == -1).astype(int), index=tgroup['mmsi'].values))
y_train_adt_16 = pd.concat(adt16_train_parts).rename('is_anomaly')
y_test_adt_16 = pd.concat(adt16_test_parts).rename('is_anomaly') if adt16_test_parts else pd.Series(dtype=int)
log(f"  16F_RELABELED ADT train anomalies: {y_train_adt_16.sum()}/{len(y_train_adt_16)} | test: {y_test_adt_16.sum()}/{len(y_test_adt_16)}")

# ---------------------------------------------------------------------------
# STEP 4b: 16F_FIXED_18F_LABELS ground truth -- reuse Phase 1's fixed 18F labels
# (recomputed deterministically from the 18-feature matrices, NOT read from a
# Phase 1 output file -- Phase 1 files are never opened for numeric reuse here)
# ---------------------------------------------------------------------------
log("Rebuilding fixed Phase 1 18F AD/ADT ground truth (for 16F_FIXED_18F_LABELS)...")
scaler_ad_18 = StandardScaler()
X_tr_18 = scaler_ad_18.fit_transform(train_agg_full_18[FEATURE_COLS_18].values)
X_te_18 = scaler_ad_18.transform(test_agg_full_18[FEATURE_COLS_18].values)
iso_ad_18 = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION, random_state=42, n_jobs=-1)
iso_ad_18.fit(X_tr_18)
y_train_ad_18fixed = pd.Series((iso_ad_18.predict(X_tr_18) == -1).astype(int), index=train_agg_full_18['mmsi'].values, name='is_anomaly')
y_test_ad_18fixed = pd.Series((iso_ad_18.predict(X_te_18) == -1).astype(int), index=test_agg_full_18['mmsi'].values, name='is_anomaly')

train_typed_18 = train_agg_full_18.merge(train_type_labels[['mmsi', 'type_group']], on='mmsi')
test_typed_18 = test_agg_full_18.merge(test_type_labels[['mmsi', 'type_group']], on='mmsi')
adt18_train_parts, adt18_test_parts = [], []
for vtype, group in train_typed_18.groupby('type_group'):
    tgroup = test_typed_18[test_typed_18['type_group'] == vtype]
    if len(group) < MIN_GROUP_SIZE:
        adt18_train_parts.append(pd.Series(0, index=group['mmsi'].values))
        adt18_test_parts.append(pd.Series(0, index=tgroup['mmsi'].values))
        continue
    sc = StandardScaler()
    Xg = sc.fit_transform(group[FEATURE_COLS_18].values)
    iso_g = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION, random_state=42, n_jobs=-1)
    iso_g.fit(Xg)
    y_g = (iso_g.predict(Xg) == -1).astype(int)
    adt18_train_parts.append(pd.Series(y_g, index=group['mmsi'].values))
    if len(tgroup) > 0:
        Xgt = sc.transform(tgroup[FEATURE_COLS_18].values)
        adt18_test_parts.append(pd.Series((iso_g.predict(Xgt) == -1).astype(int), index=tgroup['mmsi'].values))
y_train_adt_18fixed = pd.concat(adt18_train_parts).rename('is_anomaly')
y_test_adt_18fixed = pd.concat(adt18_test_parts).rename('is_anomaly') if adt18_test_parts else pd.Series(dtype=int)
log(f"  Fixed 18F AD train anomalies: {y_train_ad_18fixed.sum()}/{len(y_train_ad_18fixed)} | "
    f"ADT train anomalies: {y_train_adt_18fixed.sum()}/{len(y_train_adt_18fixed)}")

# Sanity: this MUST match Phase 1's numbers exactly (159/3166 AD, 161/3166 ADT per phase1_run.log)
log(f"  [cross-check vs Phase 1 log: expected AD train=159/3166, ADT train=161/3166]")

# ---------------------------------------------------------------------------
# Fixed evaluation matrices
# ---------------------------------------------------------------------------
X_test_vc_16 = test_agg_full_16.merge(test_type_labels[['mmsi', 'label']], on='mmsi')
X_test_fd_16 = test_agg_full_16.merge(test_type_labels[['mmsi', 'is_fishing']], on='mmsi')
test_agg_adt_full_16 = add_type_features(test_agg_full_16, test_type_labels)
X_test_adt_full_16 = test_agg_adt_full_16[test_agg_adt_full_16['mmsi'].isin(y_test_adt_16.index)].copy()

test_agg_adt_full_16_fixedlabels = add_type_features(test_agg_full_16, test_type_labels)
X_test_adt_full_16_fixed = test_agg_adt_full_16_fixedlabels[test_agg_adt_full_16_fixedlabels['mmsi'].isin(y_test_adt_18fixed.index)].copy()

# ---------------------------------------------------------------------------
# Majority-class baselines
# ---------------------------------------------------------------------------
def majority_baseline(y_train, y_test, average):
    maj_class = pd.Series(y_train).mode()[0]
    y_pred = np.full(len(y_test), maj_class)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
    return maj_class, f1

y_train_vc = train_type_labels.set_index('mmsi').loc[train_agg_full_16['mmsi'], 'label'].values
y_test_vc = X_test_vc_16['label'].values
maj_vc, f1_maj_vc = majority_baseline(y_train_vc, y_test_vc, 'macro')

y_train_fd = train_type_labels.set_index('mmsi').loc[train_agg_full_16['mmsi'], 'is_fishing'].values
y_test_fd = X_test_fd_16['is_fishing'].values
maj_fd, f1_maj_fd = majority_baseline(y_train_fd, y_test_fd, 'binary')

maj_ad16, f1_maj_ad16 = majority_baseline(y_train_ad_16.values, y_test_ad_16.values, 'binary')
maj_adt16, f1_maj_adt16 = majority_baseline(y_train_adt_16.values, y_test_adt_16.values, 'binary')

majority_info_16 = {
    'VC': {'majority_class': int(maj_vc), 'majority_class_f1': f1_maj_vc},
    'FD': {'majority_class': int(maj_fd), 'majority_class_f1': f1_maj_fd},
    'AD': {'majority_class': int(maj_ad16), 'majority_class_f1': f1_maj_ad16},
    'ADT': {'majority_class': int(maj_adt16), 'majority_class_f1': f1_maj_adt16},
}
with open(f'{OUT}/majority_baseline_info_16F.json', 'w') as f:
    json.dump(majority_info_16, f, indent=2, default=str)
log(f"Majority-class F1 (16F_RELABELED): VC={f1_maj_vc:.4f} FD={f1_maj_fd:.4f} AD={f1_maj_ad16:.4f} ADT={f1_maj_adt16:.4f}")

# ---------------------------------------------------------------------------
# fit_eval helpers
# ---------------------------------------------------------------------------
def fit_eval_relabeled(task, agg_df, seed):
    if task == 'VC':
        merged = agg_df.merge(train_type_labels[['mmsi', 'label']], on='mmsi')
        X_train, y_train = merged[FEATURE_COLS_16].values, merged['label'].values
        X_eval, y_eval, average = X_test_vc_16[FEATURE_COLS_16].values, y_test_vc, 'macro'
    elif task == 'FD':
        merged = agg_df.merge(train_type_labels[['mmsi', 'is_fishing']], on='mmsi')
        X_train, y_train = merged[FEATURE_COLS_16].values, merged['is_fishing'].values
        X_eval, y_eval, average = X_test_fd_16[FEATURE_COLS_16].values, y_test_fd, 'binary'
    elif task == 'AD':
        avail = agg_df[agg_df['mmsi'].isin(y_train_ad_16.index)]
        y_train = y_train_ad_16.loc[avail['mmsi']].values
        X_train = scaler_ad_16.transform(avail[FEATURE_COLS_16].values)
        X_eval, y_eval, average = X_te_16, y_test_ad_16.values, 'binary'
    elif task == 'ADT':
        typed = add_type_features(agg_df, train_type_labels[train_type_labels['mmsi'].isin(agg_df['mmsi'])])
        typed = typed[typed['mmsi'].isin(y_train_adt_16.index)]
        y_train = y_train_adt_16.loc[typed['mmsi']].values
        X_train = typed[FEATURE_COLS_16 + TYPE_COLS].values
        X_eval = X_test_adt_full_16[FEATURE_COLS_16 + TYPE_COLS].values
        y_eval = y_test_adt_16.loc[X_test_adt_full_16['mmsi']].values
        average = 'binary'
    else:
        raise ValueError(task)
    return _fit_and_score(X_train, y_train, X_eval, y_eval, average, seed)


def fit_eval_fixed18labels(task, agg_df, seed):
    """AD/ADT only -- 16 predictors, fixed Phase-1-equivalent 18F labels."""
    if task == 'AD':
        # scaler_ad_18 was fit on 18 columns and cannot transform a 16-column
        # matrix. For "same (fixed 18F) labels, 16 predictors," scale with
        # scaler_ad_16 (fit once on the full 16-feature training matrix) --
        # matching Phase 1's own convention of a single scaler fit on the full
        # data and reused across every condition. Only the LABELS come from the
        # fixed 18F ground truth here; the feature values and their scaling are
        # entirely 16-feature.
        avail = agg_df[agg_df['mmsi'].isin(y_train_ad_18fixed.index)]
        y_train = y_train_ad_18fixed.loc[avail['mmsi']].values
        X_train = scaler_ad_16.transform(avail[FEATURE_COLS_16].values)
        X_eval, y_eval, average = X_te_16, y_test_ad_18fixed.values, 'binary'
    elif task == 'ADT':
        typed = add_type_features(agg_df, train_type_labels[train_type_labels['mmsi'].isin(agg_df['mmsi'])])
        typed = typed[typed['mmsi'].isin(y_train_adt_18fixed.index)]
        y_train = y_train_adt_18fixed.loc[typed['mmsi']].values
        X_train = typed[FEATURE_COLS_16 + TYPE_COLS].values
        X_eval = X_test_adt_full_16_fixed[FEATURE_COLS_16 + TYPE_COLS].values
        y_eval = y_test_adt_18fixed.loc[X_test_adt_full_16_fixed['mmsi']].values
        average = 'binary'
    else:
        raise ValueError(f"fixed18labels only defined for AD/ADT, got {task}")
    return _fit_and_score(X_train, y_train, X_eval, y_eval, average, seed)


def _fit_and_score(X_train, y_train, X_eval, y_eval, average, seed):
    n_train = len(y_train)
    if n_train == 0:
        return None, 0, 0, 'FAIL', 'zero training vessels available'
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        n_pos = int((y_train == 1).sum()) if 1 in unique_classes else 0
        return None, n_pos, n_train - n_pos, 'FAIL', f'only one class present ({unique_classes.tolist()})'
    if average == 'macro':
        n_pos, n_neg = -1, -1
    else:
        n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_eval)
    f1 = f1_score(y_eval, y_pred, average=average, zero_division=0)
    return f1, n_pos, n_neg, 'OK', ''

# ---------------------------------------------------------------------------
# STEP 5a: full-data baselines, 16F_RELABELED (4 tasks x 5 seeds = 20 rows)
# ---------------------------------------------------------------------------
log("Computing 16F_RELABELED full-data baselines...")
full_baseline_16_rows = []
full_f1_16_by_task_seed = {}
for seed in SEEDS:
    for task in TASKS:
        f1, n_pos, n_neg, status, reason = fit_eval_relabeled(task, train_agg_full_16, seed)
        full_f1_16_by_task_seed[(task, seed)] = f1
        if task == 'VC':
            test_vessels, test_pos, test_prev = len(y_test_vc), -1, -1
        elif task == 'FD':
            test_vessels, test_pos, test_prev = len(y_test_fd), int(y_test_fd.sum()), float(y_test_fd.mean())
        elif task == 'AD':
            test_vessels, test_pos, test_prev = len(y_test_ad_16), int(y_test_ad_16.sum()), float(y_test_ad_16.mean())
        else:
            yt = y_test_adt_16.loc[X_test_adt_full_16['mmsi']].values
            test_vessels, test_pos, test_prev = len(yt), int(yt.sum()), float(yt.mean())
        full_baseline_16_rows.append({
            'task': task, 'seed': seed, 'full_data_f1': f1, 'status': status, 'fail_reason': reason,
            'test_vessels': test_vessels, 'test_positive_count': test_pos,
            'test_negative_count': (test_vessels - test_pos) if test_pos >= 0 else -1,
            'test_positive_prevalence': test_prev,
            'majority_class': majority_info_16[task]['majority_class'],
            'majority_class_f1': majority_info_16[task]['majority_class_f1'],
        })
    log(f"  seed={seed} done: " + ", ".join(f"{t}={full_f1_16_by_task_seed[(t,seed)]:.4f}" for t in TASKS))
full_baseline_16_df = pd.DataFrame(full_baseline_16_rows)
full_baseline_16_df.to_csv(f'{OUT}/full_baseline_f1_16_features.csv', index=False)
log(f"Saved full_baseline_f1_16_features.csv ({len(full_baseline_16_df)} rows)")

# ---------------------------------------------------------------------------
# STEP 5b: full-data baselines, 16F_FIXED_18F_LABELS (AD/ADT only, 2x5=10 rows)
# ---------------------------------------------------------------------------
log("Computing 16F_FIXED_18F_LABELS full-data baselines (AD/ADT only)...")
full_baseline_fixed_rows = []
full_f1_fixed_by_task_seed = {}
for seed in SEEDS:
    for task in ['AD', 'ADT']:
        f1, n_pos, n_neg, status, reason = fit_eval_fixed18labels(task, train_agg_full_16, seed)
        full_f1_fixed_by_task_seed[(task, seed)] = f1
        full_baseline_fixed_rows.append({
            'task': task, 'seed': seed, 'full_data_f1': f1, 'status': status, 'fail_reason': reason,
            'n_train_positive': n_pos, 'n_train_negative': n_neg,
        })
    log(f"  seed={seed}: AD={full_f1_fixed_by_task_seed[('AD',seed)]:.4f} ADT={full_f1_fixed_by_task_seed[('ADT',seed)]:.4f}")
full_baseline_fixed_df = pd.DataFrame(full_baseline_fixed_rows)
full_baseline_fixed_df.to_csv(f'{OUT}/full_baseline_16F_fixed18Flabels_AD_ADT.csv', index=False)
log(f"Saved full_baseline_16F_fixed18Flabels_AD_ADT.csv ({len(full_baseline_fixed_df)} rows)")

# ---------------------------------------------------------------------------
# STEP 6: full primary 16F_RELABELED grid (720 rows)
# ---------------------------------------------------------------------------
log("Precomputing Adaptive selection-time deltas (full training stream, 16F ablation reuses the SAME raw pings)...")
adaptive_selection_deltas = sfb.compute_selection_deltas(train_raw)

condition_rows = []
fixed_condition_rows = []
n_conditions_total = len(STRATEGIES) * len(RATES) * len(SEEDS)
cond_i = 0

for strategy in STRATEGIES:
    for rate in RATES:
        for seed in SEEDS:
            cond_i += 1
            requested_pings = int(round(total_train_pings * rate))
            unique_vessels_before = n_train_vessels_full

            if rate >= 1.0:
                agg = train_agg_full_16
                n_retained_pings = total_train_pings
                unique_vessels_with_ping = n_train_vessels_full
            else:
                agg, n_retained_pings = sfb.build_sampled_features(
                    strategy, train_raw, rate, seed, feature_cols=FEATURE_COLS_16,
                    precomputed_selection_deltas=adaptive_selection_deltas if strategy == 'adaptive' else None,
                    return_ping_count=True,
                )
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
                f1, n_pos, n_neg, status, reason = fit_eval_relabeled(task, agg, seed)
                full_f1_same_seed = full_f1_16_by_task_seed[(task, seed)]
                retention_pct = (100.0 * f1 / full_f1_same_seed) if (status == 'OK' and full_f1_same_seed) else None
                condition_rows.append({
                    'task': task, 'strategy': strategy, 'target_rate': rate, 'seed': seed,
                    'requested_training_pings': requested_pings,
                    'retained_training_pings': n_retained_pings,
                    'actual_ping_retention_percent': actual_ping_retention_pct,
                    'unique_vessels_before_sampling': unique_vessels_before,
                    'unique_vessels_with_at_least_one_sampled_ping': unique_vessels_with_ping,
                    'unique_vessels_with_valid_final_feature_vector': valid_vessels,
                    'training_positive_count': n_pos, 'training_negative_count': n_neg,
                    'sampled_f1': f1, 'full_f1_same_seed': full_f1_same_seed,
                    'retention_percent': retention_pct, 'status': status, 'fail_reason': reason,
                })

            for task in ['AD', 'ADT']:
                f1f, n_posf, n_negf, statusf, reasonf = fit_eval_fixed18labels(task, agg, seed)
                full_f1_same_seed_fixed = full_f1_fixed_by_task_seed[(task, seed)]
                retention_pct_f = (100.0 * f1f / full_f1_same_seed_fixed) if (statusf == 'OK' and full_f1_same_seed_fixed) else None
                fixed_condition_rows.append({
                    'task': task, 'strategy': strategy, 'target_rate': rate, 'seed': seed,
                    'retained_training_pings': n_retained_pings,
                    'actual_ping_retention_percent': actual_ping_retention_pct,
                    'unique_vessels_with_valid_final_feature_vector': valid_vessels,
                    'training_positive_count': n_posf, 'training_negative_count': n_negf,
                    'sampled_f1': f1f, 'full_f1_same_seed': full_f1_same_seed_fixed,
                    'retention_percent': retention_pct_f, 'status': statusf, 'fail_reason': reasonf,
                })

            if cond_i % 15 == 0 or cond_i == n_conditions_total:
                log(f"  progress: {cond_i}/{n_conditions_total} (strategy={strategy}, rate={rate}, seed={seed})")

condition_df = pd.DataFrame(condition_rows)
condition_df.to_csv(f'{OUT}/condition_metrics_16_features.csv', index=False)
log(f"Saved condition_metrics_16_features.csv ({len(condition_df)} rows)")

fixed_condition_df = pd.DataFrame(fixed_condition_rows)
fixed_condition_df.to_csv(f'{OUT}/condition_metrics_16F_fixed18Flabels_AD_ADT.csv', index=False)
log(f"Saved condition_metrics_16F_fixed18Flabels_AD_ADT.csv ({len(fixed_condition_df)} rows)")

log("PHASE 2 PRIMARY EXPERIMENT RUN COMPLETE.")
