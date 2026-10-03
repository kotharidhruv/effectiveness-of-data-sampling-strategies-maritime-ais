"""
run_phase3_experiment.py
==========================
Phase 3: Logistic Regression sensitivity analysis for ADT, at both the Phase 1
18-feature specification and the Phase 2 16-feature (timing-removed) specification.
Reuses sampled_feature_builder.py identically to Phase 1/2 -- same sampling
functions, same seeds, same lag-recomputation procedure, same fixed Isolation
Forest ADT labels (recomputed deterministically, matching Phase 1/2 exactly --
never reads a Phase 1/2 output file for numeric reuse).

Only the downstream classifier differs from Phase 1/2:
  RandomForestClassifier  ->  Pipeline(StandardScaler(), LogisticRegression(...))
"""
import sys
import time
import hashlib
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision')
import sampled_feature_builder as sfb

P1 = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase1_18feature_corrected'
P2 = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase2_16feature_ablation'
OUT = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase3_second_model'

FEATURE_COLS_18 = sfb.FEATURE_COLS_18
FEATURE_COLS_16 = sfb.FEATURE_COLS_16
STRATEGIES = ['random', 'stratified', 'spatial', 'importance', 'adaptive', 'temporal']
RATES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2, 3, 4]
MIN_GROUP_SIZE = 20
IF_CONTAMINATION = 0.05

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ---------------------------------------------------------------------------
# STEP 1: Verify checksums (STOP if mismatch)
# ---------------------------------------------------------------------------
EXPECTED_TRAIN = '5824aa4da1d63de8c128f83839f819b5b9aa1d32e4dc99ae9429ffc465b6b87d'
EXPECTED_TEST = '4b6b47051bbd0916f331e2a1b162701d52731a0ef32d5917e19ad25b89d274b2'

train_raw = pd.read_parquet(f'{P1}/train_raw_prelag.parquet')
test_raw = pd.read_parquet(f'{P1}/test_raw_prelag.parquet')
train_hash = hashlib.sha256(','.join(map(str, sorted(train_raw['mmsi'].unique().tolist()))).encode()).hexdigest()
test_hash = hashlib.sha256(','.join(map(str, sorted(test_raw['mmsi'].unique().tolist()))).encode()).hexdigest()
log(f"train checksum match={train_hash==EXPECTED_TRAIN}, test checksum match={test_hash==EXPECTED_TEST}")
if train_hash != EXPECTED_TRAIN or test_hash != EXPECTED_TEST:
    log("FATAL: checksum mismatch. STOPPING.")
    sys.exit(1)
overlap = set(train_raw['mmsi'].unique()) & set(test_raw['mmsi'].unique())
assert len(overlap) == 0
total_train_pings = len(train_raw)
n_train_vessels_full = train_raw['mmsi'].nunique()

def vessel_type_mode_labels(raw_df):
    return (raw_df.groupby('mmsi')['vessel_type_group'].agg(lambda x: x.mode()[0])
            .reset_index().rename(columns={'vessel_type_group': 'type_group'}))
train_type_labels = vessel_type_mode_labels(train_raw)
test_type_labels = vessel_type_mode_labels(test_raw)
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

def build_adt_labels(feature_cols):
    train_agg = sfb.build_sampled_features('random', train_raw, 1.0, seed=0, feature_cols=feature_cols)
    test_agg = sfb.build_sampled_features('random', test_raw, 1.0, seed=0, feature_cols=feature_cols)
    train_typed = train_agg.merge(train_type_labels[['mmsi', 'type_group']], on='mmsi')
    test_typed = test_agg.merge(test_type_labels[['mmsi', 'type_group']], on='mmsi')
    y_train_parts, y_test_parts = [], []
    for vtype, group in train_typed.groupby('type_group'):
        tgroup = test_typed[test_typed['type_group'] == vtype]
        if len(group) < MIN_GROUP_SIZE:
            y_train_parts.append(pd.Series(0, index=group['mmsi'].values))
            y_test_parts.append(pd.Series(0, index=tgroup['mmsi'].values))
            continue
        sc = StandardScaler()
        Xg = sc.fit_transform(group[feature_cols].values)
        iso_g = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION, random_state=42, n_jobs=-1)
        iso_g.fit(Xg)
        y_g = (iso_g.predict(Xg) == -1).astype(int)
        y_train_parts.append(pd.Series(y_g, index=group['mmsi'].values))
        if len(tgroup) > 0:
            Xgt = sc.transform(tgroup[feature_cols].values)
            y_test_parts.append(pd.Series((iso_g.predict(Xgt) == -1).astype(int), index=tgroup['mmsi'].values))
    y_train = pd.concat(y_train_parts).rename('is_anomaly')
    y_test = pd.concat(y_test_parts).rename('is_anomaly') if y_test_parts else pd.Series(dtype=int)
    return train_agg, test_agg, y_train, y_test

log("Building ADT ground truth (18F) -- deterministic recompute, must match Phase 1 exactly...")
train_agg_full_18, test_agg_full_18, y_train_adt_18, y_test_adt_18 = build_adt_labels(FEATURE_COLS_18)
log(f"  18F ADT train anomalies: {y_train_adt_18.sum()}/{len(y_train_adt_18)} (expect 161/3166)")

log("Building ADT ground truth (16F) -- deterministic recompute, must match Phase 2 exactly...")
train_agg_full_16, test_agg_full_16, y_train_adt_16, y_test_adt_16 = build_adt_labels(FEATURE_COLS_16)
log(f"  16F ADT train anomalies: {y_train_adt_16.sum()}/{len(y_train_adt_16)} (expect 161/3166)")

test_agg_adt_18 = add_type_features(test_agg_full_18, test_type_labels)
X_test_adt_18 = test_agg_adt_18[test_agg_adt_18['mmsi'].isin(y_test_adt_18.index)].copy()
test_agg_adt_16 = add_type_features(test_agg_full_16, test_type_labels)
X_test_adt_16 = test_agg_adt_16[test_agg_adt_16['mmsi'].isin(y_test_adt_16.index)].copy()

# ---------------------------------------------------------------------------
# Logistic Regression fit+eval helper (Pipeline: StandardScaler -> LogisticRegression)
# ---------------------------------------------------------------------------
def fit_eval_logistic(agg_df, feature_cols, y_train_full, X_test_fixed, y_test_full, seed):
    typed = add_type_features(agg_df, train_type_labels[train_type_labels['mmsi'].isin(agg_df['mmsi'])])
    typed = typed[typed['mmsi'].isin(y_train_full.index)]
    y_train = y_train_full.loc[typed['mmsi']].values
    X_train = typed[feature_cols + TYPE_COLS].values
    X_eval = X_test_fixed[feature_cols + TYPE_COLS].values
    y_eval = y_test_full.loc[X_test_fixed['mmsi']].values

    n_train = len(y_train)
    if n_train == 0:
        return None, 0, 0, 'FAIL', 'zero training vessels', False, None
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        n_pos = int((y_train == 1).sum())
        return None, n_pos, n_train - n_pos, 'FAIL', f'only one class present ({unique_classes.tolist()})', False, None

    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', C=1.0, max_iter=5000, solver='lbfgs')),
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', ConvergenceWarning)
        pipe.fit(X_train, y_train)
        converged = not any(issubclass(warn.category, ConvergenceWarning) for warn in w)
    n_iter = int(pipe.named_steps['lr'].n_iter_[0])
    y_pred = pipe.predict(X_eval)
    f1 = f1_score(y_eval, y_pred, average='binary', zero_division=0)
    return f1, n_pos, n_neg, 'OK', '', converged, n_iter

# ---------------------------------------------------------------------------
# Run both grids (18F and 16F), 180 rows each
# ---------------------------------------------------------------------------
adaptive_selection_deltas = sfb.compute_selection_deltas(train_raw)

def run_grid(feature_cols, y_train_full, X_test_fixed, y_test_full, label_tag):
    rows = []
    full_f1_by_seed = {}
    log(f"[{label_tag}] Computing full-data (100%) Logistic baseline per seed...")
    for seed in SEEDS:
        f1, n_pos, n_neg, status, reason, converged, n_iter = fit_eval_logistic(
            train_agg_full_18 if feature_cols == FEATURE_COLS_18 else train_agg_full_16,
            feature_cols, y_train_full, X_test_fixed, y_test_full, seed)
        full_f1_by_seed[seed] = f1
        log(f"  seed={seed}: full_data_f1={f1} converged={converged} n_iter={n_iter}")

    n_conditions = len(STRATEGIES) * len(RATES) * len(SEEDS)
    cond_i = 0
    for strategy in STRATEGIES:
        for rate in RATES:
            for seed in SEEDS:
                cond_i += 1
                if rate >= 1.0:
                    agg = train_agg_full_18 if feature_cols == FEATURE_COLS_18 else train_agg_full_16
                    n_retained_pings = total_train_pings
                else:
                    agg, n_retained_pings = sfb.build_sampled_features(
                        strategy, train_raw, rate, seed, feature_cols=feature_cols,
                        precomputed_selection_deltas=adaptive_selection_deltas if strategy == 'adaptive' else None,
                        return_ping_count=True,
                    )
                valid_vessels = len(agg)
                actual_ping_pct = 100.0 * n_retained_pings / total_train_pings

                f1, n_pos, n_neg, status, reason, converged, n_iter = fit_eval_logistic(
                    agg, feature_cols, y_train_full, X_test_fixed, y_test_full, seed)
                full_f1_same_seed = full_f1_by_seed[seed]
                retention_pct = (100.0 * f1 / full_f1_same_seed) if (status == 'OK' and full_f1_same_seed) else None

                rows.append({
                    'feature_set': label_tag, 'strategy': strategy, 'target_rate': rate, 'seed': seed,
                    'retained_training_pings': n_retained_pings,
                    'actual_ping_retention_percent': actual_ping_pct,
                    'valid_training_vessels': valid_vessels,
                    'training_positive_count': n_pos, 'training_negative_count': n_neg,
                    'test_vessels': len(y_test_full.loc[X_test_fixed['mmsi']]),
                    'test_positive_count': int(y_test_full.loc[X_test_fixed['mmsi']].sum()),
                    'sampled_f1': f1, 'full_f1': full_f1_same_seed, 'retention_percent': retention_pct,
                    'status': status, 'fail_reason': reason, 'converged': converged, 'iterations': n_iter,
                })
                if cond_i % 30 == 0 or cond_i == n_conditions:
                    log(f"  [{label_tag}] progress {cond_i}/{n_conditions} (strategy={strategy}, rate={rate}, seed={seed})")
    return pd.DataFrame(rows), full_f1_by_seed

log("=== Running Phase 3A: 18-feature ADT Logistic Regression grid ===")
df_18, full_f1_18 = run_grid(FEATURE_COLS_18, y_train_adt_18, X_test_adt_18, y_test_adt_18, '18F')
df_18.to_csv(f'{OUT}/logistic_ADT_18F_condition_metrics.csv', index=False)
log(f"Saved logistic_ADT_18F_condition_metrics.csv ({len(df_18)} rows)")

log("=== Running Phase 3B: 16-feature ADT Logistic Regression grid ===")
df_16, full_f1_16 = run_grid(FEATURE_COLS_16, y_train_adt_16, X_test_adt_16, y_test_adt_16, '16F')
df_16.to_csv(f'{OUT}/logistic_ADT_16F_condition_metrics.csv', index=False)
log(f"Saved logistic_ADT_16F_condition_metrics.csv ({len(df_16)} rows)")

# full_baseline_logistic_ADT.csv
baseline_rows = []
for feature_set, full_f1_dict, train_agg, y_train_full, X_test_fixed, y_test_full in [
    ('18F', full_f1_18, train_agg_full_18, y_train_adt_18, X_test_adt_18, y_test_adt_18),
    ('16F', full_f1_16, train_agg_full_16, y_train_adt_16, X_test_adt_16, y_test_adt_16),
]:
    feature_cols = FEATURE_COLS_18 if feature_set == '18F' else FEATURE_COLS_16
    for seed in SEEDS:
        f1, n_pos, n_neg, status, reason, converged, n_iter = fit_eval_logistic(
            train_agg, feature_cols, y_train_full, X_test_fixed, y_test_full, seed)
        baseline_rows.append({
            'feature_set': feature_set, 'seed': seed, 'full_data_f1': f1,
            'train_vessels': n_pos + n_neg, 'test_vessels': len(y_test_full.loc[X_test_fixed['mmsi']]),
            'train_positive_count': n_pos, 'test_positive_count': int(y_test_full.loc[X_test_fixed['mmsi']].sum()),
            'converged': converged,
        })
full_baseline_logistic = pd.DataFrame(baseline_rows)
full_baseline_logistic.to_csv(f'{OUT}/full_baseline_logistic_ADT.csv', index=False)
log(f"Saved full_baseline_logistic_ADT.csv ({len(full_baseline_logistic)} rows)")

log("PHASE 3 PRIMARY EXPERIMENT RUN COMPLETE.")
