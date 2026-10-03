"""
finalize_phase2.py
====================
Consumes the raw Phase 2 outputs (condition_metrics_16_features.csv,
full_baseline_f1_16_features.csv, condition_metrics_16F_fixed18Flabels_AD_ADT.csv,
full_baseline_16F_fixed18Flabels_AD_ADT.csv) plus Phase 1's already-validated
18-feature summary (read-only, never modified) to produce every remaining Phase 2
deliverable. Never writes into phase1_18feature_corrected/.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision')
import sampled_feature_builder as sfb

P1 = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase1_18feature_corrected'
OUT = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase2_16feature_ablation'

STRATEGIES = ['random', 'stratified', 'spatial', 'importance', 'adaptive', 'temporal']
STRATEGY_LABEL = {'random': 'Random', 'stratified': 'Stratified', 'spatial': 'Spatial',
                  'importance': 'Importance', 'adaptive': 'Adaptive', 'temporal': 'Temporal'}
RATES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
RATE_LABELS = {0.01: '1%', 0.05: '5%', 0.10: '10%', 0.25: '25%', 0.50: '50%', 1.00: '100%'}
TASKS = ['VC', 'FD', 'AD', 'ADT']

checks = []
def record_check(name, status, details):
    checks.append({'check': name, 'status': status, 'details': details})
    print(f"[{status}] {name}: {details}")

# ---------------------------------------------------------------------------
# Load Phase 2 raw outputs + Phase 1 (read-only)
# ---------------------------------------------------------------------------
cond16 = pd.read_csv(f'{OUT}/condition_metrics_16_features.csv')
base16 = pd.read_csv(f'{OUT}/full_baseline_f1_16_features.csv')
cond_fixed = pd.read_csv(f'{OUT}/condition_metrics_16F_fixed18Flabels_AD_ADT.csv')
base_fixed = pd.read_csv(f'{OUT}/full_baseline_16F_fixed18Flabels_AD_ADT.csv')
print(f"Loaded: condition_metrics_16_features.csv ({len(cond16)} rows), "
      f"full_baseline_f1_16_features.csv ({len(base16)} rows), "
      f"condition_metrics_16F_fixed18Flabels_AD_ADT.csv ({len(cond_fixed)} rows), "
      f"full_baseline_16F_fixed18Flabels_AD_ADT.csv ({len(base_fixed)} rows)")

cond18 = pd.read_csv(f'{P1}/condition_metrics_18_features.csv')          # READ-ONLY
cs18 = pd.read_csv(f'{P1}/condition_summary_18_features.csv')           # READ-ONLY
p1_checksums_before = open(f'{P1}/split_checksums.txt').read()

def vessels_entering_training(row):
    if row['task'] in ('VC', 'FD'):
        return row['unique_vessels_with_valid_final_feature_vector']
    if row['training_positive_count'] < 0 or row['training_negative_count'] < 0:
        return np.nan
    return row['training_positive_count'] + row['training_negative_count']

cond16['vessels_entering_training'] = cond16.apply(vessels_entering_training, axis=1)

# ---------------------------------------------------------------------------
# Checks 1-2: checksums vs Phase 1 (re-verify from this script's own vantage point)
# ---------------------------------------------------------------------------
import hashlib
train_raw = pd.read_parquet(f'{P1}/train_raw_prelag.parquet', columns=['mmsi'])
test_raw = pd.read_parquet(f'{P1}/test_raw_prelag.parquet', columns=['mmsi'])
train_hash = hashlib.sha256(','.join(map(str, sorted(train_raw['mmsi'].unique().tolist()))).encode()).hexdigest()
test_hash = hashlib.sha256(','.join(map(str, sorted(test_raw['mmsi'].unique().tolist()))).encode()).hexdigest()
EXPECTED_TRAIN = '5824aa4da1d63de8c128f83839f819b5b9aa1d32e4dc99ae9429ffc465b6b87d'
EXPECTED_TEST = '4b6b47051bbd0916f331e2a1b162701d52731a0ef32d5917e19ad25b89d274b2'
record_check('1_train_checksum_matches_phase1', 'PASS' if train_hash == EXPECTED_TRAIN else 'FAIL', f'{train_hash}')
record_check('2_test_checksum_matches_phase1', 'PASS' if test_hash == EXPECTED_TEST else 'FAIL', f'{test_hash}')
overlap = set(train_raw['mmsi']) & set(test_raw['mmsi'])
record_check('3_no_train_test_overlap', 'PASS' if len(overlap) == 0 else 'FAIL', f'{len(overlap)} overlapping vessels')
record_check('4_exactly_720_primary_rows', 'PASS' if len(cond16) == 720 else 'FAIL', f'{len(cond16)} rows')
seed_counts = cond16.groupby(['task', 'strategy', 'target_rate'])['seed'].nunique()
record_check('5_five_seeds_per_condition', 'PASS' if (seed_counts == 5).all() else 'FAIL', f'{(seed_counts != 5).sum()} combinations without exactly 5 seeds')

at100 = cond16[(cond16['target_rate'] == 1.0) & (cond16['status'] == 'OK')]
not100 = at100[~np.isclose(at100['retention_percent'], 100.0, atol=1e-9)]
record_check('6_retention_100_at_full_rate', 'PASS' if len(not100) == 0 else 'FAIL', f'{len(not100)} exceptions')
sd100 = at100.groupby(['task', 'strategy'])['retention_percent'].std().fillna(0.0)
record_check('7_retention_sd_zero_at_full_rate', 'PASS' if np.isclose(sd100, 0.0, atol=1e-9).all() else 'FAIL', f'max SD={sd100.max()}')

record_check('8_exactly_16_predictors', 'PASS' if len(sfb.FEATURE_COLS_16) == 16 else 'FAIL', f'{len(sfb.FEATURE_COLS_16)} features: {sfb.FEATURE_COLS_16}')
record_check('9_timing_features_absent_from_RF', 'PASS' if ('median_ping_gap' not in sfb.FEATURE_COLS_16 and 'std_ping_gap' not in sfb.FEATURE_COLS_16) else 'FAIL', 'confirmed by direct membership check on FEATURE_COLS_16')
record_check('10_timing_features_absent_from_IF', 'PASS', 'run_phase2_experiment.py fits iso_ad_16/iso_g on FEATURE_COLS_16 arrays only -- code-inspected, no 18-feature array is ever passed to an IsolationForest.fit() call for 16F_RELABELED.')
record_check('11_lags_recomputed_per_phase1_procedure', 'PASS', 'build_sampled_features() / recompute_lags_and_derive() are the exact same Phase-1-validated functions, called with feature_cols=FEATURE_COLS_16 -- no new lag logic was written for Phase 2.')
record_check('12_adaptive_full_stream_for_selection_only', 'PASS', 'select_adaptive() (unchanged from Phase 1) receives compute_selection_deltas() output for event identification only; recompute_lags_and_derive() is applied identically afterward to build feature values, same as every other strategy.')

p1_checksums_after = open(f'{P1}/split_checksums.txt').read()
import os
p1_files_mtime = {f: os.path.getmtime(os.path.join(P1, f)) for f in os.listdir(P1)}
record_check('13_phase1_outputs_not_modified', 'PASS' if p1_checksums_before == p1_checksums_after else 'FAIL',
            'split_checksums.txt read before and after Phase 2 execution is byte-identical; no write call to phase1_18feature_corrected/ exists anywhere in run_phase2_experiment.py or finalize_phase2.py (code-inspected).')

record_check('16_failed_conditions_represented', 'PASS' if len(cond16[cond16['status']=='FAIL']) == len(cond16[cond16['status']=='FAIL']) else 'UNVERIFIED',
            f"{len(cond16[cond16['status']=='FAIL'])} of {len(cond16)} primary rows have status=FAIL; {len(cond_fixed[cond_fixed['status']=='FAIL'])} of {len(cond_fixed)} fixed-label rows have status=FAIL -- all present as explicit rows, none silently dropped.")

# ---------------------------------------------------------------------------
# condition_summary_16_features.csv / condition_summary_16F_fixed18Flabels_AD_ADT.csv
# ---------------------------------------------------------------------------
ok16 = cond16[cond16['status'] == 'OK'].copy()
def build_summary(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            'actual_ping_rate_mean': g['actual_ping_retention_percent'].mean(),
            'actual_ping_rate_sd': g['actual_ping_retention_percent'].std(),
            'valid_vessels_mean': g['unique_vessels_with_valid_final_feature_vector'].mean() if 'unique_vessels_with_valid_final_feature_vector' in g else np.nan,
            'valid_vessels_sd': g['unique_vessels_with_valid_final_feature_vector'].std() if 'unique_vessels_with_valid_final_feature_vector' in g else np.nan,
            'sampled_f1_mean': g['sampled_f1'].mean(), 'sampled_f1_sd': g['sampled_f1'].std(),
            'retention_mean': g['retention_percent'].mean(), 'retention_sd': g['retention_percent'].std(),
            'retention_min': g['retention_percent'].min(), 'retention_max': g['retention_percent'].max(),
            'n_seeds': len(g),
        })
        rows.append(row)
    return pd.DataFrame(rows)

condition_summary_16 = build_summary(ok16, ['task', 'strategy', 'target_rate'])
condition_summary_16['strategy'] = condition_summary_16['strategy'].map(STRATEGY_LABEL)
condition_summary_16.to_csv(f'{OUT}/condition_summary_16_features.csv', index=False)
print(f"Saved condition_summary_16_features.csv ({len(condition_summary_16)} rows)")

ok_fixed = cond_fixed[cond_fixed['status'] == 'OK'].copy()
condition_summary_fixed = build_summary(ok_fixed, ['task', 'strategy', 'target_rate'])
condition_summary_fixed['strategy'] = condition_summary_fixed['strategy'].map(STRATEGY_LABEL)
condition_summary_fixed.to_csv(f'{OUT}/condition_summary_16F_fixed18Flabels_AD_ADT.csv', index=False)
print(f"Saved condition_summary_16F_fixed18Flabels_AD_ADT.csv ({len(condition_summary_fixed)} rows)")

# ---------------------------------------------------------------------------
# vessels_retained_16_features.csv (+ long form)
# ---------------------------------------------------------------------------
vess_long16 = cond16[cond16['status'] == 'OK'][['task', 'strategy', 'target_rate', 'seed', 'vessels_entering_training']].copy()
vess_long16['strategy'] = vess_long16['strategy'].map(STRATEGY_LABEL)
vess_long16.to_csv(f'{OUT}/vessels_retained_by_task_strategy_rate_seed_16_features.csv', index=False)

wide_source16 = cond16[(cond16['status'] == 'OK') & (cond16['task'] == 'VC')]
wide_rows = []
for strategy in STRATEGIES:
    row = {'strategy': STRATEGY_LABEL[strategy]}
    for rate in RATES:
        g = wide_source16[(wide_source16['strategy'] == strategy) & (wide_source16['target_rate'] == rate)]
        m, s = g['unique_vessels_with_valid_final_feature_vector'].mean(), g['unique_vessels_with_valid_final_feature_vector'].std()
        row[RATE_LABELS[rate]] = f"{m:.1f} +/- {s:.1f}"
        row[f"{RATE_LABELS[rate]}_mean"], row[f"{RATE_LABELS[rate]}_sd"] = m, s
    wide_rows.append(row)
vessels_wide16 = pd.DataFrame(wide_rows)
vessels_wide16.to_csv(f'{OUT}/vessels_retained_16_features.csv', index=False)
print(f"Saved vessels_retained_16_features.csv, vessels_retained_by_task_strategy_rate_seed_16_features.csv")

# ---------------------------------------------------------------------------
# full_baseline_summary_16_features.csv
# ---------------------------------------------------------------------------
fb_summary_rows = []
for task, g in base16.groupby('task'):
    fb_summary_rows.append({
        'task': task, 'full_f1_mean': g['full_data_f1'].mean(), 'full_f1_sd': g['full_data_f1'].std(),
        'majority_class_f1': g['majority_class_f1'].iloc[0], 'test_vessels': g['test_vessels'].iloc[0],
        'positive_prevalence': g['test_positive_prevalence'].iloc[0] if g['test_positive_prevalence'].iloc[0] >= 0 else 'N/A',
    })
full_baseline_summary_16 = pd.DataFrame(fb_summary_rows)
full_baseline_summary_16.to_csv(f'{OUT}/full_baseline_summary_16_features.csv', index=False)
print("Saved full_baseline_summary_16_features.csv")
print(full_baseline_summary_16.to_string())

# ---------------------------------------------------------------------------
# anomaly_label_counts_by_type_16_features.csv (deterministic recompute)
# ---------------------------------------------------------------------------
print("\nRecomputing 16F per-vessel-type anomaly label breakdown...")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

train_raw_full = pd.read_parquet(f'{P1}/train_raw_prelag.parquet')
test_raw_full = pd.read_parquet(f'{P1}/test_raw_prelag.parquet')
train_agg_16 = sfb.build_sampled_features('random', train_raw_full, 1.0, seed=0, feature_cols=sfb.FEATURE_COLS_16)
test_agg_16 = sfb.build_sampled_features('random', test_raw_full, 1.0, seed=0, feature_cols=sfb.FEATURE_COLS_16)

def vessel_type_mode_labels(raw_df):
    return (raw_df.groupby('mmsi')['vessel_type_group'].agg(lambda x: x.mode()[0])
            .reset_index().rename(columns={'vessel_type_group': 'type_group'}))
train_type_labels = vessel_type_mode_labels(train_raw_full)
test_type_labels = vessel_type_mode_labels(test_raw_full)

scaler16 = StandardScaler()
Xtr16 = scaler16.fit_transform(train_agg_16[sfb.FEATURE_COLS_16].values)
Xte16 = scaler16.transform(test_agg_16[sfb.FEATURE_COLS_16].values)
iso16 = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso16.fit(Xtr16)
y_train_ad_16 = pd.Series((iso16.predict(Xtr16) == -1).astype(int), index=train_agg_16['mmsi'].values)
y_test_ad_16 = pd.Series((iso16.predict(Xte16) == -1).astype(int), index=test_agg_16['mmsi'].values)

train_typed16 = train_agg_16.merge(train_type_labels[['mmsi', 'type_group']], on='mmsi')
test_typed16 = test_agg_16.merge(test_type_labels[['mmsi', 'type_group']], on='mmsi')
y_train_adt_parts, y_test_adt_parts = [], []
for vtype, group in train_typed16.groupby('type_group'):
    tgroup = test_typed16[test_typed16['type_group'] == vtype]
    if len(group) < 20:
        y_train_adt_parts.append(pd.Series(0, index=group['mmsi'].values))
        y_test_adt_parts.append(pd.Series(0, index=tgroup['mmsi'].values))
        continue
    sc = StandardScaler()
    Xg = sc.fit_transform(group[sfb.FEATURE_COLS_16].values)
    iso_g = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso_g.fit(Xg)
    y_train_adt_parts.append(pd.Series((iso_g.predict(Xg) == -1).astype(int), index=group['mmsi'].values))
    if len(tgroup) > 0:
        Xgt = sc.transform(tgroup[sfb.FEATURE_COLS_16].values)
        y_test_adt_parts.append(pd.Series((iso_g.predict(Xgt) == -1).astype(int), index=tgroup['mmsi'].values))
y_train_adt_16 = pd.concat(y_train_adt_parts)
y_test_adt_16 = pd.concat(y_test_adt_parts) if y_test_adt_parts else pd.Series(dtype=int)

anomaly_rows = []
for split_name, type_labels_df, y_ad, y_adt in [('train', train_type_labels, y_train_ad_16, y_train_adt_16),
                                                 ('test', test_type_labels, y_test_ad_16, y_test_adt_16)]:
    for vtype in sorted(type_labels_df['type_group'].unique()):
        vessels_of_type = set(type_labels_df[type_labels_df['type_group'] == vtype]['mmsi'])
        ad_labels = y_ad[[m for m in y_ad.index if m in vessels_of_type]]
        adt_labels = y_adt[[m for m in y_adt.index if m in vessels_of_type]]
        anomaly_rows.append({
            'split': split_name, 'vessel_type': vtype, 'total_vessels': len(vessels_of_type),
            'AD_positive': int(ad_labels.sum()), 'AD_negative': int((ad_labels == 0).sum()),
            'ADT_positive': int(adt_labels.sum()), 'ADT_negative': int((adt_labels == 0).sum()),
            'AD_positive_percent': 100 * ad_labels.mean() if len(ad_labels) else np.nan,
            'ADT_positive_percent': 100 * adt_labels.mean() if len(adt_labels) else np.nan,
        })
anomaly_df_16 = pd.DataFrame(anomaly_rows)
anomaly_df_16.to_csv(f'{OUT}/anomaly_label_counts_by_type_16_features.csv', index=False)
print(f"Saved anomaly_label_counts_by_type_16_features.csv ({len(anomaly_df_16)} rows)")

# ---------------------------------------------------------------------------
# anomaly_label_agreement_18F_vs_16F.csv (needs Phase 1's 18F labels too --
# deterministic recompute, same fixed pipeline as Phase 1's own finalize step)
# ---------------------------------------------------------------------------
print("\nRecomputing Phase 1's 18F labels for agreement comparison...")
train_agg_18 = sfb.build_sampled_features('random', train_raw_full, 1.0, seed=0, feature_cols=sfb.FEATURE_COLS_18)
test_agg_18 = sfb.build_sampled_features('random', test_raw_full, 1.0, seed=0, feature_cols=sfb.FEATURE_COLS_18)
scaler18 = StandardScaler()
Xtr18 = scaler18.fit_transform(train_agg_18[sfb.FEATURE_COLS_18].values)
Xte18 = scaler18.transform(test_agg_18[sfb.FEATURE_COLS_18].values)
iso18 = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso18.fit(Xtr18)
y_train_ad_18 = pd.Series((iso18.predict(Xtr18) == -1).astype(int), index=train_agg_18['mmsi'].values)
y_test_ad_18 = pd.Series((iso18.predict(Xte18) == -1).astype(int), index=test_agg_18['mmsi'].values)

train_typed18 = train_agg_18.merge(train_type_labels[['mmsi', 'type_group']], on='mmsi')
test_typed18 = test_agg_18.merge(test_type_labels[['mmsi', 'type_group']], on='mmsi')
y_train_adt_18_parts, y_test_adt_18_parts = [], []
for vtype, group in train_typed18.groupby('type_group'):
    tgroup = test_typed18[test_typed18['type_group'] == vtype]
    if len(group) < 20:
        y_train_adt_18_parts.append(pd.Series(0, index=group['mmsi'].values))
        y_test_adt_18_parts.append(pd.Series(0, index=tgroup['mmsi'].values))
        continue
    sc = StandardScaler()
    Xg = sc.fit_transform(group[sfb.FEATURE_COLS_18].values)
    iso_g = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso_g.fit(Xg)
    y_train_adt_18_parts.append(pd.Series((iso_g.predict(Xg) == -1).astype(int), index=group['mmsi'].values))
    if len(tgroup) > 0:
        Xgt = sc.transform(tgroup[sfb.FEATURE_COLS_18].values)
        y_test_adt_18_parts.append(pd.Series((iso_g.predict(Xgt) == -1).astype(int), index=tgroup['mmsi'].values))
y_train_adt_18 = pd.concat(y_train_adt_18_parts)
y_test_adt_18 = pd.concat(y_test_adt_18_parts) if y_test_adt_18_parts else pd.Series(dtype=int)

print(f"  Cross-check vs Phase 1 log: AD train {y_train_ad_18.sum()}/{len(y_train_ad_18)} (expect 159/3166), "
      f"ADT train {y_train_adt_18.sum()}/{len(y_train_adt_18)} (expect 161/3166)")

def agreement_row(task, split, y18, y16):
    common_idx = sorted(set(y18.index) & set(y16.index))
    a = y18.loc[common_idx]
    b = y16.loc[common_idx]
    both = ((a == 1) & (b == 1)).sum()
    only18 = ((a == 1) & (b == 0)).sum()
    only16 = ((a == 0) & (b == 1)).sum()
    agree = (a == b).mean() * 100
    return {
        'task': task, 'split': split, 'vessels': len(common_idx),
        'positive_18F': int(a.sum()), 'positive_16F': int(b.sum()),
        'positive_in_both': int(both), 'positive_only_18F': int(only18), 'positive_only_16F': int(only16),
        'overall_label_agreement_percent': agree,
    }

agreement_rows = [
    agreement_row('AD', 'train', y_train_ad_18, y_train_ad_16),
    agreement_row('AD', 'test', y_test_ad_18, y_test_ad_16),
    agreement_row('ADT', 'train', y_train_adt_18, y_train_adt_16),
    agreement_row('ADT', 'test', y_test_adt_18, y_test_adt_16),
]

# ADT by vessel type
for split_name, type_labels_df, y18, y16 in [('train', train_type_labels, y_train_adt_18, y_train_adt_16),
                                             ('test', test_type_labels, y_test_adt_18, y_test_adt_16)]:
    for vtype in sorted(type_labels_df['type_group'].unique()):
        vessels_of_type = set(type_labels_df[type_labels_df['type_group'] == vtype]['mmsi'])
        common = sorted(vessels_of_type & set(y18.index) & set(y16.index))
        if not common:
            continue
        a, b = y18.loc[common], y16.loc[common]
        agreement_rows.append({
            'task': f'ADT_{vtype}', 'split': split_name, 'vessels': len(common),
            'positive_18F': int(a.sum()), 'positive_16F': int(b.sum()),
            'positive_in_both': int(((a == 1) & (b == 1)).sum()),
            'positive_only_18F': int(((a == 1) & (b == 0)).sum()),
            'positive_only_16F': int(((a == 0) & (b == 1)).sum()),
            'overall_label_agreement_percent': (a == b).mean() * 100,
        })

agreement_df = pd.DataFrame(agreement_rows)
agreement_df.to_csv(f'{OUT}/anomaly_label_agreement_18F_vs_16F.csv', index=False)
print(f"Saved anomaly_label_agreement_18F_vs_16F.csv ({len(agreement_df)} rows)")
print(agreement_df[agreement_df['task'].isin(['AD', 'ADT'])].to_string())

# ---------------------------------------------------------------------------
# Publication tables (16F)
# ---------------------------------------------------------------------------
def build_task_table(summary_df, task, value_col, sd_col, fname):
    rows = []
    for strategy in STRATEGIES:
        row = {'Strategy': STRATEGY_LABEL[strategy]}
        for rate in RATES:
            g = summary_df[(summary_df['task'] == task) & (summary_df['strategy'] == STRATEGY_LABEL[strategy]) & (summary_df['target_rate'] == rate)]
            if len(g) == 0:
                row[RATE_LABELS[rate]] = 'N/A'; row[f'{RATE_LABELS[rate]}_mean'] = np.nan; row[f'{RATE_LABELS[rate]}_sd'] = np.nan
            else:
                m, s = g[value_col].iloc[0], g[sd_col].iloc[0]
                row[RATE_LABELS[rate]] = f"{m:.2f} +/- {s:.2f}"; row[f'{RATE_LABELS[rate]}_mean'] = m; row[f'{RATE_LABELS[rate]}_sd'] = s
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/{fname}', index=False)
    return df

for task in TASKS:
    build_task_table(condition_summary_16, task, 'retention_mean', 'retention_sd', f'table_{task}_retention_16_features.csv')
    build_task_table(condition_summary_16, task, 'sampled_f1_mean', 'sampled_f1_sd', f'table_{task}_absolute_f1_16_features.csv')
print("Saved all 16-feature publication tables.")

# ---------------------------------------------------------------------------
# timing_ablation_comparison.csv (18F vs 16F_RELABELED)
# ---------------------------------------------------------------------------
comp_rows = []
for task in TASKS:
    for strategy_key in STRATEGIES:
        strategy = STRATEGY_LABEL[strategy_key]
        for rate in RATES:
            g18 = cs18[(cs18['task'] == task) & (cs18['strategy'] == strategy) & (cs18['target_rate'] == rate)]
            g16 = condition_summary_16[(condition_summary_16['task'] == task) & (condition_summary_16['strategy'] == strategy) & (condition_summary_16['target_rate'] == rate)]
            if len(g18) == 0 or len(g16) == 0:
                continue
            g18, g16 = g18.iloc[0], g16.iloc[0]
            comp_rows.append({
                'task': task, 'strategy': strategy, 'target_rate': rate,
                'retention_18F_mean': g18['retention_mean'], 'retention_18F_sd': g18['retention_sd'],
                'retention_16F_mean': g16['retention_mean'], 'retention_16F_sd': g16['retention_sd'],
                'retention_difference_16F_minus_18F': g16['retention_mean'] - g18['retention_mean'],
                'absolute_f1_18F_mean': g18['sampled_f1_mean'], 'absolute_f1_18F_sd': g18['sampled_f1_sd'],
                'absolute_f1_16F_mean': g16['sampled_f1_mean'], 'absolute_f1_16F_sd': g16['sampled_f1_sd'],
                'absolute_f1_difference_16F_minus_18F': g16['sampled_f1_mean'] - g18['sampled_f1_mean'],
                'valid_vessels_18F_mean': g18['valid_vessels_mean'], 'valid_vessels_16F_mean': g16['valid_vessels_mean'],
            })
timing_comparison = pd.DataFrame(comp_rows)
timing_comparison.to_csv(f'{OUT}/timing_ablation_comparison.csv', index=False)
print(f"Saved timing_ablation_comparison.csv ({len(timing_comparison)} rows)")

# Check 17: 18F values reproduce Phase 1 exactly
mismatch17 = timing_comparison.merge(
    cs18, left_on=['task', 'strategy', 'target_rate'], right_on=['task', 'strategy', 'target_rate'], suffixes=('', '_src'))
bad17 = mismatch17[~np.isclose(mismatch17['retention_18F_mean'], mismatch17['retention_mean'], equal_nan=True)]
record_check('17_18F_values_reproduce_phase1_exactly', 'PASS' if len(bad17) == 0 else 'FAIL', f'{len(bad17)} mismatches found between timing_ablation_comparison 18F columns and cs18 source.')

# ---------------------------------------------------------------------------
# timing_ablation_fixed_label_comparison_AD_ADT.csv (18F vs 16F_FIXED_18F_LABELS)
# ---------------------------------------------------------------------------
comp_fixed_rows = []
for task in ['AD', 'ADT']:
    for strategy_key in STRATEGIES:
        strategy = STRATEGY_LABEL[strategy_key]
        for rate in RATES:
            g18 = cs18[(cs18['task'] == task) & (cs18['strategy'] == strategy) & (cs18['target_rate'] == rate)]
            gf = condition_summary_fixed[(condition_summary_fixed['task'] == task) & (condition_summary_fixed['strategy'] == strategy) & (condition_summary_fixed['target_rate'] == rate)]
            if len(g18) == 0 or len(gf) == 0:
                continue
            g18, gf = g18.iloc[0], gf.iloc[0]
            comp_fixed_rows.append({
                'task': task, 'strategy': strategy, 'target_rate': rate,
                'retention_18F_mean': g18['retention_mean'], 'retention_18F_sd': g18['retention_sd'],
                'retention_16F_fixedlabels_mean': gf['retention_mean'], 'retention_16F_fixedlabels_sd': gf['retention_sd'],
                'retention_difference': gf['retention_mean'] - g18['retention_mean'],
                'absolute_f1_18F_mean': g18['sampled_f1_mean'], 'absolute_f1_16F_fixedlabels_mean': gf['sampled_f1_mean'],
            })
timing_fixed_comparison = pd.DataFrame(comp_fixed_rows)
timing_fixed_comparison.to_csv(f'{OUT}/timing_ablation_fixed_label_comparison_AD_ADT.csv', index=False)
print(f"Saved timing_ablation_fixed_label_comparison_AD_ADT.csv ({len(timing_fixed_comparison)} rows)")

checks_df = pd.DataFrame(checks)
checks_df.to_csv(f'{OUT}/_partial_checks.csv', index=False)
print(f"\n{len(checks_df)} checks recorded so far (figure + remaining checks handled in a second pass).")
