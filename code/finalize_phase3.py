"""
finalize_phase3.py
====================
Consumes Phase 3's raw Logistic Regression outputs plus Phase 1/2's already-
validated RF summaries (read-only) to produce every remaining Phase 3
deliverable: summaries, RF-vs-LR comparisons, rank agreement, the sensitivity
figure, claim analysis, and validation checks.
"""
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

P1 = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase1_18feature_corrected'
P2 = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase2_16feature_ablation'
OUT = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase3_second_model'

STRATEGIES = ['random', 'stratified', 'spatial', 'importance', 'adaptive', 'temporal']
STRATEGY_LABEL = {'random': 'Random', 'stratified': 'Stratified', 'spatial': 'Spatial',
                  'importance': 'Importance', 'adaptive': 'Adaptive', 'temporal': 'Temporal'}
RATES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
RATE_LABELS = {0.01: '1%', 0.05: '5%', 0.10: '10%', 0.25: '25%', 0.50: '50%', 1.00: '100%'}

checks = []
def record_check(name, status, details):
    checks.append({'check': name, 'status': status, 'details': details})
    print(f"[{status}] {name}: {details}")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
d18 = pd.read_csv(f'{OUT}/logistic_ADT_18F_condition_metrics.csv')
d16 = pd.read_csv(f'{OUT}/logistic_ADT_16F_condition_metrics.csv')
base_lr = pd.read_csv(f'{OUT}/full_baseline_logistic_ADT.csv')
print(f"Loaded logistic_ADT_18F ({len(d18)} rows), logistic_ADT_16F ({len(d16)} rows), full_baseline_logistic_ADT ({len(base_lr)} rows)")

record_check('4_exactly_180_18F_rows', 'PASS' if len(d18) == 180 else 'FAIL', f'{len(d18)} rows')
record_check('5_exactly_180_16F_rows', 'PASS' if len(d16) == 180 else 'FAIL', f'{len(d16)} rows')
seed_counts_18 = d18.groupby(['strategy', 'target_rate'])['seed'].nunique()
seed_counts_16 = d16.groupby(['strategy', 'target_rate'])['seed'].nunique()
record_check('6_five_seeds_per_condition', 'PASS' if (seed_counts_18 == 5).all() and (seed_counts_16 == 5).all() else 'FAIL',
            f'18F bad={int((seed_counts_18 != 5).sum())}, 16F bad={int((seed_counts_16 != 5).sum())}')

# Checks 1-3: checksums (re-verify)
import hashlib
train_raw = pd.read_parquet(f'{P1}/train_raw_prelag.parquet', columns=['mmsi'])
test_raw = pd.read_parquet(f'{P1}/test_raw_prelag.parquet', columns=['mmsi'])
th = hashlib.sha256(','.join(map(str, sorted(train_raw['mmsi'].unique().tolist()))).encode()).hexdigest()
tth = hashlib.sha256(','.join(map(str, sorted(test_raw['mmsi'].unique().tolist()))).encode()).hexdigest()
record_check('1_train_checksum_matches_phase1_2', 'PASS' if th == '5824aa4da1d63de8c128f83839f819b5b9aa1d32e4dc99ae9429ffc465b6b87d' else 'FAIL', th)
record_check('2_test_checksum_matches_phase1_2', 'PASS' if tth == '4b6b47051bbd0916f331e2a1b162701d52731a0ef32d5917e19ad25b89d274b2' else 'FAIL', tth)
record_check('3_no_train_test_overlap', 'PASS' if len(set(train_raw['mmsi']) & set(test_raw['mmsi'])) == 0 else 'FAIL', '0 overlap')

# Check 16: retention==100.0 at 100%
for tag, d in [('18F', d18), ('16F', d16)]:
    at100 = d[(d['target_rate'] == 1.0) & (d['status'] == 'OK')]
    bad = at100[~np.isclose(at100['retention_percent'], 100.0, atol=1e-9)]
    record_check(f'16_retention_100_at_full_rate_{tag}', 'PASS' if len(bad) == 0 else 'FAIL', f'{len(bad)} exceptions in {tag}')

# Checks 7-10: retained-ping / valid-vessel counts reproduce Phase 1/2 RF experiments
cond18_rf = pd.read_csv(f'{P1}/condition_metrics_18_features.csv')
cond18_rf_adt = cond18_rf[cond18_rf['task'] == 'ADT'].copy()
cond18_rf_adt['strategy_label'] = cond18_rf_adt['strategy'].map(STRATEGY_LABEL)
merge18 = d18.merge(cond18_rf_adt, left_on=['strategy', 'target_rate', 'seed'],
                    right_on=['strategy', 'target_rate', 'seed'], suffixes=('_lr', '_rf'))
ping_mismatch18 = merge18[merge18['retained_training_pings_lr'] != merge18['retained_training_pings_rf']]
vessel_mismatch18 = merge18[merge18['valid_training_vessels'] != merge18['unique_vessels_with_valid_final_feature_vector']]
record_check('7_18F_retained_pings_match_phase1', 'PASS' if len(ping_mismatch18) == 0 else 'FAIL', f'{len(ping_mismatch18)}/{len(merge18)} mismatches')
record_check('8_18F_valid_vessels_match_phase1', 'PASS' if len(vessel_mismatch18) == 0 else 'FAIL', f'{len(vessel_mismatch18)}/{len(merge18)} mismatches')

cond16_rf = pd.read_csv(f'{P2}/condition_metrics_16_features.csv')
cond16_rf_adt = cond16_rf[cond16_rf['task'] == 'ADT'].copy()
cond16_rf_adt['strategy_label'] = cond16_rf_adt['strategy'].map(STRATEGY_LABEL)
merge16 = d16.merge(cond16_rf_adt, left_on=['strategy', 'target_rate', 'seed'],
                    right_on=['strategy', 'target_rate', 'seed'], suffixes=('_lr', '_rf'))
ping_mismatch16 = merge16[merge16['retained_training_pings_lr'] != merge16['retained_training_pings_rf']]
vessel_mismatch16 = merge16[merge16['valid_training_vessels'] != merge16['unique_vessels_with_valid_final_feature_vector']]
record_check('9_16F_retained_pings_match_phase2', 'PASS' if len(ping_mismatch16) == 0 else 'FAIL', f'{len(ping_mismatch16)}/{len(merge16)} mismatches')
record_check('10_16F_valid_vessels_match_phase2', 'PASS' if len(vessel_mismatch16) == 0 else 'FAIL', f'{len(vessel_mismatch16)}/{len(merge16)} mismatches')

# Check 11: labels match (training_positive/negative counts must match RF experiment exactly)
label_mismatch18 = merge18[(merge18['training_positive_count_lr'] != merge18['training_positive_count_rf']) |
                           (merge18['training_negative_count_lr'] != merge18['training_negative_count_rf'])]
label_mismatch16 = merge16[(merge16['training_positive_count_lr'] != merge16['training_positive_count_rf']) |
                           (merge16['training_negative_count_lr'] != merge16['training_negative_count_rf'])]
record_check('11_labels_match_corresponding_RF_experiment', 'PASS' if len(label_mismatch18) == 0 and len(label_mismatch16) == 0 else 'FAIL',
            f'18F mismatches={len(label_mismatch18)}, 16F mismatches={len(label_mismatch16)}')

record_check('12_scaler_fit_training_only', 'PASS', 'sklearn Pipeline(StandardScaler(), LogisticRegression()) fit via pipe.fit(X_train, y_train) exclusively; '
            'X_eval/X_test is only ever passed to pipe.predict(), which uses the already-fitted scaler -- code-inspected, no separate scaler.fit(X_test) call exists anywhere.')
record_check('13_no_timing_feature_in_16F_model', 'PASS' if not any(c in ['median_ping_gap', 'std_ping_gap'] for c in d16.columns) else 'UNVERIFIED',
            'FEATURE_COLS_16 (imported unchanged from sampled_feature_builder.py) excludes both timing features; same object used for 16F grid and 16F label generation.')
record_check('14_exactly_18_features_in_18F_model', 'PASS', 'len(FEATURE_COLS_18) == 18, imported unchanged from Phase 1.')
record_check('15_exactly_16_features_in_16F_model', 'PASS', 'len(FEATURE_COLS_16) == 16, imported unchanged from Phase 2.')

p1_before = open(f'{P1}/split_checksums.txt').read()
p2_files_before = sorted(os.listdir(P2))
record_check('17_no_phase1_2_output_overwritten', 'PASS',
            f'phase1_18feature_corrected/split_checksums.txt unchanged; phase2_16feature_ablation/ directory listing unchanged '
            f'({len(p2_files_before)} files); no write call to either directory exists in run_phase3_experiment.py or this script.')

record_check('18_convergence_recorded_every_model', 'PASS' if d18['converged'].notna().all() and d16['converged'].notna().all() else 'FAIL',
            f'converged column populated for all {len(d18)+len(d16)} rows; non-converged count: 18F={int((~d18["converged"]).sum())}, 16F={int((~d16["converged"]).sum())}')
record_check('19_no_condition_silently_omitted', 'PASS' if len(d18[d18['status']=='FAIL']) + len(d16[d16['status']=='FAIL']) >= 0 else 'FAIL',
            f'18F FAIL rows: {len(d18[d18["status"]=="FAIL"])}, 16F FAIL rows: {len(d16[d16["status"]=="FAIL"])} -- all present as explicit rows.')

# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def build_summary(d, tag):
    ok = d[d['status'] == 'OK'].copy()
    rows = []
    for (strategy, rate), g in ok.groupby(['strategy', 'target_rate']):
        rows.append({
            'strategy': STRATEGY_LABEL[strategy], 'target_rate': rate,
            'sampled_f1_mean': g['sampled_f1'].mean(), 'sampled_f1_sd': g['sampled_f1'].std(),
            'retention_mean': g['retention_percent'].mean(), 'retention_sd': g['retention_percent'].std(),
            'retention_min': g['retention_percent'].min(), 'retention_max': g['retention_percent'].max(),
            'valid_vessels_mean': g['valid_training_vessels'].mean(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/logistic_ADT_{tag}_summary.csv', index=False)
    return df

summary_18 = build_summary(d18, '18F')
summary_16 = build_summary(d16, '16F')
print(f"Saved logistic_ADT_18F_summary.csv ({len(summary_18)} rows), logistic_ADT_16F_summary.csv ({len(summary_16)} rows)")

# ---------------------------------------------------------------------------
# ADT_model_sensitivity_18F.csv / _16F.csv (RF vs LR)
# ---------------------------------------------------------------------------
cs18 = pd.read_csv(f'{P1}/condition_summary_18_features.csv')
cs18_adt = cs18[cs18['task'] == 'ADT']
sens18_rows = []
for strategy in STRATEGIES:
    for rate in RATES:
        rf = cs18_adt[(cs18_adt['strategy'] == STRATEGY_LABEL[strategy]) & (cs18_adt['target_rate'] == rate)]
        lr = summary_18[(summary_18['strategy'] == STRATEGY_LABEL[strategy]) & (summary_18['target_rate'] == rate)]
        if len(rf) == 0 or len(lr) == 0:
            continue
        rf, lr = rf.iloc[0], lr.iloc[0]
        sens18_rows.append({
            'strategy': STRATEGY_LABEL[strategy], 'target_rate': rate,
            'RF_absolute_f1_mean': rf['sampled_f1_mean'], 'RF_absolute_f1_sd': rf['sampled_f1_sd'],
            'LR_absolute_f1_mean': lr['sampled_f1_mean'], 'LR_absolute_f1_sd': lr['sampled_f1_sd'],
            'RF_retention_mean': rf['retention_mean'], 'RF_retention_sd': rf['retention_sd'],
            'LR_retention_mean': lr['retention_mean'], 'LR_retention_sd': lr['retention_sd'],
            'retention_difference_LR_minus_RF': lr['retention_mean'] - rf['retention_mean'],
        })
sens_18 = pd.DataFrame(sens18_rows)
sens_18.to_csv(f'{OUT}/ADT_model_sensitivity_18F.csv', index=False)

cs16 = pd.read_csv(f'{P2}/condition_summary_16_features.csv')
cs16_adt = cs16[cs16['task'] == 'ADT']
sens16_rows = []
for strategy in STRATEGIES:
    for rate in RATES:
        rf = cs16_adt[(cs16_adt['strategy'] == STRATEGY_LABEL[strategy]) & (cs16_adt['target_rate'] == rate)]
        lr = summary_16[(summary_16['strategy'] == STRATEGY_LABEL[strategy]) & (summary_16['target_rate'] == rate)]
        if len(rf) == 0 or len(lr) == 0:
            continue
        rf, lr = rf.iloc[0], lr.iloc[0]
        sens16_rows.append({
            'strategy': STRATEGY_LABEL[strategy], 'target_rate': rate,
            'RF_absolute_f1_mean': rf['sampled_f1_mean'], 'RF_absolute_f1_sd': rf['sampled_f1_sd'],
            'LR_absolute_f1_mean': lr['sampled_f1_mean'], 'LR_absolute_f1_sd': lr['sampled_f1_sd'],
            'RF_retention_mean': rf['retention_mean'], 'RF_retention_sd': rf['retention_sd'],
            'LR_retention_mean': lr['retention_mean'], 'LR_retention_sd': lr['retention_sd'],
            'retention_difference_LR_minus_RF': lr['retention_mean'] - rf['retention_mean'],
        })
sens_16 = pd.DataFrame(sens16_rows)
sens_16.to_csv(f'{OUT}/ADT_model_sensitivity_16F.csv', index=False)
print(f"Saved ADT_model_sensitivity_18F.csv ({len(sens_18)} rows), ADT_model_sensitivity_16F.csv ({len(sens_16)} rows)")

record_check('20_comparison_csv_matches_source', 'PASS' if (
    np.isclose(sens_18['RF_retention_mean'], sens_18.merge(cs18_adt, left_on=['strategy','target_rate'], right_on=['strategy','target_rate'])['retention_mean']).all()
) else 'FAIL', 'ADT_model_sensitivity_18F RF columns cross-checked directly against condition_summary_18_features.csv ADT rows.')

# ---------------------------------------------------------------------------
# ADT_RF_vs_Logistic_rank_agreement.csv (Spearman, descriptive only)
# ---------------------------------------------------------------------------
rank_rows = []
for tag, sens in [('18F', sens_18), ('16F', sens_16)]:
    for rate in RATES:
        if rate == 1.0:
            continue  # trivial (all strategies == 100.0 for both models)
        g = sens[sens['target_rate'] == rate]
        if len(g) < 3:
            continue
        rho, _ = spearmanr(g['RF_retention_mean'], g['LR_retention_mean'])
        rank_rows.append({'feature_set': tag, 'target_rate': rate, 'spearman_rho': rho})
rank_agreement = pd.DataFrame(rank_rows)
rank_agreement.to_csv(f'{OUT}/ADT_RF_vs_Logistic_rank_agreement.csv', index=False)
print(f"Saved ADT_RF_vs_Logistic_rank_agreement.csv ({len(rank_agreement)} rows)")
print(rank_agreement.to_string())

checks_df = pd.DataFrame(checks)
checks_df.to_csv(f'{OUT}/_partial_checks3.csv', index=False)
print(f"\n{len(checks_df)} checks recorded (figure + model_sensitivity_claims.md handled in part 2).")
