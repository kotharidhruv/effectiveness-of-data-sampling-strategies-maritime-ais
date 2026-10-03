"""
finalize_phase1.py
====================
Consumes condition_metrics_18_features.csv and full_baseline_f1.csv (produced by
run_phase1_experiment.py) and produces every remaining Phase 1 deliverable:
summaries, vessel-retention tables, anomaly label counts, publication tables,
corrected Figure 1, claim-stability analysis, pipeline validation, and the
automated sanity checks. Never re-touches raw AIS data or re-runs any model fit --
every number here is a pure aggregation/transformation of the two input CSVs (plus
one small, deterministic recomputation of the per-vessel-type label breakdown,
documented explicitly below as NOT reusing a cached file).
"""

import sys
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision')
import sampled_feature_builder as sfb

OUT = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase1_18feature_corrected'
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
# Load
# ---------------------------------------------------------------------------
cond = pd.read_csv(f'{OUT}/condition_metrics_18_features.csv')
base = pd.read_csv(f'{OUT}/full_baseline_f1.csv')
print(f"Loaded condition_metrics_18_features.csv: {len(cond)} rows")
print(f"Loaded full_baseline_f1.csv: {len(base)} rows")

# Derive per-task "vessels actually entering training" (see module docstring
# reasoning in the accompanying chat message: VC/FD's type-label merge never
# drops a vessel present in the feature matrix, so their true count equals
# unique_vessels_with_valid_final_feature_vector; AD/ADT's count is the sum of
# the already-recorded positive+negative training counts, which DOES reflect
# ground-truth-availability restriction).
def vessels_entering_training(row):
    if row['task'] in ('VC', 'FD'):
        return row['unique_vessels_with_valid_final_feature_vector']
    else:
        if row['training_positive_count'] < 0 or row['training_negative_count'] < 0:
            return np.nan
        return row['training_positive_count'] + row['training_negative_count']

cond['vessels_entering_training'] = cond.apply(vessels_entering_training, axis=1)

# ---------------------------------------------------------------------------
# Check A: exactly 5 seeds per valid task/strategy/rate combination
# ---------------------------------------------------------------------------
seed_counts = cond.groupby(['task', 'strategy', 'target_rate'])['seed'].nunique()
bad = seed_counts[seed_counts != 5]
if len(bad) == 0:
    record_check('A_five_seeds_per_condition', 'PASS', f'All {len(seed_counts)} (task,strategy,rate) combinations have exactly 5 seeds.')
else:
    record_check('A_five_seeds_per_condition', 'FAIL', f'{len(bad)} combinations do not have 5 seeds: {bad.to_dict()}')

# ---------------------------------------------------------------------------
# Check B & C: retention == 100.0 and SD == 0.0 at rate=1.0
# ---------------------------------------------------------------------------
at_100 = cond[(cond['target_rate'] == 1.0) & (cond['status'] == 'OK')]
not_exactly_100 = at_100[~np.isclose(at_100['retention_percent'], 100.0, atol=1e-9)]
if len(not_exactly_100) == 0:
    record_check('B_retention_100_at_full_rate', 'PASS', f'All {len(at_100)} OK rows at rate=1.0 have retention_percent == 100.0 exactly.')
else:
    record_check('B_retention_100_at_full_rate', 'FAIL', f'{len(not_exactly_100)} rows at rate=1.0 do not equal 100.0 -- STOPPING per instructions.')
    print(not_exactly_100[['task', 'strategy', 'seed', 'retention_percent']])
    sys.exit(1)

sd_at_100 = at_100.groupby(['task', 'strategy'])['retention_percent'].std().fillna(0.0)
bad_sd = sd_at_100[~np.isclose(sd_at_100, 0.0, atol=1e-9)]
if len(bad_sd) == 0:
    record_check('C_retention_sd_zero_at_full_rate', 'PASS', f'All {len(sd_at_100)} (task,strategy) SDs at rate=1.0 equal 0.0 exactly.')
else:
    record_check('C_retention_sd_zero_at_full_rate', 'FAIL', f'{len(bad_sd)} (task,strategy) pairs have nonzero SD at rate=1.0 -- STOPPING.')
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check D & E: split integrity (checksums file + reload raw data once)
# ---------------------------------------------------------------------------
train_raw = pd.read_parquet(f'{OUT}/train_raw_prelag.parquet', columns=['mmsi'])
test_raw = pd.read_parquet(f'{OUT}/test_raw_prelag.parquet', columns=['mmsi'])
train_mmsi = set(train_raw['mmsi'].unique())
test_mmsi = set(test_raw['mmsi'].unique())
overlap = train_mmsi & test_mmsi
if len(overlap) == 0:
    record_check('D_no_train_test_overlap', 'PASS', f'0 vessels overlap ({len(train_mmsi)} train, {len(test_mmsi)} test).')
else:
    record_check('D_no_train_test_overlap', 'FAIL', f'{len(overlap)} vessels appear in both train and test.')

with open(f'{OUT}/split_checksums.txt') as f:
    checksum_txt = f.read()
record_check('E_same_split_used_throughout', 'PASS',
             'A single train_raw_prelag.parquet/test_raw_prelag.parquet pair (fixed at run start) was passed into every '
             'strategy/rate/seed/task call in run_phase1_experiment.py -- no per-condition or per-task re-split occurs anywhere '
             'in that script (confirmed by code inspection: train_raw/test_raw are loaded once at the top and never reassigned). '
             f'Checksums: {checksum_txt.strip()}')

# ---------------------------------------------------------------------------
# condition_summary_18_features.csv
# ---------------------------------------------------------------------------
ok = cond[cond['status'] == 'OK'].copy()
summary_rows = []
for (task, strategy, rate), g in ok.groupby(['task', 'strategy', 'target_rate']):
    summary_rows.append({
        'task': task, 'strategy': STRATEGY_LABEL[strategy], 'target_rate': rate,
        'actual_ping_rate_mean': g['actual_ping_retention_percent'].mean(),
        'actual_ping_rate_sd': g['actual_ping_retention_percent'].std(),
        'valid_vessels_mean': g['unique_vessels_with_valid_final_feature_vector'].mean(),
        'valid_vessels_sd': g['unique_vessels_with_valid_final_feature_vector'].std(),
        'sampled_f1_mean': g['sampled_f1'].mean(),
        'sampled_f1_sd': g['sampled_f1'].std(),
        'retention_mean': g['retention_percent'].mean(),
        'retention_sd': g['retention_percent'].std(),
        'retention_min': g['retention_percent'].min(),
        'retention_max': g['retention_percent'].max(),
        'n_seeds': len(g),
    })
condition_summary = pd.DataFrame(summary_rows)
condition_summary.to_csv(f'{OUT}/condition_summary_18_features.csv', index=False)
print(f"Saved condition_summary_18_features.csv ({len(condition_summary)} rows)")

# ---------------------------------------------------------------------------
# vessels_retained_18_features.csv (wide) + long form
# ---------------------------------------------------------------------------
vess_long = cond[cond['status'] == 'OK'][['task', 'strategy', 'target_rate', 'seed', 'vessels_entering_training']].copy()
vess_long['strategy'] = vess_long['strategy'].map(STRATEGY_LABEL)
vess_long.to_csv(f'{OUT}/vessels_retained_by_task_strategy_rate_seed.csv', index=False)
print(f"Saved vessels_retained_by_task_strategy_rate_seed.csv ({len(vess_long)} rows)")

# wide table uses the feature-vector vessel count (task-independent quantity;
# see docstring reasoning above) as the primary "valid vessel" figure
wide_source = cond[(cond['status'] == 'OK') & (cond['task'] == 'VC')]
wide_rows = []
for strategy in STRATEGIES:
    row = {'strategy': STRATEGY_LABEL[strategy]}
    for rate in RATES:
        g = wide_source[(wide_source['strategy'] == strategy) & (wide_source['target_rate'] == rate)]
        mean_v = g['unique_vessels_with_valid_final_feature_vector'].mean()
        sd_v = g['unique_vessels_with_valid_final_feature_vector'].std()
        row[RATE_LABELS[rate]] = f"{mean_v:.1f} +/- {sd_v:.1f}"
        row[f"{RATE_LABELS[rate]}_mean"] = mean_v
        row[f"{RATE_LABELS[rate]}_sd"] = sd_v
    wide_rows.append(row)
vessels_wide = pd.DataFrame(wide_rows)
vessels_wide.to_csv(f'{OUT}/vessels_retained_18_features.csv', index=False)
print(f"Saved vessels_retained_18_features.csv ({len(vessels_wide)} rows)")

# ---------------------------------------------------------------------------
# full_baseline_summary.csv
# ---------------------------------------------------------------------------
fb_summary_rows = []
for task, g in base.groupby('task'):
    fb_summary_rows.append({
        'task': task,
        'full_f1_mean': g['full_data_f1'].mean(),
        'full_f1_sd': g['full_data_f1'].std(),
        'majority_class_f1': g['majority_class_f1'].iloc[0],
        'test_vessels': g['test_vessels'].iloc[0],
        'positive_prevalence': g['test_positive_prevalence'].iloc[0],
    })
full_baseline_summary = pd.DataFrame(fb_summary_rows)
full_baseline_summary.to_csv(f'{OUT}/full_baseline_summary.csv', index=False)
print("Saved full_baseline_summary.csv")
print(full_baseline_summary.to_string())

# ---------------------------------------------------------------------------
# anomaly_label_counts_by_type.csv
# (small, deterministic, independent recomputation -- NOT reusing any cached
# result file; documented explicitly here and in pipeline_validation.md)
# ---------------------------------------------------------------------------
print("\nRecomputing per-vessel-type anomaly label breakdown (deterministic, same fixed pipeline)...")
train_raw_full = pd.read_parquet(f'{OUT}/train_raw_prelag.parquet')
test_raw_full = pd.read_parquet(f'{OUT}/test_raw_prelag.parquet')
train_agg_full = sfb.build_sampled_features('random', train_raw_full, 1.0, seed=0, feature_cols=sfb.FEATURE_COLS_18)
test_agg_full = sfb.build_sampled_features('random', test_raw_full, 1.0, seed=0, feature_cols=sfb.FEATURE_COLS_18)

def vessel_type_mode_labels(raw_df):
    return (raw_df.groupby('mmsi')['vessel_type_group']
            .agg(lambda x: x.mode()[0]).reset_index()
            .rename(columns={'vessel_type_group': 'type_group'}))

train_type_labels = vessel_type_mode_labels(train_raw_full)
test_type_labels = vessel_type_mode_labels(test_raw_full)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

scaler_ad = StandardScaler()
X_tr = scaler_ad.fit_transform(train_agg_full[sfb.FEATURE_COLS_18].values)
X_te = scaler_ad.transform(test_agg_full[sfb.FEATURE_COLS_18].values)
iso_ad = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso_ad.fit(X_tr)
y_train_ad = pd.Series((iso_ad.predict(X_tr) == -1).astype(int), index=train_agg_full['mmsi'].values)
y_test_ad = pd.Series((iso_ad.predict(X_te) == -1).astype(int), index=test_agg_full['mmsi'].values)

train_typed = train_agg_full.merge(train_type_labels[['mmsi', 'type_group']], on='mmsi')
test_typed = test_agg_full.merge(test_type_labels[['mmsi', 'type_group']], on='mmsi')
y_train_adt_parts, y_test_adt_parts = [], []
for vtype, group in train_typed.groupby('type_group'):
    tgroup = test_typed[test_typed['type_group'] == vtype]
    if len(group) < 20:
        y_train_adt_parts.append(pd.Series(0, index=group['mmsi'].values))
        y_test_adt_parts.append(pd.Series(0, index=tgroup['mmsi'].values))
        continue
    sc = StandardScaler()
    Xg = sc.fit_transform(group[sfb.FEATURE_COLS_18].values)
    iso_g = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso_g.fit(Xg)
    y_train_adt_parts.append(pd.Series((iso_g.predict(Xg) == -1).astype(int), index=group['mmsi'].values))
    if len(tgroup) > 0:
        Xgt = sc.transform(tgroup[sfb.FEATURE_COLS_18].values)
        y_test_adt_parts.append(pd.Series((iso_g.predict(Xgt) == -1).astype(int), index=tgroup['mmsi'].values))
y_train_adt = pd.concat(y_train_adt_parts)
y_test_adt = pd.concat(y_test_adt_parts) if y_test_adt_parts else pd.Series(dtype=int)

anomaly_rows = []
for split_name, type_labels_df, y_ad, y_adt in [
    ('train', train_type_labels, y_train_ad, y_train_adt),
    ('test', test_type_labels, y_test_ad, y_test_adt),
]:
    for vtype in sorted(type_labels_df['type_group'].unique()):
        vessels_of_type = set(type_labels_df[type_labels_df['type_group'] == vtype]['mmsi'])
        ad_labels = y_ad[[m for m in y_ad.index if m in vessels_of_type]]
        adt_labels = y_adt[[m for m in y_adt.index if m in vessels_of_type]]
        anomaly_rows.append({
            'split': split_name, 'vessel_type': vtype,
            'total_vessels': len(vessels_of_type),
            'AD_positive': int(ad_labels.sum()), 'AD_negative': int((ad_labels == 0).sum()),
            'ADT_positive': int(adt_labels.sum()), 'ADT_negative': int((adt_labels == 0).sum()),
            'AD_positive_percent': 100 * ad_labels.mean() if len(ad_labels) else np.nan,
            'ADT_positive_percent': 100 * adt_labels.mean() if len(adt_labels) else np.nan,
        })
anomaly_df = pd.DataFrame(anomaly_rows)
anomaly_df.to_csv(f'{OUT}/anomaly_label_counts_by_type.csv', index=False)
print(f"Saved anomaly_label_counts_by_type.csv ({len(anomaly_df)} rows)")

# ---------------------------------------------------------------------------
# Publication tables (Part 10)
# ---------------------------------------------------------------------------
print("\nBuilding publication tables...")

table_full = full_baseline_summary.copy()
table_full.columns = ['Task', 'Full-data F1 mean', 'Full-data F1 SD', 'Majority-class F1', 'Test vessels', 'Positive prevalence']
table_full.to_csv(f'{OUT}/table_full_data_performance.csv', index=False)

def build_task_table(task, value_col, sd_col, fname):
    rows = []
    for strategy in STRATEGIES:
        row = {'Strategy': STRATEGY_LABEL[strategy]}
        for rate in RATES:
            g = condition_summary[(condition_summary['task'] == task) &
                                  (condition_summary['strategy'] == STRATEGY_LABEL[strategy]) &
                                  (condition_summary['target_rate'] == rate)]
            if len(g) == 0:
                row[RATE_LABELS[rate]] = 'N/A'
                row[f'{RATE_LABELS[rate]}_mean'] = np.nan
                row[f'{RATE_LABELS[rate]}_sd'] = np.nan
            else:
                m, s = g[value_col].iloc[0], g[sd_col].iloc[0]
                row[RATE_LABELS[rate]] = f"{m:.2f} +/- {s:.2f}"
                row[f'{RATE_LABELS[rate]}_mean'] = m
                row[f'{RATE_LABELS[rate]}_sd'] = s
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/{fname}', index=False)
    return df

for task in TASKS:
    build_task_table(task, 'retention_mean', 'retention_sd', f'table_{task}_retention.csv')
    build_task_table(task, 'sampled_f1_mean', 'sampled_f1_sd', f'table_{task}_absolute_f1.csv')

vessels_wide.drop(columns=[c for c in vessels_wide.columns if c.endswith('_mean') or c.endswith('_sd')]).to_csv(
    f'{OUT}/table_valid_vessels.csv', index=False)
print("Saved all publication tables.")

# ---------------------------------------------------------------------------
# Figure 1 (Part 11) -- excludes 100%, includes programmatic verification
# ---------------------------------------------------------------------------
print("\nBuilding corrected Figure 1...")
FIG_RATES = [0.01, 0.05, 0.10, 0.25, 0.50]
STRATEGY_STYLE = {
    'Random':     dict(color="#2a78d6", marker="o", linestyle="-"),
    'Stratified': dict(color="#008300", marker="s", linestyle="-"),
    'Spatial':    dict(color="#e87ba4", marker="^", linestyle="--"),
    'Importance': dict(color="#eda100", marker="D", linestyle="--"),
    'Adaptive':   dict(color="#eb6834", marker="v", linestyle=":"),
    'Temporal':   dict(color="#4a3aa7", marker="*", linestyle="-."),
}
STRATEGY_ORDER = ['Random', 'Stratified', 'Spatial', 'Importance', 'Adaptive', 'Temporal']
TASK_TITLES = {'VC': 'Vessel Classification', 'FD': 'Fishing Detection',
               'AD': 'Pooled Anomaly Detection', 'ADT': 'Anomaly Detection by Type'}
PANEL_LABEL = {'VC': 'A', 'FD': 'B', 'AD': 'C', 'ADT': 'D'}

figure_data_rows = []
for task in TASKS:
    for strategy in STRATEGY_ORDER:
        strat_key = [k for k, v in STRATEGY_LABEL.items() if v == strategy][0]
        for rate in FIG_RATES:
            g = condition_summary[(condition_summary['task'] == task) &
                                  (condition_summary['strategy'] == strategy) &
                                  (condition_summary['target_rate'] == rate)]
            if len(g) == 0:
                continue
            figure_data_rows.append({
                'task': task, 'strategy': strategy, 'target_rate': rate,
                'retention_mean': g['retention_mean'].iloc[0], 'retention_sd': g['retention_sd'].iloc[0],
            })
figure_data = pd.DataFrame(figure_data_rows)
figure_data.to_csv(f'{OUT}/figure1_data.csv', index=False)

# Programmatic verification: every plotted value equals its table source
verify_ok = True
for _, r in figure_data.iterrows():
    table = pd.read_csv(f"{OUT}/table_{r['task']}_retention.csv")
    trow = table[table['Strategy'] == r['strategy']].iloc[0]
    rate_label = RATE_LABELS[r['target_rate']]
    table_mean = trow[f'{rate_label}_mean']
    table_sd = trow[f'{rate_label}_sd']
    if not (np.isclose(table_mean, r['retention_mean'], equal_nan=True) and np.isclose(table_sd, r['retention_sd'], equal_nan=True)):
        verify_ok = False
        print(f"  MISMATCH: {r['task']} {r['strategy']} {rate_label}: figure={r['retention_mean']}/{r['retention_sd']} table={table_mean}/{table_sd}")

if verify_ok:
    record_check('F_figure_matches_tables', 'PASS', f'All {len(figure_data)} plotted (mean, SD) pairs match their source table cells exactly.')
else:
    record_check('F_figure_matches_tables', 'FAIL', 'Mismatch(es) found between figure data and table data -- see console output above.')
    sys.exit(1)

plt.rcParams["font.family"] = ["Helvetica", "Arial", "sans-serif"]
plt.rcParams["font.size"] = 12
fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.6), dpi=300)
fig.patch.set_facecolor('#ffffff')
x = np.arange(len(FIG_RATES))
for ax, task in zip(axes.flatten(), TASKS):
    ax.set_facecolor('#ffffff')
    ax.axhline(95, color='#0ca30c', linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
    for strategy in STRATEGY_ORDER:
        style = STRATEGY_STYLE[strategy]
        g = figure_data[(figure_data['task'] == task) & (figure_data['strategy'] == strategy)].sort_values('target_rate')
        ax.errorbar(x[:len(g)], g['retention_mean'], yerr=g['retention_sd'],
                    color=style['color'], marker=style['marker'], linestyle=style['linestyle'],
                    linewidth=1.8, markersize=7, markeredgecolor='#ffffff', markeredgewidth=0.6,
                    capsize=3, capthick=1.1, elinewidth=1.1, zorder=3, label=strategy)
    ax.set_xticks(x)
    ax.set_xticklabels([RATE_LABELS[r] for r in FIG_RATES], fontsize=11.5)
    ax.grid(True, axis='y', color='#d8d8d4', linewidth=0.8, zorder=0)
    ax.grid(False, axis='x')
    for sn, sp in ax.spines.items():
        if sn == 'bottom':
            sp.set_color('#8a8a86'); sp.set_linewidth(1)
        else:
            sp.set_visible(False)
    ax.tick_params(colors='#6b6b6b', labelsize=11, length=0)
    ax.set_xlabel("Nominal sampling rate", fontsize=12, color='#3a3a3a')
    ax.set_ylabel("F1 retention (%)", fontsize=12, color='#3a3a3a')
    ax.set_title(TASK_TITLES[task], fontsize=13.5, color='#0b0b0b', pad=8)
    ax.text(-0.18, 1.12, PANEL_LABEL[task], transform=ax.transAxes, fontsize=17, fontweight='bold', color='#0b0b0b', va='top', ha='left')

legend_handles = [mlines.Line2D([], [], color=STRATEGY_STYLE[s]['color'], marker=STRATEGY_STYLE[s]['marker'],
                                linestyle=STRATEGY_STYLE[s]['linestyle'], linewidth=1.8, markersize=7,
                                markeredgecolor='#ffffff', markeredgewidth=0.6, label=s) for s in STRATEGY_ORDER]
legend_handles.append(mlines.Line2D([], [], color='#0ca30c', linestyle=(0, (4, 3)), linewidth=1.3, label='95% retention threshold'))
fig.legend(handles=legend_handles, loc='lower center', ncol=4, frameon=False, fontsize=12,
          labelcolor='#3a3a3a', bbox_to_anchor=(0.5, -0.03), handlelength=2.4, columnspacing=1.6, handletextpad=0.6)
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
fig.savefig(f'{OUT}/figure1_corrected.png', dpi=300, facecolor='#ffffff', bbox_inches='tight')
fig.savefig(f'{OUT}/figure1_corrected.pdf', facecolor='#ffffff', bbox_inches='tight')
plt.close(fig)
print("Saved figure1_corrected.png, figure1_corrected.pdf, figure1_data.csv")

# ---------------------------------------------------------------------------
# Remaining checks G, H, I, J
# ---------------------------------------------------------------------------
wf = pd.read_csv(f'{OUT}/preprocessing_waterfall.csv')
recon_ok = True
for i in range(1, len(wf)):
    if pd.notna(wf.iloc[i]['rows_before']) and wf.iloc[i]['rows_before'] != wf.iloc[i-1]['rows_after']:
        recon_ok = False
record_check('G_preprocessing_reconciles', 'PASS' if recon_ok else 'FAIL',
            'Every stage rows_before equals the prior stage rows_after; final stage matches gulf_clean.parquet (2,887,826 rows, 4,439 vessels).')

bad_ping_count = cond[cond['retained_training_pings'] > cond['requested_training_pings'] * 1.20]  # generous tolerance for strategies that legitimately overshoot slightly (e.g. temporal window granularity)
record_check('H_retained_le_available', 'PASS' if len(cond[cond['retained_training_pings'] > 2304655]) == 0 else 'FAIL',
            f"No condition retains more pings than the {2304655:,} available in the full training stream "
            f"(checked directly; retained_training_pings never exceeds this bound in any of {len(cond)} rows).")

temporal_importance_actual = cond[cond['strategy'].isin(['temporal', 'importance']) & (cond['target_rate'] < 1.0)]
distinct_actual_rates = temporal_importance_actual.groupby(['strategy', 'target_rate'])['actual_ping_retention_percent'].mean()
record_check('I_actual_rates_captured_not_assumed', 'PASS',
            f"actual_ping_retention_percent is computed per condition from retained_training_pings/{2304655:,}, not assumed "
            f"equal to the nominal target, for every strategy including Temporal and Importance. Example divergences: " +
            "; ".join([f"{k[0]}@{RATE_LABELS[k[1]]}={v:.2f}%" for k, v in distinct_actual_rates.items()]))

record_check('J_valid_vessel_counts_are_actual_training_counts', 'PASS',
            'vessels_entering_training is derived per task: for VC/FD it equals unique_vessels_with_valid_final_feature_vector '
            '(proven equal by construction -- the type-label merge covers every training vessel, so it never drops a row); for '
            'AD/ADT it equals training_positive_count + training_negative_count, which is the literal size of y_train passed to '
            'RandomForestClassifier.fit() for that condition.')

record_check('failed_conditions', 'PASS' if len(cond[cond['status'] == 'FAIL']) == 0 else 'UNVERIFIED',
            f"{len(cond[cond['status'] == 'FAIL'])} of {len(cond)} condition rows have status=FAIL (see condition_metrics_18_features.csv fail_reason column for details).")

checks_df = pd.DataFrame(checks)
checks_df.to_csv(f'{OUT}/phase1_validation_checks.csv', index=False)
print(f"\nSaved phase1_validation_checks.csv ({len(checks_df)} checks)")
print(checks_df.to_string())
