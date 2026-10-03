"""
finalize_phase3_part2.py
==========================
Second pass: model_sensitivity_claims.md, figure_ADT_model_sensitivity, and the
final validation checks CSV. Run AFTER finalize_phase3.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

OUT = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/phase3_second_model'
STRATEGIES_ORDER = ['Random', 'Stratified', 'Spatial', 'Importance', 'Adaptive', 'Temporal']
STRATEGY_COLOR = {'Random': "#2a78d6", 'Stratified': "#008300", 'Spatial': "#e87ba4",
                  'Importance': "#eda100", 'Adaptive': "#eb6834", 'Temporal': "#4a3aa7"}
STRATEGY_MARKER = {'Random': 'o', 'Stratified': 's', 'Spatial': '^', 'Importance': 'D', 'Adaptive': 'v', 'Temporal': '*'}
RATES_NO100 = [0.01, 0.05, 0.10, 0.25, 0.50]
RATE_LABELS = {0.01: '1%', 0.05: '5%', 0.10: '10%', 0.25: '25%', 0.50: '50%', 1.00: '100%'}

sens_18 = pd.read_csv(f'{OUT}/ADT_model_sensitivity_18F.csv')
sens_16 = pd.read_csv(f'{OUT}/ADT_model_sensitivity_16F.csv')
rank_agree = pd.read_csv(f'{OUT}/ADT_RF_vs_Logistic_rank_agreement.csv')
checks = pd.read_csv(f'{OUT}/_partial_checks3.csv').to_dict('records')

def record_check(name, status, details):
    checks.append({'check': name, 'status': status, 'details': details})
    print(f"[{status}] {name}: {details}")

# ---------------------------------------------------------------------------
# model_sensitivity_claims.md
# ---------------------------------------------------------------------------
lines = ["# Model Sensitivity Claims -- ADT, Random Forest vs. Logistic Regression (Part 10)\n"]
lines.append("Evidence-limited language only: 'consistent', 'directionally similar', 'different', "
             "'not clearly distinguishable given variability'. No significance claims.\n")

for tag, sens in [('18F', sens_18), ('16F', sens_16)]:
    lines.append(f"\n## {tag}\n")
    for rate in RATES_NO100:
        g = sens[sens['target_rate'] == rate].copy()
        if len(g) == 0:
            continue
        g_rf = g.sort_values('RF_retention_mean', ascending=False)
        g_lr = g.sort_values('LR_retention_mean', ascending=False)
        lines.append(f"\n**{RATE_LABELS[rate]}:**")
        lines.append(f"- RF: most retained = {g_rf.iloc[0]['strategy']} ({g_rf.iloc[0]['RF_retention_mean']:.1f}), "
                     f"least retained = {g_rf.iloc[-1]['strategy']} ({g_rf.iloc[-1]['RF_retention_mean']:.1f})")
        lines.append(f"- LR: most retained = {g_lr.iloc[0]['strategy']} ({g_lr.iloc[0]['LR_retention_mean']:.1f}), "
                     f"least retained = {g_lr.iloc[-1]['strategy']} ({g_lr.iloc[-1]['LR_retention_mean']:.1f})")
        same_top = g_rf.iloc[0]['strategy'] == g_lr.iloc[0]['strategy']
        lines.append(f"- Top strategy {'matches' if same_top else 'differs'} between RF and LR at this rate.")
        rho_row = rank_agree[(rank_agree['feature_set'] == tag) & (rank_agree['target_rate'] == rate)]
        if len(rho_row) > 0:
            lines.append(f"- Spearman rank agreement (descriptive only, not a significance test): rho = {rho_row.iloc[0]['spearman_rho']:.2f}")
        any_95 = (g['LR_retention_mean'] >= 95).any()
        lines.append(f"- Does any strategy reach >=95% LR retention at this rate? {'Yes -- ' + ', '.join(g[g['LR_retention_mean']>=95]['strategy'].tolist()) if any_95 else 'No.'}")
        # spread vs SD
        spread = g['LR_retention_mean'].max() - g['LR_retention_mean'].min()
        mean_sd = g['LR_retention_sd'].mean()
        lines.append(f"- LR retention spread across strategies: {spread:.1f} points; mean per-strategy SD: {mean_sd:.1f} points "
                     f"-> {'spread exceeds typical SD (differences may be more than noise)' if spread > mean_sd else 'spread is comparable to typical SD -- not clearly distinguishable given variability'}.")

lines.append("\n## Overall pattern\n")
mean_diff_18 = sens_18[sens_18['target_rate'] < 1.0]['retention_difference_LR_minus_RF'].mean()
mean_diff_16 = sens_16[sens_16['target_rate'] < 1.0]['retention_difference_LR_minus_RF'].mean()
mean_rho_18 = rank_agree[rank_agree['feature_set'] == '18F']['spearman_rho'].mean()
mean_rho_16 = rank_agree[rank_agree['feature_set'] == '16F']['spearman_rho'].mean()
lines.append(f"- Mean retention difference (LR - RF) across all non-100% conditions: 18F = {mean_diff_18:+.1f}, 16F = {mean_diff_16:+.1f}.")
lines.append(f"- Mean Spearman rank agreement across reduced rates: 18F = {mean_rho_18:.2f}, 16F = {mean_rho_16:.2f} -- "
             f"rank agreement varies substantially by rate (see ADT_RF_vs_Logistic_rank_agreement.csv for the full per-rate breakdown, "
             f"which ranges from {rank_agree['spearman_rho'].min():.2f} to {rank_agree['spearman_rho'].max():.2f}). "
             f"This indicates strategy rankings are **not consistently reproduced** across the two models at every individual rate, "
             f"even though the broad qualitative pattern (retention increasing with rate; ADT being difficult to recover from low samples) "
             f"is directionally similar between RF and LR.")
lines.append(f"- The general finding that sampling-strategy effectiveness is task/rate-dependent, rather than fixed, remains observable "
             f"under Logistic Regression -- retention still varies substantially by rate and by strategy for LR, similar in kind (though "
             f"not identical in magnitude or exact ranking) to what Random Forest shows.")

with open(f'{OUT}/model_sensitivity_claims.md', 'w') as f:
    f.write('\n'.join(lines))
print("Saved model_sensitivity_claims.md")

# ---------------------------------------------------------------------------
# figure_ADT_model_sensitivity.png/.pdf/_data.csv
# Design: per-strategy LR-minus-RF retention DIFFERENCE by rate (avoids a
# 12-line unreadable plot) -- two panels, 18F and 16F.
# ---------------------------------------------------------------------------
fig_rows = []
for tag, sens in [('18F', sens_18), ('16F', sens_16)]:
    for _, r in sens[sens['target_rate'] < 1.0].iterrows():
        fig_rows.append({'feature_set': tag, 'strategy': r['strategy'], 'target_rate': r['target_rate'],
                         'retention_difference_LR_minus_RF': r['retention_difference_LR_minus_RF']})
fig_data = pd.DataFrame(fig_rows)
fig_data.to_csv(f'{OUT}/figure_ADT_model_sensitivity_data.csv', index=False)

verify_ok = True
for _, r in fig_data.iterrows():
    src = (sens_18 if r['feature_set'] == '18F' else sens_16)
    match = src[(src['strategy'] == r['strategy']) & (src['target_rate'] == r['target_rate'])]
    if len(match) == 0 or not np.isclose(match.iloc[0]['retention_difference_LR_minus_RF'], r['retention_difference_LR_minus_RF']):
        verify_ok = False
record_check('figure_matches_source_csv', 'PASS' if verify_ok else 'FAIL', f'{len(fig_data)} plotted points checked against ADT_model_sensitivity_*.csv.')

plt.rcParams["font.family"] = ["Helvetica", "Arial", "sans-serif"]
plt.rcParams["font.size"] = 12
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=300)
fig.patch.set_facecolor('#ffffff')
x = np.arange(len(RATES_NO100))

for ax, tag, panel in zip(axes, ['18F', '16F'], ['A', 'B']):
    ax.set_facecolor('#ffffff')
    ax.axhline(0, color='#8a8a86', linewidth=1.2, linestyle='-', zorder=2)
    g_all = fig_data[fig_data['feature_set'] == tag]
    for strategy in STRATEGIES_ORDER:
        g = g_all[g_all['strategy'] == strategy].sort_values('target_rate')
        ax.plot(x[:len(g)], g['retention_difference_LR_minus_RF'], color=STRATEGY_COLOR[strategy],
               marker=STRATEGY_MARKER[strategy], linewidth=1.8, markersize=7,
               markeredgecolor='#ffffff', markeredgewidth=0.6, label=strategy)
    ax.set_xticks(x); ax.set_xticklabels([RATE_LABELS[r] for r in RATES_NO100], fontsize=11.5)
    ax.grid(True, axis='y', color='#d8d8d4', linewidth=0.8, zorder=0); ax.grid(False, axis='x')
    for sn, sp in ax.spines.items():
        if sn == 'bottom': sp.set_color('#8a8a86'); sp.set_linewidth(1)
        else: sp.set_visible(False)
    ax.tick_params(colors='#6b6b6b', labelsize=11, length=0)
    ax.set_xlabel("Nominal sampling rate", fontsize=12, color='#3a3a3a')
    ax.set_ylabel("LR retention - RF retention (points)", fontsize=11.5, color='#3a3a3a')
    ax.set_title(f"ADT, {tag}", fontsize=13.5, color='#0b0b0b', pad=8)
    ax.text(-0.14, 1.12, panel, transform=ax.transAxes, fontsize=17, fontweight='bold', color='#0b0b0b', va='top', ha='left')

legend_handles = [mlines.Line2D([], [], color=STRATEGY_COLOR[s], marker=STRATEGY_MARKER[s], linewidth=1.8, markersize=7,
                                markeredgecolor='#ffffff', markeredgewidth=0.6, label=s) for s in STRATEGIES_ORDER]
fig.legend(handles=legend_handles, loc='lower center', ncol=6, frameon=False, fontsize=10.5, labelcolor='#3a3a3a',
          bbox_to_anchor=(0.5, -0.05), handlelength=2.0, columnspacing=1.2, handletextpad=0.5)
fig.suptitle("")
fig.tight_layout(rect=[0, 0.10, 1, 1.0])
fig.savefig(f'{OUT}/figure_ADT_model_sensitivity.png', dpi=300, facecolor='#ffffff', bbox_inches='tight')
fig.savefig(f'{OUT}/figure_ADT_model_sensitivity.pdf', facecolor='#ffffff', bbox_inches='tight')
plt.close(fig)
print("Saved figure_ADT_model_sensitivity.png/.pdf/_data.csv")

checks_df = pd.DataFrame(checks)
checks_df.to_csv(f'{OUT}/phase3_validation_checks.csv', index=False)
import os
os.remove(f'{OUT}/_partial_checks3.csv')
print(f"\nSaved phase3_validation_checks.csv ({len(checks_df)} checks)")
print(checks_df[['check', 'status']].to_string())
