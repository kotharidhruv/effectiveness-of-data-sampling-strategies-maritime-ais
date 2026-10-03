"""
Regenerates Figure 1 for the revised manuscript: 4-panel retention chart
(VC, FD, AD, ADT), rates 1/5/10/25/50% only (100% dropped per Minor Issue #7,
since retention there is exactly 100.0 +/- 0.0 by construction under the
corrected per-seed retention formula and carries no information).
Data is transcribed directly from phase1_18feature_corrected/table_*_retention.csv
(mean, sd), which is itself the output of the corrected pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

RATES = [0.01, 0.05, 0.10, 0.25, 0.50]
RATE_LABELS = ['1%', '5%', '10%', '25%', '50%']
STRATEGIES = ['Random', 'Stratified', 'Spatial', 'Importance', 'Adaptive', 'Temporal']
COLOR = {'Random': "#2a78d6", 'Stratified': "#008300", 'Spatial': "#e87ba4",
         'Importance': "#eda100", 'Adaptive': "#eb6834", 'Temporal': "#4a3aa7"}
MARKER = {'Random': 'o', 'Stratified': 's', 'Spatial': '^', 'Importance': 'D', 'Adaptive': 'v', 'Temporal': '*'}

# mean, sd tuples per strategy, in RATES order -- from table_VC/FD/AD/ADT_retention.csv
DATA = {
'VC': {
 'Random':     [(77.54,2.28),(65.41,4.39),(64.42,6.76),(69.26,5.33),(86.64,2.42)],
 'Stratified': [(75.60,4.13),(67.33,4.87),(64.91,5.78),(66.84,2.73),(87.59,2.51)],
 'Spatial':    [(65.74,5.17),(72.89,3.10),(68.29,5.63),(74.40,1.55),(91.75,1.92)],
 'Importance': [(19.12,2.92),(15.37,0.94),(17.60,1.17),(28.43,1.73),(46.34,1.06)],
 'Adaptive':   [(67.46,3.10),(68.84,2.25),(72.95,3.25),(84.27,2.31),(91.35,3.14)],
 'Temporal':   [(82.19,2.56),(81.32,2.67),(83.94,3.27),(87.11,1.54),(95.26,2.05)],
},
'FD': {
 'Random':     [(33.47,21.45),(24.12,17.46),(23.23,8.39),(36.73,7.38),(79.61,3.56)],
 'Stratified': [(37.24,22.48),(26.89,12.75),(29.21,15.01),(46.75,9.12),(79.93,6.27)],
 'Spatial':    [(10.92,10.40),(19.73,23.57),(9.90,11.91),(24.60,7.67),(43.45,2.25)],
 'Importance': [(25.78,2.39),(24.13,1.51),(24.24,1.02),(25.88,0.49),(31.86,1.46)],
 'Adaptive':   [(64.83,10.37),(22.65,11.96),(31.65,13.95),(58.00,8.79),(83.03,6.19)],
 'Temporal':   [(53.15,9.76),(52.95,7.52),(55.75,4.33),(84.90,9.20),(93.74,5.51)],
},
'AD': {
 'Random':     [(59.84,5.92),(75.04,4.59),(84.69,4.40),(94.79,3.06),(98.09,2.37)],
 'Stratified': [(59.74,4.04),(75.96,6.33),(84.89,5.62),(90.48,3.09),(98.19,3.41)],
 'Spatial':    [(31.75,26.71),(73.19,9.22),(84.64,4.43),(88.36,2.91),(92.76,4.71)],
 'Importance': [(60.81,2.89),(72.04,4.10),(75.35,6.37),(86.96,3.07),(91.56,4.41)],
 'Adaptive':   [(14.36,18.74),(41.18,13.95),(71.88,15.03),(93.53,3.27),(100.53,1.08)],
 'Temporal':   [(69.12,1.19),(85.58,6.82),(82.66,4.37),(91.17,3.14),(100.56,1.62)],
},
'ADT': {
 'Random':     [(7.21,9.92),(54.50,16.95),(68.85,8.14),(67.15,12.71),(70.58,8.62)],
 'Stratified': [(23.40,20.90),(39.07,13.93),(52.97,19.48),(61.08,14.13),(62.38,10.35)],
 'Spatial':    [(5.86,5.36),(34.52,15.66),(58.31,8.65),(74.73,11.25),(53.30,16.05)],
 'Importance': [(0.00,0.00),(34.73,11.95),(36.82,6.87),(40.53,5.09),(59.70,6.83)],
 'Adaptive':   [(7.32,7.48),(33.58,18.51),(41.96,20.18),(55.47,9.31),(79.42,13.68)],
 'Temporal':   [(11.25,3.94),(44.66,12.52),(79.94,7.26),(66.17,5.51),(81.05,9.32)],
},
}

TITLES = {'VC': 'Vessel Classification', 'FD': 'Fishing Detection',
          'AD': 'Pooled Anomaly Detection', 'ADT': 'Anomaly Detection by Type'}
PANELS = {'VC': 'A', 'FD': 'B', 'AD': 'C', 'ADT': 'D'}

plt.rcParams["font.family"] = ["Helvetica", "Arial", "sans-serif"]
plt.rcParams["font.size"] = 12
fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), dpi=300)
fig.patch.set_facecolor('#ffffff')
x = np.arange(len(RATES))

for ax, task in zip(axes.flat, ['VC', 'FD', 'AD', 'ADT']):
    ax.set_facecolor('#ffffff')
    ax.axhline(95, color='#8a8a86', linewidth=1.2, linestyle='--', zorder=2)
    for strat in STRATEGIES:
        means = [v[0] for v in DATA[task][strat]]
        sds = [v[1] for v in DATA[task][strat]]
        ax.errorbar(x, means, yerr=sds, color=COLOR[strat], marker=MARKER[strat],
                    linewidth=1.8, markersize=7, markeredgecolor='#ffffff', markeredgewidth=0.6,
                    capsize=3, elinewidth=1.0, label=strat)
    ax.set_xticks(x); ax.set_xticklabels(RATE_LABELS, fontsize=11.5)
    ax.set_ylim(-5, 115)
    ax.grid(True, axis='y', color='#d8d8d4', linewidth=0.8, zorder=0); ax.grid(False, axis='x')
    for sn, sp in ax.spines.items():
        if sn == 'bottom': sp.set_color('#8a8a86'); sp.set_linewidth(1)
        else: sp.set_visible(False)
    ax.tick_params(colors='#6b6b6b', labelsize=11, length=0)
    ax.set_xlabel("Nominal sampling rate", fontsize=12, color='#3a3a3a')
    ax.set_ylabel("F1 retention (%)", fontsize=11.5, color='#3a3a3a')
    ax.set_title(TITLES[task], fontsize=13.5, color='#0b0b0b', pad=8)
    ax.text(-0.14, 1.08, PANELS[task], transform=ax.transAxes, fontsize=17, fontweight='bold',
            color='#0b0b0b', va='top', ha='left')

legend_handles = [mlines.Line2D([], [], color=COLOR[s], marker=MARKER[s], linewidth=1.8, markersize=7,
                                markeredgecolor='#ffffff', markeredgewidth=0.6, label=s) for s in STRATEGIES]
legend_handles.append(mlines.Line2D([], [], color='#8a8a86', linewidth=1.2, linestyle='--', label='95% retention threshold'))
fig.legend(handles=legend_handles, loc='lower center', ncol=4, frameon=False, fontsize=10.5, labelcolor='#3a3a3a',
          bbox_to_anchor=(0.5, -0.02), handlelength=2.0, columnspacing=1.2, handletextpad=0.5)
fig.tight_layout(rect=[0, 0.06, 1, 1.0])
OUT = '/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/exploratory_data_analysis/reviewer_revision/final_submission'
fig.savefig(f'{OUT}/figure1_revised.png', dpi=300, facecolor='#ffffff', bbox_inches='tight')
fig.savefig(f'{OUT}/figure1_revised.pdf', facecolor='#ffffff', bbox_inches='tight')
plt.close(fig)
print("Saved figure1_revised.png/.pdf")
