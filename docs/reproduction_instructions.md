# Reproduction Instructions — Phase 1 (18-Feature Corrected Analysis)

## Environment

- Python 3.11.2
- scikit-learn 1.9.0
- pandas 3.0.3
- numpy 2.4.6
- duckdb 1.5.4

All package versions are exactly what is installed in the project's `.venv`
(`/Users/mehek/research/Smart-Sampling-for-Maritime-Detection/.venv`), confirmed via
`pip show`. No other library materially affects the experiment (matplotlib is used
only for Figure 1 rendering, not for any numeric result).

## Required input files

- `datasets/ais-2024-10-06_gom.parquet` — raw AIS data (source of the preprocessing
  waterfall).
- `exploratory_data_analysis/gulf_clean.parquet` — pre-existing cleaned (but pre-lag)
  ping data; used to cross-validate the waterfall reconciliation, though the Phase 1
  pipeline rebuilds this stage's logic directly from the raw parquet rather than
  depending on this cached file.

## Exact script execution order

Run from `exploratory_data_analysis/`:

1. **Build the preprocessing waterfall and confirm reconciliation:**
   the waterfall generation logic is documented in
   `reviewer_revision/preprocessing_waterfall.csv`'s own `rule_applied` column;
   the raw-count derivation queries are recorded in `audit_report.md` and this
   file's own construction (no separate script file — see
   `pipeline_validation.md` for the exact commands used).

2. **Build the Phase 1 train/test split and raw pre-lag ping sets** (creates
   `phase1_18feature_corrected/train_raw_prelag.parquet`,
   `test_raw_prelag.parquet`, `split_checksums.txt`):
   - Load `gulf_clean.parquet`.
   - Assign `vessel_type_group` (raw `vessel_type` -> 7 groups).
   - Compute `dist_to_port` (Haversine to 8 fixed reference ports -- named "Gulf
     ports" in earlier code comments, but corrected here since one of the eight,
     Miami, is on the Atlantic side, not the Gulf).
   - Split by vessel: `train_test_split(all_mmsi, test_size=0.2, random_state=42)`
     over the **full 4,439-vessel cleaned universe** (not the smaller,
     lag-dropna-reduced universe the pre-correction pipeline used — see
     `audit_report.md` and `pipeline_validation.md` for why this changed).

3. **Run the full corrected experiment grid:**
   ```
   python reviewer_revision/run_phase1_experiment.py
   ```
   This single script: builds the full (100%) training/test feature matrices,
   the vessel-type labels, the fixed Isolation Forest ground truth (pooled AD and
   per-type ADT), the majority-class baselines, the per-seed full-data baselines,
   and the complete 6-strategy x 6-rate x 5-seed corrected grid (using
   `sampled_feature_builder.py` for every sampling/lag-recomputation/aggregation
   step). Produces `full_baseline_f1.csv` and `condition_metrics_18_features.csv`.

4. **Finalize all downstream tables, the corrected figure, and validation
   artifacts** (reads only the two files from step 3, never re-touches raw data):
   ```
   python reviewer_revision/finalize_phase1.py
   ```
   Produces every remaining Phase 1 deliverable listed below.

## Expected output files (all under `exploratory_data_analysis/reviewer_revision/`)

```
audit_report.md
analysis_notes.md
reproduction_instructions.md          (this file)
sampled_feature_builder.py
run_phase1_experiment.py
finalize_phase1.py
phase1_18feature_corrected/
    train_raw_prelag.parquet
    test_raw_prelag.parquet
    split_checksums.txt
    adt_group_report.json
    majority_baseline_info.json
    preprocessing_waterfall.csv
    full_baseline_f1.csv
    full_baseline_summary.csv
    condition_metrics_18_features.csv
    condition_summary_18_features.csv
    vessels_retained_18_features.csv
    vessels_retained_by_task_strategy_rate_seed.csv
    anomaly_label_counts_by_type.csv
    table_full_data_performance.csv
    table_VC_retention.csv
    table_FD_retention.csv
    table_AD_retention.csv
    table_ADT_retention.csv
    table_VC_absolute_f1.csv
    table_FD_absolute_f1.csv
    table_AD_absolute_f1.csv
    table_ADT_absolute_f1.csv
    table_valid_vessels.csv
    figure1_corrected.png
    figure1_corrected.pdf
    figure1_data.csv
    retention_audit.md
    claim_stability.md
    pipeline_validation.md
    phase1_validation_checks.csv
```

## Expected number of experiment rows

- `full_baseline_f1.csv`: 4 tasks x 5 seeds = **20 rows**.
- `condition_metrics_18_features.csv`: 6 strategies x 6 rates x 5 seeds x 4 tasks
  = **720 rows** (unless any condition fails to train, in which case that row is
  still present with `status='FAIL'` and a `fail_reason` — no row is ever silently
  omitted).
- `condition_summary_18_features.csv`: 6 strategies x 4 tasks x 6 rates =
  **144 rows** (one summary row per condition, aggregating its 5 seeds).
