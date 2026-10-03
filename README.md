# AIS Sampling Pipeline — Code and Results

Sampling, feature-recomputation, and evaluation code used to produce every result reported in
"Task-Dependent Efficacy of Data Sampling Strategies for Maritime AIS Analytics" (NHSJS submission).

## What this is

This repository contains the corrected analysis pipeline referenced in the manuscript's Methods
(Evaluation Consistency Check) and Data and Code Availability sections: six ping-level AIS sampling
strategies, evaluated across four vessel-level ML tasks, with every lag-dependent feature recomputed
from each sampling condition's own retained pings rather than computed once globally before sampling.

## Repository structure

- `code/` — the pipeline itself.
  - `sampled_feature_builder.py` — core module: sampling strategies, per-condition lag recomputation,
    vessel-level feature aggregation.
  - `run_phase1_experiment.py`, `finalize_phase1.py` — primary 18-feature experiment (720 conditions:
    6 strategies x 6 rates x 5 seeds x 4 tasks) and all derived tables/figures.
  - `run_phase2_experiment.py`, `finalize_phase2.py`, `finalize_phase2_part2.py` — 16-feature ablation
    (ping-timing features removed), freezing Phase 1's train/test split.
  - `run_phase3_experiment.py`, `finalize_phase3.py`, `finalize_phase3_part2.py` — second-model
    (Logistic Regression) sensitivity check for the Anomaly-Detection-by-Type task.
- `figures/make_figure1.py` — regenerates the manuscript's Figure 1 from the corrected retention tables.
- `docs/` — audit trail: how the pre-correction pipeline was diagnosed, how the corrected train/test
  split and retention formula were validated, and full reproduction instructions.
- `phase1_18feature_corrected/`, `phase2_16feature_ablation/`, `phase3_second_model/` — every numeric
  result (CSV/JSON) and figure these scripts produce, included so the manuscript's tables can be
  checked against their source without rerunning the full grid.

## Data

Raw input is public AIS data from NOAA MarineCadastre.gov for October 6, 2024, Gulf of Mexico
(18 degrees N-31 degrees N, 97 degrees W-80 degrees W): https://marinecadastre.gov/ais/

The two intermediate files `train_raw_prelag.parquet` and `test_raw_prelag.parquet` (the cleaned,
vessel-level train/test split prior to any sampling or lag computation) are not included in this
repository: the training file is 109 MB, over GitHub's 100 MB per-file limit. They are fully
reproducible from the public source above by following the cleaning and split logic documented in
`docs/pipeline_validation.md` and `docs/audit_report.md` (filter SOG <= 50 and valid MMSI and >= 30
pings/vessel, then split by vessel with `random_state=42`, `test_size=0.2`).

## Environment

Python 3.11.2, scikit-learn 1.9.0, pandas 3.0.3, numpy 2.4.6, duckdb 1.5.4, matplotlib (figure
generation only). See `docs/reproduction_instructions.md` for exact run order and expected output
row counts.

## Run order

1. Build `train_raw_prelag.parquet` / `test_raw_prelag.parquet` as described above.
2. `python code/run_phase1_experiment.py` then `python code/finalize_phase1.py`
3. `python code/run_phase2_experiment.py` then `python code/finalize_phase2.py` and
   `python code/finalize_phase2_part2.py`
4. `python code/run_phase3_experiment.py` then `python code/finalize_phase3.py` and
   `python code/finalize_phase3_part2.py`
5. `python figures/make_figure1.py`
