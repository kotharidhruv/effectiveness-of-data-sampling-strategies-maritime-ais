# Pipeline Validation (Part 14)

## Train/test split identity

| Item | Value |
|---|---|
| Split universe | `gulf_clean.parquet` restricted to rows passing SOG<=50, valid-MMSI, >=30-pings-per-vessel filters only (4,439 vessels) -- **not** the smaller, lag-dropna-reduced 4,026-vessel universe the pre-correction pipeline used (see "Why the split changed" below) |
| Split method | `train_test_split(all_mmsi, test_size=0.2, random_state=42)`, unstratified |
| Number of train vessels | 3,551 |
| Number of test vessels | 888 |
| Train vessel-ID list SHA-256 | `5824aa4da1d63de8c128f83839f819b5b9aa1d32e4dc99ae9429ffc465b6b87d` |
| Test vessel-ID list SHA-256 | `4b6b47051bbd0916f331e2a1b162701d52731a0ef32d5917e19ad25b89d274b2` |

Both hashes are computed over the sorted, comma-joined vessel-ID list (see
`phase1_18feature_corrected/split_checksums.txt`, generated at the same time the
split itself was created — not recomputed after the fact).

## Confirmation: same train IDs used for every strategy/task; same test IDs used for every strategy/task

`run_phase1_experiment.py` loads `train_raw_prelag.parquet` and `test_raw_prelag.parquet`
exactly once, at the top of the script, into `train_raw` and `test_raw`. These two
dataframes are never reassigned, resplit, or reloaded anywhere else in the script.
Every one of the 6 strategies x 6 rates x 5 seeds x 4 tasks calls into
`sampled_feature_builder.build_sampled_features()` (or, for the fixed 100% baseline,
consumes `train_agg_full`/`test_agg_full`, themselves built once from the same
`train_raw`/`test_raw`) — there is no code path in this pipeline that constructs an
alternate split. This was confirmed by direct code inspection of
`run_phase1_experiment.py` (not inferred).

## Confirmation: all tables and Figure 1 use outputs from this corrected run

Every downstream file (`condition_summary_18_features.csv`,
`vessels_retained_18_features.csv`, all `table_*.csv` files, `figure1_data.csv`,
`figure1_corrected.png/.pdf`) is produced by `finalize_phase1.py`, which reads
**only** `condition_metrics_18_features.csv` and `full_baseline_f1.csv` — the two
files `run_phase1_experiment.py` writes directly from this run. `finalize_phase1.py`
additionally recomputes the per-vessel-type anomaly label breakdown
(`anomaly_label_counts_by_type.csv`) independently, from the same
`train_raw_prelag.parquet`/`test_raw_prelag.parquet` and the same fixed
Isolation Forest configuration (200 estimators, contamination=0.05, random_state=42)
— this is a deterministic recomputation of the same pipeline, not a reuse of a
cached result, and is called out explicitly rather than silently reused.
Figure 1's plotted values are additionally verified programmatically against their
source table cells before the figure file is written (see Check F,
`phase1_validation_checks.csv`) — the script halts rather than saving a figure if
any mismatch is found.

**No file from the pre-correction pipeline (`sampling_utils.py`, any
`*_sampling_all_tasks.py` script, `final_paper_data.json`, or any `*_v2.csv` result
file) is read anywhere in `run_phase1_experiment.py` or `finalize_phase1.py`.**
Confirmed by inspection: neither script imports from or opens a path under
`exploratory_data_analysis/` other than `gulf_clean.parquet` (read once, at split-
construction time, before this pipeline existed as a script) and files under
`reviewer_revision/phase1_18feature_corrected/` that this pipeline itself wrote.

## Why the split changed from the original 4,026-vessel universe to 4,439

The original pipeline's train/test split (`sampling_utils.load_and_split()`) was
built from `gulf_features.parquet`'s vessel list -- but that file's vessel universe
(4,026) is itself the *result* of applying a global lag computation and its
associated null-value dropna cascade to the full 4,439-vessel cleaned universe
*before* any split occurred. Part 3 of this revision explicitly states: "Do not
remove raw rows merely because pre-sampling lag values are unavailable if those
lags are supposed to be recomputed after sampling." Since lag recomputation now
happens per sampling condition (after the split, after sampling), there is no
longer a valid reason to let lag-availability determine which vessels enter the
train/test split in the first place. The corrected pipeline therefore builds the
split from the full, lag-independent 4,439-vessel cleaned universe. This is a
deliberate, documented consequence of the Part 3 correction, not an oversight --
see `audit_report.md` Q2/Q3 and the "critically WITHOUT dropping any row" comment
in `sampled_feature_builder.compute_selection_deltas()`.

## Exact explanation of the original Random/ADT split mismatch

**Source: `anomaly_detection_by_vessel_type.py`, lines 113-121** — a standalone,
pre-unification script that builds its own train/test split
`stratify=vessel_type_labels['type_group'].values`), different from the
unstratified split (`sampling_utils.load_and_split()`, no `stratify=`) used by the
current, unified six-strategy pipeline.

**Why it affected only (Random, ADT) and no other branch:**
1. This script only ever implemented the ADT task -- it contains no VC, FD, or
   pooled-AD logic, so the discrepancy could never appear outside ADT.
2. Within ADT, this script implements only plain random ping sampling (confirmed by
   its own docstring: "The random-sampling procedure used to test robustness
   afterward is the same simple random ping sampling as
   `anomaly_detection_sampling.py`") -- it has no Stratified/Spatial/Importance/
   Adaptive/Temporal logic. Every other strategy's ADT numbers were therefore
   always sourced exclusively from the current unified pipeline. Only Random had
   two independent implementations (this legacy script, and
   `random_sampling_all_tasks.py`) that could disagree.

**Confirmation the fix was already in place before this revision began:**
`random_sampling_all_tasks.py` lines 8-20 (its own docstring) documents the
correction directly: routing Random through the shared `sampling_utils.py`
infrastructure "removes that mismatch: its 100% row now lands on exactly the same
number as every other strategy's 100% row."

**Confirmation no legacy/cached result was mixed into this revision's corrected
output:** verified by direct inspection of every import and file path opened in
`run_phase1_experiment.py`, `finalize_phase1.py`, and `sampled_feature_builder.py`
(listed above). `anomaly_detection_by_vessel_type.py` itself is never imported or
executed by this revision's pipeline.

## Items I can confirm vs. cannot confirm

| Item | Status |
|---|---|
| Train vessel ID hash | CONFIRMED (computed directly, see above) |
| Test vessel ID hash | CONFIRMED (computed directly, see above) |
| No train/test vessel overlap | CONFIRMED (Check D, `phase1_validation_checks.csv`) |
| Same split used by every strategy/task | CONFIRMED (single load, no reassignment -- code-inspected) |
| All tables/figure from this corrected run | CONFIRMED (single-source read, programmatically verified for the figure) |
| Original Random/ADT mismatch source and scope | CONFIRMED (exact file/lines cited, mechanism explained) |
| No legacy pipeline file read by this revision | CONFIRMED (all file paths in the three scripts enumerated) |

No item in this document is marked UNVERIFIED; every claim above was checked
directly against the code or the generated output rather than assumed.
