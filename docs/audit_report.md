# Audit Report — Pre-Correction State of the Sampling Pipeline

This document answers the seven audit questions (Part A of the revision request) by
citing exact files, functions, and line numbers in the codebase as it exists today,
before any correction. No numbers in this report are estimated — every claim is a
direct reading of the code, or (where marked) a live re-execution of it.

---

## 1. Exact current pipeline order

The pipeline is split across two disjoint stages that never talk to each other again
once the first stage finishes: **(a)** a one-time dataset-construction stage in
`eda.ipynb`, executed once, whose output is saved to disk; and **(b)** the six
per-strategy experiment scripts, which only ever read that saved output.

**Stage (a) — `eda.ipynb`, run once, in this order:**

| Step | Cell | What happens |
|---|---|---|
| Cleaning | Cell 16 | SQL filter `sog <= 50 AND mmsi BETWEEN 100000000 AND 999999999` on the raw parquet, then drop vessels with `< 30` pings. Saves `gulf_clean.parquet`. |
| Sorting | Cell 18 | `df = df.sort_values(['mmsi', 'base_date_time']).reset_index(drop=True)` |
| Lag SOG | Cell 18 | `df['prev_sog'] = df.groupby('mmsi')['sog'].shift(1)` |
| Lag COG | Cell 18 | `df['prev_cog'] = df.groupby('mmsi')['cog'].shift(1)` |
| Lag timestamp | Cell 18 | `df['prev_time'] = df.groupby('mmsi')['base_date_time'].shift(1)` |
| Delta/derived features | Cell 20 | `delta_sog = sog - prev_sog`; `delta_cog = ((cog - prev_cog + 180) % 360) - 180`; `ping_gap = (base_date_time - prev_time).dt.total_seconds()` |
| Row drop (dropna on lag cols) | Cell 20 | `df.dropna(subset=['prev_sog','prev_cog','prev_time'])` |
| Vessel type groups | Cell 24 | `assign_type_group()` applied to raw `vessel_type` |
| Distance to port | Cell 26 | Haversine to 8 hardcoded ports, `.min(axis=1)` |
| Save | Cell 28 | Writes `gulf_features.parquet` — this file already contains `delta_sog`, `delta_cog`, `ping_gap` as ping-level columns, computed once against the **full, unsampled, cleaned** ping sequence. |

**Stage (b) — every one of the six `*_sampling_all_tasks.py` scripts, in this order:**

| Step | Function / line | What happens |
|---|---|---|
| Load | `sampling_utils.load_and_split()` (line 73) | Reads `gulf_features.parquet` directly — the lag/delta columns from Stage (a) are read as-is, never recomputed. |
| Train/test split | `load_and_split()` line 79 | `train_test_split(df['mmsi'].unique(), test_size=0.2, random_state=42)` — vessel-level, unstratified. |
| Sampling | strategy-specific `*_sample(df_train, rate, seed)` | Selects a subset of **already-lagged** rows from `df_train`. No lag/delta value is ever recomputed here — the function only filters/subsets rows. |
| Lag recomputation | **does not exist** | See Q2/Q3 below. |
| Vessel-level aggregation | `aggregate_features()` (`sampling_utils.py` line 50) | `groupby('mmsi').agg(...)` over whatever rows survived sampling, using the columns `delta_sog`/`delta_cog`/`ping_gap` that were computed in Stage (a) against the full sequence. |
| Drop missing-feature rows | `run_experiment()` line 234: `.dropna(subset=FEATURE_COLS).fillna(0)` | (Note: the `.fillna(0)` here is inert — see prior audit finding in this conversation; `dropna` already removes every row it could touch.) |
| Isolation Forest labeling | `build_ad_labels()` line 108, `build_ad_by_type_labels()` line 151 | Fit once, before the sampling-rate loop, on the **full unsampled** training aggregation. Held fixed as ground truth for every subsequent (rate, seed, strategy). |
| Random Forest training | `run_experiment()` line 255 (approx.) | `RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed).fit(X_train, y_train)`, once per (task, strategy, rate, seed). |

---

## 2 & 3. Are lags computed before sampling and carried into sampled records, or recomputed after sampling from retained pings only?

**Confirmed: lags are computed once, before sampling, on the full unsampled sequence, and are simply carried through unchanged into whatever subset of rows a sampling strategy retains. They are never recomputed after sampling.**

Evidence: `delta_sog`, `delta_cog`, and `ping_gap` are ping-level columns that already exist in `gulf_features.parquet` by the time any of the six `*_sampling_all_tasks.py` scripts even opens the file (`load_and_split()`, `sampling_utils.py` line 73-84, just calls `pd.read_parquet(parquet_path)`). Every `*_sample(df, rate, seed)` function I inspected (`random_sample`, `stratified_sample`, `spatial_sample`, `importance_sample`, `adaptive_sample` in their respective scripts; `temporal_sample` in `temporal_sampling_all_tasks.py`) does row selection only — filtering, grouping, or `.sample()` calls on existing columns — and none of them touch `sog`, `cog`, `base_date_time`, or recompute `prev_sog`/`prev_cog`/`delta_sog`/`delta_cog`/`ping_gap`. `aggregate_features()` (`sampling_utils.py` line 50) then aggregates whatever `delta_sog`/`delta_cog` values happen to survive into the sampled subset — but those values were computed against each vessel's **full** chronological sequence, not against the reduced sequence a downstream consumer would actually have.

**Concretely, this means:** for a vessel whose full sequence is `[p1, p2, p3, p4, p5]` and a sampling strategy that retains `[p1, p3, p5]`, the retained `p3`'s `delta_sog` still reflects `p3 - p2` (the *unsampled* previous ping), not `p3 - p1` (the previous *retained* ping) — information from a ping that is not actually present in the sampled training stream. This is a genuine discrepancy from the pipeline Part C specifies as correct, and is **not** merely a documentation gap — it is the single largest correction needed in this revision, since it affects every one of the 15 features (all except `mean_lat`/`mean_lon`/`mean_dist_port`/`std_dist_port`/`frac_near_port`, which don't depend on lag) for five of the six strategies at every rate below 100% (at rate=100%, the retained sequence *is* the full sequence, so no discrepancy exists there — this is also directly relevant to Q6 below).

---

## 4. Adaptive sampling — exact deltas used for event pings

`adaptive_sampling_all_tasks.py`, function `adaptive_sample()` (lines 33-52):

```python
SOG_THRESHOLD = 1.0   # knots
COG_THRESHOLD = 5.0   # degrees

events = df[
    (df['delta_sog'].abs() > SOG_THRESHOLD) |
    (df['delta_cog'].abs() > COG_THRESHOLD)
]
```

Exactly two columns, `delta_sog` and `delta_cog` (both pre-computed in Stage (a), against the full sequence — see Q2/3), combined with a logical OR. A ping qualifies as an "event" if its speed changed by more than 1.0 knot **or** its course changed by more than 5.0 degrees, relative to its immediately preceding ping in the *full, unsampled* per-vessel sequence.

Per Part C's explicit carve-out ("it is acceptable to use the complete cleaned TRAINING stream to determine which pings qualify as event pings"), this specific use of full-sequence deltas for **event identification** is already compliant with the corrected pipeline and does not need to change. What does need to change (per Part C steps 5-6) is that the resulting retained pings' **feature values** (`mean_abs_delta_sog`, `mean_abs_delta_cog`, etc., computed downstream in `aggregate_features()`) are currently still the same full-sequence-derived values, rather than being recomputed from the retained subsequence's own internal lags.

---

## 5. Exact current retention calculation across seeds

`sampling_utils.py`, inside `run_experiment()` (line 279):

```python
summary['f1_retention'] = summary['mean_f1'] / baseline_f1 * 100
```

where `summary['mean_f1']` is the **mean of F1 across the 5 seeds** for that (task, strategy, rate), computed a few lines earlier via `results_df.groupby('rate').agg(mean_f1=('f1','mean'), ...)`, and `baseline_f1` is one of the fixed module-level constants (line 37-47: `BASELINE_VC_F1 = 0.5962`, `BASELINE_FD_F1 = 0.5950`, `BASELINE_AD_F1 = 0.8653`, `BASELINE_ADT_F1 = 0.5723`) — **a single number shared across all 5 seeds, not a per-seed value.**

**This directly violates the corrected rule specified in Part D** ("Do NOT divide seed-level sampled scores by an averaged full-data F1"). The current code does exactly that: every seed's contribution is absorbed into a single mean before division, and the divisor itself is a fixed constant rather than that seed's own 100%-condition F1. This is confirmed, not assumed, and is the second major correction this revision requires (see Part D deliverables).

---

## 6. Why Figure 1 can have nonzero error bars at the 100% condition

At `rate == 1.0`, `run_experiment()` (line 231) uses `sampled = df_train` directly — no `sample_fn` is ever called, so the training data is bit-identical across all 5 seeds at 100%. However, the `RandomForestClassifier` is still constructed with `random_state=seed` (line 255-256) for `seed` in `range(5)`, and Random Forest training is stochastic even on identical input (bootstrap resampling per tree, random feature subsampling per split). Five different seeds therefore produce five slightly different fitted forests, hence five slightly different F1 scores on the fixed test set — nonzero variance at 100% is pure **model variance**, since sampling variance is exactly zero there by construction.

Under the *current* retention formula (Q5), dividing each of those 5 (slightly different) F1 values by the single fixed baseline constant produces 5 slightly different retention values, hence a nonzero SD at 100% (I measured this directly in earlier verification work: e.g. VC retention SD at 100% = 0.7 across all six strategies, ADT = 5.5). Under the **corrected** per-seed rule required by Part D — divide each seed's F1 by that *same seed's own* 100%-condition F1 — the 100% condition becomes `retention_seed = 100 × F1_seed / F1_seed = 100.0` **exactly**, for every seed, giving **SD = 0.0 by construction**. This is precisely why Part D states "For the 100% condition... retention = 100.0 for every seed, SD of retention = 0.0" and why Part J asks to drop the 100% condition from Figure 1 entirely — under the corrected formula it is a trivial, uninformative flat line at every strategy simultaneously, carrying zero information.

---

## 7. Exact source of the historical Random/ADT train-test split mismatch

**Source located: `anomaly_detection_by_vessel_type.py`, lines 113-121:**

```python
# ── 4. Train/test split by vessel ─────────────────────────────
# Stratified by type (unlike the pooled anomaly_detection_sampling.py) so
# every type is proportionally represented in both splits — needed since
# ground truth is now built per type.
all_mmsi = vessel_type_labels['mmsi'].values
mmsi_train, mmsi_test = train_test_split(
    all_mmsi, test_size=0.2, random_state=42,
    stratify=vessel_type_labels['type_group'].values
)
```

This is a **standalone, pre-unification script** (predates `sampling_utils.py` and the six current `*_sampling_all_tasks.py` scripts). Its own docstring (lines 1-39) states it implements the ADT anomaly-ground-truth construction, and explicitly: *"The random-sampling procedure used to test robustness afterward is the same simple random ping sampling as `anomaly_detection_sampling.py`."* — i.e., this file's own random-sampling numbers for the ADT task were computed against a **vessel-type-stratified** split, not the plain unstratified split (`train_test_split(all_mmsi, test_size=0.2, random_state=42)`, no `stratify=`) that `sampling_utils.load_and_split()` (line 79) uses for every strategy and every task in the current, unified pipeline.

**Why this affected only the (Random, ADT) case and nothing else:**
1. This script only ever computed the **ADT** task — it has no VC, FD, or pooled-AD logic at all. So the discrepancy is architecturally confined to ADT from the start.
2. Within ADT, this script implements only **plain random ping sampling** — it contains no Stratified/Spatial/Importance/Adaptive/Temporal sampling logic whatsoever. The other five strategies' ADT numbers were therefore *always* sourced exclusively from the current unified pipeline (`build_ad_by_type_labels()` + `run_experiment()`, both using the unstratified split), since no alternate implementation of those five strategies for ADT ever existed.
3. Only **Random** had two independent implementations for ADT that could disagree: this older standalone stratified-split script, and the newer `random_sampling_all_tasks.py` (unified, unstratified split). Every other (strategy, task) pair only ever had one implementation, so there was nothing for it to mismatch against.

**Confirmation the fix already exists and is documented in-code:** `random_sampling_all_tasks.py`, lines 8-20 (its own docstring):

> *"Unlike the original standalone random-sampling scripts (vessel_classification_sampling.py, fishing_detection_sampling.py, anomaly_detection_sampling.py, anomaly_detection_by_vessel_type.py), this version shares sampling_utils.py's config and infra with every other strategy script: the same load_and_split() train/test split, the same label builders, the same run_experiment() loop, and the same baseline constants... Running Random through the shared infra removes that mismatch: its 100% row now lands on exactly the same number as every other strategy's 100% row, because it's now genuinely the same computation."*

This confirms both the diagnosis above and that the fix (routing Random through the unified `sampling_utils.py` pipeline, same as all five other strategies) is already the current, in-place state of `random_sampling_all_tasks.py` — the standalone script above is legacy and is not read by any of the six current `*_sampling_all_tasks.py` scripts or by `final_paper_data.json`'s generation path (independently confirmed in prior verification work in this project: all 144 task×strategy×rate values in `final_paper_data.json` match direct recomputation from the six `_v2.csv` result files with zero discrepancies).

---

## Summary of corrections this audit shows are required

| # | Issue | Required by | Status |
|---|---|---|---|
| 1 | Lags/deltas computed once against the full sequence, never recomputed after sampling | Part C | **Confirmed broken** — affects 5 of 6 strategies, all rates below 100%, 15 of 18 features |
| 2 | Retention divides mean-of-5-seeds by a fixed baseline constant, not each seed's own matched full-data F1 | Part D | **Confirmed broken** — affects every task/strategy/rate cell in every published table and figure |
| 3 | Random/ADT split mismatch (historical) | Part M | **Already fixed** in the current codebase; legacy script identified and isolated, not part of the live pipeline |

Corrections 1 and 2 require rebuilding the feature-construction and retention-calculation logic (Parts C, N) and re-running the full experiment grid (Parts D through K) — this is the majority of the remaining work in this revision and is addressed next.
