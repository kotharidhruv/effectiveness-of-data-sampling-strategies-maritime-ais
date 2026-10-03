# Retention Calculation Audit (Part 5)

## The old (pre-correction) formula

`sampling_utils.py`, inside `run_experiment()`, line 279:

```python
summary['f1_retention'] = summary['mean_f1'] / baseline_f1 * 100
```

where `summary['mean_f1']` is the mean of F1 **across the 5 seeds** for that
(task, strategy, rate), and `baseline_f1` is one fixed module-level constant per
task (`BASELINE_VC_F1 = 0.5962`, `BASELINE_FD_F1 = 0.5950`, `BASELINE_AD_F1 = 0.8653`,
`BASELINE_ADT_F1 = 0.5723`) — a single number shared identically across all 5 seeds,
not derived from any individual seed's own 100%-rate run.

This is a mean-first, cross-seed-averaged-baseline formula. It divides an
already-averaged sampled score by an already-fixed (and itself originally derived
from a separate averaging step) full-data score.

## The corrected formula (this revision)

```python
retention_seed = 100 * sampled_F1_seed / full_F1_same_seed
```

computed independently for each of the 5 seeds, where `full_F1_same_seed` is that
**exact same seed's** full-data (rate=100%) F1 for that task — not a cross-seed
average, not a fixed constant. Only after this per-seed division is the mean and SD
of `retention_seed` computed across the 5 seeds, for reporting.

Implemented in `run_phase1_experiment.py`: `full_f1_by_task_seed[(task, seed)]` is
looked up per condition and used as the denominator for that condition's own seed.

## Why the old Figure 1 could show nonzero error bars at the 100% condition

At `rate == 1.0`, no sampling function is ever invoked (`sampled = df_train`
directly) — the training data is bit-identical across all 5 seeds. However, the
`RandomForestClassifier` is still constructed with `random_state=seed` for each of
the 5 seeds, and Random Forest fitting is stochastic even on identical input data
(bootstrap resampling per tree, random feature subsampling per split). Five
different seeds therefore produce five slightly different fitted models and five
slightly different F1 scores on the fixed test set — this is pure **model
variance**, since sampling variance is exactly zero at 100% by construction.

Under the **old** formula, each of those 5 (slightly different) F1 values was
divided by the *same fixed baseline constant* — producing 5 slightly different
retention values, hence a nonzero SD at 100%. (Measured directly in this project's
prior verification work: e.g. VC retention SD at 100% = 0.7 across every strategy,
ADT = 5.5 — identical across all six strategies at 100%, because they all reduce to
the same unsampled training data divided by the same fixed constant.)

Under the **corrected** formula, each seed's retention is `100 * F1_seed / F1_seed`
— exactly 100.0, individually, for every seed, because the numerator and
denominator are now literally the same F1 value for that seed at rate=100%
(same training data, same seed, therefore the same fitted model, therefore the same
score, evaluated once and used as both "sampled" and "full-data" for that condition).
**SD of retention at 100% is therefore exactly 0.0 by construction, not an
empirical observation** — it cannot be otherwise under this formula.

## Verification performed

The corrected pipeline's actual output was checked, not assumed: every
(task, strategy, seed) row at `target_rate == 1.0` in
`condition_metrics_18_features.csv` was confirmed to have `retention_percent`
exactly equal to `100.0` (see `phase1_validation_checks.csv`, Checks B and C). If
this had not held exactly, Phase 1 execution would have stopped for debugging
before any table or figure was generated, per this revision's explicit instruction.

## Consequence for Figure 1

Since the 100% condition is now trivially `100.0 +/- 0.0` for every strategy
simultaneously under the corrected formula, it carries zero discriminating
information between strategies. This is the basis for removing it from the
corrected Figure 1 (Part 11) — not a stylistic choice, but a direct consequence of
the formula correction making that condition uninformative by construction.
