# Claim Stability Analysis (Part 12)

**Headline finding:** the lag-recomputation correction (Part 2/3) is not a small
numerical adjustment — it materially changes the shape of the results, especially
at low sampling rates, because delta-based features (speed-change, course-change)
computed from a sparse *retained* subsequence are much noisier than the same
features inherited from the *full* sequence (the pre-correction bug). Several of
the manuscript's central comparative claims do not survive this correction. This is
reported plainly below, claim by claim, with the actual corrected mean +/- SD and an
explicit overlap check — never as a significance test (none was run).

All numbers below are from `condition_summary_18_features.csv` (this corrected
Phase 1 run), retention_mean +/- retention_sd, 5 seeds per cell.

---

## Claim 1: "Temporal is the only strategy reaching 95% ADT retention below 100%"

**Verdict: REMOVE.**

Under the corrected pipeline, **no strategy reaches anywhere near 95% ADT
retention at any rate below 100%** — the single highest value in the entire ADT
grid below 100% is Temporal at 50% (81.1% +/- 9.3%), still 14 points short of the
95% bar. The claim as written is no longer true even for its one named exception.

| Strategy | ADT retention @ 50% |
|---|---|
| Temporal | 81.1 +/- 9.3 |
| Adaptive | 79.4 +/- 13.7 |
| Random | 70.6 +/- 8.6 |
| Stratified | 62.4 +/- 10.3 |
| Importance | 59.7 +/- 6.8 |
| Spatial | 53.3 +/- 16.0 |

Temporal and Adaptive overlap heavily at 50% (81.1 +/- 9.3 spans [71.8, 90.4];
79.4 +/- 13.7 spans [65.7, 93.1]) — even a *relative* "Temporal is best at 50%"
framing would not be clearly distinguishable from Adaptive at this rate.

## Claim 2: "Adaptive performs highest/strongest for pooled AD [at moderate-to-high rates]"

**Verdict: QUALIFY — no longer a clean single-strategy claim.**

| Strategy | AD retention @ 25% | AD retention @ 50% |
|---|---|---|
| Adaptive | 93.5 +/- 3.3 | 100.5 +/- 1.1 |
| Random | **94.8** +/- 3.1 | 98.1 +/- 2.4 |
| Temporal | 91.2 +/- 3.1 | **100.6** +/- 1.6 |
| Stratified | 90.5 +/- 3.1 | 98.2 +/- 3.4 |
| Spatial | 88.4 +/- 2.9 | 92.8 +/- 4.7 |
| Importance | 87.0 +/- 3.1 | 91.6 +/- 4.4 |

At 25%, Random's point estimate (94.8) is actually slightly *higher* than
Adaptive's (93.5), though their +/-1 SD intervals overlap substantially
([91.7, 97.9] vs [90.2, 96.8]) — not distinguishable. At 50%, Temporal's point
estimate (100.6) edges out Adaptive's (100.5) by a trivial amount, and their
intervals overlap almost completely. **Adaptive remains competitive and among the
top strategies at moderate-to-high AD rates, but "Adaptive is the strongest" is not
supportable as a clean, distinguishable ranking under the corrected pipeline** — at
best it is statistically tied with Random (25%) and Temporal (50%).

At 1%, the original claim of an early "worst-then-best" reversal is still directionally
present (Adaptive 14.4 +/- 18.7 is the lowest of all six at 1%, and its very high SD
makes even that ranking uncertain), but the magnitude is far more extreme under the
corrected pipeline than originally reported (was 60.7%, is now 14.4%).

## Claim 3: "Adaptive is among the weakest for ADT"

**Verdict: REMOVE — directionally reversed at moderate-to-high rates.**

| Strategy | ADT retention @ 25% | ADT retention @ 50% |
|---|---|---|
| Adaptive | 55.5 +/- 9.3 | **79.4** +/- 13.7 |
| Spatial | **74.7** +/- 11.2 | 53.3 +/- 16.0 |
| Random | 67.1 +/- 12.7 | 70.6 +/- 8.6 |
| Temporal | 66.2 +/- 5.5 | 81.1 +/- 9.3 |
| Stratified | 61.1 +/- 14.1 | 62.4 +/- 10.3 |
| Importance | 40.5 +/- 5.1 | 59.7 +/- 6.8 |

At 50%, Adaptive (79.4) is the **second-highest** of all six strategies, not the
weakest — Spatial and Importance are now the weakest. At 25%, Adaptive is
mid-to-low-ranked but not the extreme minimum (Importance is). At 10% and 1%,
Adaptive's ranking is closer to the original claim (near the bottom), but the very
large SDs at low rates (e.g. +/-20.2 at 10%) mean most pairwise strategy comparisons
at those rates are not clearly distinguishable at all. **This claim should be
removed rather than qualified** — the direction of the effect actually reverses
between low and moderate-to-high rates under the corrected pipeline, which is the
opposite of "consistently among the weakest."

## Claim 4: Random / Stratified / Spatial relative comparisons

**Verdict: QUALIFY across all four tasks — rankings among these three are mostly
not distinguishable given seed variability.**

Representative check, AD @ 10%: Random 84.7 +/- 4.4, Stratified 84.9 +/- 5.6,
Spatial 84.6 +/- 4.4 — all three point estimates fall within 0.3 points of each
other, well inside one another's +/-1 SD bands. The same near-total overlap holds
at most rates across VC, FD, AD, and ADT for this trio; where a point-estimate
ordering exists, it rarely survives an overlap check. Any manuscript language
implying a clear ranking among Random, Stratified, and Spatial specifically should
be qualified as "not clearly distinguishable given seed-to-seed variability" rather
than presented as a ranking.

## General note on VC and FD

Both tasks show far larger retention variability at low rates under the corrected
pipeline than originally reported — e.g. FD retention SDs of 10-24 points at
1-10% sampling are common (Random: 21.4 SD at 1%; Spatial: 23.6 SD at 5%). Any
claim comparing two strategies' *absolute position* at a specific low rate for
VC/FD should be treated as unstable unless the gap between their means clearly
exceeds the sum of their SDs — which is the exception, not the rule, at rates
below 25% in this corrected run.

## Summary table

| Claim | Task/Rate | A mean +/- SD | B mean +/- SD | Overlap? | Verdict |
|---|---|---|---|---|---|
| Temporal only ADT strategy >=95% below 100% | ADT, all rates <100% | Temporal max 81.1 +/- 9.3 @ 50% | (95% threshold) | N/A — max value itself is below 95 | **REMOVE** |
| Adaptive highest for AD | AD @ 25% | Adaptive 93.5 +/- 3.3 | Random 94.8 +/- 3.1 | Yes | **QUALIFY** |
| Adaptive highest for AD | AD @ 50% | Adaptive 100.5 +/- 1.1 | Temporal 100.6 +/- 1.6 | Yes | **QUALIFY** |
| Adaptive among weakest for ADT | ADT @ 50% | Adaptive 79.4 +/- 13.7 | Spatial 53.3 +/- 16.0 (weakest) | No — Adaptive is *higher* | **REMOVE** |
| Random/Stratified/Spatial ranking | AD @ 10% | 84.7 / 84.9 / 84.6 | all mutually overlapping | Yes | **QUALIFY** |

No statistical significance test was performed anywhere in this analysis; "overlap"
above refers strictly to whether the +/-1 SD intervals (5 seeds) intersect, per the
explicit instruction not to conduct or claim a significance test without separate
justification.
