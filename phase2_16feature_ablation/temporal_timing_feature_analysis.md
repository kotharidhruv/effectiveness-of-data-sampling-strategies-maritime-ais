# Temporal-Specific Timing-Feature Analysis (Part 9)

All values: mean +/- SD across 5 seeds. Overlap = whether the +/-1 SD intervals of the 18F and 16F rows intersect.


## VC

| Rate | 18F abs. F1 | 16F abs. F1 | 18F retention | 16F retention | Retention diff (16F-18F) | Overlap |
|---|---|---|---|---|---|---|
| 1% | 0.4875 +/- 0.0171 | 0.4970 +/- 0.0176 | 82.2 +/- 2.6 | 87.1 +/- 3.8 | +4.9 | Yes |
| 5% | 0.4823 +/- 0.0147 | 0.4778 +/- 0.0048 | 81.3 +/- 2.7 | 83.7 +/- 1.1 | +2.4 | Yes |
| 10% | 0.4977 +/- 0.0138 | 0.4884 +/- 0.0042 | 83.9 +/- 3.3 | 85.6 +/- 1.4 | +1.6 | Yes |
| 25% | 0.5167 +/- 0.0102 | 0.5479 +/- 0.0044 | 87.1 +/- 1.5 | 96.0 +/- 1.0 | +8.9 | No |
| 50% | 0.5650 +/- 0.0116 | 0.5787 +/- 0.0063 | 95.3 +/- 2.1 | 101.4 +/- 2.2 | +6.2 | No |

## FD

| Rate | 18F abs. F1 | 16F abs. F1 | 18F retention | 16F retention | Retention diff (16F-18F) | Overlap |
|---|---|---|---|---|---|---|
| 1% | 0.2917 +/- 0.0535 | 0.2816 +/- 0.0463 | 53.2 +/- 9.8 | 53.2 +/- 8.6 | +0.0 | Yes |
| 5% | 0.2904 +/- 0.0381 | 0.3071 +/- 0.0325 | 52.9 +/- 7.5 | 58.0 +/- 6.0 | +5.0 | Yes |
| 10% | 0.3057 +/- 0.0166 | 0.3171 +/- 0.0070 | 55.7 +/- 4.3 | 59.9 +/- 1.0 | +4.1 | Yes |
| 25% | 0.4656 +/- 0.0428 | 0.4514 +/- 0.0138 | 84.9 +/- 9.2 | 85.3 +/- 3.2 | +0.4 | Yes |
| 50% | 0.5143 +/- 0.0212 | 0.5085 +/- 0.0158 | 93.7 +/- 5.5 | 96.0 +/- 3.2 | +2.3 | Yes |

## AD

| Rate | 18F abs. F1 | 16F abs. F1 | 18F retention | 16F retention | Retention diff (16F-18F) | Overlap |
|---|---|---|---|---|---|---|
| 1% | 0.6206 +/- 0.0098 | 0.8377 +/- 0.0106 | 69.1 +/- 1.2 | 89.6 +/- 1.7 | +20.5 | No |
| 5% | 0.7685 +/- 0.0621 | 0.8852 +/- 0.0458 | 85.6 +/- 6.8 | 94.7 +/- 4.9 | +9.1 | Yes |
| 10% | 0.7420 +/- 0.0354 | 0.9048 +/- 0.0193 | 82.7 +/- 4.4 | 96.8 +/- 2.2 | +14.1 | No |
| 25% | 0.8187 +/- 0.0315 | 0.9155 +/- 0.0116 | 91.2 +/- 3.1 | 97.9 +/- 2.0 | +6.8 | No |
| 50% | 0.9028 +/- 0.0095 | 0.9155 +/- 0.0202 | 100.6 +/- 1.6 | 97.9 +/- 3.0 | -2.6 | Yes |

## ADT

| Rate | 18F abs. F1 | 16F abs. F1 | 18F retention | 16F retention | Retention diff (16F-18F) | Overlap |
|---|---|---|---|---|---|---|
| 1% | 0.0579 +/- 0.0197 | 0.2332 +/- 0.0452 | 11.2 +/- 3.9 | 40.4 +/- 7.7 | +29.1 | No |
| 5% | 0.2302 +/- 0.0641 | 0.4050 +/- 0.0254 | 44.7 +/- 12.5 | 70.2 +/- 3.7 | +25.6 | No |
| 10% | 0.4120 +/- 0.0344 | 0.5081 +/- 0.0255 | 79.9 +/- 7.3 | 88.2 +/- 5.8 | +8.3 | Yes |
| 25% | 0.3413 +/- 0.0293 | 0.4911 +/- 0.0302 | 66.2 +/- 5.5 | 85.1 +/- 3.9 | +19.0 | No |
| 50% | 0.4179 +/- 0.0465 | 0.5729 +/- 0.0294 | 81.1 +/- 9.3 | 99.3 +/- 3.5 | +18.3 | No |

## AD / ADT — 16F_FIXED_18F_LABELS (Temporal), for comparison

| Task | Rate | 18F retention | 16F_FIXED_18F_LABELS retention | Difference |
|---|---|---|---|---|
| AD | 1% | 69.1 +/- 1.2 | 77.0 +/- 4.4 | +7.8 |
| AD | 5% | 85.6 +/- 6.8 | 86.7 +/- 1.9 | +1.1 |
| AD | 10% | 82.7 +/- 4.4 | 87.3 +/- 2.0 | +4.7 |
| AD | 25% | 91.2 +/- 3.1 | 93.2 +/- 1.3 | +2.1 |
| AD | 50% | 100.6 +/- 1.6 | 98.7 +/- 1.9 | -1.8 |
| ADT | 1% | 11.2 +/- 3.9 | 36.1 +/- 9.5 | +24.8 |
| ADT | 5% | 44.7 +/- 12.5 | 72.3 +/- 9.6 | +27.7 |
| ADT | 10% | 79.9 +/- 7.3 | 82.8 +/- 8.4 | +2.9 |
| ADT | 25% | 66.2 +/- 5.5 | 93.6 +/- 4.5 | +27.4 |
| ADT | 50% | 81.1 +/- 9.3 | 92.0 +/- 6.9 | +11.0 |

## Numerical answers

1. **Does Temporal performance decrease after removing timing features?** In 1 of 24 (task, rate) cells, 16F retention is lower than 18F; in 19 it is higher. Mean difference across all cells: +7.7 points.
2. **Does it increase?** Yes, in 19 cells (see above) -- the direction is not uniform.
3. **Is the difference small relative to seed variability?** In 15/24 (62%) of (task, rate) cells, the 18F and 16F +/-1 SD intervals overlap -- i.e. the difference is not distinguishable from seed-to-seed variability in the large majority of cells.
4. **Does Temporal retain any apparent advantage over competing strategies at 16 features?** See claim_stability_phase2.md for the per-task, per-rate comparison against the other five strategies under 16F.
5. **Do +/-1 SD intervals overlap?** 62% of cells overlap (see table above for the per-cell breakdown).