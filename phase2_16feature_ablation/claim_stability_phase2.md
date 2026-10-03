# Claim Stability After Phase 2 (Part 11)

All 16F values from `condition_summary_16_features.csv` (16F_RELABELED for AD/ADT); 18F values from Phase 1's validated `condition_summary_18_features.csv`.


## Temporal performance -- VC

| Rate | 18F mean +/- SD | 16F mean +/- SD | Diff | Overlap? |
|---|---|---|---|---|
| 1% | 82.2 +/- 2.6 | 87.1 +/- 3.8 | +4.9 | Yes |
| 5% | 81.3 +/- 2.7 | 83.7 +/- 1.1 | +2.4 | Yes |
| 10% | 83.9 +/- 3.3 | 85.6 +/- 1.4 | +1.6 | Yes |
| 25% | 87.1 +/- 1.5 | 96.0 +/- 1.0 | +8.9 | No |
| 50% | 95.3 +/- 2.1 | 101.4 +/- 2.2 | +6.2 | No |

**Does Temporal's 18F ranking among the six strategies survive at 16F for VC?**

- 1%: Temporal rank 3/6 at 16F (top strategy: Stratified at 87.6); Temporal = 87.1
- 5%: Temporal rank 5/6 at 16F (top strategy: Spatial at 92.2); Temporal = 83.7
- 10%: Temporal rank 5/6 at 16F (top strategy: Spatial at 96.2); Temporal = 85.6
- 25%: Temporal rank 4/6 at 16F (top strategy: Random at 98.6); Temporal = 96.0
- 50%: Temporal rank 1/6 at 16F (top strategy: Temporal at 101.4); Temporal = 101.4

## Temporal performance -- FD

| Rate | 18F mean +/- SD | 16F mean +/- SD | Diff | Overlap? |
|---|---|---|---|---|
| 1% | 53.2 +/- 9.8 | 53.2 +/- 8.6 | +0.0 | Yes |
| 5% | 52.9 +/- 7.5 | 58.0 +/- 6.0 | +5.0 | Yes |
| 10% | 55.7 +/- 4.3 | 59.9 +/- 1.0 | +4.1 | Yes |
| 25% | 84.9 +/- 9.2 | 85.3 +/- 3.2 | +0.4 | Yes |
| 50% | 93.7 +/- 5.5 | 96.0 +/- 3.2 | +2.3 | Yes |

**Does Temporal's 18F ranking among the six strategies survive at 16F for FD?**

- 1%: Temporal rank 4/6 at 16F (top strategy: Stratified at 86.3); Temporal = 53.2
- 5%: Temporal rank 5/6 at 16F (top strategy: Spatial at 82.3); Temporal = 58.0
- 10%: Temporal rank 6/6 at 16F (top strategy: Adaptive at 81.8); Temporal = 59.9
- 25%: Temporal rank 3/6 at 16F (top strategy: Stratified at 95.4); Temporal = 85.3
- 50%: Temporal rank 3/6 at 16F (top strategy: Stratified at 101.9); Temporal = 96.0

## Temporal performance -- AD

| Rate | 18F mean +/- SD | 16F mean +/- SD | Diff | Overlap? |
|---|---|---|---|---|
| 1% | 69.1 +/- 1.2 | 89.6 +/- 1.7 | +20.5 | No |
| 5% | 85.6 +/- 6.8 | 94.7 +/- 4.9 | +9.1 | Yes |
| 10% | 82.7 +/- 4.4 | 96.8 +/- 2.2 | +14.1 | No |
| 25% | 91.2 +/- 3.1 | 97.9 +/- 2.0 | +6.8 | No |
| 50% | 100.6 +/- 1.6 | 97.9 +/- 3.0 | -2.6 | Yes |

**Does Temporal's 18F ranking among the six strategies survive at 16F for AD?**

- 1%: Temporal rank 1/6 at 16F (top strategy: Temporal at 89.6); Temporal = 89.6
- 5%: Temporal rank 2/6 at 16F (top strategy: Importance at 94.9); Temporal = 94.7
- 10%: Temporal rank 1/6 at 16F (top strategy: Temporal at 96.8); Temporal = 96.8
- 25%: Temporal rank 1/6 at 16F (top strategy: Temporal at 97.9); Temporal = 97.9
- 50%: Temporal rank 2/6 at 16F (top strategy: Spatial at 98.7); Temporal = 97.9

## Temporal performance -- ADT

| Rate | 18F mean +/- SD | 16F mean +/- SD | Diff | Overlap? |
|---|---|---|---|---|
| 1% | 11.2 +/- 3.9 | 40.4 +/- 7.7 | +29.1 | No |
| 5% | 44.7 +/- 12.5 | 70.2 +/- 3.7 | +25.6 | No |
| 10% | 79.9 +/- 7.3 | 88.2 +/- 5.8 | +8.3 | Yes |
| 25% | 66.2 +/- 5.5 | 85.1 +/- 3.9 | +19.0 | No |
| 50% | 81.1 +/- 9.3 | 99.3 +/- 3.5 | +18.3 | No |

**Does Temporal's 18F ranking among the six strategies survive at 16F for ADT?**

- 1%: Temporal rank 2/6 at 16F (top strategy: Stratified at 42.2); Temporal = 40.4
- 5%: Temporal rank 3/6 at 16F (top strategy: Random at 75.8); Temporal = 70.2
- 10%: Temporal rank 1/6 at 16F (top strategy: Temporal at 88.2); Temporal = 88.2
- 25%: Temporal rank 4/6 at 16F (top strategy: Random at 94.5); Temporal = 85.1
- 50%: Temporal rank 1/6 at 16F (top strategy: Temporal at 99.3); Temporal = 99.3