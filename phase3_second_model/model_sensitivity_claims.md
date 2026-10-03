# Model Sensitivity Claims -- ADT, Random Forest vs. Logistic Regression (Part 10)

Evidence-limited language only: 'consistent', 'directionally similar', 'different', 'not clearly distinguishable given variability'. No significance claims.


## 18F


**1%:**
- RF: most retained = Stratified (23.4), least retained = Importance (0.0)
- LR: most retained = Stratified (94.5), least retained = Importance (64.5)
- Top strategy matches between RF and LR at this rate.
- Spearman rank agreement (descriptive only, not a significance test): rho = 0.83
- Does any strategy reach >=95% LR retention at this rate? No.
- LR retention spread across strategies: 29.9 points; mean per-strategy SD: 10.0 points -> spread exceeds typical SD (differences may be more than noise).

**5%:**
- RF: most retained = Random (54.5), least retained = Adaptive (33.6)
- LR: most retained = Stratified (106.1), least retained = Temporal (83.1)
- Top strategy differs between RF and LR at this rate.
- Spearman rank agreement (descriptive only, not a significance test): rho = 0.26
- Does any strategy reach >=95% LR retention at this rate? Yes -- Random, Stratified, Spatial
- LR retention spread across strategies: 23.0 points; mean per-strategy SD: 5.9 points -> spread exceeds typical SD (differences may be more than noise).

**10%:**
- RF: most retained = Temporal (79.9), least retained = Importance (36.8)
- LR: most retained = Random (111.9), least retained = Temporal (23.6)
- Top strategy differs between RF and LR at this rate.
- Spearman rank agreement (descriptive only, not a significance test): rho = 0.03
- Does any strategy reach >=95% LR retention at this rate? Yes -- Random, Stratified, Spatial, Importance
- LR retention spread across strategies: 88.3 points; mean per-strategy SD: 5.1 points -> spread exceeds typical SD (differences may be more than noise).

**25%:**
- RF: most retained = Spatial (74.7), least retained = Importance (40.5)
- LR: most retained = Random (117.0), least retained = Temporal (95.7)
- Top strategy differs between RF and LR at this rate.
- Spearman rank agreement (descriptive only, not a significance test): rho = 0.09
- Does any strategy reach >=95% LR retention at this rate? Yes -- Random, Stratified, Spatial, Importance, Adaptive, Temporal
- LR retention spread across strategies: 21.4 points; mean per-strategy SD: 3.1 points -> spread exceeds typical SD (differences may be more than noise).

**50%:**
- RF: most retained = Temporal (81.1), least retained = Spatial (53.3)
- LR: most retained = Temporal (116.5), least retained = Spatial (101.9)
- Top strategy matches between RF and LR at this rate.
- Spearman rank agreement (descriptive only, not a significance test): rho = 0.89
- Does any strategy reach >=95% LR retention at this rate? Yes -- Random, Stratified, Spatial, Importance, Adaptive, Temporal
- LR retention spread across strategies: 14.7 points; mean per-strategy SD: 3.1 points -> spread exceeds typical SD (differences may be more than noise).

## 16F


**1%:**
- RF: most retained = Stratified (42.2), least retained = Importance (22.4)
- LR: most retained = Temporal (87.1), least retained = Spatial (65.1)
- Top strategy differs between RF and LR at this rate.
- Spearman rank agreement (descriptive only, not a significance test): rho = 0.77
- Does any strategy reach >=95% LR retention at this rate? No.
- LR retention spread across strategies: 22.0 points; mean per-strategy SD: 9.1 points -> spread exceeds typical SD (differences may be more than noise).

**5%:**
- RF: most retained = Random (75.8), least retained = Adaptive (54.7)
- LR: most retained = Random (105.3), least retained = Adaptive (86.0)
- Top strategy matches between RF and LR at this rate.
- Spearman rank agreement (descriptive only, not a significance test): rho = 0.94
- Does any strategy reach >=95% LR retention at this rate? Yes -- Random, Stratified, Importance, Temporal
- LR retention spread across strategies: 19.2 points; mean per-strategy SD: 4.0 points -> spread exceeds typical SD (differences may be more than noise).

**10%:**
- RF: most retained = Temporal (88.2), least retained = Adaptive (53.4)
- LR: most retained = Temporal (106.5), least retained = Adaptive (87.5)
- Top strategy matches between RF and LR at this rate.
- Spearman rank agreement (descriptive only, not a significance test): rho = 0.71
- Does any strategy reach >=95% LR retention at this rate? Yes -- Random, Stratified, Spatial, Importance, Temporal
- LR retention spread across strategies: 19.0 points; mean per-strategy SD: 3.8 points -> spread exceeds typical SD (differences may be more than noise).

**25%:**
- RF: most retained = Random (94.5), least retained = Adaptive (57.6)
- LR: most retained = Importance (107.4), least retained = Temporal (99.6)
- Top strategy differs between RF and LR at this rate.
- Spearman rank agreement (descriptive only, not a significance test): rho = 0.31
- Does any strategy reach >=95% LR retention at this rate? Yes -- Random, Stratified, Spatial, Importance, Adaptive, Temporal
- LR retention spread across strategies: 7.8 points; mean per-strategy SD: 2.4 points -> spread exceeds typical SD (differences may be more than noise).

**50%:**
- RF: most retained = Temporal (99.3), least retained = Importance (85.9)
- LR: most retained = Importance (106.4), least retained = Temporal (99.1)
- Top strategy differs between RF and LR at this rate.
- Spearman rank agreement (descriptive only, not a significance test): rho = -0.94
- Does any strategy reach >=95% LR retention at this rate? Yes -- Random, Stratified, Spatial, Importance, Adaptive, Temporal
- LR retention spread across strategies: 7.3 points; mean per-strategy SD: 1.8 points -> spread exceeds typical SD (differences may be more than noise).

## Overall pattern

- Mean retention difference (LR - RF) across all non-100% conditions: 18F = +49.5, 16F = +26.2.
- Mean Spearman rank agreement across reduced rates: 18F = 0.42, 16F = 0.36 -- rank agreement varies substantially by rate (see ADT_RF_vs_Logistic_rank_agreement.csv for the full per-rate breakdown, which ranges from -0.94 to 0.94). This indicates strategy rankings are **not consistently reproduced** across the two models at every individual rate, even though the broad qualitative pattern (retention increasing with rate; ADT being difficult to recover from low samples) is directionally similar between RF and LR.
- The general finding that sampling-strategy effectiveness is task/rate-dependent, rather than fixed, remains observable under Logistic Regression -- retention still varies substantially by rate and by strategy for LR, similar in kind (though not identical in magnitude or exact ranking) to what Random Forest shows.