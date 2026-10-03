# Analysis Notes — Label-Informed vs. Label-Agnostic Sampling (Part 13)

This note documents a methodological distinction in the study design: two of the six
sampling strategies use vessel type (the ground-truth label for VC, and the basis
for FD) while selecting which pings to retain; the other four do not.

## The distinction

**Label-informed strategies:**
- **Stratified** — samples each `vessel_type_group` independently at the same rate,
  so every type contributes proportionally to the sampled set (`select_stratified()`,
  `sampled_feature_builder.py`).
- **Importance** — allocates sampling budget per vessel type using fixed weights
  (fishing 3.0x, tanker 2.0x, cargo 1.5x, passenger 1.0x, towing_tug/pleasure 0.5x,
  other 0.3x) (`select_importance()`).

**Label-agnostic strategies:**
- **Random** — uniform draw over all training pings, no reference to vessel type.
- **Spatial** — budget allocated by distance-to-port zone only.
- **Adaptive** — budget allocated by speed/course-change magnitude only.
- **Temporal** — budget allocated by time-window bucketing only.

## Why this matters

- **Vessel type is the direct target label for VC** (7-class classification of
  `vessel_type_group`).
- **FD is derived from vessel type** (`is_fishing = (vessel_type_group == 'fishing')`).
- Stratified and Importance therefore have access to information correlated with
  (in FD's case, definitionally equal to a transformation of) the label they are
  later evaluated against for VC and FD, before sampling occurs. Random, Spatial,
  Adaptive, and Temporal select pings with zero reference to vessel type.
- For AD and ADT, this distinction is less direct: AD's ground truth does not use
  vessel type at all, while ADT's ground truth is defined per vessel type (though
  Stratified/Importance's *sampling* still doesn't use the anomaly label itself,
  only the type label, which for ADT is also used to define ground truth — a
  different, separate use of vessel type from what sampling does).

## What this note does NOT claim

This is a methodological distinction for the reader to weigh, not a validity
judgment. Stratified and Importance's use of vessel type does not invalidate their
retention numbers for VC/FD — it is a legitimate, explicitly-designed feature of
those two strategies (guaranteeing every type is represented, respectively
prioritizing analytically important types). The paper's tables should simply make
this categorization available so a reader forming conclusions about "why does
Strategy X do well on VC" can consider whether ping-selection information overlap
with the label is a plausible contributing factor, without this note asserting
that it is the actual explanation in any specific case.

## Category summary

| Category | Strategies |
|---|---|
| Label-informed | Stratified, Importance |
| Label-agnostic | Random, Spatial, Adaptive, Temporal |
