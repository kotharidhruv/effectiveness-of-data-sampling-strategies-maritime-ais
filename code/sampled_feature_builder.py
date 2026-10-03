"""
sampled_feature_builder.py
===========================
Corrected sampled-condition feature builder (Part C / N of the reviewer revision).

Fixes the issue documented in audit_report.md (Q2/Q3): the pre-correction pipeline
computed delta_sog / delta_cog / ping_gap ONCE against the full, unsampled per-vessel
ping sequence (in eda.ipynb), and every sampling strategy then simply carried those
stale, full-sequence-derived lag values through into whatever subset of rows it kept.

This module instead implements, for every strategy:

    select retained training pings -> sort by (mmsi, time) -> recompute lags
    WITHIN EACH VESSEL using ONLY the retained pings -> derive delta_sog/delta_cog/
    ping_gap from those recomputed lags -> drop each vessel's now-first ping (no
    prior retained ping to diff against) -> aggregate to 18 vessel-level features.

Adaptive sampling is the one documented exception (Part C carve-out): it is allowed
to use the ORIGINAL full-sequence delta_sog/delta_cog (already present in
gulf_features.parquet) to decide which pings qualify as "events", since a real
deployment could reasonably compute that decision online, ping-by-ping, before
deciding whether to keep or drop each one. Once pings are selected, this module
still recomputes every model feature from the retained subsequence's own lags,
exactly like every other strategy -- selection criterion and feature values are
kept strictly separate.

This file intentionally does NOT change any strategy's row-selection logic (which
vessel/zone/type/time-window/event budget each strategy uses) -- only what happens
to the rows AFTER they are selected. Row-selection logic is imported unchanged from
each strategy's own script where practical, or re-implemented identically where a
strategy script does not expose its selector as an importable function.
"""

import pandas as pd
import numpy as np

FEATURE_COLS_18 = [
    'mean_sog', 'std_sog', 'median_sog', 'max_sog', 'sog_range',
    'mean_abs_delta_sog', 'std_delta_sog',
    'mean_abs_delta_cog', 'std_delta_cog', 'max_abs_delta_cog',
    'mean_dist_port', 'std_dist_port', 'frac_near_port',
    'median_ping_gap', 'std_ping_gap',
    'frac_stationary',
    'mean_lat', 'mean_lon',
]

# 16-feature ablation set (Part H): drop the two ping-timing features.
TIMING_FEATURES = ['median_ping_gap', 'std_ping_gap']
FEATURE_COLS_16 = [c for c in FEATURE_COLS_18 if c not in TIMING_FEATURES]

# Raw (non-lag-derived) columns every strategy's row-selector is allowed to use.
# Explicitly excludes delta_sog/delta_cog/ping_gap/prev_* -- those are stale,
# full-sequence artifacts from eda.ipynb and must not leak into the corrected
# per-condition feature build.
RAW_COLS = ['mmsi', 'base_date_time', 'sog', 'cog', 'latitude', 'longitude',
            'dist_to_port', 'vessel_type_group']

# Adaptive is the one strategy whose SELECTION step is allowed to see
# full-sequence deltas (Part C carve-out). Unlike the pre-correction pipeline,
# these are NOT read from a pre-existing column (train_raw_prelag.parquet is
# pure pre-lag raw data -- it has no delta_sog/delta_cog at all). They are
# computed fresh, here, from the COMPLETE unsampled training stream, and
# critically WITHOUT dropping any row for a missing lag (Part 3: "do not
# remove raw rows merely because pre-sampling lag values are unavailable").
# A vessel's true first ping simply gets delta_sog/delta_cog = NaN, and
# NaN > threshold evaluates to False in pandas, so it is correctly treated as
# "stable" (not an event) rather than being dropped from the selection pool.
ADAPTIVE_SELECTION_COLS = RAW_COLS + ['delta_sog', 'delta_cog']

SOG_THRESHOLD = 1.0   # knots  (unchanged from adaptive_sampling_all_tasks.py)
COG_THRESHOLD = 5.0   # degrees


def compute_selection_deltas(df_train_full_raw: pd.DataFrame) -> pd.DataFrame:
    """
    For Adaptive's event-selection step ONLY. Computes delta_sog/delta_cog over
    the COMPLETE unsampled training stream, per vessel, sorted by time -- but
    does NOT drop any row (no dropna), unlike recompute_lags_and_derive(). A
    vessel's first ping gets NaN deltas (never an event); every other ping's
    delta reflects its true previous ping in the full training sequence, exactly
    as Part C's carve-out permits. Output still has every input row -- callers
    must not treat this as a filtering step.
    """
    df = df_train_full_raw[RAW_COLS].copy()
    df = df.sort_values(['mmsi', 'base_date_time']).reset_index(drop=True)
    prev_sog = df.groupby('mmsi')['sog'].shift(1)
    prev_cog = df.groupby('mmsi')['cog'].shift(1)
    df['delta_sog'] = df['sog'] - prev_sog
    df['delta_cog'] = ((df['cog'] - prev_cog + 180) % 360) - 180
    return df


def recompute_lags_and_derive(sampled_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Step: sort sampled pings -> recompute lags -> derive delta/gap features.

    `sampled_raw` must contain only RAW_COLS (or a superset that will be reduced
    to RAW_COLS internally) -- i.e. it must NOT carry any pre-existing
    delta_sog/delta_cog/ping_gap/prev_* column, so there is no possibility of a
    stale value silently surviving into the output.

    Returns the same pings, sorted, with delta_sog/delta_cog/ping_gap recomputed
    from ONLY this retained subsequence, and with each vessel's first ping in the
    retained subsequence dropped (mirroring eda.ipynb's original cleaning logic,
    now re-applied per sampled condition instead of once globally).
    """
    df = sampled_raw[RAW_COLS].copy()
    df = df.sort_values(['mmsi', 'base_date_time']).reset_index(drop=True)

    df['prev_sog'] = df.groupby('mmsi')['sog'].shift(1)
    df['prev_cog'] = df.groupby('mmsi')['cog'].shift(1)
    df['prev_time'] = df.groupby('mmsi')['base_date_time'].shift(1)

    df['delta_sog'] = df['sog'] - df['prev_sog']
    # Circular wraparound correction, identical formula to eda.ipynb cell 20.
    df['delta_cog'] = ((df['cog'] - df['prev_cog'] + 180) % 360) - 180
    df['ping_gap'] = (df['base_date_time'] - df['prev_time']).dt.total_seconds()

    df = df.dropna(subset=['prev_sog', 'prev_cog', 'prev_time']).copy()
    return df.drop(columns=['prev_sog', 'prev_cog', 'prev_time'])


def aggregate_features(ping_df: pd.DataFrame) -> pd.DataFrame:
    """Vessel-level aggregation -- identical formulas to sampling_utils.aggregate_features,
    operating here on RECOMPUTED (not stale) ping-level delta_sog/delta_cog/ping_gap."""
    return ping_df.groupby('mmsi').agg(
        mean_sog            = ('sog',          'mean'),
        std_sog             = ('sog',          'std'),
        median_sog          = ('sog',          'median'),
        max_sog             = ('sog',          'max'),
        sog_range           = ('sog',          lambda x: x.max() - x.min()),
        mean_abs_delta_sog  = ('delta_sog',    lambda x: x.abs().mean()),
        std_delta_sog       = ('delta_sog',    'std'),
        mean_abs_delta_cog  = ('delta_cog',    lambda x: x.abs().mean()),
        std_delta_cog       = ('delta_cog',    'std'),
        max_abs_delta_cog   = ('delta_cog',    lambda x: x.abs().max()),
        mean_dist_port      = ('dist_to_port', 'mean'),
        std_dist_port       = ('dist_to_port', 'std'),
        frac_near_port      = ('dist_to_port', lambda x: (x < 5).mean()),
        median_ping_gap     = ('ping_gap',     'median'),
        std_ping_gap        = ('ping_gap',     'std'),
        frac_stationary     = ('sog',          lambda x: (x < 0.5).mean()),
        mean_lat            = ('latitude',     'mean'),
        mean_lon            = ('longitude',    'mean'),
    ).reset_index()


# ---------------------------------------------------------------------------
# Strategy row-selectors. Selection CRITERIA are unchanged from the original
# scripts (random_sampling_all_tasks.py, stratified_..., spatial_..., etc.) --
# only their inputs/outputs are adapted to the raw-column contract above.
# ---------------------------------------------------------------------------

def select_random(df_train_raw, rate, seed):
    return df_train_raw.sample(frac=rate, random_state=seed)


def select_stratified(df_train_raw, rate, seed):
    # NOTE: a naive groupby('vessel_type_group').apply(lambda x: x.sample(...))
    # silently drops the vessel_type_group column from the result under this
    # project's pandas version (3.0.3) -- confirmed via a smoke test that
    # raised KeyError('vessel_type_group not in index') downstream. The
    # original (pre-correction) stratified_sample() never surfaced this
    # because its consumer (the old aggregate_features) never needed that
    # column to survive. Using an explicit per-group loop instead, matching
    # the pattern already used by select_importance/select_spatial, avoids
    # relying on groupby.apply's version-dependent column-inclusion behavior.
    parts = []
    for vtype, group in df_train_raw.groupby('vessel_type_group'):
        parts.append(group.sample(frac=rate, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def select_spatial(df_train_raw, rate, seed):
    port = df_train_raw[df_train_raw['dist_to_port'] < 5]
    coastal = df_train_raw[(df_train_raw['dist_to_port'] >= 5) & (df_train_raw['dist_to_port'] < 50)]
    offshore = df_train_raw[df_train_raw['dist_to_port'] >= 50]
    target = int(len(df_train_raw) * rate)
    n_port = min(len(port), int(target * 0.60))
    n_coastal = min(len(coastal), int(target * 0.30))
    n_offshore = min(len(offshore), int(target * 0.10))
    leftover = target - (n_port + n_coastal + n_offshore)
    if leftover > 0:
        extra_port = min(leftover, len(port) - n_port)
        n_port += extra_port
        leftover -= extra_port
        extra_coastal = min(leftover, len(coastal) - n_coastal)
        n_coastal += extra_coastal
    parts = []
    if n_port > 0: parts.append(port.sample(n=n_port, random_state=seed))
    if n_coastal > 0: parts.append(coastal.sample(n=n_coastal, random_state=seed))
    if n_offshore > 0: parts.append(offshore.sample(n=n_offshore, random_state=seed))
    return pd.concat(parts)


IMPORTANCE_WEIGHTS = {'fishing': 3.0, 'tanker': 2.0, 'cargo': 1.5, 'passenger': 1.0,
                      'towing_tug': 0.5, 'pleasure': 0.5, 'other': 0.3}


def select_importance(df_train_raw, rate, seed):
    target = int(len(df_train_raw) * rate)
    type_sizes = df_train_raw.groupby('vessel_type_group').size()
    weighted = {t: IMPORTANCE_WEIGHTS.get(t, 1.0) * n for t, n in type_sizes.items()}
    total_w = sum(weighted.values())
    parts = []
    for vtype, w_size in weighted.items():
        subset = df_train_raw[df_train_raw['vessel_type_group'] == vtype]
        n = min(len(subset), max(1, round(target * w_size / total_w)))
        parts.append(subset.sample(n=n, random_state=seed))
    result = pd.concat(parts)
    if len(result) > target:
        result = result.sample(n=target, random_state=seed)
    return result


def select_adaptive(df_train_with_full_deltas, rate, seed):
    """
    The ONE selector permitted to look at delta_sog/delta_cog (Part C carve-out),
    and only to decide which pings qualify as events -- not to build features.
    `df_train_with_full_deltas` must include ADAPTIVE_SELECTION_COLS.
    """
    events = df_train_with_full_deltas[
        (df_train_with_full_deltas['delta_sog'].abs() > SOG_THRESHOLD) |
        (df_train_with_full_deltas['delta_cog'].abs() > COG_THRESHOLD)
    ]
    stable = df_train_with_full_deltas.drop(events.index)
    target = int(len(df_train_with_full_deltas) * rate)
    if len(events) >= target:
        selected = events.sample(n=target, random_state=seed)
    else:
        n_stable = target - len(events)
        r_stable = min(1.0, n_stable / max(1, len(stable)))
        parts = [events]
        if r_stable > 0 and len(stable) > 0:
            parts.append(stable.sample(frac=r_stable, random_state=seed))
        selected = pd.concat(parts)
    # Drop the selection-only columns immediately -- everything downstream of
    # this function must only ever see RAW_COLS.
    return selected[RAW_COLS]


CANDIDATE_WINDOWS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30,
                     45, 60, 90, 120, 180, 240, 300, 360, 480, 720]


def find_temporal_window(df_train_raw, rate):
    target = int(len(df_train_raw) * rate)
    best_N, best_diff = 1, float('inf')
    for N in CANDIDATE_WINDOWS:
        tb = df_train_raw['base_date_time'].dt.floor(f'{N}min')
        n_pings = df_train_raw.groupby(['mmsi', tb]).ngroups
        diff = abs(n_pings - target)
        if diff < best_diff:
            best_diff, best_N = diff, N
        if n_pings < target * 0.5:
            break
    return best_N


def select_temporal(df_train_raw, N):
    df = df_train_raw.copy()
    df['tb'] = df['base_date_time'].dt.floor(f'{N}min')
    sampled = (df.sort_values('base_date_time').groupby(['mmsi', 'tb'])
               .first().reset_index().drop(columns=['tb']))
    return sampled


# ---------------------------------------------------------------------------
# Top-level entry point used by the experiment runner.
# ---------------------------------------------------------------------------

STRATEGY_SELECTORS = {
    'random': select_random,
    'stratified': select_stratified,
    'spatial': select_spatial,
    'importance': select_importance,
    # 'adaptive' and 'temporal' handled specially -- see build_sampled_features()
}


def build_sampled_features(strategy, df_train_full, rate, seed, feature_cols=None,
                            precomputed_selection_deltas=None, return_ping_count=False):
    """
    Full corrected pipeline for one (strategy, rate, seed) condition:
      select -> sort -> recompute lags -> derive -> aggregate -> return
      vessel-level feature dataframe (mmsi + feature_cols).

    `df_train_full` must be the FULL cleaned, already-vessel-split, PRE-LAG
    training ping set (RAW_COLS only -- e.g. train_raw_prelag.parquet). It must
    NOT carry any pre-existing delta_sog/delta_cog/ping_gap column.

    For strategy=='adaptive', full-sequence selection deltas are computed via
    compute_selection_deltas() (Part C carve-out). Since that computation is
    identical across every (rate, seed) sharing the same df_train_full, callers
    running many conditions should precompute it once with
    compute_selection_deltas(df_train_full) and pass it as
    `precomputed_selection_deltas` to avoid redundant recomputation.

    `rate >= 1.0` returns df_train_full directly for every strategy (no
    sampling function is invoked), matching the original pipeline's behavior.

    feature_cols: FEATURE_COLS_18 (default) or FEATURE_COLS_16 for the ablation.
    If return_ping_count=True, returns (agg_df, n_retained_pings) instead of
    just agg_df -- needed for condition_metrics' ping-retention accounting.
    """
    feature_cols = feature_cols or FEATURE_COLS_18

    if rate >= 1.0:
        selected_raw = df_train_full[RAW_COLS]
    elif strategy == 'adaptive':
        with_deltas = precomputed_selection_deltas
        if with_deltas is None:
            with_deltas = compute_selection_deltas(df_train_full)
        selected_raw = select_adaptive(with_deltas, rate, seed)
    elif strategy == 'temporal':
        N = find_temporal_window(df_train_full[RAW_COLS], rate)
        selected_raw = select_temporal(df_train_full[RAW_COLS], N)
    else:
        selector = STRATEGY_SELECTORS[strategy]
        selected_raw = selector(df_train_full[RAW_COLS], rate, seed)

    n_retained_pings = len(selected_raw)
    recomputed = recompute_lags_and_derive(selected_raw)
    agg = aggregate_features(recomputed)
    agg = agg.dropna(subset=[c for c in feature_cols if c in agg.columns])
    if return_ping_count:
        return agg, n_retained_pings
    return agg
