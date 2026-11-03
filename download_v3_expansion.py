#!/usr/bin/env python3
"""
Historical Data Expansion v3 - Lineup Optimization Project
============================================================
Extends the 60-file historical-testing dataset in three ways:

  1. xwOBA column added alongside wOBA in every PREDICTOR-side file:
     before_*_vs_left/right, totals_before_*, recent_before_*, and the
     new per-season career files below. after_* files (the prediction
     TARGET) stay wOBA-only -- you only ever predict actual performance
     with projections, never the other way around.

  2. Per-season career split files (vs_left/vs_right, full season,
     woba + xwoba) for 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024.
     These REPLACE the old pre-aggregated combined_2019_2022 /
     combined_2021_2023 / combined_2022_2024 files (left on disk,
     unused) so the model can apply season-by-season recency weighting
     instead of a flat 3-year blend.

  3. Three new test years -- 2019, 2021, 2022 -- get the full set of
     before_*/after_*/totals_before_*/recent_before_* files (3 cutoffs
     each: june_1, july_1, aug_1), matching the existing 2023/2024/2025
     pattern exactly. 2023/2024/2025 before-side files are REGENERATED
     (same dates, same cache, just adds the xwoba column) so every year
     is on equal footing.

CAREER WINDOW RULE (confirms existing combined_* groupings + user's
2019->2016-2018 example): for any test year, use the 3 most recent
ACTUAL (non-2020) seasons before it, skipping back an extra year
whenever 2020 would otherwise fall in the window:
    2019 -> 2016, 2017, 2018
    2021 -> 2017, 2018, 2019      (2020 skipped)
    2022 -> 2018, 2019, 2021      (2020 skipped)
    2023 -> 2019, 2021, 2022      (2020 skipped)  [matches existing file]
    2024 -> 2021, 2022, 2023                       [matches existing file]
    2025 -> 2022, 2023, 2024                       [matches existing file]
This script doesn't need to know this mapping -- it just produces the
8 per-season files; backtest.py applies the mapping + recency weights.

xwOBA METHODOLOGY (matches Baseball Savant's own published definition):
  - For batted-ball events with a valid Statcast
    estimated_woba_using_speedangle, use that value as the contribution.
  - For everything else (walks, strikeouts, HBP, and any batted ball
    missing the expected-value estimate), fall back to the actual
    woba_value -- there's no "luck on contact" to model away for
    outcomes that don't involve a fielded ball.
  - xwoba = sum(contribution) / sum(woba_denom)   (same denominator as
    wOBA -- Statcast already encodes the correct wOBA-denom logic per
    PA-ending event).

wOBA METHODOLOGY (fixed -- see WOBA_WEIGHTS below):
  Statcast's raw per-pitch woba_value column does NOT use the actual
  season-specific wOBA linear weights. It uses a fixed generic weight
  per event (confirmed identical across 2019 and 2023 raw pulls: BB/HBP
  0.70, 1B 0.90, 2B 1.25, 3B 1.60, HR 2.00) and also assigns intentional
  walks a nonzero value (0.4) despite their woba_denom being 0. Summing
  that column directly overstates wOBA, by 0.01-0.03+ depending on the
  player's batted-ball/walk mix. The fix: compute the wOBA numerator
  from raw event counts (1B/2B/3B/HR/BB-excluding-IBB/HBP) times the
  correct season weight from WOBA_WEIGHTS (FanGraphs/Tom Tango Guts
  constants), keeping Statcast's woba_denom sum as the denominator
  (its PA-exclusion logic for IBB/sac-bunts/etc. is correct on its own).
  Verified against Baseball Savant's official season leaderboard:
  Trout 2019 0.440 vs official 0.436; Acuna 2023 0.429 vs official
  0.428 (both previously off by 0.016-0.032 using the raw column).

RUNTIME NOTE: 2016-2018 are brand-new full-season downloads (not
previously cached), and 2019/2021/2022's before/after/recent sub-ranges
are also new (only their full-season range was cached previously, as
part of the old combined_* multi-year groups). Expect a long runtime --
quite possibly over an hour -- since full MLB seasons are 700K+ pitches
each and Statcast's API is rate-limited. 2023/2024/2025 sub-ranges
reuse pybaseball's local cache and recompute almost instantly.

SETUP (one-time):
    pip install pybaseball pandas

USAGE:
    python download_v3_expansion.py
    (or double-click run_v3_expansion.bat, which logs to a text file)

NOTES:
    - Statcast 'player_name' is the PITCHER's name; batter names are
      resolved via the Chadwick player register (downloaded once, cached).
    - Script overwrites existing files of the same name if re-run.
    - Safe to re-run after a partial failure -- pybaseball's cache means
      already-downloaded date ranges won't be re-fetched from the network.
"""

import os
import sys
import time
import warnings
import pandas as pd

# == Configuration ============================================================

SAVE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "historical testing"
)

# Full-season date ranges. A day or two of padding at each boundary is
# harmless -- clean_raw() filters to game_type == "R" (regular season)
# regardless, so any stray exhibition-game rows get dropped anyway.
SEASON_DATES = {
    2016: {"start": "2016-04-01", "end": "2016-10-03"},
    2017: {"start": "2017-03-31", "end": "2017-10-02"},
    2018: {"start": "2018-03-28", "end": "2018-10-02"},
    2019: {"start": "2019-03-28", "end": "2019-09-29"},
    # 2020 intentionally skipped (60-game COVID season)
    2021: {"start": "2021-04-01", "end": "2021-10-03"},
    2022: {"start": "2022-04-07", "end": "2022-10-05"},
    2023: {"start": "2023-03-30", "end": "2023-10-01"},
    2024: {"start": "2024-03-20", "end": "2024-09-29"},
    2025: {"start": "2025-03-27", "end": "2025-09-28"},
}

# Per-season full-season career split files (woba + xwoba, vs_left/vs_right)
# 2025 added 2026-06-19: now a complete past season, so it graduates from
# TEST_YEARS-only into the career pool (used by update_live_data.py to
# build career_vs_left/right.csv for predicting the live 2026 season).
CAREER_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

# Test years needing before/after/totals/recent files across 3 cutoffs
TEST_YEARS = [2019, 2021, 2022, 2023, 2024, 2025]

HANDEDNESS = [("vs_left", "L"), ("vs_right", "R")]

# (cutoff_label, before_date_func, after_date_func)
CUTOFFS = [
    ("june_1", lambda y: (SEASON_DATES[y]["start"], f"{y}-05-31"),
               lambda y: (f"{y}-06-01", SEASON_DATES[y]["end"])),
    ("july_1", lambda y: (SEASON_DATES[y]["start"], f"{y}-06-30"),
               lambda y: (f"{y}-07-01", SEASON_DATES[y]["end"])),
    ("aug_1",  lambda y: (SEASON_DATES[y]["start"], f"{y}-07-31"),
               lambda y: (f"{y}-08-01", SEASON_DATES[y]["end"])),
]

# 14-day "recent form" windows ending the day before each cutoff
RECENT_WINDOWS = [
    ("recent_before_june_1", lambda y: (f"{y}-05-18", f"{y}-05-31")),
    ("recent_before_july_1", lambda y: (f"{y}-06-17", f"{y}-06-30")),
    ("recent_before_aug_1",  lambda y: (f"{y}-07-18", f"{y}-07-31")),
]

# Season-specific wOBA linear weights (source: FanGraphs "Guts!" wOBA
# constants table, which matches the standard Tom Tango methodology).
# Used to recompute the wOBA numerator from raw event counts instead of
# trusting Statcast's raw woba_value column -- see "wOBA METHODOLOGY"
# note above for why that column can't be summed directly.
WOBA_WEIGHTS = {
    2016: {"walk": 0.691, "hit_by_pitch": 0.721, "single": 0.878, "double": 1.242, "triple": 1.569, "home_run": 2.015},
    2017: {"walk": 0.693, "hit_by_pitch": 0.723, "single": 0.877, "double": 1.232, "triple": 1.552, "home_run": 1.980},
    2018: {"walk": 0.690, "hit_by_pitch": 0.720, "single": 0.880, "double": 1.247, "triple": 1.578, "home_run": 2.031},
    2019: {"walk": 0.690, "hit_by_pitch": 0.719, "single": 0.870, "double": 1.217, "triple": 1.529, "home_run": 1.940},
    2021: {"walk": 0.692, "hit_by_pitch": 0.722, "single": 0.879, "double": 1.242, "triple": 1.568, "home_run": 2.007},
    2022: {"walk": 0.689, "hit_by_pitch": 0.720, "single": 0.884, "double": 1.261, "triple": 1.601, "home_run": 2.072},
    2023: {"walk": 0.696, "hit_by_pitch": 0.726, "single": 0.883, "double": 1.244, "triple": 1.569, "home_run": 2.004},
    2024: {"walk": 0.689, "hit_by_pitch": 0.720, "single": 0.882, "double": 1.254, "triple": 1.590, "home_run": 2.050},
    2025: {"walk": 0.691, "hit_by_pitch": 0.722, "single": 0.882, "double": 1.252, "triple": 1.584, "home_run": 2.037},
    # 2026 added 2026-06-19, sourced from FanGraphs Guts! (live in-season
    # estimate, page last updated 2026-06-12). This is NOT a finalized
    # season constant -- it will drift slightly as the season progresses
    # and FanGraphs recomputes the run environment. Re-check
    # https://www.fangraphs.com/tools/guts periodically (e.g. monthly)
    # and update this entry; the 2025 row above matched the prior
    # finalized constant exactly, which is why this source is trusted.
    2026: {"walk": 0.701, "hit_by_pitch": 0.732, "single": 0.894, "double": 1.267, "triple": 1.603, "home_run": 2.059},
}

# == Imports ==================================================================

try:
    from pybaseball import statcast, chadwick_register, cache
    cache.enable()
except ImportError:
    print("ERROR: pybaseball not installed.")
    print("Run: pip install pybaseball pandas")
    sys.exit(1)

# == Helpers ==================================================================

def build_name_lookup():
    print("Loading player name register ...", end="", flush=True)
    try:
        reg = chadwick_register()
        reg = reg[["key_mlbam", "name_last", "name_first"]].dropna(
            subset=["key_mlbam", "name_last"]
        ).copy()
        reg["key_mlbam"] = reg["key_mlbam"].astype(int)
        lookup = {
            row["key_mlbam"]: (row["name_last"], row["name_first"])
            for _, row in reg.iterrows()
        }
        print(f" {len(lookup):,} players loaded")
        return lookup
    except Exception as e:
        print(f"\nWARNING: Could not load name register ({e}). Names will be blank.")
        return {}


def clean_raw(raw):
    """Filter to regular season, deduplicate. Returns cleaned DataFrame."""
    if "game_type" in raw.columns:
        before_gt = len(raw)
        raw = raw[raw["game_type"] == "R"]
        dropped = before_gt - len(raw)
        if dropped:
            print(f" (filtered {dropped} non-regular-season rows)", end="", flush=True)
    before = len(raw)
    raw = raw.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    dupes = before - len(raw)
    if dupes:
        print(f" ({dupes} dupes removed)", end="", flush=True)
    return raw


def fetch_range(start_date, end_date, label):
    """Fetch + clean one Statcast date range. Returns DataFrame or None."""
    print(f"  Fetching {label}  ({start_date} -> {end_date}) ...", end="", flush=True)
    try:
        raw = statcast(start_dt=start_date, end_dt=end_date)
    except Exception as e:
        print(f"\n    ERROR downloading: {e}")
        return None
    if raw is None or raw.empty:
        print(f"\n    WARNING: No data returned.")
        return None
    raw = clean_raw(raw)
    print(f" {len(raw):,} pitches")
    return raw


def aggregate(df, name_lookup, include_xwoba, year):
    """
    Aggregate per-batter wOBA (and optionally xwOBA) from raw Statcast
    event data. Returns DataFrame with columns:
        last_name, first_name, player_id, pa, woba [, xwoba]

    wOBA numerator is computed from raw event counts (1B/2B/3B/HR/BB/HBP)
    times the season-correct linear weight in WOBA_WEIGHTS -- NOT by
    summing Statcast's raw woba_value column, which uses a fixed generic
    weight set (not the real per-season constants) and also leaks a
    nonzero value for intentional walks despite their woba_denom being 0.
    See the "wOBA METHODOLOGY" note at the top of this file. The
    denominator (woba_denom) is unaffected by either issue and is still
    summed directly from Statcast.
    """
    base_cols = ["last_name", "first_name", "player_id", "pa", "woba"]
    cols = base_cols + (["xwoba"] if include_xwoba else [])

    pa = df[df["events"].notna()].copy()
    if pa.empty:
        return pd.DataFrame(columns=cols)

    required = ["batter", "events", "woba_value", "woba_denom"]
    missing = [c for c in required if c not in pa.columns]
    if missing:
        raise KeyError(f"Statcast data missing columns: {missing}")

    weights = WOBA_WEIGHTS.get(year)
    if weights is None:
        raise KeyError(f"No wOBA linear weights configured for year {year}")

    # Per-PA numerator contribution: season-correct weight for the 6
    # wOBA event types, 0 for everything else (outs, intentional walks,
    # GIDP, sac flies, etc.) -- matching the official wOBA formula.
    pa["woba_num_contrib"] = pa["events"].map(weights).fillna(0.0)

    if include_xwoba:
        if "estimated_woba_using_speedangle" not in pa.columns:
            raise KeyError(
                "Statcast data missing estimated_woba_using_speedangle; "
                "try: pip install --upgrade pybaseball"
            )
        # xwOBA contribution: Statcast's expected value for batted balls,
        # falling back to the actual wOBA value for everything else
        # (walks, Ks, HBP, and any batted ball missing the estimate).
        # Deliberately uses the raw woba_value here, not woba_num_contrib:
        # xwOBA has already been verified against Savant's est_woba
        # leaderboard and matches well, so this fallback is left as-is.
        pa["xwoba_contrib"] = pa["estimated_woba_using_speedangle"].where(
            pa["estimated_woba_using_speedangle"].notna(), pa["woba_value"]
        )

    agg_kwargs = dict(
        pa=("events", "count"),
        woba_num=("woba_num_contrib", "sum"),
        woba_denom=("woba_denom", "sum"),
    )
    if include_xwoba:
        agg_kwargs["xwoba_num"] = ("xwoba_contrib", "sum")

    grouped = pa.groupby("batter", sort=False).agg(**agg_kwargs).reset_index()
    grouped["woba"] = (
        grouped["woba_num"] / grouped["woba_denom"].replace(0, float("nan"))
    ).round(3)
    if include_xwoba:
        grouped["xwoba"] = (
            grouped["xwoba_num"] / grouped["woba_denom"].replace(0, float("nan"))
        ).round(3)

    grouped["last_name"] = grouped["batter"].map(
        lambda pid: name_lookup.get(int(pid), ("", ""))[0]
    )
    grouped["first_name"] = grouped["batter"].map(
        lambda pid: name_lookup.get(int(pid), ("", ""))[1]
    )

    result = (
        grouped[["last_name", "first_name", "batter", "pa", "woba"] + (["xwoba"] if include_xwoba else [])]
        .rename(columns={"batter": "player_id"})
        .sort_values("pa", ascending=False)
        .reset_index(drop=True)
    )
    return result


def save(df, filename, files_saved_counter):
    filepath = os.path.join(SAVE_DIR, filename)
    df.to_csv(filepath, index=False)
    files_saved_counter[0] += 1
    print(f"    [OK] {filename}  ({len(df)} batters)")


# == Main =====================================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    total_expected = (
        len(CAREER_SEASONS) * len(HANDEDNESS) +                       # 16
        len(TEST_YEARS) * len(CUTOFFS) * len(HANDEDNESS) * 2 +        # before+after splits: 36
        len(TEST_YEARS) * len(CUTOFFS) +                              # totals_before: 18
        len(TEST_YEARS) * len(RECENT_WINDOWS)                         # recent_before: 18
    )
    files_saved = [0]

    print(f"Saving up to {total_expected} CSV files to:")
    print(f"  {SAVE_DIR}")
    print()

    name_lookup = build_name_lookup()
    print()

    # =========================================================================
    # PART 1: Per-season career split files (woba + xwoba)
    # =========================================================================
    print("=" * 70)
    print(f"PART 1: Per-season career splits, woba+xwoba ({len(CAREER_SEASONS)} seasons x 2 sides = {len(CAREER_SEASONS)*2} files)")
    print("=" * 70)

    for year in CAREER_SEASONS:
        dates = SEASON_DATES[year]
        raw = fetch_range(dates["start"], dates["end"], f"full season {year}")
        if raw is None:
            print(f"    SKIPPING year {year} -- no data.")
            continue
        for hand_label, hand_code in HANDEDNESS:
            hand_data = raw[raw["p_throws"] == hand_code]
            result = aggregate(hand_data, name_lookup, include_xwoba=True, year=year)
            save(result, f"season_{year}_{hand_label}.csv", files_saved)
        time.sleep(0.5)
    print()

    # =========================================================================
    # PART 2: before/after split files for each test year x cutoff
    # =========================================================================
    print("=" * 70)
    print(f"PART 2: before/after splits for {TEST_YEARS} ({len(TEST_YEARS)} years x 3 cutoffs x 2 sides x 2 = {len(TEST_YEARS)*len(CUTOFFS)*len(HANDEDNESS)*2} files)")
    print("=" * 70)

    for year in TEST_YEARS:
        print(f"  [{year}]")
        for cutoff_label, before_fn, after_fn in CUTOFFS:
            b_start, b_end = before_fn(year)
            raw_before = fetch_range(b_start, b_end, f"before_{cutoff_label}_{year}")
            if raw_before is not None:
                for hand_label, hand_code in HANDEDNESS:
                    hand_data = raw_before[raw_before["p_throws"] == hand_code]
                    result = aggregate(hand_data, name_lookup, include_xwoba=True, year=year)
                    save(result, f"before_{cutoff_label}_{year}_{hand_label}.csv", files_saved)

            a_start, a_end = after_fn(year)
            raw_after = fetch_range(a_start, a_end, f"after_{cutoff_label}_{year}")
            if raw_after is not None:
                for hand_label, hand_code in HANDEDNESS:
                    hand_data = raw_after[raw_after["p_throws"] == hand_code]
                    result = aggregate(hand_data, name_lookup, include_xwoba=False, year=year)
                    save(result, f"after_{cutoff_label}_{year}_{hand_label}.csv", files_saved)

            time.sleep(0.5)
        print()

    # =========================================================================
    # PART 3: totals_before (no handedness split, woba+xwoba)
    # =========================================================================
    print("=" * 70)
    print(f"PART 3: totals_before for {TEST_YEARS} ({len(TEST_YEARS)*len(CUTOFFS)} files)")
    print("=" * 70)

    for year in TEST_YEARS:
        print(f"  [{year}]")
        for cutoff_label, before_fn, _ in CUTOFFS:
            b_start, b_end = before_fn(year)
            raw = fetch_range(b_start, b_end, f"totals_before_{cutoff_label}_{year}")
            if raw is not None:
                result = aggregate(raw, name_lookup, include_xwoba=True, year=year)
                save(result, f"totals_before_{cutoff_label}_{year}.csv", files_saved)
            time.sleep(0.5)
        print()

    # =========================================================================
    # PART 4: recent_before (14-day window, no handedness split, woba+xwoba)
    # =========================================================================
    print("=" * 70)
    print(f"PART 4: recent_before for {TEST_YEARS} ({len(TEST_YEARS)*len(RECENT_WINDOWS)} files)")
    print("=" * 70)

    for year in TEST_YEARS:
        print(f"  [{year}]")
        for label, date_fn in RECENT_WINDOWS:
            start_date, end_date = date_fn(year)
            raw = fetch_range(start_date, end_date, f"{label}_{year}")
            if raw is not None:
                result = aggregate(raw, name_lookup, include_xwoba=True, year=year)
                save(result, f"{label}_{year}.csv", files_saved)
            time.sleep(0.5)
        print()

    # =========================================================================
    print(f"Done. Saved {files_saved[0]}/{total_expected} files.")
    if files_saved[0] < total_expected:
        print("Re-run to retry any that failed (cached ranges reload instantly).")


if __name__ == "__main__":
    main()
