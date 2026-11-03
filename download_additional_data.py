#!/usr/bin/env python3
"""
Additional Data Downloader - Baseball Lineup Optimization Project
=================================================================
Generates 24 new CSV files using the same Statcast source as the
original 36 split files, ensuring consistent data across the project.

OUTPUT: All files saved to the 'historical testing' subfolder.

PART 1 - Multi-year combined splits (6 files):
    Combined wOBA vs LHP and vs RHP across multiple seasons.
    Useful for multi-season stability estimates.

    combined_2019_2022_vs_left.csv   -- 2019, 2021, 2022 (skips COVID 2020)
    combined_2019_2022_vs_right.csv
    combined_2021_2023_vs_left.csv   -- 2021, 2022, 2023
    combined_2021_2023_vs_right.csv
    combined_2022_2024_vs_left.csv   -- 2022, 2023, 2024
    combined_2022_2024_vs_right.csv

PART 2 - Full-season totals up to cutoff dates (9 files):
    Combined (both LHP and RHP) wOBA from season start through each
    cutoff date. Same methodology as split files but no handedness filter.

    totals_before_june_1_2023.csv
    totals_before_july_1_2023.csv
    totals_before_aug_1_2023.csv
    totals_before_june_1_2024.csv
    ...etc...

PART 3 - Recent 14-day totals before each cutoff (9 files):
    Combined wOBA from the 14 days immediately before each cutoff date.
    Used as the "recent form" component in the backtest model.

    recent_before_june_1_2023.csv    -- May 18-31
    recent_before_july_1_2023.csv    -- Jun 17-30
    recent_before_aug_1_2023.csv     -- Jul 18-31
    recent_before_june_1_2024.csv
    ...etc...

NOTES:
    - 2020 is skipped (60-game COVID season; 2019 added instead).
    - Statcast cache reused for 2023/2024/2025 date ranges already
      downloaded by the original script. New downloads for 2019/2021/2022.
    - All files: last_name, first_name, player_id, pa, woba (same format
      as the 36 split files).
    - wOBA computed as sum(woba_value) / sum(woba_denom) from raw pitches,
      identical to the original script.
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

# Full-season date ranges
SEASON_DATES = {
    2019: {"start": "2019-03-28", "end": "2019-09-29"},
    # 2020 intentionally skipped (COVID-shortened 60-game season)
    2021: {"start": "2021-04-01", "end": "2021-10-03"},
    2022: {"start": "2022-04-07", "end": "2022-10-05"},
    2023: {"start": "2023-03-30", "end": "2023-10-01"},
    2024: {"start": "2024-03-20", "end": "2024-09-29"},
    2025: {"start": "2025-03-27", "end": "2025-09-28"},
}

# Multi-year groups: (file_label, [years to combine])
# Note: "2019_2022" uses 2019+2021+2022, skipping 2020
MULTI_YEAR_GROUPS = [
    ("2019_2022", [2019, 2021, 2022]),
    ("2021_2023", [2021, 2022, 2023]),
    ("2022_2024", [2022, 2023, 2024]),
]

# Cutoff splits for full-season totals
CUTOFF_SPLITS = [
    ("before_june_1", lambda y: (SEASON_DATES[y]["start"], f"{y}-05-31")),
    ("before_july_1", lambda y: (SEASON_DATES[y]["start"], f"{y}-06-30")),
    ("before_aug_1",  lambda y: (SEASON_DATES[y]["start"], f"{y}-07-31")),
]

TOTALS_YEARS   = [2023, 2024, 2025]
HANDEDNESS     = [("vs_left", "L"), ("vs_right", "R")]

# 14-day windows ending the day before each cutoff
# (start date is 13 days before the last day of the pre-period, inclusive)
RECENT_SPLITS = [
    ("recent_before_june_1", lambda y: (f"{y}-05-18", f"{y}-05-31")),
    ("recent_before_july_1", lambda y: (f"{y}-06-17", f"{y}-06-30")),
    ("recent_before_aug_1",  lambda y: (f"{y}-07-18", f"{y}-07-31")),
]

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
    print("  Loading player name register ...", end="", flush=True)
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
        print(f"\n  WARNING: Could not load name register ({e}). Names will be blank.")
        return {}


def clean_raw(raw, label=""):
    """Filter to regular season, deduplicate. Returns cleaned DataFrame."""
    if "game_type" in raw.columns:
        before_gt = len(raw)
        raw = raw[raw["game_type"] == "R"]
        dropped = before_gt - len(raw)
        if dropped:
            print(f"    filtered {dropped} non-regular-season rows", end="", flush=True)
    before = len(raw)
    raw = raw.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    dupes = before - len(raw)
    if dupes:
        print(f"    {dupes} dupes removed", end="", flush=True)
    return raw


def aggregate_woba(df, name_lookup):
    """
    Aggregate per-batter wOBA from raw Statcast event data.
    Returns DataFrame with columns: last_name, first_name, player_id, pa, woba.
    """
    pa = df[df["events"].notna()].copy()
    if pa.empty:
        return pd.DataFrame(columns=["last_name", "first_name", "player_id", "pa", "woba"])

    missing = [c for c in ("batter", "woba_value", "woba_denom") if c not in pa.columns]
    if missing:
        raise KeyError(f"Statcast data missing columns: {missing}")

    grouped = (
        pa.groupby("batter", sort=False)
        .agg(
            pa=("events", "count"),
            woba_num=("woba_value", "sum"),
            woba_denom=("woba_denom", "sum"),
        )
        .reset_index()
    )
    grouped["woba"] = (
        grouped["woba_num"] / grouped["woba_denom"].replace(0, float("nan"))
    ).round(3)
    grouped["last_name"]  = grouped["batter"].map(
        lambda pid: name_lookup.get(int(pid), ("", ""))[0]
    )
    grouped["first_name"] = grouped["batter"].map(
        lambda pid: name_lookup.get(int(pid), ("", ""))[1]
    )
    result = (
        grouped[["last_name", "first_name", "batter", "pa", "woba"]]
        .rename(columns={"batter": "player_id"})
        .sort_values("pa", ascending=False)
        .reset_index(drop=True)
    )
    return result


def download_season(year):
    """Download and clean a full season. Returns DataFrame or None on error."""
    dates = SEASON_DATES[year]
    print(f"    Downloading {year} ({dates['start']} -> {dates['end']}) ...", end="", flush=True)
    try:
        raw = statcast(start_dt=dates["start"], end_dt=dates["end"])
    except Exception as e:
        print(f"\n    ERROR: {e}")
        return None
    if raw is None or raw.empty:
        print(f"\n    WARNING: No data returned.")
        return None
    raw = clean_raw(raw)
    print(f" {len(raw):,} pitches")
    return raw


# == Main =====================================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    total_expected = (len(MULTI_YEAR_GROUPS) * len(HANDEDNESS) +
                      len(CUTOFF_SPLITS) * len(TOTALS_YEARS) +
                      len(RECENT_SPLITS) * len(TOTALS_YEARS))  # 6 + 9 + 9 = 24
    files_saved = 0

    print(f"Saving 24 additional CSV files to:")
    print(f"  {SAVE_DIR}")
    print()

    name_lookup = build_name_lookup()
    print()

    # =========================================================================
    # PART 1: Multi-year combined splits
    # =========================================================================
    print("=" * 60)
    print("PART 1: Multi-year combined splits (6 files)")
    print("=" * 60)
    print()

    # Identify all unique years across all groups and download each once
    unique_years = sorted(set(y for _, years in MULTI_YEAR_GROUPS for y in years))
    print(f"  Seasons needed: {unique_years}")
    print(f"  (2020 is intentionally skipped; 2019 added to compensate)")
    print(f"  (2023/2024 should load from cache - already downloaded)")
    print()

    season_data = {}
    for year in unique_years:
        df = download_season(year)
        season_data[year] = df
        time.sleep(0.5)

    print()

    for group_label, years in MULTI_YEAR_GROUPS:
        year_str = ", ".join(str(y) for y in years)
        print(f"  Group: combined_{group_label}  (years: {year_str})")

        frames = [season_data[y] for y in years
                  if y in season_data and season_data[y] is not None]
        if not frames:
            print("    No data available. Skipping.")
            continue

        combined = pd.concat(frames, ignore_index=True)
        # game_pk values are unique per game and don't repeat across years,
        # so no cross-year duplicate risk, but deduplicate within each year's
        # data just to be safe.
        before = len(combined)
        combined = combined.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
        print(f"    Combined: {len(combined):,} pitches ({before - len(combined)} dupes removed)")

        for hand_label, hand_code in HANDEDNESS:
            filename = f"combined_{group_label}_{hand_label}.csv"
            filepath = os.path.join(SAVE_DIR, filename)
            try:
                hand_data = combined[combined["p_throws"] == hand_code]
                result = aggregate_woba(hand_data, name_lookup)
                result.to_csv(filepath, index=False)
                files_saved += 1
                print(f"    [OK] {filename}  ({len(result)} batters)")
            except Exception as e:
                print(f"    [ERROR] {filename}: {e}")

        print()

    # =========================================================================
    # PART 2: Full-season totals up to cutoff dates
    # =========================================================================
    print("=" * 60)
    print("PART 2: Full-season totals up to cutoff dates (9 files)")
    print("=" * 60)
    print("  (No handedness split -- combines all pitcher types)")
    print("  (2023/2024/2025 date ranges cached from original script)")
    print()

    for year in TOTALS_YEARS:
        print(f"  [{year}]")
        for split_label, date_func in CUTOFF_SPLITS:
            start_date, end_date = date_func(year)
            filename = f"totals_{split_label}_{year}.csv"
            filepath = os.path.join(SAVE_DIR, filename)

            print(f"    {split_label}  ({start_date} -> {end_date})", end="", flush=True)

            try:
                raw = statcast(start_dt=start_date, end_dt=end_date)
            except Exception as e:
                print(f"\n    ERROR downloading: {e}")
                continue

            if raw is None or raw.empty:
                print(f"\n    WARNING: No data returned. Skipping.")
                continue

            raw = clean_raw(raw)
            print(f" -- {len(raw):,} pitches")

            try:
                result = aggregate_woba(raw, name_lookup)
                result.to_csv(filepath, index=False)
                files_saved += 1
                print(f"    [OK] {filename}  ({len(result)} batters)")
            except Exception as e:
                print(f"    [ERROR] {filename}: {e}")

            time.sleep(0.5)

        print()

    # =========================================================================
    # PART 3: Recent 14-day totals before each cutoff
    # =========================================================================
    print("=" * 60)
    print("PART 3: Recent 14-day totals before each cutoff (9 files)")
    print("=" * 60)
    print("  (New downloads -- 14-day windows not previously cached)")
    print()

    for year in TOTALS_YEARS:
        print(f"  [{year}]")
        for split_label, date_func in RECENT_SPLITS:
            start_date, end_date = date_func(year)
            filename = f"{split_label}_{year}.csv"
            filepath = os.path.join(SAVE_DIR, filename)

            print(f"    {split_label}  ({start_date} -> {end_date})", end="", flush=True)

            try:
                raw = statcast(start_dt=start_date, end_dt=end_date)
            except Exception as e:
                print(f"\n    ERROR downloading: {e}")
                continue

            if raw is None or raw.empty:
                print(f"\n    WARNING: No data returned. Skipping.")
                continue

            raw = clean_raw(raw)
            print(f" -- {len(raw):,} pitches")

            try:
                result = aggregate_woba(raw, name_lookup)
                result.to_csv(filepath, index=False)
                files_saved += 1
                print(f"    [OK] {filename}  ({len(result)} batters)")
            except Exception as e:
                print(f"    [ERROR] {filename}: {e}")

            time.sleep(0.5)

        print()

    # =========================================================================
    print(f"Done. Saved {files_saved}/{total_expected} files.")
    if files_saved < total_expected:
        print("Re-run to retry failed files (cached data reused for completed ranges).")


if __name__ == "__main__":
    main()
