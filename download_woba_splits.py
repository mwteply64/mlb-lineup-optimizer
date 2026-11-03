#!/usr/bin/env python3
"""
wOBA Splits Downloader - Baseball Lineup Optimization Project
=============================================================
Downloads wOBA splits vs LHP and RHP for custom date ranges,
using Baseball Savant Statcast data via pybaseball.

SETUP (one-time):
    pip install pybaseball pandas

USAGE:
    python download_woba_splits.py

OUTPUT:
    36 CSV files in the 'historical testing' subfolder, e.g.:
        before_june_1_2023_vs_left.csv
        after_aug_1_2025_vs_right.csv

NOTES:
    - Statcast 'player_name' is the PITCHER's name; batter names are
      resolved via the Chadwick player register (downloaded once, cached).
    - wOBA computed from woba_value / woba_denom (same as FanGraphs).
    - Statcast downloads are cached in ~/.pybaseball/cache/ so re-runs
      for the same date range are instant.
    - Script overwrites existing files if re-run.
"""

import os
import sys
import time
import warnings
import pandas as pd

# == Configuration ==

SAVE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "historical testing"
)

SEASON_DATES = {
    2023: {"start": "2023-03-30", "end": "2023-10-01"},
    2024: {"start": "2024-03-20", "end": "2024-09-29"},
    2025: {"start": "2025-03-27", "end": "2025-09-28"},
}

SPLITS = [
    ("before_june_1", lambda y: (SEASON_DATES[y]["start"], f"{y}-05-31")),
    ("after_june_1",  lambda y: (f"{y}-06-01", SEASON_DATES[y]["end"])),
    ("before_july_1", lambda y: (SEASON_DATES[y]["start"], f"{y}-06-30")),
    ("after_july_1",  lambda y: (f"{y}-07-01", SEASON_DATES[y]["end"])),
    ("before_aug_1",  lambda y: (SEASON_DATES[y]["start"], f"{y}-07-31")),
    ("after_aug_1",   lambda y: (f"{y}-08-01", SEASON_DATES[y]["end"])),
]

YEARS = [2023, 2024, 2025]
HANDEDNESS = [("vs_left", "L"), ("vs_right", "R")]

# -- Imports --------------------------------------------------------------------

try:
    from pybaseball import statcast, chadwick_register, cache
    cache.enable()
except ImportError:
    print("ERROR: pybaseball not installed.")
    print("Run: pip install pybaseball pandas")
    sys.exit(1)

# -- Player Name Lookup --------------------------------------------------------

def build_name_lookup():
    """
    Download the Chadwick player register once and return a dict:
        mlbam_id (int) -> (last_name, first_name)

    NOTE: In Statcast data, 'player_name' is the PITCHER's name.
    Batter names are not directly included; we look them up by MLBAM ID.
    """
    print("  Loading player name register ...", end="", flush=True)
    try:
        reg = chadwick_register()
        # Keep only rows with an MLBAM id and a name
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

# -- wOBA Aggregation -----------------------------------------------------------

def aggregate_woba(df, name_lookup):
    """
    Given a Statcast event DataFrame filtered to one pitcher handedness,
    return a per-batter summary: last_name, first_name, player_id, pa, woba.
    """
    # Keep only plate-appearance-ending events
    pa = df[df["events"].notna()].copy()

    if pa.empty:
        return pd.DataFrame(
            columns=["last_name", "first_name", "player_id", "pa", "woba"]
        )

    # Check required columns (batter ID and wOBA fields)
    missing = [c for c in ("batter", "woba_value", "woba_denom") if c not in pa.columns]
    if missing:
        raise KeyError(
            f"Statcast data missing columns: {missing}\n"
            "Try: pip install --upgrade pybaseball"
        )

    # Group by BATTER ID only - player_name is the pitcher's name in Statcast data
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

    # Look up batter names from the Chadwick register
    grouped["last_name"] = grouped["batter"].map(
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


# -- Main -----------------------------------------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    total_files = len(YEARS) * len(SPLITS) * len(HANDEDNESS)  # 36
    files_saved = 0

    print(f"Saving {total_files} CSV files to:\n  {SAVE_DIR}\n")

    # Load player name register once up front
    name_lookup = build_name_lookup()
    print()

    for year in YEARS:
        for split_label, date_func in SPLITS:
            start_date, end_date = date_func(year)

            print(f"[{year}] {split_label}  ({start_date} -> {end_date})")
            print(f"  Downloading Statcast data ...", end="", flush=True)

            try:
                raw = statcast(start_dt=start_date, end_dt=end_date)
            except Exception as e:
                print(f"\n  ERROR downloading: {e}")
                continue

            if raw is None or raw.empty:
                print(f"\n  WARNING: No data returned. Skipping.")
                continue

            # Keep regular season only (drops spring training / exhibition games
            # that Statcast sometimes includes at season boundaries).
            if "game_type" in raw.columns:
                before_gt = len(raw)
                raw = raw[raw["game_type"] == "R"]
                dropped_gt = before_gt - len(raw)
                if dropped_gt:
                    print(f" (filtered {dropped_gt} non-regular-season rows)", end="", flush=True)

            # Deduplicate: pybaseball can return duplicate rows near monthly chunk
            # boundaries. Drop exact pitch duplicates before aggregating.
            before = len(raw)
            raw = raw.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
            dupes = before - len(raw)
            if dupes:
                print(f" {before:,} events ({dupes} dupes removed - {len(raw):,})")
            else:
                print(f" {len(raw):,} events")

            for hand_label, hand_code in HANDEDNESS:
                filename = f"{split_label}_{year}_{hand_label}.csv"
                filepath = os.path.join(SAVE_DIR, filename)

                try:
                    hand_data = raw[raw["p_throws"] == hand_code]
                    result = aggregate_woba(hand_data, name_lookup)
                    result.to_csv(filepath, index=False)
                    files_saved += 1
                    print(f"  [OK] {filename}  ({len(result)} batters)")
                except Exception as e:
                    print(f"  [ERROR] {filename}  ERROR: {e}")

            time.sleep(1)

    print(f"\nDone. Saved {files_saved}/{total_files} files.")
    if files_saved < total_files:
        print("Re-run to retry any that failed (cached data reused for completed ranges).")


if __name__ == "__main__":
    main()
