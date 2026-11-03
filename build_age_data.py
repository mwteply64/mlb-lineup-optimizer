#!/usr/bin/env python3
"""
Aging Curve Project - Step 1: Player Age Lookup
=================================================
Pulls the Chadwick Bureau player register DIRECTLY from its source CSVs on
GitHub (raw.githubusercontent.com/chadwickbureau/register), rather than via
pybaseball's chadwick_register() wrapper. That wrapper is also used elsewhere
in this project (download_woba_splits.py / download_additional_data.py) for
batter name lookups, but on the installed pybaseball version it turned out
to return only 8 trimmed columns (key_mlbam, key_fangraphs, names, played
years) -- no birth date fields at all. The raw register CSVs it's built from
still carry full birth dates (birth_year/birth_month/birth_day) keyed by the
exact same key_mlbam id that appears as "MLBAMID" in every CSV this project
produces, so this script reads those files itself instead.

WHAT THIS SCRIPT DOES
----------------------
1. Downloads the 16 register data files (people-0.csv ... people-9.csv,
   people-a.csv ... people-f.csv -- the register is split into 16 chunks by
   the first hex digit of an internal UUID) directly from GitHub, keeping
   only the columns this script needs.
2. Keeps rows that have both an MLBAM id and a complete birth date.
3. For every season this project uses, computes each player's "baseball
   age" as of June 30 of that season -- the same reference-date convention
   Baseball-Reference and FanGraphs use for their own "Age" columns. (This
   is a convention choice, not a measured quantity: it only affects which
   side of a birthday a player lands on for players born in the back half
   of the year. It doesn't need to be exact to the day for curve-building
   purposes, just consistent.)
4. Writes player_ages.csv: one row per player, with mlbam_id, fangraphs_id
   (key_fangraphs, as a backup join key against the "PlayerId" column FYI
   -- not the primary key, MLBAMID is), birth date, and one age_<season>
   column per season.

This file is meant to be joined onto any existing project file via:
    existing_df.merge(ages_df, left_on="MLBAMID", right_on="mlbam_id")

WHY THIS NEEDS TO RUN LOCALLY
-------------------------------
This assistant's own sandbox has no general internet access (outbound
requests are allowlist-restricted), so the download has to happen on your
machine, which already has open internet access and pandas installed (same
as everything else in this project). Total download is 16 files, a few MB
each -- should take well under a minute on a normal connection.

USAGE
-----
    python build_age_data.py
    (or double-click run_build_age_data.bat, which logs to console + pauses)
"""
import sys
import datetime

import pandas as pd

REGISTER_BASE_URL = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/"
REGISTER_SUFFIXES = list("0123456789abcdef")
REGISTER_COLUMNS = ["key_mlbam", "key_fangraphs", "name_last", "name_first",
                     "birth_year", "birth_month", "birth_day"]

# Every season this project currently uses anywhere (career windows, test
# years, the live 2026 projection year). 2020 is intentionally excluded --
# the rest of the codebase already treats it as a skip-year (shortened/
# anomalous COVID season), and the aging-curve step will too.
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]

# Standard "baseball age" reference date: age as of June 30 of the season
# (matches Baseball-Reference's and FanGraphs' own "Age" column convention).
REF_MONTH, REF_DAY = 6, 30

OUTPUT_FILE = "player_ages.csv"


def compute_age(birth_year, birth_month, birth_day, season):
    """Age as of REF_MONTH/REF_DAY of `season`, given a full birth date."""
    if pd.isna(birth_year) or pd.isna(birth_month) or pd.isna(birth_day):
        return None
    try:
        birth_date = datetime.date(int(birth_year), int(birth_month), int(birth_day))
        ref_date = datetime.date(season, REF_MONTH, REF_DAY)
    except ValueError:
        # Handles malformed register rows (e.g. day=0 or out-of-range values
        # that occasionally show up in historical/incomplete entries).
        return None
    age = ref_date.year - birth_date.year
    if (ref_date.month, ref_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age


def load_register():
    """Download and concatenate all 16 register chunk files."""
    frames = []
    for i, suffix in enumerate(REGISTER_SUFFIXES, start=1):
        url = f"{REGISTER_BASE_URL}people-{suffix}.csv"
        print(f"  [{i}/{len(REGISTER_SUFFIXES)}] {url}")
        try:
            chunk = pd.read_csv(url, usecols=REGISTER_COLUMNS)
        except Exception as e:
            print(f"ERROR: failed to download {url}")
            print(f"  {type(e).__name__}: {e}")
            sys.exit(1)
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


def main():
    print("Downloading Chadwick Bureau player register (16 files)...")
    reg = load_register()
    print(f"  {len(reg):,} total register entries loaded")

    reg = reg.dropna(subset=["key_mlbam", "birth_year", "birth_month", "birth_day"]).copy()
    reg["key_mlbam"] = reg["key_mlbam"].astype(int)
    print(f"  {len(reg):,} entries have both an MLBAM id and a complete birth date")

    out = reg[[
        "key_mlbam", "key_fangraphs", "name_last", "name_first",
        "birth_year", "birth_month", "birth_day",
    ]].copy()
    out = out.rename(columns={"key_mlbam": "mlbam_id", "key_fangraphs": "fangraphs_id"})

    # The register has occasional duplicate key_mlbam rows (rare data-entry
    # dupes upstream). Keep the first and drop the rest so the join below
    # stays one-row-per-player.
    before = len(out)
    out = out.drop_duplicates(subset=["mlbam_id"], keep="first")
    if len(out) != before:
        print(f"  Dropped {before - len(out)} duplicate mlbam_id rows")

    for season in SEASONS:
        out[f"age_{season}"] = out.apply(
            lambda r: compute_age(r["birth_year"], r["birth_month"], r["birth_day"], season),
            axis=1,
        )

    out.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(out):,} players with ages for seasons {SEASONS}")
    print(f"  -> {OUTPUT_FILE}")

    # Quick sanity check on a few well-known players, if present, so you can
    # eyeball this before it gets used for anything.
    sample_ids = {
        "Bobby Witt Jr.": 677951,
        "Mike Trout": 545361,
        "Aaron Judge": 592450,
        "Shohei Ohtani": 660271,
    }
    print("\nSanity check:")
    for name, mlbam_id in sample_ids.items():
        row = out[out["mlbam_id"] == mlbam_id]
        if len(row):
            r = row.iloc[0]
            print(
                f"  {name} (MLBAM {mlbam_id}): born "
                f"{int(r.birth_year)}-{int(r.birth_month):02d}-{int(r.birth_day):02d}, "
                f"age_2025={r.age_2025}, age_2026={r.age_2026}"
            )
        else:
            print(f"  {name} (MLBAM {mlbam_id}): NOT FOUND in register")


if __name__ == "__main__":
    main()
