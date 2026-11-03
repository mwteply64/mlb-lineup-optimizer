#!/usr/bin/env python3
"""
Aging Curve Project - Step 2: Delta-Method Curve Estimation
==============================================================
Estimates a wOBA aging curve (expected year-over-year change in performance
as a function of age) using the standard "delta method" (Tom Tango / Jeff
Zimmerman style): for every player who appears in BOTH of two adjacent
seasons, compute how much their wOBA changed, and bin those deltas by the
player's age in the first of the two seasons. Averaging within an age bin,
weighted toward players with more playing time, gives an estimate of the
typical performance change at that age that is far less contaminated by
selection bias than just comparing the average performance of, say, all
25-year-olds vs all 35-year-olds in different player pools (that comparison
is badly biased: players who age poorly leave the league, so the surviving
35-year-old pool is disproportionately the ones who aged WELL).

The delta method still has a milder version of the same bias (a player has
to be good enough to get meaningful PA in BOTH years to show up in a pair
at all), which is a known, accepted limitation of this method in the
sabermetric literature, not a bug here -- it's the standard approach because
nothing fully solves survivorship bias with observational data alone.

DATA SOURCE -- no new downloads needed
----------------------------------------
Uses files already on disk in this project:
  - historical testing/season_<YYYY>_vs_left.csv + _vs_right.csv
    for YYYY in 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024
    (full-season per-player pa/woba, split by opposing-pitcher hand --
    combined here into a full-season total per player)
  - historical testing/before_aug_1_2025_vs_left/right.csv +
    after_aug_1_2025_vs_left/right.csv, summed together to reconstruct
    the full 2025 season (before Aug 1 + after Aug 1 = the whole season;
    no separate full-season 2025 file exists yet, so this rebuilds it from
    the cutoff-split files that do)
  - player_ages.csv (produced by build_age_data.py / run_build_age_data.bat
    -- must be run first; this script will exit with a clear error if it's
    missing)

2020 is excluded everywhere, consistent with the rest of this codebase
(shortened/anomalous COVID season). That means no 2019->2020 or 2020->2021
pair -- the year-pairs used are:
    2016-17, 2017-18, 2018-19, 2021-22, 2022-23, 2023-24, 2024-25

METHOD
------
For each adjacent year pair (yA, yB):
  1. Inner-join players who have >= MIN_PA_PER_SEASON plate appearances in
     BOTH yA and yB (the PA filter reduces noise from tiny-sample seasons;
     it does not change WHICH players are eligible to pair, just excludes
     pairs where one side is too small to be informative).
  2. age = that player's age (as of June 30) in yA, from player_ages.csv.
  3. delta = woba(yB) - woba(yA)
  4. weight = harmonic_mean(PA(yA), PA(yB))  -- Tango's convention; this
     downweights pairs where either season has thin playing time, without
     letting one huge season fully dominate either.
Pool every pair from every year-transition together, then group by
integer age and report the PA-weighted mean delta per age, along with N
(number of player-pairs) so thin, unreliable bins are visible rather than
hidden.

OUTPUT
------
  - Console report: per-age table (age, N pairs, total weight, weighted
    mean delta, cumulative curve relative to the youngest reliable age).
  - aging_curve_results_<date>.txt: same report, saved.
  - aging_curve_pairs.csv: the full underlying pair-level dataset (one row
    per player per year-transition), for transparency / re-analysis.

USAGE
-----
    python build_aging_curve.py
    (or double-click run_build_aging_curve.bat)
"""
import datetime
import os
import sys

import pandas as pd

DATA_DIR = "."
HIST_DIR = os.path.join(DATA_DIR, "historical testing")
AGE_FILE = os.path.join(DATA_DIR, "player_ages.csv")

FULL_SEASON_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
RECONSTRUCT_2025 = True  # before_aug_1_2025 + after_aug_1_2025, both hands

YEAR_PAIRS = [(2016, 2017), (2017, 2018), (2018, 2019),
              (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]

MIN_PA_PER_SEASON = 100  # each side of a pair must clear this to be used

TODAY = datetime.date.today().isoformat()
RESULTS_FILE = f"aging_curve_results_{TODAY}.txt"
PAIRS_FILE = "aging_curve_pairs.csv"


class Tee:
    """Mirrors everything printed to both the console and a results file."""
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def load_full_season(year):
    """
    Combine vs_left + vs_right into one full-season per-player pa/woba row.
    PA-weighted average of the two split wOBAs approximates the true
    full-season wOBA closely enough for curve estimation (this is not
    backtest.py's production scoring path -- no need for event-level
    reconstruction here).
    """
    left = pd.read_csv(os.path.join(HIST_DIR, f"season_{year}_vs_left.csv"))
    right = pd.read_csv(os.path.join(HIST_DIR, f"season_{year}_vs_right.csv"))
    return _combine_splits([left, right])


def load_2025_reconstructed():
    """Full 2025 season = before_aug_1_2025 + after_aug_1_2025, both hands."""
    frames = []
    for cutoff in ("before_aug_1_2025", "after_aug_1_2025"):
        for hand in ("vs_left", "vs_right"):
            path = os.path.join(HIST_DIR, f"{cutoff}_{hand}.csv")
            frames.append(pd.read_csv(path))
    return _combine_splits(frames)


def _combine_splits(frames):
    """
    Given 2+ dataframes with columns [last_name, first_name, player_id, pa,
    woba, (xwoba)], sum PA per player and take the PA-weighted average wOBA.
    """
    parts = []
    for df in frames:
        d = df[["player_id", "pa", "woba"]].copy()
        d["pa_woba"] = d["pa"] * d["woba"]
        parts.append(d)
    combined = pd.concat(parts, ignore_index=True)
    grouped = combined.groupby("player_id", as_index=False).agg(
        pa=("pa", "sum"), pa_woba=("pa_woba", "sum")
    )
    grouped["woba"] = grouped["pa_woba"] / grouped["pa"]
    return grouped[["player_id", "pa", "woba"]]


def harmonic_mean(a, b):
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def main():
    if not os.path.exists(AGE_FILE):
        print(f"ERROR: {AGE_FILE} not found.")
        print("Run build_age_data.py (or run_build_age_data.bat) first --")
        print("this script depends on its output.")
        sys.exit(1)

    sys.stdout = Tee(RESULTS_FILE)

    print("=" * 70)
    print("AGING CURVE ESTIMATION (delta method)")
    print(f"Min PA per season to qualify a pair: {MIN_PA_PER_SEASON}")
    print(f"Year pairs used: {YEAR_PAIRS}")
    print("=" * 70)
    print()

    ages = pd.read_csv(AGE_FILE)

    print("Loading full-season pa/woba per player per year...")
    season_data = {}
    for year in FULL_SEASON_YEARS:
        season_data[year] = load_full_season(year)
        print(f"  {year}: {len(season_data[year]):,} players")
    if RECONSTRUCT_2025:
        season_data[2025] = load_2025_reconstructed()
        print(f"  2025 (reconstructed): {len(season_data[2025]):,} players")
    print()

    all_pairs = []
    print("Building year-over-year pairs...")
    for (yA, yB) in YEAR_PAIRS:
        dfA = season_data[yA].rename(columns={"pa": "pa_A", "woba": "woba_A"})
        dfB = season_data[yB].rename(columns={"pa": "pa_B", "woba": "woba_B"})
        merged = dfA.merge(dfB, on="player_id", how="inner")
        before_pa_filter = len(merged)
        merged = merged[(merged["pa_A"] >= MIN_PA_PER_SEASON) &
                         (merged["pa_B"] >= MIN_PA_PER_SEASON)]

        age_col = f"age_{yA}"
        if age_col not in ages.columns:
            print(f"  WARNING: {age_col} not in {AGE_FILE}, skipping pair {yA}-{yB}")
            continue
        merged = merged.merge(
            ages[["mlbam_id", age_col]].rename(columns={"mlbam_id": "player_id", age_col: "age"}),
            on="player_id", how="inner",
        )
        merged = merged.dropna(subset=["age"])

        merged["delta"] = merged["woba_B"] - merged["woba_A"]
        merged["weight"] = merged.apply(lambda r: harmonic_mean(r["pa_A"], r["pa_B"]), axis=1)
        merged["year_pair"] = f"{yA}-{yB}"

        print(f"  {yA}-{yB}: {before_pa_filter} players in both years -> "
              f"{len(merged)} after PA filter + age lookup")
        all_pairs.append(merged[["player_id", "age", "delta", "weight", "year_pair",
                                  "pa_A", "pa_B", "woba_A", "woba_B"]])
    print()

    pairs_df = pd.concat(all_pairs, ignore_index=True)
    pairs_df["age"] = pairs_df["age"].astype(int)
    pairs_df.to_csv(PAIRS_FILE, index=False)
    print(f"Saved {len(pairs_df):,} player-pair rows to {PAIRS_FILE}")
    print()

    print("Pooling across all year-pairs, grouping by age...")
    pairs_df["weighted_delta"] = pairs_df["delta"] * pairs_df["weight"]
    grouped = pairs_df.groupby("age", as_index=False).agg(
        n_pairs=("delta", "size"),
        total_weight=("weight", "sum"),
        total_weighted_delta=("weighted_delta", "sum"),
    )
    grouped["weighted_mean_delta"] = grouped["total_weighted_delta"] / grouped["total_weight"]
    grouped = grouped.sort_values("age")

    print()
    print("=" * 70)
    print("AGING CURVE (PA-weighted mean wOBA delta, by age)")
    print("=" * 70)
    print(f"{'age':>4}  {'N pairs':>8}  {'tot weight':>11}  {'mean delta':>11}  {'cum (vs youngest)':>18}")
    cum = 0.0
    first = True
    for _, row in grouped.iterrows():
        if first:
            cum = 0.0
            first = False
        else:
            cum += row["weighted_mean_delta"]
        flag = "  <- THIN" if row["n_pairs"] < 20 else ""
        print(f"{int(row['age']):>4}  {int(row['n_pairs']):>8}  {row['total_weight']:>11.0f}  "
              f"{row['weighted_mean_delta']:>+11.4f}  {cum:>+18.4f}{flag}")

    print()
    print("Bins flagged THIN (N < 20 pairs) are shown but should not be")
    print("trusted on their own -- consider smoothing or wider bins before")
    print("using them in any production adjustment.")
    print("=" * 70)

    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout
    print(f"\nDone. Full report saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
