#!/usr/bin/env python3
"""
Team-Tagging Download - Lineup Optimization Project
======================================================
Companion to download_v3_expansion.py. Re-pulls the SAME 18 pre-cutoff
date ranges already used for totals_before_<cutoff>_<year>.csv (6 test
years x 3 cutoffs: june_1, july_1, aug_1) and re-aggregates with each
player's TEAM retained, so the new lineup_forward_test.py script can
group players by roster instead of treating the whole league as one
pool.

WHY THIS EXISTS
---------------
backtest.py's historical CSVs (season_*, before_*/after_*, totals_before_*,
recent_before_*) have no team column at all -- aggregate() in
download_v3_expansion.py groups by player_id only. That's fine for
backtest.py's job (does the model's ranking correlate with future
performance, league-wide?), but it means we can't build a roster-
constrained test ("would the optimizer's pick have beaten THIS team's
real regular?") without re-pulling with team retained.

Deliberately narrow scope: this script does NOT touch
download_v3_expansion.py, does NOT overwrite any existing validated
CSV, and does NOT re-pull the handedness-split, career, or recent-form
files -- none of those need team, since team only matters for figuring
out which roster a player belonged to heading into the cutoff. That's
exactly what the existing totals_before_<cutoff>_<year> date range
already represents (the player's whole workload before the cutoff,
unsplit by pitcher hand), so this script reuses that exact date range
and just adds a team column to the aggregation step.

OUTPUT
------
totals_before_<cutoff>_<year>_with_team.csv, columns:
    last_name, first_name, player_id, team, pa, woba
18 files total (6 test years x 3 cutoffs). Saved alongside the existing
historical-testing files but with a distinct "_with_team" suffix, so
nothing already-validated gets overwritten.

TEAM DERIVATION
----------------
Statcast's pitch-level export carries home_team / away_team /
inning_topbot per pitch (standard, long-documented Statcast/Savant CSV
columns). Batting team follows the standard baseball convention: top of
the inning -> the AWAY team is at bat; bottom of the inning -> the HOME
team is at bat. If a player changed teams mid-window (a trade), this
naturally splits him into two rows -- one per team -- each with just
that team's share of his PA/wOBA, since team is derived per-pitch before
grouping.

IMPORTANT -- this script's sandbox has no network access to install
pybaseball or reach Statcast, so the home_team/away_team/inning_topbot
columns could not be live-verified before writing this. They are
well-established, documented Statcast columns, but run the smoke test
below on a single small date range FIRST and inspect the output before
committing to the full 18-window pull.

SMOKE TEST (run this first):
    python download_team_tags.py --smoke-test
This pulls just 2026-04-01 to 2026-04-02 and prints the team-tagged
result for inspection, without touching the historical-testing folder.

FULL RUN:
    python download_team_tags.py
    (or double-click run_download_team_tags.bat, which logs to a file)

SETUP: same as download_v3_expansion.py (pip install pybaseball pandas).
If you've already run download_v3_expansion.py locally, pybaseball's
cache may still have these exact date ranges cached, which would make
this re-pull much faster than a cold pull.
"""

import os
import sys
import time
import warnings
import pandas as pd

# Reuse constants/helpers from the validated download script instead of
# duplicating them -- read-only import, does not modify or re-run
# download_v3_expansion.py's own main().
from download_v3_expansion import (
    SAVE_DIR, TEST_YEARS, CUTOFFS, WOBA_WEIGHTS,
    build_name_lookup, clean_raw,
)

try:
    from pybaseball import statcast, cache
    cache.enable()
except ImportError:
    print("ERROR: pybaseball not installed.")
    print("Run: pip install pybaseball pandas")
    sys.exit(1)


# == Helpers ==================================================================

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


def aggregate_with_team(df, name_lookup, year):
    """
    Like download_v3_expansion.aggregate(), but groups by (player, team)
    instead of player alone. Team is derived per-pitch from
    home_team/away_team + inning_topbot before grouping, so a player
    who was traded mid-window comes out as two rows, each correctly
    scoped to that team's share of his PA/wOBA.

    Returns columns: last_name, first_name, player_id, team, pa, woba
    """
    out_cols = ["last_name", "first_name", "player_id", "team", "pa", "woba"]

    pa = df[df["events"].notna()].copy()
    if pa.empty:
        return pd.DataFrame(columns=out_cols)

    required = ["batter", "events", "woba_denom", "home_team", "away_team", "inning_topbot"]
    missing = [c for c in required if c not in pa.columns]
    if missing:
        raise KeyError(
            "Statcast data missing columns needed for team tagging: "
            + ", ".join(missing) +
            ". Run with --smoke-test and inspect df.columns before "
            "assuming this script's column names are still correct."
        )

    weights = WOBA_WEIGHTS.get(year)
    if weights is None:
        raise KeyError(f"No wOBA linear weights configured for year {year}")

    pa["woba_num_contrib"] = pa["events"].map(weights).fillna(0.0)

    # Batting team: top of inning -> away team bats; bottom -> home team bats.
    pa["team"] = pa["away_team"].where(pa["inning_topbot"] == "Top", pa["home_team"])

    grouped = (
        pa.groupby(["batter", "team"], sort=False)
        .agg(pa=("events", "count"),
             woba_num=("woba_num_contrib", "sum"),
             woba_denom=("woba_denom", "sum"))
        .reset_index()
    )
    grouped["woba"] = (
        grouped["woba_num"] / grouped["woba_denom"].replace(0, float("nan"))
    ).round(3)

    grouped["last_name"] = grouped["batter"].map(
        lambda pid: name_lookup.get(int(pid), ("", ""))[0]
    )
    grouped["first_name"] = grouped["batter"].map(
        lambda pid: name_lookup.get(int(pid), ("", ""))[1]
    )

    result = (
        grouped[["last_name", "first_name", "batter", "team", "pa", "woba"]]
        .rename(columns={"batter": "player_id"})
        .sort_values(["player_id", "pa"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return result


def save(df, filename, files_saved_counter):
    filepath = os.path.join(SAVE_DIR, filename)
    df.to_csv(filepath, index=False)
    files_saved_counter[0] += 1
    print(f"    [OK] {filename}  ({len(df)} player-team rows)")


# == Smoke test ===============================================================

def smoke_test():
    print("=== SMOKE TEST: team tagging on a tiny date range ===")
    print("(2026-04-01 -> 2026-04-02, not saved to disk)")
    print()
    raw = fetch_range("2026-04-01", "2026-04-02", "smoke test")
    if raw is None:
        print("No data returned -- can't verify column schema. Check network/pybaseball.")
        return
    missing = [c for c in ("home_team", "away_team", "inning_topbot") if c not in raw.columns]
    if missing:
        print(f"PROBLEM: raw pull is missing expected columns: {missing}")
        print("Available columns:")
        print(list(raw.columns))
        return
    print("Columns OK: home_team, away_team, inning_topbot all present.")
    name_lookup = build_name_lookup()
    result = aggregate_with_team(raw, name_lookup, 2026)
    print()
    print(f"Aggregated {len(result)} player-team rows. Sample:")
    print(result.head(15).to_string(index=False))
    print()
    print("Inspect the 'team' column above -- should be 2/3-letter MLB team")
    print("codes (NYY, LAD, etc.), and any player who appears twice should")
    print("be a real trade, not a bug. If this looks right, run the full pull.")


# == Main =====================================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    total_expected = len(TEST_YEARS) * len(CUTOFFS)
    files_saved = [0]

    print(f"Re-pulling {total_expected} pre-cutoff windows with team tagging.")
    print(f"Saving to: {SAVE_DIR}")
    print("Output files use a '_with_team' suffix -- nothing already-")
    print("validated gets overwritten.")
    print()

    name_lookup = build_name_lookup()
    print()

    for year in TEST_YEARS:
        print(f"  [{year}]")
        for cutoff_label, before_fn, _after_fn in CUTOFFS:
            b_start, b_end = before_fn(year)
            raw = fetch_range(b_start, b_end, f"totals_before_{cutoff_label}_{year} (team-tagged)")
            if raw is not None:
                result = aggregate_with_team(raw, name_lookup, year)
                save(result, f"totals_before_{cutoff_label}_{year}_with_team.csv", files_saved)
            time.sleep(0.5)
        print()

    print(f"Done. Saved {files_saved[0]}/{total_expected} files.")
    if files_saved[0] < total_expected:
        print("Re-run to retry any that failed (cached ranges reload instantly).")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        main()
