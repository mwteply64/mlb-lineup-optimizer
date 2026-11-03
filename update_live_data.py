#!/usr/bin/env python3
"""
Live Data Update - Lineup Optimization Project
================================================
Automates the data feed for test.py (the live in-season lineup
optimizer), replacing the manual FanGraphs CSV exports with the same
Savant/pybaseball pipeline already validated for backtest.py.

Produces, in the PROJECT ROOT (overwriting the old manual exports):
    2026_vs_left.csv    -- 2026 season-to-date vs LHP, woba + xwoba
    2026_vs_right.csv   -- 2026 season-to-date vs RHP, woba + xwoba
    2026_total.csv       -- 2026 season-to-date, no hand split, woba + xwoba
    recent.csv            -- last 14 days, no hand split, woba + xwoba
    career_vs_left.csv  -- career (2016-2025) vs LHP, decay-weighted
    career_vs_right.csv -- career (2016-2025) vs RHP, decay-weighted

And, in the "historical testing" folder (one-time backfill, idempotent):
    season_2025_vs_left.csv / season_2025_vs_right.csv
        -- 2025 is now a complete past season, so it graduates into the
           career pool. download_v3_expansion.py's CAREER_SEASONS was
           updated 2026-06-19 to include it; this script generates the
           two files if they aren't already on disk.

WHY THE COLUMN SHAPE CHANGES:
The files this replaces (manual FanGraphs exports) had ~50 columns
(Team, G, HR, R, RBI, BsR, Off, Def, WAR, wRC+, ...) and -- critically --
had ZERO xwOBA coverage on every handedness-split file (checked
2026-06-19: 0/356-389 rows across 2026_vs_left/right.csv and
career_vs_left/right.csv). That's a hard blocker for porting backtest.py's
wOBA+xwOBA blend logic into test.py, which is the whole point of this
data step. The new files use the same simple shape as backtest.py's own
pipeline instead: last_name, first_name, player_id, pa, woba[, xwoba].

This means test.py's CURRENT code (which reads a "wRC+" column) will NOT
run against these new files as-is. That's expected -- the next step is
porting test.py's scoring engine to consume wOBA/xwOBA the way
backtest.py does (already agreed/designed; not part of this script).

CAREER AGGREGATION METHODOLOGY:
career_vs_left/right.csv are built by combining every
season_<year>_vs_<hand>.csv file in CAREER_SEASONS, using the EXACT
formula backtest.py's build_career_data() uses in production: each
season weighted by its own PA times decay ** (2026 - season_year).
DECAY_CAREER_LHP/RHP below are copied from backtest.py's current tuned
constants (0.8001 / 1.0) -- NOT imported, since backtest.py's globals are
entangled with its own CV-tuning framework. If backtest.py's tuned decay
values ever change, update the two constants below to match.
aging_strength is fixed at 0.0 here, matching backtest.py's current
production AGING_STRENGTH constant (a validated no-op), so no
player-age lookup is needed for this script.

2026-06-24 change: the current season is now ALSO folded into
career_vs_left/right.csv as one more weighted "season," using the
already-pulled 2026-vs-hand aggregate from STEP 2 below (no extra
Statcast pull) -- same role as season_<year>_vs_<hand>.csv for every
prior year, but sourced from memory instead of a per-season CSV.
Under the exact same decay ** (test_year - season_year) formula, the
current season's weight is decay ** (2026 - 2026) = decay ** 0 == 1.0
no matter what DECAY_CAREER_LHP/RHP are set to -- i.e. this season
always counts at full PA weight, never decayed, which is exactly the
"no decay (1.0) for the current-season component" behavior discussed
with the user, achieved with no new constant. Its PA-weighted share of
the blend still grows naturally over the season as 2026 PA accumulates
relative to the (fixed) historical seasons -- decay only ever discounts
COMPLETED prior seasons, not the current one. backtest.py's own
build_career_data() is UNCHANGED -- it still excludes test_year by
construction, which is correct for honest historical CV (including a
backtest fold's own future-relative-to-cutoff data would be the same
leakage the project already eliminated elsewhere). This live-only
change does not touch test.py either -- test.py just reads whatever is
in career_vs_left/right.csv, so the new behavior is entirely contained
in this script. WEIGHT_CAREER_*/REGRESS_PA_CAREER_* (in test.py /
backtest.py) were tuned under the OLD career definition (history only);
they are NOT retuned as part of this change -- see the user discussion
2026-06-24 for the trade-off (retuning would need a new rolling
within-season CV scheme, not a rerun of the existing annual-holdout
tuner, since the existing tuner's test windows are complete prior
seasons with no "in-progress season" to fold in).

WOBA_WEIGHTS[2026]:
Sourced from FanGraphs Guts! (page last updated 2026-06-12), added to
download_v3_expansion.py on 2026-06-19. This is a LIVE in-season
estimate, not a finalized season constant -- it will drift slightly as
2026 progresses. Re-check https://www.fangraphs.com/tools/guts
periodically (e.g. monthly) and update that entry; trust in this source
comes from the 2025 row matching the prior finalized constant exactly.

SETUP (one-time):
    pip install pybaseball pandas

USAGE:
    python update_live_data.py
    (or double-click run_update_live_data.bat)

NOTES:
    - Safe to re-run any time (e.g. daily/weekly) -- the live files are
      meant to be overwritten every run. Only the season_2025 backfill
      is skip-if-exists.
    - Pulls "through yesterday" by default, not through today, since
      Statcast data for the current calendar day may still be partial.
"""

import os
import sys
import csv
import time
import warnings
from datetime import date, timedelta

import pandas as pd

import download_v3_expansion as dv3

# == Configuration ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

LIVE_TEST_YEAR = 2026
SEASON_2026_START = "2026-03-25"  # 2026 Opening Night (Giants @ Yankees)

# Copied from backtest.py's production constants (see module docstring).
DECAY_CAREER_LHP = 0.8001
DECAY_CAREER_RHP = 1.0
AGING_STRENGTH = 0.0  # matches backtest.py's current neutral placeholder

CAREER_SEASONS = dv3.CAREER_SEASONS  # now includes 2025 (edited 2026-06-19)


def get_live_end_date():
    """Yesterday, to avoid pulling a partial/still-settling game day."""
    return (date.today() - timedelta(days=1)).isoformat()


def save_to_root(df, filename):
    path = os.path.join(PROJECT_ROOT, filename)
    df.to_csv(path, index=False)
    print(f"    [OK] {filename}  ({len(df)} batters)  -> project root")


# == Step 1: backfill missing per-season career files =========================

def backfill_missing_seasons(name_lookup):
    """
    Idempotent: fetch+save any season_<year>_vs_<hand>.csv in
    CAREER_SEASONS that isn't already on disk. As of 2026-06-19 this is
    just 2025 (both hands); already-present years are skipped entirely
    (no network call).
    """
    for year in CAREER_SEASONS:
        paths = {
            hand_label: os.path.join(dv3.SAVE_DIR, f"season_{year}_{hand_label}.csv")
            for hand_label, _ in dv3.HANDEDNESS
        }
        if all(os.path.exists(p) for p in paths.values()):
            continue

        print(f"  season_{year} files missing -- backfilling...")
        dates = dv3.SEASON_DATES[year]
        raw = dv3.fetch_range(dates["start"], dates["end"], f"full season {year}")
        if raw is None:
            print(f"    SKIPPING year {year} -- no data.")
            continue
        for hand_label, hand_code in dv3.HANDEDNESS:
            if os.path.exists(paths[hand_label]):
                continue
            hand_data = raw[raw["p_throws"] == hand_code]
            result = dv3.aggregate(hand_data, name_lookup, include_xwoba=True, year=year)
            dv3.save(result, f"season_{year}_{hand_label}.csv", [0])
        time.sleep(0.5)


# == Step 2: career aggregation (mirrors backtest.py build_career_data) ======

def load_season_csv(path):
    """
    Load a season_<year>_vs_<hand>.csv. Returns dict:
        player_id -> {last_name, first_name, pa, woba, xwoba (or None)}
    Mirrors backtest.py's load_file() row-parsing exactly (skips
    rows with missing/invalid wOBA).
    """
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pid = str(row["player_id"]).strip()
                pa = int(float(row["pa"]))
                woba_str = str(row.get("woba", "")).strip()
                if not woba_str or woba_str.lower() in ("nan", "none", ""):
                    continue
                woba = float(woba_str)
                last = str(row.get("last_name", "")).strip()
                first = str(row.get("first_name", "")).strip()
                xwoba_str = str(row.get("xwoba", "")).strip()
                xwoba = None
                if xwoba_str and xwoba_str.lower() not in ("nan", "none"):
                    try:
                        xwoba = float(xwoba_str)
                    except ValueError:
                        xwoba = None
                result[pid] = {
                    "last_name": last, "first_name": first,
                    "pa": pa, "woba": woba, "xwoba": xwoba,
                }
            except (ValueError, KeyError):
                continue
    return result


def current_season_to_dict(df):
    """
    Convert the in-memory 2026-vs-hand aggregate (same shape pull_2026_live()
    saves to 2026_vs_<hand>.csv) into the same
    {player_id -> {last_name, first_name, pa, woba, xwoba}} shape
    load_season_csv() returns, so build_career_file() can blend it in via
    the exact same loop as every historical season -- no special-casing.
    """
    result = {}
    if df is None or df.empty:
        return result
    has_xwoba = "xwoba" in df.columns
    for row in df.itertuples(index=False):
        try:
            pid = str(row.player_id).strip()
            pa = int(row.pa)
            woba = float(row.woba)
        except (ValueError, TypeError):
            continue
        if pd.isna(woba):
            continue
        xwoba = None
        if has_xwoba:
            xval = getattr(row, "xwoba", None)
            if xval is not None and not pd.isna(xval):
                xwoba = float(xval)
        result[pid] = {
            "last_name": str(row.last_name), "first_name": str(row.first_name),
            "pa": pa, "woba": woba, "xwoba": xwoba,
        }
    return result


def build_career_file(hand_label, decay, current_season_df=None, test_year=LIVE_TEST_YEAR):
    """
    Combine every season_<year>_<hand_label>.csv in CAREER_SEASONS into
    one career file. Each season is weighted by its own PA times
    decay ** (test_year - season_year) -- identical math to backtest.py's
    build_career_data() with aging_strength=0.0 (a validated no-op, so
    skipped here entirely rather than re-implemented).

    current_season_df: optional in-memory DataFrame (2026-vs-hand_label,
    same shape as a season_<year>_<hand_label>.csv) to fold in as one more
    weighted "season" at test_year. Its weight under the formula above is
    decay ** (test_year - test_year) == 1.0 always -- the current season
    is never decayed, no matter what decay is -- so this naturally
    implements "current-season component of career has no decay" with no
    extra constant. Pass None (default) to reproduce the old
    history-only career file exactly.
    """
    num, den, xnum, xden = {}, {}, {}, {}
    pa_sum, last_map, first_map = {}, {}, {}

    seasons_to_blend = list(CAREER_SEASONS)
    current_season = current_season_to_dict(current_season_df)
    if current_season:
        seasons_to_blend.append(test_year)

    for y in seasons_to_blend:
        if y == test_year and current_season:
            season = current_season
        else:
            path = os.path.join(dv3.SAVE_DIR, f"season_{y}_{hand_label}.csv")
            season = load_season_csv(path)
            if not season:
                print(f"    WARNING: season_{y}_{hand_label}.csv missing or empty -- skipped.")
                continue
        w = decay ** (test_year - y)
        for pid, d in season.items():
            wpa = w * d["pa"]
            num[pid] = num.get(pid, 0.0) + wpa * d["woba"]
            den[pid] = den.get(pid, 0.0) + wpa
            pa_sum[pid] = pa_sum.get(pid, 0) + d["pa"]
            last_map[pid] = d["last_name"]
            first_map[pid] = d["first_name"]
            if d.get("xwoba") is not None:
                xnum[pid] = xnum.get(pid, 0.0) + wpa * d["xwoba"]
                xden[pid] = xden.get(pid, 0.0) + wpa

    rows = []
    for pid, denom in den.items():
        if denom <= 0:
            continue
        row = {
            "last_name": last_map[pid],
            "first_name": first_map[pid],
            "player_id": pid,
            "pa": pa_sum[pid],
            "woba": round(num[pid] / denom, 3),
        }
        if xden.get(pid, 0) > 0:
            row["xwoba"] = round(xnum[pid] / xden[pid], 3)
        rows.append(row)

    cols = ["last_name", "first_name", "player_id", "pa", "woba", "xwoba"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df.reindex(columns=cols).sort_values("pa", ascending=False).reset_index(drop=True)


# == Step 3/4: live 2026 + recent pulls =======================================

def pull_2026_live(name_lookup, end_date):
    """
    Returns {hand_label: df} (in addition to saving 2026_vs_<hand>.csv /
    2026_total.csv as before) so build_career_file() can reuse this same
    in-memory aggregate to fold the current season into career_vs_<hand>.csv
    -- one Statcast pull instead of two.
    """
    raw = dv3.fetch_range(SEASON_2026_START, end_date, "2026 season-to-date")
    if raw is None:
        print("    SKIPPING 2026 live pull -- no data.")
        return {}
    by_hand = {}
    for hand_label, hand_code in dv3.HANDEDNESS:
        hand_data = raw[raw["p_throws"] == hand_code]
        result = dv3.aggregate(hand_data, name_lookup, include_xwoba=True, year=LIVE_TEST_YEAR)
        save_to_root(result, f"2026_{hand_label}.csv")
        by_hand[hand_label] = result
    total_result = dv3.aggregate(raw, name_lookup, include_xwoba=True, year=LIVE_TEST_YEAR)
    save_to_root(total_result, "2026_total.csv")
    return by_hand


def pull_recent(name_lookup, end_date):
    start_date = (date.fromisoformat(end_date) - timedelta(days=13)).isoformat()
    raw = dv3.fetch_range(start_date, end_date, "recent (last 14 days)")
    if raw is None:
        print("    SKIPPING recent pull -- no data.")
        return
    result = dv3.aggregate(raw, name_lookup, include_xwoba=True, year=LIVE_TEST_YEAR)
    save_to_root(result, "recent.csv")


# == Main ======================================================================

def main():
    warnings.filterwarnings("ignore")
    end_date = get_live_end_date()

    print("=" * 70)
    print("=== LIVE DATA UPDATE STARTED ===")
    print(f"    Pulling 2026 data through {end_date} (yesterday)")
    print("=" * 70)
    print()

    name_lookup = dv3.build_name_lookup()
    print()

    print("=" * 70)
    print("STEP 1: Backfill missing career seasons (currently: 2025)")
    print("=" * 70)
    backfill_missing_seasons(name_lookup)
    print()

    print("=" * 70)
    print(f"STEP 2: 2026 season-to-date ({SEASON_2026_START} -> {end_date})")
    print("=" * 70)
    live_by_hand = pull_2026_live(name_lookup, end_date)
    print()

    print("=" * 70)
    print("STEP 3: Build career_vs_left/right.csv (decay-weighted, mirrors")
    print("        backtest.py's build_career_data(), PLUS the current 2026")
    print("        season-to-date pulled in STEP 2 above, folded in at full/")
    print("        no-decay weight -- see module docstring, 2026-06-24 change)")
    print("=" * 70)
    for hand_label, decay in [("vs_left", DECAY_CAREER_LHP), ("vs_right", DECAY_CAREER_RHP)]:
        df = build_career_file(hand_label, decay, current_season_df=live_by_hand.get(hand_label))
        save_to_root(df, f"career_{hand_label}.csv")
    print()

    print("=" * 70)
    print("STEP 4: Recent form (last 14 days)")
    print("=" * 70)
    pull_recent(name_lookup, end_date)
    print()

    print("=" * 70)
    print("Done.")
    print("NOTE: test.py's current code still expects a wRC+ column and")
    print("won't run against these files until its scoring engine is")
    print("ported to consume wOBA/xwOBA (the next planned step).")
    print("=" * 70)


if __name__ == "__main__":
    main()
