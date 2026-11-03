#!/usr/bin/env python3
"""
Lineup Forward Test - Lineup Optimization Project
==================================================
Team-constrained, non-circular forward test of the stabilized-wOBA
model against real historical outcomes.

WHY THIS EXISTS
---------------
test.py's run_comparison() grades both the "real" and "optimal" lineup
using the model's OWN current Valuation_wRC+ estimate -- self-grading,
not an independent check. backtest.py's existing validation (Spearman
rank correlation between predicted and actual wOBA) IS independent, but
it's player-level only -- it never asks "would the model's pick have
beaten this TEAM's actual regular," because the historical CSVs had no
team column to restrict the player pool to one roster.

This script closes that gap using the new
totals_before_<cutoff>_<year>_with_team.csv files (built by
download_team_tags.py):

  1. ROSTER (no circularity): each player's team for a given window is
     resolved from PRE-cutoff PA only -- a real-world fact that would
     have been observable before the cutoff, not something defined
     using the same window that grades performance.
  2. REAL lineup = that team's top-9 roster players by PRE-cutoff PA.
     A simple, observable proxy for "who actually played every day."
  3. OPTIMAL lineup = top-9 roster players by the model's PRE-cutoff
     predicted stabilized wOBA (stabilize() from backtest.py, completely
     unmodified -- same weights/regression/xWOBA-blend constants
     already validated there).
  4. GRADE both lineups using REAL POST-cutoff actual wOBA (the after_*
     ground-truth files backtest.py already uses).

No window is reused for two purposes anywhere in this chain, which is
the structural fix for the circularity an in-season holdout would have
had (defining "real" via the same window used to grade).

This mirrors the "X of 30 teams had a better lineup" framing from the
older backtest.py run, but now across every available team x window x
hand combination instead of one snapshot.

WHAT THIS DOES NOT DO
----------------------
- Does not change ANY weight, regression-PA, league-average, decay,
  platoon, or xWOBA-blend constant. All scoring math (stabilize,
  build_career_data, blended_woba, load_file) is imported READ-ONLY
  from backtest.py.
- Does not touch download_v3_expansion.py's validated totals_before_*
  .csv files -- those stay the single source of truth for each
  player's combined (all-teams) PA/wOBA, fed into stabilize() exactly
  as backtest.py itself does. The new _with_team files are used ONLY
  to determine roster membership, never to alter a prediction input.
- backtest.py was given a small structural edit (its run-on-import
  main() call wrapped in `if __name__ == "__main__":`) so it can be
  imported here without re-running the full backtest as a side effect.
  That edit changes nothing about how `python backtest.py` behaves.

OUTPUT
------
lineup_forward_test_results_<date>.txt -- per-team-window-hand gap and
win/loss, plus aggregate win rate, average wOBA gap, a per-team
rollup, and a flagged-approximate runs/600 PA translation.

USAGE
  python lineup_forward_test.py
  (or double-click run_lineup_forward_test.bat)

2026-07-09 CHANGE -- hand-specific real lineup is now the DEFAULT
------------------------------------------------------------------
The real-lineup benchmark is now the team's top-9 by PRE-cutoff PA
**vs the specific pitcher hand** (from the new
before_<cutoff>_<year>_vs_<hand>_with_team.csv files built by
download_team_tags_by_hand.py), not top-9 by total PA. This mirrors
live_forward_validation.py's benchmark exactly, so historical and live
numbers are finally measuring the same question: does the model's
hand-specific top-9 beat the manager's actual hand-specific deployment?
The hand-specific benchmark is strictly stronger -- it encodes real
platooning -- so expect a lower win rate than the old hand-agnostic
baseline (65.8%, n=1048).

Flag: --hand-agnostic-real reproduces the old benchmark (top-9 by total
PA, same 9 graded vs both hands). Outputs from that mode carry a
"_handagnostic" filename suffix so the two run types never overwrite
each other. Team MEMBERSHIP always comes from total-PA tags either way
(max-PA team wins); only the ranking within the team differs.
"""

import os
import sys
import csv
import datetime
import statistics

from backtest import (
    DATA_DIR,
    TEST_YEARS, CUTOFFS,
    load_file, build_career_data, blended_woba, stabilize,
    DECAY_CAREER_LHP, DECAY_CAREER_RHP, AGING_STRENGTH, PLAYER_AGES,
    XWOBA_BLEND_SPLITS_LHP, XWOBA_BLEND_CAREER_LHP,
    XWOBA_BLEND_TOTAL_LHP, XWOBA_BLEND_RECENT_LHP,
    XWOBA_BLEND_SPLITS_RHP, XWOBA_BLEND_CAREER_RHP,
    XWOBA_BLEND_TOTAL_RHP, XWOBA_BLEND_RECENT_RHP,
    MIN_PA_TOTAL, MIN_PA_POST,
)

# -------------------------------------------------------
# OUTPUT SETUP
# -------------------------------------------------------

FOLDER = os.path.dirname(os.path.abspath(__file__))
today  = datetime.date.today().strftime("%Y-%m-%d")

# --hand-agnostic-real reproduces the pre-2026-07-09 benchmark (top-9 by
# total PA). Suffixed outputs keep the two run types separate on disk.
HAND_AGNOSTIC_REAL = "--hand-agnostic-real" in sys.argv
RUN_SUFFIX = "_handagnostic" if HAND_AGNOSTIC_REAL else ""

OUTPUT_FILE = os.path.join(FOLDER, "lineup_forward_test_results_" + today + RUN_SUFFIX + ".txt")


class Tee:
    """Write to both console and file simultaneously."""
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file    = open(filepath, "w", encoding="utf-8")
    def write(self, text):
        self.console.write(text)
        self.file.write(text)
    def flush(self):
        self.console.flush()
        self.file.flush()
    def close(self):
        self.file.close()


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

LINEUP_SIZE = 9

# Require at least this many of the 9 lineup slots to have usable
# post-cutoff ground truth before counting a team-window-hand in the
# aggregate -- avoids one lucky/unlucky player swinging a "9-man"
# average computed from only 1-2 graded players.
MIN_GRADED = 5

# Illustrative-only runs translation. NOT used for win/loss (that's
# decided purely on the actual measured wOBA gap). FanGraphs' real
# wOBA scale drifts ~1.15-1.25 year to year and this project hasn't
# validated a per-year value the way it has the wOBA model itself, so
# this is a rough, flagged approximation for intuition only.
ROUGH_WOBA_SCALE = 1.20
ILLUSTRATIVE_PA  = 600   # a full-time everyday-player season


# -------------------------------------------------------
# TEAM ROSTER DATA (new, additive -- not part of backtest.py)
# -------------------------------------------------------

def load_team_map(year, cutoff_label):
    """
    Load totals_before_<cutoff>_<year>_with_team.csv and resolve each
    player's PRIMARY team for this window: whichever team they
    accumulated the most PRE-cutoff PA with. That's a real, observable,
    non-circular fact -- nothing about the post-cutoff grading window
    is used here. A player traded mid-window shows up as two rows in
    the source file; only the larger-PA row determines their roster
    for this test.

    Returns dict: player_id (str) -> {"team": str, "team_pa": int}
    Empty dict (with a warning) if the file is missing.
    """
    filename = "totals_before_" + cutoff_label + "_" + str(year) + "_with_team.csv"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print("  WARNING: " + filename + " not found -- skipping this window "
              "(no roster data). Re-run run_team_tags.bat to backfill it.")
        return {}

    best = {}  # pid -> (pa, team)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pid  = str(row["player_id"]).strip()
                team = str(row["team"]).strip()
                pa   = int(float(row["pa"]))
            except (ValueError, KeyError):
                continue
            if not pid or not team:
                continue
            prev = best.get(pid)
            if prev is None or pa > prev[0]:
                best[pid] = (pa, team)

    return {pid: {"team": team, "team_pa": pa} for pid, (pa, team) in best.items()}


def build_rosters(team_map):
    """team -> [player_id, ...] for players with >= MIN_PA_TOTAL PA on
    that specific team pre-cutoff (same PA bar backtest.py already
    uses to trust a sample, applied here per-team instead of league-wide)."""
    rosters = {}
    for pid, info in team_map.items():
        if info["team_pa"] < MIN_PA_TOTAL:
            continue
        rosters.setdefault(info["team"], []).append(pid)
    return rosters


def load_hand_team_pa(year, cutoff_label, hand_label):
    """
    Load before_<cutoff>_<year>_<hand_label>_with_team.csv (built by
    download_team_tags_by_hand.py): each player's PRE-cutoff PA vs this
    pitcher hand, per team. Returns dict: player_id -> {team: pa}.
    A traded player has one entry per team, each scoped to that team's
    share of his PA. Returns None (with a warning) if the file is
    missing, so the caller can skip the window rather than silently
    grade against a wrong benchmark.
    """
    filename = ("before_" + cutoff_label + "_" + str(year) + "_" +
                hand_label + "_with_team.csv")
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print("  WARNING: " + filename + " not found -- run "
              "run_team_tags_by_hand.bat to build it. Skipping this window.")
        return None
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pid  = str(row["player_id"]).strip()
                team = str(row["team"]).strip()
                pa   = int(float(row["pa"]))
            except (ValueError, KeyError):
                continue
            if not pid or not team:
                continue
            result.setdefault(pid, {})[team] = result.get(pid, {}).get(team, 0) + pa
    return result


# -------------------------------------------------------
# SCORING (mirrors backtest.py's run_window() scoring loop exactly --
# same function calls, same constants. Duplicated only because
# run_window() returns name-keyed pairs with no player_id, which isn't
# enough to join against team rosters; every actual scoring building
# block below is imported read-only from backtest.py.)
# -------------------------------------------------------

def score_window(year, cutoff_label):
    pre_l  = load_file("before_" + cutoff_label + "_" + str(year) + "_vs_left.csv")
    pre_r  = load_file("before_" + cutoff_label + "_" + str(year) + "_vs_right.csv")
    car_l  = build_career_data(year, "left",  DECAY_CAREER_LHP, AGING_STRENGTH, PLAYER_AGES)
    car_r  = build_career_data(year, "right", DECAY_CAREER_RHP, AGING_STRENGTH, PLAYER_AGES)
    total  = load_file("totals_before_" + cutoff_label + "_" + str(year) + ".csv")
    recent = load_file("recent_before_" + cutoff_label + "_" + str(year) + ".csv")
    post_l = load_file("after_" + cutoff_label + "_" + str(year) + "_vs_left.csv")
    post_r = load_file("after_" + cutoff_label + "_" + str(year) + "_vs_right.csv")

    all_pids = set(pre_l.keys()) | set(pre_r.keys())

    scored = {}
    for pid in all_pids:
        sl  = pre_l.get(pid)
        sr  = pre_r.get(pid)
        cl  = car_l.get(pid)
        cr  = car_r.get(pid)
        tot = total.get(pid)
        rec = recent.get(pid)
        name = (sl or sr or tot or {}).get("name", pid)

        stab_l = stabilize(
            sl["pa"]  if sl  else 0, blended_woba(sl,  XWOBA_BLEND_SPLITS_LHP),
            cl["pa"]  if cl  else 0, blended_woba(cl,  XWOBA_BLEND_CAREER_LHP),
            tot["pa"] if tot else 0, blended_woba(tot, XWOBA_BLEND_TOTAL_LHP),
            rec["pa"] if rec else 0, blended_woba(rec, XWOBA_BLEND_RECENT_LHP),
            side='L',
            career_cross_pa=cr["pa"] if cr else 0,
            career_cross_woba=blended_woba(cr, XWOBA_BLEND_CAREER_LHP),
            pid=pid,
        )
        stab_r = stabilize(
            sr["pa"]  if sr  else 0, blended_woba(sr,  XWOBA_BLEND_SPLITS_RHP),
            cr["pa"]  if cr  else 0, blended_woba(cr,  XWOBA_BLEND_CAREER_RHP),
            tot["pa"] if tot else 0, blended_woba(tot, XWOBA_BLEND_TOTAL_RHP),
            rec["pa"] if rec else 0, blended_woba(rec, XWOBA_BLEND_RECENT_RHP),
            side='R',
            career_cross_pa=cl["pa"] if cl else 0,
            career_cross_woba=blended_woba(cl, XWOBA_BLEND_CAREER_RHP),
            pid=pid,
        )
        scored[pid] = {"name": name, "stab_l": stab_l, "stab_r": stab_r}

    return scored, post_l, post_r


# -------------------------------------------------------
# LINEUP CONSTRUCTION + GRADING
# -------------------------------------------------------

def real_lineup(team_pids, team_map):
    """Top-9 by PRE-cutoff TOTAL PA on this team -- the hand-agnostic
    playing-time proxy (pre-2026-07-09 default; now behind
    --hand-agnostic-real)."""
    return sorted(team_pids, key=lambda pid: team_map[pid]["team_pa"], reverse=True)[:LINEUP_SIZE]


def real_lineup_hand(team, team_pids, hand_pa_map):
    """Top-9 by PRE-cutoff PA vs this SPECIFIC pitcher hand on this team
    -- mirrors live_forward_validation.py's real_lineup_pool(). Players
    with zero PA vs this hand are excluded (they weren't part of the
    manager's deployment vs this hand), so the list can be shorter than
    9; the MIN_GRADED gate downstream protects against thin lists, same
    as on the live side."""
    ranked = sorted(
        ((pid, hand_pa_map.get(pid, {}).get(team, 0)) for pid in team_pids),
        key=lambda t: t[1], reverse=True)
    return [pid for pid, pa in ranked if pa > 0][:LINEUP_SIZE]


def optimal_lineup(team_pids, scored, stab_key):
    """Top-9 by the model's PRE-cutoff predicted stabilized wOBA,
    restricted to this team's own roster."""
    eligible = [pid for pid in team_pids if scored.get(pid, {}).get(stab_key) is not None]
    return sorted(eligible, key=lambda pid: scored[pid][stab_key], reverse=True)[:LINEUP_SIZE]


def grade_lineup(pids, post_dict):
    """Average actual POST-cutoff wOBA over lineup players with enough
    post-cutoff PA to trust (mirrors backtest.py's MIN_PA_POST gate).
    Returns (avg_or_None, n_graded)."""
    vals = []
    for pid in pids:
        post = post_dict.get(pid)
        if post is not None and post["pa"] >= MIN_PA_POST:
            vals.append(post["woba"])
    if not vals:
        return None, 0
    return statistics.mean(vals), len(vals)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    print("=== LINEUP FORWARD TEST -- team-constrained, non-circular ===")
    print("Results saved to: " + OUTPUT_FILE)
    print()
    if HAND_AGNOSTIC_REAL:
        print("*** BENCHMARK MODE: --hand-agnostic-real (legacy) ***")
        print("Real lineup  = team's top-9 roster players by PRE-cutoff TOTAL PA")
        print("               (same 9 players graded vs both hands)")
    else:
        print("Real lineup  = team's top-9 roster players by PRE-cutoff PA vs the")
        print("               SPECIFIC pitcher hand (mirrors live_forward_validation.py)")
    print("Optimal lineup = team's top-9 roster players by the model's")
    print("                 PRE-cutoff predicted stabilized wOBA")
    print("Both graded on REAL POST-cutoff actual wOBA. No window is")
    print("reused for two purposes anywhere in this chain.")
    print()

    results = []
    skipped_windows = []
    thin_roster_teams = 0   # team-windows dropped for <9 eligible roster players

    for year in TEST_YEARS:
        for cutoff_label in CUTOFFS:
            team_map = load_team_map(year, cutoff_label)
            if not team_map:
                skipped_windows.append(str(year) + " " + cutoff_label)
                continue

            # Hand-specific real-lineup PA maps (default mode). Missing
            # files skip the window -- never silently fall back to the
            # weaker hand-agnostic benchmark mid-run.
            hand_pa_l = hand_pa_r = None
            if not HAND_AGNOSTIC_REAL:
                hand_pa_l = load_hand_team_pa(year, cutoff_label, "vs_left")
                hand_pa_r = load_hand_team_pa(year, cutoff_label, "vs_right")
                if hand_pa_l is None or hand_pa_r is None:
                    skipped_windows.append(str(year) + " " + cutoff_label + " (hand-tag files)")
                    continue

            scored, post_l, post_r = score_window(year, cutoff_label)
            rosters = build_rosters(team_map)

            for team, pids in rosters.items():
                if len(pids) < LINEUP_SIZE:
                    thin_roster_teams += 1
                    continue
                real9_agnostic = real_lineup(pids, team_map)

                for hand_label, stab_key, post_dict, hand_pa_map in (
                    ("LHP", "stab_l", post_l, hand_pa_l),
                    ("RHP", "stab_r", post_r, hand_pa_r),
                ):
                    if HAND_AGNOSTIC_REAL:
                        real9 = real9_agnostic
                    else:
                        real9 = real_lineup_hand(team, pids, hand_pa_map)
                    opt9 = optimal_lineup(pids, scored, stab_key)
                    if len(opt9) < LINEUP_SIZE:
                        continue

                    # How many of the 9 optimal-lineup players are NOT in the
                    # real lineup (2026-06-26 addition) -- a personnel-change
                    # magnitude metric, separate from win/loss or gap size.
                    # Lets a downstream check ask "does the model only help
                    # when it makes big swaps, or even with small tweaks?"
                    n_diff = len(set(opt9) - set(real9))

                    real_avg, n_real = grade_lineup(real9, post_dict)
                    opt_avg,  n_opt  = grade_lineup(opt9,  post_dict)
                    if real_avg is None or opt_avg is None:
                        continue
                    if n_real < MIN_GRADED or n_opt < MIN_GRADED:
                        continue

                    gap = opt_avg - real_avg
                    results.append({
                        "year": year, "cutoff": cutoff_label, "team": team, "hand": hand_label,
                        "real_avg": real_avg, "opt_avg": opt_avg, "gap": gap,
                        "n_real": n_real, "n_opt": n_opt, "win": gap > 0,
                        "n_diff": n_diff,
                    })

    if skipped_windows:
        print("Skipped windows (missing team-tag file): " + ", ".join(skipped_windows))
        print()

    if not results:
        print("No team-window-hand combinations had enough data to grade. Stopping.")
        return

    # -- Per-combo detail CSV (2026-06-26 addition) --
    # Purely additive: writes the same `results` list already held in memory
    # to disk, one row per team-window-hand combo, so downstream uncertainty
    # tooling (bootstrap CIs, permutation tests) can resample the actual
    # graded outcomes instead of only the aggregate .txt summary. Does not
    # change any scoring logic, constant, or existing printed output above
    # or below this block.
    detail_path = os.path.join(FOLDER, "lineup_forward_test_detail_" + today + RUN_SUFFIX + ".csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "cutoff", "team", "hand", "real_avg",
                                           "opt_avg", "gap", "n_real", "n_opt", "win", "n_diff"])
        w.writeheader()
        w.writerows(results)
    print("Per-combo detail written to: " + detail_path)
    print()

    # -- Overall aggregate --
    n_total = len(results)
    n_wins  = sum(1 for r in results if r["win"])
    gaps    = [r["gap"] for r in results]
    mean_gap   = statistics.mean(gaps)
    median_gap = statistics.median(gaps)

    print("=" * 60)
    print("OVERALL  (" + str(n_total) + " team-window-hand combinations graded)")
    print("=" * 60)
    print("  Optimal lineup beat real lineup: " + str(n_wins) + "/" + str(n_total) +
          "  (" + str(round(100 * n_wins / n_total, 1)) + "%)")
    print("  Mean wOBA gap (optimal - real):   " + format(mean_gap, "+.4f"))
    print("  Median wOBA gap:                  " + format(median_gap, "+.4f"))
    illustrative_runs = mean_gap / ROUGH_WOBA_SCALE * ILLUSTRATIVE_PA
    print("  Illustrative runs/" + str(ILLUSTRATIVE_PA) + " PA @ wOBA scale~" +
          str(ROUGH_WOBA_SCALE) + ":  " + format(illustrative_runs, "+.2f") +
          "   (rough translation only -- NOT a validated per-year constant,")
    print("    not used for win/loss above -- intuition aid only)")
    print("  Team-windows dropped for thin roster (<9 eligible players): " + str(thin_roster_teams))
    print()

    # -- By handedness --
    print("-" * 60)
    print("BY PITCHER HANDEDNESS")
    print("-" * 60)
    for hand_label in ("LHP", "RHP"):
        subset = [r for r in results if r["hand"] == hand_label]
        if not subset:
            continue
        w = sum(1 for r in subset if r["win"])
        g = statistics.mean(r["gap"] for r in subset)
        print("  vs " + hand_label + ":  " + str(w) + "/" + str(len(subset)) +
              "  (" + str(round(100 * w / len(subset), 1)) + "%)" +
              "   mean gap " + format(g, "+.4f"))
    print()

    # -- By cutoff --
    print("-" * 60)
    print("BY CUTOFF DATE (more in-season data as the season progresses)")
    print("-" * 60)
    for cutoff_label in CUTOFFS:
        subset = [r for r in results if r["cutoff"] == cutoff_label]
        if not subset:
            continue
        w = sum(1 for r in subset if r["win"])
        g = statistics.mean(r["gap"] for r in subset)
        print("  " + cutoff_label.replace("_", " ") + ":  " + str(w) + "/" + str(len(subset)) +
              "  (" + str(round(100 * w / len(subset), 1)) + "%)" +
              "   mean gap " + format(g, "+.4f"))
    print()

    # -- Per-team rollup (mirrors the "X of 30 teams" framing) --
    print("-" * 60)
    print("PER-TEAM ROLLUP (win rate across all its graded windows/hands)")
    print("-" * 60)
    by_team = {}
    for r in results:
        by_team.setdefault(r["team"], []).append(r)
    team_rows = []
    for team, rows in by_team.items():
        w = sum(1 for r in rows if r["win"])
        team_rows.append((team, w, len(rows), statistics.mean(r["gap"] for r in rows)))
    team_rows.sort(key=lambda t: t[3], reverse=True)
    for team, w, n, g in team_rows:
        print("  " + team.ljust(5) + str(w).rjust(3) + "/" + str(n).ljust(4) +
              "  mean gap " + format(g, "+.4f"))
    n_teams_favoring_optimal = sum(1 for _, w, n, _ in team_rows if w / n > 0.5)
    print()
    print("  Optimal favored more often than not in " + str(n_teams_favoring_optimal) +
          " of " + str(len(team_rows)) + " teams tested.")
    print()

    # -- Most lopsided cases each direction --
    print("-" * 60)
    print("MOST LOPSIDED CASES")
    print("-" * 60)
    by_gap_desc = sorted(results, key=lambda r: r["gap"], reverse=True)[:10]
    by_gap_asc  = sorted(results, key=lambda r: r["gap"])[:10]
    print("  Optimal lineup won BIGGEST:")
    for r in by_gap_desc:
        print("    " + str(r["year"]) + " " + r["cutoff"].ljust(8) + " " + r["team"].ljust(5) +
              " vs " + r["hand"].ljust(3) + "  gap " + format(r["gap"], "+.4f") +
              "  (real " + format(r["real_avg"], ".3f") + " n=" + str(r["n_real"]) +
              ", opt " + format(r["opt_avg"], ".3f") + " n=" + str(r["n_opt"]) + ")")
    print("  Real lineup won BIGGEST:")
    for r in by_gap_asc:
        print("    " + str(r["year"]) + " " + r["cutoff"].ljust(8) + " " + r["team"].ljust(5) +
              " vs " + r["hand"].ljust(3) + "  gap " + format(r["gap"], "+.4f") +
              "  (real " + format(r["real_avg"], ".3f") + " n=" + str(r["n_real"]) +
              ", opt " + format(r["opt_avg"], ".3f") + " n=" + str(r["n_opt"]) + ")")
    print()
    print("=== DONE ===")


if __name__ == "__main__":
    tee = Tee(OUTPUT_FILE)
    sys.stdout = tee
    try:
        main()
    except Exception as e:
        print("ERROR: " + str(e))
        import traceback
        traceback.print_exc()
    tee.close()
    sys.stdout = tee.console
