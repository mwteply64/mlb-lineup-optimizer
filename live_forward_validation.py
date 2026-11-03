#!/usr/bin/env python3
"""
Live Forward-Validation - Lineup Optimization Project
========================================================
Restructures run_comparison()'s "runs left on table" check so it stops
grading lineups on the same metric used to build them.

THE PROBLEM THIS FIXES
-----------------------
test.py's run_comparison() builds the "optimal" lineup by maximizing the
model's own Valuation_wRC+ estimate, then grades BOTH the real and
optimal lineups using that same Valuation_wRC+ lookup. A lineup picked
to maximize a metric will almost always score highest on that exact
metric -- so the comparison can essentially never show the real lineup
winning. Today's actual results_2026-06-23.txt: 28/30 teams (93%) show
a positive "season composite runs left on table," averaging ~56
runs/season/team. lineup_forward_test.py (the validated, non-circular
HISTORICAL check, graded on real outcomes from 2019-2025) instead shows
the model beating the real lineup 66% of the time, with a much smaller
implied magnitude -- i.e. today's self-graded number is inflated mostly
because it's tautological, not because the model is that much better.

User's explicit direction (2026-06-23): pursue the full structural fix
-- stop self-grading entirely -- rather than a cheaper reframe/discount.

DESIGN (non-circular by construction, mirrors lineup_forward_test.py's
already-validated approach, but scoped to the CURRENT 2026 season's
actual personnel/decisions instead of 2019-2025 history):

For each rolling cutoff date D (computed automatically below, spaced
CUTOFF_SPACING_DAYS apart across the 2026 season-to-date):
  1. Use ONLY Statcast data from Opening Day through D ("pre-D") --
     exactly what a manager and this optimizer would have known on
     date D, nothing later.
  2. From that pre-D snapshot, build:
       - team_map_D: which team each player actually played for,
         derived straight from Statcast (home_team/away_team +
         inning_topbot), NOT from roster_resource.xlsx's current Team
         column. Avoids any staleness from in-season trades between D
         and today -- one consistent source of truth for team
         membership in both halves of the comparison below.
       - REAL lineup as of D, per team per hand: the top-9 players by
         OBSERVED plate appearances against that specific pitcher hand.
         FanGraphs RosterResource has no historical archive of past
         lineup cards (confirmed via web search 2026-06-23 -- Excel
         downloads and depth-chart pages are current-state only), so
         this Statcast-derived proxy -- the same one already validated
         in lineup_forward_test.py -- is the only viable option for
         grading a PAST decision point. Per the user's 2026-06-23
         decision.
       - OPTIMAL lineup as of D, per team per hand: test.py's own,
         COMPLETELY UNCHANGED build_master() / build_fallback_pool() /
         assign_lineup_positions() / balance_handedness() pipeline, fed
         this pre-D snapshot instead of today's live files. No tuned
         constant, scoring formula, or ordering rule is touched here --
         only the input data's as-of date changes.
  3. Grade BOTH lineups against REAL post-D outcomes (D+1 through
     yesterday) -- now fully known, since D is in the past. Neither
     lineup was built using any post-D data, so neither can game this
     grading step. Grading metric is plain average actual wOBA across
     graded players (MIN_PA_POST qualifying PA each), not a slot-
     weighted run value -- deliberately avoids assuming a batting-order
     for the real lineup, which the PA-share proxy can't give us. Same
     choice already made and explained to the user for
     lineup_forward_test.py.

Win rate and mean wOBA gap accumulate across every (cutoff, team, hand)
combination tested so far -- same statistics lineup_forward_test.py
already reports, except scoped to THIS season's actual rosters and
decision points, which is what matters for judging this season's
actual lineup calls. The rolling schedule automatically grows by one
more cutoff every CUTOFF_SPACING_DAYS as the season progresses -- just
re-run this script periodically (see CADENCE below).

WHAT THIS DELIBERATELY DOES NOT DO
------------------------------------
- Does not change DECAY_CAREER_*, REGRESS_PA_*, XWOBA_BLEND_*,
  AGING_STRENGTH, WEIGHT_*, or any other tuned constant, in test.py or
  anywhere else.
- Does not touch backtest.py, lineup_forward_test.py, or any already-
  validated historical CSV.
- Does not modify roster_resource.xlsx or rely on its Team or Role
  columns at all (deliberate, per the historical-archive limitation
  above) -- it IS still used, read-only, for position eligibility
  (Positions) and bats-handedness, which don't need a historical
  snapshot since they rarely change mid-season.
- Does NOT read career_vs_left.csv / career_vs_right.csv off disk for
  scoring (see 2026-06-24 note below for why).

2026-06-24 UPDATE -- career is now rebuilt per cutoff, not read from disk
--------------------------------------------------------------------------
update_live_data.py changed on 2026-06-24 to fold the current 2026
season-to-date into career_vs_left/right.csv (at full/no-decay weight),
so those files now depend on whatever day update_live_data.py was last
run. Reusing one static copy of them across every rolling cutoff here
(as this script did before) would leak post-cutoff 2026 data into
earlier folds -- e.g. the April 15 cutoff would see career numbers
informed by games played in May and June.

Fix: run_one_cutoff() no longer takes career_right_csv/career_left_csv
as arguments. It rebuilds career-as-of-D itself, every cutoff, by
calling update_live_data.py's build_career_file() with
current_season_df=splits_r/splits_l -- i.e. THIS cutoff's own pre-D
2026 splits (already computed by build_as_of_inputs() from raw_pre),
not today's. The static historical seasons (2016-2025) folded in
underneath are unaffected by cutoff D, so this stays cheap and exactly
mirrors live production's math, just evaluated as of D instead of as
of today.

WHY test.py NEEDED A ONE-LINE FIX FIRST
------------------------------------------
test.py's entry point ran its main()-equivalent code unconditionally at
import time -- same issue already fixed in backtest.py (Task #116) so
lineup_forward_test.py could import it safely. Added the identical
`if __name__ == "__main__":` guard to test.py on 2026-06-23 so this
script can import its scoring functions read-only. Zero behavior change
for `python test.py` directly.

OUTPUT
------
live_forward_validation_results_<date>.txt -- full human-readable report,
    same style as lineup_forward_test.py's report.
live_forward_validation_summary.json -- compact per-team rolling
    aggregate, OVERWRITTEN each run (not dated, since it's a rolling
    cache, not a point-in-time snapshot). test.py's run_comparison()
    reads this file, if present, to display forward-validated numbers
    in place of the old self-graded "runs left on table" table.

CADENCE
-------
This pulls several weeks of full-league Statcast data and is meant to
be re-run periodically (e.g. weekly), NOT every time you run test.py --
that's why it's a separate script that writes a cached summary file
rather than being wired directly into test.py's daily run. test.py
stays fast, local-file-only, and network-free for everyday use.

SETUP: same as the other download scripts (pip install pybaseball pandas).
This sandbox has no network access, so none of the Statcast pulls below
have been live-verified end-to-end. Run with --smoke-test FIRST (runs
the complete pipeline for just the earliest cutoff and one team) and
inspect the output before trusting a full run.

USAGE:
    python live_forward_validation.py --smoke-test
    python live_forward_validation.py
    (or double-click run_live_forward_validation.bat)

EXPERIMENT FLAG (added 2026-07-08): --history-only-career
    Rebuilds each cutoff's career files from HISTORICAL seasons only
    (current_season_df=None), i.e. the pre-2026-06-24 career definition
    that every weight/regression constant was actually tuned under --
    isolating whether folding the in-progress 2026 season into career
    (where it double-counts with the splits/total/recent components) is
    hurting live accuracy. Every output gets a "_histcareer" suffix
    (results .txt, detail .csv, build log .csv, summary .json), so an
    experiment run can NEVER overwrite the production summary JSON that
    test.py's run_comparison() displays, nor a same-day normal run's
    results. Compare overall and by-cutoff win rates against the normal
    run: if the decline across cutoffs flattens under this flag, the
    career fold-in is the culprit.

EXPERIMENT FLAG (added 2026-07-08): --hand-agnostic-real
    Builds the REAL-lineup benchmark from top-9 by TOTAL pre-D PA (same
    9 players graded vs both hands) instead of top-9 by PA vs the
    specific pitcher hand. This mirrors lineup_forward_test.py's
    historical methodology exactly -- the historical 66.1% baseline was
    measured against a hand-AGNOSTIC real lineup, while this script's
    default benchmark is hand-specific (it encodes the manager's actual
    platoon deployment, a much stronger opponent whose information
    content grows as the season progresses). Run with this flag to get
    the live number that is actually comparable to the historical
    baseline. Outputs get a "_handagnostic" suffix; combinable with
    --history-only-career (suffixes stack).

2026-07-09 CHANGE -- symmetric measurement is now the DEFAULT
--------------------------------------------------------------
Two changes make the default run measure exactly what
lineup_forward_test.py measures historically:

1. OPTIMAL lineup = plain top-9 by the model's Stabilized score vs this
   hand, position-agnostic -- mirrors lineup_forward_test.py's
   optimal_lineup() exactly. Combos where fewer than 9 team players
   could be scored are skipped (same full-9 requirement as the
   historical test). The old behavior -- test.py's full position-DP /
   fallback-pool / ordering pipeline -- is behind the --full-pipeline
   flag (outputs suffixed "_fullpipeline"). Grading was always
   order-insensitive, so what --full-pipeline actually adds is the
   position-feasibility constraint and fallback capping, i.e. it tests
   the production machinery rather than the model's ranking skill.

2. IL LEAKAGE FIX: the roster is now loaded with
   exclude_unavailable=False. Players on the IL *today* were previously
   excluded from every past cutoff's optimal pool -- future injury
   knowledge leaking into a past decision, and the cause of most
   "Could not fill slot" warnings. A player hurt now was a perfectly
   valid pick in April; if he truly didn't play after a cutoff, the
   MIN_PA_POST grading gate already handles it. Production test.py is
   unaffected (its default stays exclude_unavailable=True -- you don't
   field injured players TODAY).

With these defaults, this script's real benchmark (top-9 by PA vs the
specific hand) matches lineup_forward_test.py's new default benchmark,
and both "optimal" definitions are unconstrained model top-9s -- past
and present finally measure the same thing.
"""

import os
import sys
import csv
import json
import warnings
from datetime import date, timedelta

import pandas as pd

import download_v3_expansion as dv3
from update_live_data import (
    SEASON_2026_START, LIVE_TEST_YEAR, build_career_file,
    DECAY_CAREER_LHP, DECAY_CAREER_RHP,
)
from backtest import MIN_PA_POST
import test as opt  # read-only import of the live optimizer's own functions/constants

try:
    from pybaseball import cache
    cache.enable()
except ImportError:
    print("ERROR: pybaseball not installed.")
    print("Run: pip install pybaseball pandas")
    sys.exit(1)


# == Configuration ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Experiment flags (see module docstring). Each suffixes every output
# filename so an experiment run stays fully separate from production
# outputs -- in particular the summary JSON test.py displays.
HISTORY_ONLY_CAREER = "--history-only-career" in sys.argv
HAND_AGNOSTIC_REAL = "--hand-agnostic-real" in sys.argv
POSITION_CONSTRAINED = "--full-pipeline" in sys.argv
RUN_SUFFIX = (("_histcareer" if HISTORY_ONLY_CAREER else "") +
              ("_handagnostic" if HAND_AGNOSTIC_REAL else "") +
              ("_fullpipeline" if POSITION_CONSTRAINED else ""))

# --as-of YYYY-MM-DD pins "today" (default: the real date). After the
# season ends, pass the day after the final regular-season game so the
# cutoff schedule stops where post-cutoff grading data actually exists.
AS_OF = (date.fromisoformat(sys.argv[sys.argv.index("--as-of") + 1])
         if "--as-of" in sys.argv else date.today())

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT, "live_forward_validation_results_" + date.today().isoformat() + RUN_SUFFIX + ".txt"
)
SUMMARY_JSON = os.path.join(PROJECT_ROOT, "live_forward_validation_summary" + RUN_SUFFIX + ".json")

CUTOFF_SPACING_DAYS  = 14  # rolling cutoffs every 2 weeks
MIN_PRE_CUTOFF_DAYS  = 21  # need >= 3 weeks of pre-D PA before splits mean much
MIN_POST_CUTOFF_DAYS = 21  # need >= 3 weeks of post-D PA to grade reliably

LINEUP_SIZE   = opt.LINEUP_SIZE   # 9, imported rather than re-stated
MIN_TEAM_POOL = LINEUP_SIZE       # need at least a full lineup's worth of tagged players
MIN_GRADED    = 5                 # same convention as lineup_forward_test.py


# == Console + file logging (same pattern as the other scripts) =============

class Tee:
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# == Step 0: rolling cutoff schedule =========================================

def build_cutoff_schedule(today, season_start=SEASON_2026_START):
    """
    Every CUTOFF_SPACING_DAYS days, starting MIN_PRE_CUTOFF_DAYS after
    Opening Day, ending MIN_POST_CUTOFF_DAYS before today. Grows by one
    more cutoff automatically as the season progresses -- no manual
    date updates needed on re-runs.
    """
    start = date.fromisoformat(season_start)
    earliest = start + timedelta(days=MIN_PRE_CUTOFF_DAYS)
    latest = today - timedelta(days=MIN_POST_CUTOFF_DAYS)
    cutoffs = []
    d = earliest
    while d <= latest:
        cutoffs.append(d.isoformat())
        d += timedelta(days=CUTOFF_SPACING_DAYS)
    return cutoffs


# == Step 1: team tagging (mirrors download_team_tags.py's aggregate_with_team,
#            proven on live 2026 dates by that script's own --smoke-test) ====

def team_tag(raw, hand_code=None):
    """
    Aggregate raw Statcast rows to (player_id, team) -> pa, optionally
    restricted to one pitcher hand (p_throws == hand_code). Team is
    derived per-pitch from home_team/away_team/inning_topbot -- a
    player traded mid-window naturally splits into two rows, one per
    team, exactly like download_team_tags.py's aggregate_with_team().
    """
    df = raw if hand_code is None else raw[raw["p_throws"] == hand_code]
    pa = df[df["events"].notna()].copy()
    cols = ["player_id", "team", "pa"]
    if pa.empty:
        return pd.DataFrame(columns=cols)
    required = ["batter", "events", "home_team", "away_team", "inning_topbot"]
    missing = [c for c in required if c not in pa.columns]
    if missing:
        raise KeyError("Statcast data missing columns needed for team tagging: " + ", ".join(missing))
    pa["team"] = pa["away_team"].where(pa["inning_topbot"] == "Top", pa["home_team"])
    grouped = (
        pa.groupby(["batter", "team"], sort=False)
        .agg(pa=("events", "count"))
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )
    grouped["player_id"] = grouped["player_id"].astype(str)
    return grouped[cols]


def resolve_team_map(team_tagged_total):
    """
    Each player -> the single team they accumulated the most pre-D PA
    for (handles trades the same way lineup_forward_test.py's
    load_team_map() does -- max-PA row wins).
    """
    if team_tagged_total.empty:
        return {}
    idx = team_tagged_total.groupby("player_id")["pa"].idxmax()
    resolved = team_tagged_total.loc[idx]
    return dict(zip(resolved["player_id"], resolved["team"]))


def real_lineup_pool(team_tagged_hand, team_map, team, n=LINEUP_SIZE):
    """Top-n players by OBSERVED PA against this specific pitcher hand, restricted to this team."""
    if team_tagged_hand.empty:
        return []
    df = team_tagged_hand.copy()
    df["resolved_team"] = df["player_id"].map(team_map)
    pool = df[df["resolved_team"] == team].sort_values("pa", ascending=False)
    return pool.head(n)["player_id"].tolist()


# == Step 2: build the optimizer's own "as-of-D" inputs ======================

def build_as_of_inputs(raw_pre, end_date, name_lookup):
    """
    Derive build_master()'s four data components (splits vs each hand,
    season total, last-14-days recent) entirely from the already-
    fetched pre-D raw Statcast rows -- no extra network calls. Mirrors
    update_live_data.py's pull_2026_live()/pull_recent(), just sliced
    from one shared raw pull instead of two separate fetch_range() calls.
    """
    splits_r = dv3.aggregate(raw_pre[raw_pre["p_throws"] == "R"], name_lookup, True, LIVE_TEST_YEAR)
    splits_l = dv3.aggregate(raw_pre[raw_pre["p_throws"] == "L"], name_lookup, True, LIVE_TEST_YEAR)
    total_df = dv3.aggregate(raw_pre, name_lookup, True, LIVE_TEST_YEAR)
    recent_start = (date.fromisoformat(end_date) - timedelta(days=13)).isoformat()
    recent_raw = raw_pre[raw_pre["game_date"].astype(str) >= recent_start]
    recent_df = dv3.aggregate(recent_raw, name_lookup, True, LIVE_TEST_YEAR)
    return splits_r, splits_l, total_df, recent_df


def build_post_lookup(raw_post, name_lookup, hand_code):
    """Real, now-known post-D outcomes for grading. Mirrors backtest.py's 'after' data."""
    df = raw_post[raw_post["p_throws"] == hand_code]
    agg = dv3.aggregate(df, name_lookup, False, LIVE_TEST_YEAR)
    if agg.empty:
        return {}
    agg["player_id"] = agg["player_id"].astype(str)
    return {row.player_id: {"pa": row.pa, "woba": row.woba} for row in agg.itertuples()}


def grade(pids, post_lookup):
    vals = []
    for pid in pids:
        info = post_lookup.get(str(pid))
        if info and info["pa"] >= MIN_PA_POST and info["woba"] == info["woba"]:  # not NaN
            vals.append(info["woba"])
    if len(vals) < MIN_GRADED:
        return None, len(vals)
    return sum(vals) / len(vals), len(vals)


# == Step 3: one cutoff, all teams, both hands ===============================

def run_one_cutoff(cutoff, raw_pre, raw_post, name_lookup, roster_base,
                    debug_team=None, build_log=None):
    """
    Returns a list of result dicts, one per (team, hand) combination
    that had enough data to grade on both sides.

    debug_team: if set, prints raw pre-grading counts for that team only
    (slots actually filled by assign_lineup_positions, and size of the
    real-lineup candidate pool) so smoke_test() can tell apart "the
    optimizer couldn't fill a slot" from "it filled the lineup fine but
    a player didn't clear MIN_PA_POST against this specific hand post-
    cutoff" -- both can produce the same n_opt/n_real, but only the first
    is a build problem.

    build_log: if a list is passed in, every (team, hand) combo this
    cutoff actually iterates over (regardless of whether it ends up
    graded) gets one record appended -- {cutoff, team, hand,
    filled_slots, total_slots, fallback_slots_used, real_pool_size}.
    Lets a downstream check cross-reference which teams had an
    incomplete optimal-lineup build (filled_slots < total_slots, i.e. a
    "Could not fill slot" warning) against that team's win/gap numbers,
    instead of guessing from the aggregate warning counts alone.
    fallback_slots_used (2026-06-25 addition) separately counts slots
    that WERE filled, but only via opt.build_fallback_pool()'s
    deliberately conservative, hard-capped (FALLBACK_WRC_CAP) tier --
    distinct from filled_slots < total_slots, which only catches total
    failure to fill. A team can be filled_slots == total_slots == 9
    every time and still be leaning on fallback picks every cutoff;
    this field is what makes that visible.

    career_right_csv/career_left_csv are NOT passed in (see module
    docstring, 2026-06-24 update) -- they're rebuilt fresh below, every
    call, from THIS cutoff's own pre-D splits, so career never sees
    anything past D.
    """
    results = []

    team_tagged_total = team_tag(raw_pre, hand_code=None)
    team_map = resolve_team_map(team_tagged_total)
    if not team_map:
        print("  No team-taggable PA before " + cutoff + " -- skipping this cutoff.")
        return results

    team_tagged_r = team_tag(raw_pre, hand_code="R")
    team_tagged_l = team_tag(raw_pre, hand_code="L")

    splits_r, splits_l, total_df, recent_df = build_as_of_inputs(raw_pre, cutoff, name_lookup)

    # Career-as-of-D: historical seasons (unaffected by D) + THIS cutoff's
    # own pre-D 2026 splits, folded in at the current season's always-1.0
    # weight (decay ** 0) -- same call update_live_data.py's main() makes
    # for "today," just fed pre-D splits instead of today's.
    # Under --history-only-career (experiment, see module docstring) the
    # 2026 fold-in is dropped and career is rebuilt from completed prior
    # seasons only -- the definition the constants were tuned under.
    current_r = None if HISTORY_ONLY_CAREER else splits_r
    current_l = None if HISTORY_ONLY_CAREER else splits_l
    career_right_csv = build_career_file("vs_right", DECAY_CAREER_RHP, current_season_df=current_r)
    career_left_csv = build_career_file("vs_left", DECAY_CAREER_LHP, current_season_df=current_l)

    roster_d = roster_base.copy()
    roster_d[opt.TEAM_COL] = roster_d[opt.ROSTER_ID_COL].astype(str).map(team_map)

    print("  Building master tables as of " + cutoff + " ...")
    master_right = opt.build_master(splits_r.copy(), career_right_csv.copy(), career_left_csv.copy(),
                                     total_df.copy(), recent_df.copy(), roster_d, side="R")
    master_left = opt.build_master(splits_l.copy(), career_left_csv.copy(), career_right_csv.copy(),
                                    total_df.copy(), recent_df.copy(), roster_d, side="L")

    # League-average Stabilized_wRC+ per hand, as of this cutoff -- passed
    # into opt.build_fallback_pool() below as the placeholder score for a
    # true zero-data roster player (mirrors test.py's run_optimizer()).
    league_avg_right = master_right["Stabilized_wRC+"].mean()
    league_avg_left  = master_left["Stabilized_wRC+"].mean()

    post_lookup_r = build_post_lookup(raw_post, name_lookup, "R")
    post_lookup_l = build_post_lookup(raw_post, name_lookup, "L")

    teams = sorted(set(team_map.values()))
    for team in teams:
        team_pool_size = sum(1 for t in team_map.values() if t == team)
        if team_pool_size < MIN_TEAM_POOL:
            continue

        # Under --hand-agnostic-real (experiment, see module docstring)
        # the real-lineup benchmark ranks by TOTAL pre-D PA -- the same
        # hand-agnostic proxy lineup_forward_test.py's historical
        # baseline uses -- instead of PA vs this specific hand.
        real_source_r = team_tagged_total if HAND_AGNOSTIC_REAL else team_tagged_r
        real_source_l = team_tagged_total if HAND_AGNOSTIC_REAL else team_tagged_l
        for hand, master, team_tagged_hand, post_lookup in (
            ("RHP", master_right, real_source_r, post_lookup_r),
            ("LHP", master_left,  real_source_l, post_lookup_l),
        ):
            team_data = master[master[opt.TEAM_COL] == team] if len(master) else master
            if len(team_data) == 0:
                continue
            if POSITION_CONSTRAINED:
                # Legacy --full-pipeline mode: test.py's production
                # machinery (position DP, fallback pool, ordering rules).
                career_for_fallback = career_right_csv if hand == "RHP" else career_left_csv
                league_avg_for_hand = league_avg_right if hand == "RHP" else league_avg_left
                fb_pool = opt.build_fallback_pool(team, total_df.copy(), career_for_fallback.copy(), roster_d,
                                                   league_avg_wrc=league_avg_for_hand)
                assigned = opt.assign_lineup_positions(team_data, fb_pool)
                ordered = opt.balance_handedness(assigned)
                optimal_pids = [str(p["PlayerId"]) for p in ordered if p is not None]
                fallback_slots_used = sum(1 for p in ordered if p is not None and opt.is_fallback(p))
                total_slots = len(assigned)
            else:
                # Default (2026-07-09): position-agnostic top-9 by the
                # model's Stabilized score vs this hand -- mirrors
                # lineup_forward_test.py's optimal_lineup() exactly,
                # including the full-9 requirement. Grading is order-
                # insensitive, so no ordering step is needed.
                ranked = team_data.sort_values("Stabilized_wRC+", ascending=False)
                optimal_pids = [str(pid) for pid in ranked["PlayerId"].head(LINEUP_SIZE)]
                if len(optimal_pids) < LINEUP_SIZE:
                    continue
                assigned = None
                fallback_slots_used = 0
                total_slots = LINEUP_SIZE

            real_pids = real_lineup_pool(team_tagged_hand, team_map, team)

            # How many of the optimal-lineup picks are NOT in the real
            # lineup pool (2026-06-26 addition) -- personnel-change
            # magnitude metric, mirrors lineup_forward_test.py's same-day
            # addition. Independent of win/loss or gap size.
            n_diff = len(set(optimal_pids) - set(real_pids))

            if build_log is not None:
                build_log.append({
                    "cutoff": cutoff, "team": team, "hand": hand,
                    "filled_slots": len(optimal_pids), "total_slots": total_slots,
                    "fallback_slots_used": fallback_slots_used,
                    "real_pool_size": len(real_pids),
                })

            if debug_team is not None and team == debug_team:
                if assigned is not None:
                    filled_slots = [s for s, p in assigned.items() if p is not None]
                    empty_slots = [s for s, p in assigned.items() if p is None]
                    print("  [debug] " + team + " " + hand + ": optimal lineup filled " +
                          str(len(filled_slots)) + "/" + str(len(assigned)) + " slots" +
                          (" (empty: " + ", ".join(empty_slots) + ")" if empty_slots else "") +
                          "; real-lineup candidate pool pre-grade = " + str(len(real_pids)) + " players.")
                else:
                    print("  [debug] " + team + " " + hand + ": position-agnostic top-" +
                          str(LINEUP_SIZE) + " selected " + str(len(optimal_pids)) + " players" +
                          "; real-lineup candidate pool pre-grade = " + str(len(real_pids)) + " players.")

            opt_avg, n_opt = grade(optimal_pids, post_lookup)
            real_avg, n_real = grade(real_pids, post_lookup)
            if opt_avg is None or real_avg is None:
                continue

            gap = round(opt_avg - real_avg, 5)
            results.append({
                "cutoff": cutoff, "team": team, "hand": hand,
                "real_avg": round(real_avg, 5), "opt_avg": round(opt_avg, 5),
                "gap": gap, "n_real": n_real, "n_opt": n_opt,
                "win": gap > 0, "n_diff": n_diff,
            })

    return results


# == Reporting ================================================================

def write_reports(all_results, cutoffs_tested, skipped_cutoffs):
    print()
    print("=" * 70)
    print("LIVE FORWARD-VALIDATION RESULTS")
    print("=" * 70)
    print()
    print("Cutoffs tested: " + ", ".join(cutoffs_tested))
    if skipped_cutoffs:
        print("Cutoffs skipped (no data): " + ", ".join(skipped_cutoffs))
    print("Team-hand-cutoff combinations graded: " + str(len(all_results)))
    print()

    if not all_results:
        print("No combinations could be graded. Nothing to report.")
        _write_summary_json([], cutoffs_tested)
        return

    # -- Per-combo detail CSV (2026-06-26 addition) --
    # Purely additive, mirrors lineup_forward_test.py's same-day addition:
    # writes the already-computed all_results list to disk, one row per
    # team-hand-cutoff combo, so downstream uncertainty tooling (bootstrap
    # CIs, permutation tests) can resample the actual graded outcomes
    # instead of only the aggregated .txt/.json. Changes nothing about the
    # existing printed report or summary JSON below.
    detail_path = os.path.join(
        PROJECT_ROOT, "live_forward_validation_detail_" + date.today().isoformat() + RUN_SUFFIX + ".csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cutoff", "team", "hand", "real_avg",
                                           "opt_avg", "gap", "n_real", "n_opt", "win", "n_diff"])
        w.writeheader()
        w.writerows(all_results)
    print("Per-combo detail written to: " + detail_path)
    print()

    wins = sum(1 for r in all_results if r["win"])
    win_rate = wins / len(all_results)
    mean_gap = sum(r["gap"] for r in all_results) / len(all_results)
    print("Overall: optimal beat real in " + str(wins) + "/" + str(len(all_results)) +
          " (" + str(round(100 * win_rate, 1)) + "%)")
    print("Overall mean wOBA gap (optimal - real): " + str(round(mean_gap, 5)))
    print()

    for hand_label in ("RHP", "LHP"):
        sub = [r for r in all_results if r["hand"] == hand_label]
        if not sub:
            continue
        w = sum(1 for r in sub if r["win"])
        g = sum(r["gap"] for r in sub) / len(sub)
        print("By hand -- vs " + hand_label + ": " + str(w) + "/" + str(len(sub)) +
              " (" + str(round(100 * w / len(sub), 1)) + "%), mean gap " + str(round(g, 5)))
    print()

    print("By cutoff:")
    for cutoff in cutoffs_tested:
        sub = [r for r in all_results if r["cutoff"] == cutoff]
        if not sub:
            continue
        w = sum(1 for r in sub if r["win"])
        g = sum(r["gap"] for r in sub) / len(sub)
        print("  " + cutoff + ": " + str(w) + "/" + str(len(sub)) +
              " (" + str(round(100 * w / len(sub), 1)) + "%), mean gap " + str(round(g, 5)))
    print()

    by_team = {}
    for r in all_results:
        by_team.setdefault(r["team"], []).append(r)

    team_rollup = []
    for team, rows in by_team.items():
        w = sum(1 for r in rows if r["win"])
        g = sum(r["gap"] for r in rows) / len(rows)
        team_rollup.append({"team": team, "n": len(rows), "wins": w, "win_rate": w / len(rows), "mean_gap": g})
    team_rollup.sort(key=lambda r: r["mean_gap"], reverse=True)

    print("Per-team rollup (sorted by mean gap, descending):")
    print("Team    N   Wins  Win%    Mean Gap")
    print("----  ---  -----  ------  --------")
    for r in team_rollup:
        print(r["team"].ljust(6) + str(r["n"]).rjust(3) + "  " +
              str(r["wins"]).rjust(5) + "  " +
              (str(round(100 * r["win_rate"], 1)) + "%").rjust(6) + "  " +
              str(round(r["mean_gap"], 5)).rjust(8))
    n_favoring = sum(1 for r in team_rollup if r["mean_gap"] > 0)
    print()
    print(str(n_favoring) + "/" + str(len(team_rollup)) + " teams favor the optimal lineup on average.")
    print()

    _write_summary_json(team_rollup, cutoffs_tested, wins, len(all_results), mean_gap)


def _write_summary_json(team_rollup, cutoffs_tested, wins=0, n_total=0, mean_gap=None):
    payload = {
        "generated": date.today().isoformat(),
        "cutoffs_tested": cutoffs_tested,
        "n_windows_overall": n_total,
        "win_rate_overall": (wins / n_total) if n_total else None,
        "mean_gap_overall": mean_gap,
        "by_team": {
            r["team"]: {
                "n_windows": r["n"], "wins": r["wins"],
                "win_rate": round(r["win_rate"], 4), "mean_gap": round(r["mean_gap"], 5),
            }
            for r in team_rollup
        },
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Summary written to: " + SUMMARY_JSON)
    print("(test.py's run_comparison() reads this file to display forward-validated numbers.)")


# == Smoke test ===============================================================

def smoke_test():
    print("=== SMOKE TEST: one cutoff, one team, full pipeline ===")
    print()
    today = date.today()
    schedule = build_cutoff_schedule(today)
    if not schedule:
        print("No valid cutoff yet this season (too early / too close to today). Nothing to smoke-test.")
        return
    cutoff = schedule[0]
    print("Using earliest scheduled cutoff: " + cutoff)
    print("Full schedule would be: " + ", ".join(schedule))
    print()

    name_lookup = dv3.build_name_lookup()
    print()

    yesterday = (today - timedelta(days=1)).isoformat()
    raw_full = dv3.fetch_range(SEASON_2026_START, yesterday, "season-to-date through yesterday")
    if raw_full is None:
        print("No data returned -- check network/pybaseball.")
        return
    raw_full["game_date"] = raw_full["game_date"].astype(str)
    raw_pre = raw_full[raw_full["game_date"] <= cutoff]
    raw_post = raw_full[raw_full["game_date"] > cutoff]
    print("Pre-cutoff rows: " + str(len(raw_pre)) + "   Post-cutoff rows: " + str(len(raw_post)))
    print()

    team_tagged_total = team_tag(raw_pre, hand_code=None)
    team_map = resolve_team_map(team_tagged_total)
    if not team_map:
        print("PROBLEM: no team-taggable PA before this cutoff.")
        return
    sample_team = sorted(set(team_map.values()))[0]
    print("Inspecting team: " + sample_team)
    print()

    roster_base, _, _ = opt.load_roster(opt.FILE_ROSTER, exclude_unavailable=False)
    # exclude_unavailable=False (2026-07-09): current IL status must not
    # leak into past-cutoff candidate pools -- see module docstring.
    print()

    # --- Diagnostic: who from TODAY's active roster didn't get pre-cutoff
    # team-tagged to this team, and why (no PA at all yet, vs. tagged
    # elsewhere)? Distinguishes "correctly reflects roster churn since the
    # cutoff" from "a real player_id/team matching bug." Smoke-test only --
    # not part of the scored pipeline.
    current_roster_ids = set(
        roster_base[roster_base[opt.ROSTER_TEAM_COL] == sample_team][opt.ROSTER_ID_COL].astype(str)
    )
    tagged_to_team = {pid for pid, t in team_map.items() if t == sample_team}
    name_map = dict(zip(roster_base[opt.ROSTER_ID_COL].astype(str), roster_base[opt.ROSTER_NAME_COL]))
    missing = current_roster_ids - tagged_to_team
    print("Today's roster_resource.xlsx position players listed under " + sample_team +
          ": " + str(len(current_roster_ids)))
    print("Of those, pre-cutoff (" + cutoff + ") team-tagging assigned to " + sample_team +
          ": " + str(len(current_roster_ids & tagged_to_team)))
    if missing:
        print("Missing (" + str(len(missing)) + "):")
        for pid in sorted(missing):
            rows = team_tagged_total[team_tagged_total["player_id"] == pid]
            if rows.empty:
                detail = "zero recorded PA anywhere pre-cutoff"
            else:
                best = rows.loc[rows["pa"].idxmax()]
                detail = "resolved to team '" + str(best["team"]) + "' (" + str(int(best["pa"])) + " PA) -- NOT " + sample_team
            print("  " + name_map.get(pid, pid) + " (id " + pid + "): " + detail)
    print()

    results = run_one_cutoff(cutoff, raw_pre, raw_post, name_lookup, roster_base,
                              debug_team=sample_team)
    sample_results = [r for r in results if r["team"] == sample_team]
    if not sample_results:
        print("PROBLEM: no graded result for " + sample_team + " at this cutoff -- inspect thresholds.")
        return
    for r in sample_results:
        print(r)
    print()
    print("If the numbers above look sane (real_avg/opt_avg in plausible wOBA range, ~0.200-0.450,")
    print("n_real/n_opt >= " + str(MIN_GRADED) + "), the full pipeline is wired correctly.")
    print("Run without --smoke-test for the full rolling validation across all cutoffs/teams.")


# == Main ======================================================================

def main():
    warnings.filterwarnings("ignore")
    today = AS_OF
    schedule = build_cutoff_schedule(today)

    print("=" * 70)
    print("=== LIVE FORWARD-VALIDATION STARTED ===")
    print("=" * 70)
    if HISTORY_ONLY_CAREER:
        print("*** EXPERIMENT MODE: --history-only-career ***")
        print("Career files rebuilt from completed prior seasons only (no 2026")
        print("fold-in). All outputs carry a _histcareer suffix; the production")
        print("summary JSON is NOT touched by this run.")
        print()
    if HAND_AGNOSTIC_REAL:
        print("*** EXPERIMENT MODE: --hand-agnostic-real ***")
        print("Real-lineup benchmark = top-9 by TOTAL pre-D PA (hand-agnostic,")
        print("legacy benchmark). All outputs carry a _handagnostic suffix; the")
        print("production summary JSON is NOT touched by this run.")
        print()
    if POSITION_CONSTRAINED:
        print("*** LEGACY MODE: --full-pipeline ***")
        print("Optimal lineup built by test.py's position-DP / fallback-pool /")
        print("ordering pipeline (tests the production machinery, not just the")
        print("model's ranking). Outputs carry a _fullpipeline suffix.")
        print()
    else:
        print("Optimal lineup = position-agnostic top-9 by model score vs hand")
        print("(mirrors lineup_forward_test.py -- default since 2026-07-09).")
        print("Roster loaded with IL/restricted players INCLUDED (retrospective")
        print("mode -- current injury status must not leak into past cutoffs).")
        print()
    print("Rolling cutoff schedule (" + str(len(schedule)) + " windows): " + ", ".join(schedule))
    print()

    if not schedule:
        print("Season too young (or too close to today) for any valid rolling cutoff yet.")
        print("Needs >= " + str(MIN_PRE_CUTOFF_DAYS) + " days after Opening Day and >= " +
              str(MIN_POST_CUTOFF_DAYS) + " days before today.")
        return

    name_lookup = dv3.build_name_lookup()
    print()

    roster_base, _, _ = opt.load_roster(opt.FILE_ROSTER, exclude_unavailable=False)
    # exclude_unavailable=False (2026-07-09): current IL status must not
    # leak into past-cutoff candidate pools -- see module docstring.
    print()

    yesterday = (today - timedelta(days=1)).isoformat()
    raw_full = dv3.fetch_range(SEASON_2026_START, yesterday, "full season-to-date (shared across all cutoffs)")
    if raw_full is None:
        print("ERROR: could not fetch season-to-date data. Aborting.")
        return
    raw_full["game_date"] = raw_full["game_date"].astype(str)
    print()

    all_results = []
    cutoffs_tested = []
    skipped_cutoffs = []
    build_log = []
    for cutoff in schedule:
        print("--- Cutoff: " + cutoff + " ---")
        raw_pre = raw_full[raw_full["game_date"] <= cutoff]
        raw_post = raw_full[raw_full["game_date"] > cutoff]
        results = run_one_cutoff(cutoff, raw_pre, raw_post, name_lookup, roster_base,
                                  build_log=build_log)
        if results:
            all_results.extend(results)
            cutoffs_tested.append(cutoff)
            print("  " + str(len(results)) + " team-hand combinations graded.")
        else:
            skipped_cutoffs.append(cutoff)
            print("  No combinations could be graded for this cutoff.")
        print()

    if build_log:
        log_path = os.path.join(PROJECT_ROOT, "live_forward_validation_build_log" + RUN_SUFFIX + ".csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["cutoff", "team", "hand", "filled_slots",
                                               "total_slots", "fallback_slots_used",
                                               "real_pool_size"])
            w.writeheader()
            w.writerows(build_log)
        n_incomplete = sum(1 for r in build_log if r["filled_slots"] < r["total_slots"])
        n_fallback = sum(1 for r in build_log if r.get("fallback_slots_used", 0) > 0)
        print("Build-completeness log written to: " + log_path)
        print("(" + str(n_incomplete) + "/" + str(len(build_log)) +
              " team-hand-cutoff builds had at least one unfilled slot.)")
        print("(" + str(n_fallback) + "/" + str(len(build_log)) +
              " team-hand-cutoff builds used at least one fallback-pool slot.)")
        print()

    write_reports(all_results, cutoffs_tested, skipped_cutoffs)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        tee = Tee(os.path.join(PROJECT_ROOT, "live_forward_validation_smoketest_log.txt"))
        sys.stdout = tee
        try:
            smoke_test()
        except Exception as e:
            print("ERROR: " + str(e))
            import traceback
            traceback.print_exc()
        tee.close()
        sys.stdout = tee.console
    else:
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
