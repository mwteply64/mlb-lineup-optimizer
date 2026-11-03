#!/usr/bin/env python3
"""
Lineup Ablation Test - Lineup Optimization Project
====================================================
Turns off one model component/mechanism at a time and reruns the same
team-constrained, non-circular forward test lineup_forward_test.py
already validates (real lineup = top-9 by pre-cutoff PA, optimal
lineup = top-9 by the model's predicted stabilized wOBA, both graded
on real post-cutoff actual wOBA) -- to see which knobs are actually
carrying their weight in the END-TO-END team-lineup metric, as opposed
to the Spearman-r metric most of them were originally tuned on.

WHY A SEPARATE SCRIPT, NOT A FLAG IN lineup_forward_test.py
-------------------------------------------------------------
lineup_forward_test.py is itself currently affected by a known sandbox
bash-mount stale-file-cache bug from earlier this session (it was read
by bash before its last edit, and the sandbox keeps serving the old
bytes no matter how long you wait or rename the file -- inode-keyed,
not path-keyed). Importing it here would risk silently running that
stale, syntactically-broken copy inside THIS process. So this script
inlines its own verbatim copies of the small roster/grading helpers
(load_team_map, build_rosters, real_lineup, grade_lineup) instead of
importing them, and imports ONLY from backtest.py, which has not been
edited this session and is confirmed fresh. Nothing about the actual
validated logic of those helpers is changed by inlining them.

HOW EACH ABLATION WORKS
-------------------------------------------------------------
Every config still calls the REAL, completely unmodified
backtest.stabilize() -- ablation never rewrites its math. Two
different mechanisms are used, chosen per-knob for correctness:

  - The four core components (splits/career/total/recent) are
    disabled by feeding stabilize() pa=0 / woba=None for that
    component, for every player, before calling it. stabilize()'s own
    "if woba is not None and pa >= MIN_PA: ... weights.append(W)"
    gating then drops that component and renormalizes the remaining
    weights exactly the way it already does for any player who's
    simply missing that data -- this is the only way to fully zero out
    a component's influence without risking a divide-by-zero (setting
    its WEIGHT constant to 0 instead would crash for any player whose
    *only* usable data source was the ablated one, since stabilize()
    renormalizes by sum(weights), and 0/0 is undefined).
  - Platoon blend and the PA clamp are read by stabilize() directly
    from module-level constants (not passed as arguments), so those
    two are ablated by temporarily monkeypatching backtest.PLATOON_
    BLEND_LHP/RHP or backtest.NO_CLAMP for the duration of each scoring
    call, then restoring the original value immediately after.
  - xWOBA blend ratio and career decay ARE passed as explicit
    arguments (blend_woba()'s xwoba_blend param, build_career_data()'s
    decay param), so those are ablated by simply passing 0.0 / 1.0
    instead of the production constant -- no monkeypatching needed.

ABLATIONS TESTED
  FULL              -- baseline, no ablation (should reproduce
                       lineup_forward_test.py's own numbers)
  no_splits         -- drop current-year pre-cutoff hand-split data
  no_career         -- drop career data (including the cross-hand
                       career data used by the platoon prior)
  no_total          -- drop current-season-total data
  no_recent         -- drop recent-form data
  no_platoon        -- platoon tendency prior off (blend=0 both hands)
  no_prior          -- v16 batter-hand platoon prior off (PLATOON_PRIOR_PA_LHP=0)
  no_xwoba          -- pure actual wOBA, no xWOBA blending at all
  no_decay          -- career recency decay off (LHP forced to 1.0;
                       RHP unaffected, already 1.0 in production)
  clamp_on          -- re-enable the PA-based clamp (NO_CLAMP=False)
                       that production currently disables

AGING_STRENGTH is already 0.0 (off) in production, so there is nothing
to ablate there -- it would just reproduce FULL exactly. Not included.

METRIC
  Same one used everywhere else in this project: win rate and mean
  wOBA gap of the (ablated) optimal lineup vs the real lineup, graded
  on actual post-cutoff wOBA. A component that "carries its weight"
  should show a real drop in both numbers when it's removed; a
  component that's dead weight (or actively harmful) won't, or will
  even improve them.

USAGE
  python lineup_ablation_test.py
"""
import os
import csv
import datetime
import statistics

import numpy as np
import backtest as bt
from backtest import (
    DATA_DIR, TEST_YEARS, CUTOFFS,
    load_file, build_career_data, blend_woba, stabilize,
    DECAY_CAREER_LHP, DECAY_CAREER_RHP, AGING_STRENGTH, PLAYER_AGES,
    XWOBA_BLEND_SPLITS_LHP, XWOBA_BLEND_CAREER_LHP,
    XWOBA_BLEND_TOTAL_LHP, XWOBA_BLEND_RECENT_LHP,
    XWOBA_BLEND_SPLITS_RHP, XWOBA_BLEND_CAREER_RHP,
    XWOBA_BLEND_TOTAL_RHP, XWOBA_BLEND_RECENT_RHP,
    MIN_PA_TOTAL, MIN_PA_POST,
)

FOLDER = os.path.dirname(os.path.abspath(__file__))
today  = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(FOLDER, "lineup_ablation_test_results_" + today + ".txt")

LINEUP_SIZE = 9
MIN_GRADED  = 5
N_BOOT      = 10000
RNG_SEED    = 20260629  # fixed seed -- reproducible CIs run to run
rng = np.random.default_rng(RNG_SEED)


class Tee:
    def __init__(self, filepath):
        self.console = __import__("sys").stdout
        self.file = open(filepath, "w", encoding="utf-8")
    def write(self, text):
        self.console.write(text)
        self.file.write(text)
    def flush(self):
        self.console.flush()
        self.file.flush()
    def close(self):
        self.file.close()


# -------------------------------------------------------
# ROSTER HELPERS
# Verbatim copies of lineup_forward_test.py's logic (see module
# docstring for why these are inlined instead of imported).
# -------------------------------------------------------

def load_team_map(year, cutoff_label):
    filename = "totals_before_" + cutoff_label + "_" + str(year) + "_with_team.csv"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {}
    best = {}
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
    rosters = {}
    for pid, info in team_map.items():
        if info["team_pa"] < MIN_PA_TOTAL:
            continue
        rosters.setdefault(info["team"], []).append(pid)
    return rosters


def real_lineup(team_pids, team_map):
    return sorted(team_pids, key=lambda pid: team_map[pid]["team_pa"], reverse=True)[:LINEUP_SIZE]


def optimal_lineup(team_pids, scored, stab_key):
    eligible = [pid for pid in team_pids if scored.get(pid, {}).get(stab_key) is not None]
    return sorted(eligible, key=lambda pid: scored[pid][stab_key], reverse=True)[:LINEUP_SIZE]


def grade_lineup(pids, post_dict):
    vals = []
    for pid in pids:
        post = post_dict.get(pid)
        if post is not None and post["pa"] >= MIN_PA_POST:
            vals.append(post["woba"])
    if not vals:
        return None, 0
    return statistics.mean(vals), len(vals)


# -------------------------------------------------------
# ABLATION-AWARE SCORING
# -------------------------------------------------------

def blended(entry, blend):
    if entry is None:
        return None
    return blend_woba(entry["woba"], entry.get("xwoba"), blend)


def score_window_ablated(year, cutoff_label, ablation):
    pre_l  = load_file("before_" + cutoff_label + "_" + str(year) + "_vs_left.csv")
    pre_r  = load_file("before_" + cutoff_label + "_" + str(year) + "_vs_right.csv")

    decay_l = 1.0 if ablation.get("no_decay") else DECAY_CAREER_LHP
    decay_r = DECAY_CAREER_RHP  # already 1.0 in production either way
    car_l  = build_career_data(year, "left",  decay_l, AGING_STRENGTH, PLAYER_AGES)
    car_r  = build_career_data(year, "right", decay_r, AGING_STRENGTH, PLAYER_AGES)

    total  = load_file("totals_before_" + cutoff_label + "_" + str(year) + ".csv")
    recent = load_file("recent_before_" + cutoff_label + "_" + str(year) + ".csv")
    post_l = load_file("after_" + cutoff_label + "_" + str(year) + "_vs_left.csv")
    post_r = load_file("after_" + cutoff_label + "_" + str(year) + "_vs_right.csv")

    no_splits = ablation.get("no_splits", False)
    no_career = ablation.get("no_career", False)
    no_total  = ablation.get("no_total",  False)
    no_recent = ablation.get("no_recent", False)
    xb_override = 0.0 if ablation.get("no_xwoba") else None

    def xb(default):
        return xb_override if xb_override is not None else default

    all_pids = set(pre_l.keys()) | set(pre_r.keys())

    # -- platoon / clamp ablations: monkeypatch, always restore --
    orig_pb_l, orig_pb_r = bt.PLATOON_BLEND_LHP, bt.PLATOON_BLEND_RHP
    orig_no_clamp = bt.NO_CLAMP
    orig_prior_pa_l = bt.PLATOON_PRIOR_PA_LHP
    if ablation.get("no_platoon"):
        bt.PLATOON_BLEND_LHP = 0.0
        bt.PLATOON_BLEND_RHP = 0.0
    if ablation.get("no_prior"):
        # v16 batter-hand platoon prior off (LHP only -- RHP ships at 0 anyway)
        bt.PLATOON_PRIOR_PA_LHP = 0
    if ablation.get("clamp_on"):
        bt.NO_CLAMP = False

    try:
        scored = {}
        for pid in all_pids:
            sl  = pre_l.get(pid)
            sr  = pre_r.get(pid)
            cl  = car_l.get(pid)
            cr  = car_r.get(pid)
            tot = total.get(pid)
            rec = recent.get(pid)
            name = (sl or sr or tot or {}).get("name", pid)

            splits_pa_l   = 0 if no_splits else (sl["pa"] if sl else 0)
            splits_woba_l = None if no_splits else blended(sl, xb(XWOBA_BLEND_SPLITS_LHP))
            career_pa_l   = 0 if no_career else (cl["pa"] if cl else 0)
            career_woba_l = None if no_career else blended(cl, xb(XWOBA_BLEND_CAREER_LHP))
            total_pa_l    = 0 if no_total  else (tot["pa"] if tot else 0)
            total_woba_l  = None if no_total  else blended(tot, xb(XWOBA_BLEND_TOTAL_LHP))
            recent_pa_l   = 0 if no_recent else (rec["pa"] if rec else 0)
            recent_woba_l = None if no_recent else blended(rec, xb(XWOBA_BLEND_RECENT_LHP))
            cross_pa_l    = 0 if no_career else (cr["pa"] if cr else 0)
            cross_woba_l  = None if no_career else blended(cr, xb(XWOBA_BLEND_CAREER_LHP))

            stab_l = stabilize(
                splits_pa_l, splits_woba_l,
                career_pa_l, career_woba_l,
                total_pa_l,  total_woba_l,
                recent_pa_l, recent_woba_l,
                side='L',
                career_cross_pa=cross_pa_l,
                career_cross_woba=cross_woba_l,
                pid=pid,
            )

            splits_pa_r   = 0 if no_splits else (sr["pa"] if sr else 0)
            splits_woba_r = None if no_splits else blended(sr, xb(XWOBA_BLEND_SPLITS_RHP))
            career_pa_r   = 0 if no_career else (cr["pa"] if cr else 0)
            career_woba_r = None if no_career else blended(cr, xb(XWOBA_BLEND_CAREER_RHP))
            total_pa_r    = 0 if no_total  else (tot["pa"] if tot else 0)
            total_woba_r  = None if no_total  else blended(tot, xb(XWOBA_BLEND_TOTAL_RHP))
            recent_pa_r   = 0 if no_recent else (rec["pa"] if rec else 0)
            recent_woba_r = None if no_recent else blended(rec, xb(XWOBA_BLEND_RECENT_RHP))
            cross_pa_r    = 0 if no_career else (cl["pa"] if cl else 0)
            cross_woba_r  = None if no_career else blended(cl, xb(XWOBA_BLEND_CAREER_RHP))

            stab_r = stabilize(
                splits_pa_r, splits_woba_r,
                career_pa_r, career_woba_r,
                total_pa_r,  total_woba_r,
                recent_pa_r, recent_woba_r,
                side='R',
                career_cross_pa=cross_pa_r,
                career_cross_woba=cross_woba_r,
                pid=pid,
            )

            scored[pid] = {"name": name, "stab_l": stab_l, "stab_r": stab_r}
    finally:
        bt.PLATOON_BLEND_LHP, bt.PLATOON_BLEND_RHP = orig_pb_l, orig_pb_r
        bt.NO_CLAMP = orig_no_clamp
        bt.PLATOON_PRIOR_PA_LHP = orig_prior_pa_l

    return scored, post_l, post_r


# -------------------------------------------------------
# ONE FULL SWEEP FOR A GIVEN ABLATION CONFIG
# -------------------------------------------------------

def run_sweep(ablation):
    results = []
    for year in TEST_YEARS:
        for cutoff_label in CUTOFFS:
            team_map = load_team_map(year, cutoff_label)
            if not team_map:
                continue
            scored, post_l, post_r = score_window_ablated(year, cutoff_label, ablation)
            rosters = build_rosters(team_map)

            for team, pids in rosters.items():
                if len(pids) < LINEUP_SIZE:
                    continue
                real9 = real_lineup(pids, team_map)

                for hand_label, stab_key, post_dict in (
                    ("LHP", "stab_l", post_l),
                    ("RHP", "stab_r", post_r),
                ):
                    opt9 = optimal_lineup(pids, scored, stab_key)
                    if len(opt9) < LINEUP_SIZE:
                        continue
                    real_avg, n_real = grade_lineup(real9, post_dict)
                    opt_avg,  n_opt  = grade_lineup(opt9,  post_dict)
                    if real_avg is None or opt_avg is None:
                        continue
                    if n_real < MIN_GRADED or n_opt < MIN_GRADED:
                        continue
                    gap = opt_avg - real_avg
                    results.append({"year": year, "gap": gap, "win": gap > 0})
    return results


def summarize(results):
    n = len(results)
    if n == 0:
        return {"n": 0, "win_rate": None, "mean_gap": None}
    wins = sum(1 for r in results if r["win"])
    mean_gap = statistics.mean(r["gap"] for r in results)
    return {"n": n, "win_rate": wins / n, "mean_gap": mean_gap}


def by_year_wins(results):
    by_year = {}
    for r in results:
        by_year.setdefault(r["year"], []).append(r["win"])
    return {y: np.array(v, dtype=bool) for y, v in by_year.items()}


def paired_year_clustered_diff(full_results, ablated_results, n_boot=N_BOOT):
    """Is (ablated win rate - FULL win rate) distinguishable from year-to-
    year noise? Draws the SAME resampled year list each iteration and
    applies it to both configs' year-grouped win arrays (mirrors
    analyze_hand_difference's pairing logic in
    historical_uncertainty_analysis.py), then differences the two
    resulting pooled win rates. Each config's own combo set is used (they
    differ slightly -- some combos drop below MIN_GRADED once a data
    source is removed -- so this isn't a row-for-row pairing, but the
    shared year draw still captures most of the cross-config year
    correlation that matters)."""
    full_by_year = by_year_wins(full_results)
    abl_by_year  = by_year_wins(ablated_results)
    years = sorted(set(full_by_year) | set(abl_by_year))
    n_years = len(years)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, n_years, size=n_years)
        full_w = np.concatenate([full_by_year[years[i]] for i in chosen if years[i] in full_by_year])
        abl_w  = np.concatenate([abl_by_year[years[i]]  for i in chosen if years[i] in abl_by_year])
        diffs[b] = abl_w.mean() - full_w.mean()
    full_rate = np.concatenate([full_by_year[y] for y in years if y in full_by_year]).mean()
    abl_rate  = np.concatenate([abl_by_year[y]  for y in years if y in abl_by_year]).mean()
    observed = abl_rate - full_rate
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return observed, lo, hi


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

ABLATIONS = [
    ("FULL (baseline)",                {}),
    ("no_splits",                      {"no_splits": True}),
    ("no_career",                      {"no_career": True}),
    ("no_total",                       {"no_total": True}),
    ("no_recent",                      {"no_recent": True}),
    ("no_platoon",                     {"no_platoon": True}),
    ("no_prior",                       {"no_prior": True}),
    ("no_xwoba",                       {"no_xwoba": True}),
    ("no_decay (LHP only, RHP=1.0 already)", {"no_decay": True}),
    ("clamp_on (re-enable PA clamp)",  {"clamp_on": True}),
]


def main():
    print("=== LINEUP ABLATION TEST ===")
    print("Same team-constrained forward-test metric as lineup_forward_test.py.")
    print("Results saved to: " + OUTPUT_FILE)
    print()

    rows = []
    raw_by_name = {}
    baseline = None
    full_results = None
    for name, ablation in ABLATIONS:
        print("Running: " + name + " ...")
        results = run_sweep(ablation)
        raw_by_name[name] = results
        s = summarize(results)
        rows.append((name, s))
        if name.startswith("FULL"):
            baseline = s
            full_results = results
        print("  n=" + str(s["n"]) +
              "  win rate=" + (format(s["win_rate"], ".1%") if s["win_rate"] is not None else "N/A") +
              "  mean gap=" + (format(s["mean_gap"], "+.4f") if s["mean_gap"] is not None else "N/A"))

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("{:<42s} {:>6s} {:>10s} {:>12s} {:>10s} {:>10s}".format(
        "config", "n", "win rate", "mean gap", "d win%", "d gap"))
    print("-" * 78)
    for name, s in rows:
        if s["win_rate"] is None:
            print("{:<42s} {:>6d}   (no combos graded)".format(name, s["n"]))
            continue
        d_win = (s["win_rate"] - baseline["win_rate"]) * 100 if baseline["win_rate"] is not None else None
        d_gap = (s["mean_gap"] - baseline["mean_gap"]) if baseline["mean_gap"] is not None else None
        d_win_str = (format(d_win, "+.1f") + "pp") if d_win is not None and not name.startswith("FULL") else "--"
        d_gap_str = format(d_gap, "+.4f") if d_gap is not None and not name.startswith("FULL") else "--"
        print("{:<42s} {:>6d} {:>9.1%} {:>+12.4f} {:>10s} {:>10s}".format(
            name, s["n"], s["win_rate"], s["mean_gap"], d_win_str, d_gap_str))

    print()
    print("-" * 78)
    print("SIGNIFICANCE -- year-clustered bootstrap on (ablated win rate - FULL win rate)")
    print("(6 year-clusters, " + str(N_BOOT) + " draws, seed=" + str(RNG_SEED) + ")")
    print("-" * 78)
    sig_lines = []
    for name, ablation in ABLATIONS:
        if name.startswith("FULL"):
            continue
        observed, lo, hi = paired_year_clustered_diff(full_results, raw_by_name[name])
        excludes_zero = not (lo <= 0 <= hi)
        line = "  {:<42s} {:>+7.1%}   95% CI [{:>+7.1%}, {:>+7.1%}]   {}".format(
            name, observed, lo, hi,
            "distinguishable from noise" if excludes_zero else "NOT distinguishable from noise")
        print(line)
        sig_lines.append(line)

    print()
    print("Reading the table: a component/mechanism that \"carries its weight\"")
    print("should show a clear NEGATIVE d-win% / d-gap when removed (performance")
    print("drops without it). A flat or positive delta means that knob isn't")
    print("contributing in this end-to-end metric, whatever its earlier")
    print("rank-correlation tuning showed.")
    print()
    print("=== DONE ===")

    out_lines = []
    out_lines.append("=" * 78)
    out_lines.append("SUMMARY")
    out_lines.append("=" * 78)
    out_lines.append("{:<42s} {:>6s} {:>10s} {:>12s} {:>10s} {:>10s}".format(
        "config", "n", "win rate", "mean gap", "d win%", "d gap"))
    out_lines.append("-" * 78)
    for name, s in rows:
        if s["win_rate"] is None:
            out_lines.append("{:<42s} {:>6d}   (no combos graded)".format(name, s["n"]))
            continue
        d_win = (s["win_rate"] - baseline["win_rate"]) * 100 if baseline["win_rate"] is not None else None
        d_gap = (s["mean_gap"] - baseline["mean_gap"]) if baseline["mean_gap"] is not None else None
        d_win_str = (format(d_win, "+.1f") + "pp") if d_win is not None and not name.startswith("FULL") else "--"
        d_gap_str = format(d_gap, "+.4f") if d_gap is not None and not name.startswith("FULL") else "--"
        out_lines.append("{:<42s} {:>6d} {:>9.1%} {:>+12.4f} {:>10s} {:>10s}".format(
            name, s["n"], s["win_rate"], s["mean_gap"], d_win_str, d_gap_str))
    out_lines.append("")
    out_lines.append("-" * 78)
    out_lines.append("SIGNIFICANCE -- year-clustered bootstrap on (ablated win rate - FULL win rate)")
    out_lines.append("-" * 78)
    out_lines.extend(sig_lines)
    with open(os.path.join(FOLDER, "lineup_ablation_test_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    import sys
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
