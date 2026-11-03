#!/usr/bin/env python3
"""
LHP Win-Rate Tuner - Lineup Optimization Project
===================================================
First tuner in this project scored on the SYMMETRIC WIN-RATE metric
(does the model's top-9 vs LHP beat the manager's actual top-9 by PA vs
LHP, graded on post-cutoff wOBA?) instead of league-wide Spearman r.

WHY (2026-07-10, follows the LHP diagnostics -- see PROJECT_HANDOFF.md
section 0):
- The historical RHP-LHP win-rate gap (+11.5pp, 68.2% vs 56.7%) is
  year-cluster significant under the symmetric framework.
- Every LHP constant was tuned (v10) on Spearman r, a league-wide
  ranking metric. The deployed question is different: beat the
  manager's platoon deployment, which is decided by marginal picks with
  thin vs-LHP samples (swap-ins average ~27 pre-cutoff PA vs LHP).
- This tuner re-searches the dominant LHP knobs directly on the
  deployed metric, with the same held-out-CV shipping bar as every
  other tuner in this project.

NEW KNOB -- PLATOON_PRIOR_PA (batter-hand platoon prior, evidence-weighted)
----------------------------------------------------------------------------
The existing platoon factor is built purely from the player's OWN
career splits -- deliberately, because reverse-split hitters exist.
This tuner adds a REGRESSION TARGET, not an override: the player's raw
platoon ratio is blended with the league-mean ratio for his batter side
(vs LHP), weighted by career evidence:

    est_ratio = (own_ratio * own_pa + prior_ratio * PRIOR_PA)
                / (own_pa + PRIOR_PA)

own_pa is the player's combined career PA behind his own ratio, so an
established (including reverse-split) hitter keeps his own ratio almost
untouched, while a thin-sample hitter is pulled toward the typical
same-side/opposite-side factor. PRIOR_PA = 0 reproduces current
production bit-for-bit. A player whose batter side is unknown, or with
no career data at all when PRIOR_PA == 0, falls back to current
behavior. League prior ratios are estimated ONCE from the 2016-2018
seasons only -- strictly before every test year (2019+), so no fold
ever sees a prior fitted on its own data. Batter sides come from
"historical testing/batter_sides.csv" (download_batter_sides.py).

METHOD
------
- Preloads all 18 windows (pre/career/total/recent/post files, team
  rosters, hand-specific real lineups and their grades) once; each
  parameter evaluation re-scores only the LHP side in memory.
- Baseline correctness gate: with production constants and
  PRIOR_PA=0, per-combo results must match
  lineup_forward_test_detail_2026-07-09.csv's LHP rows exactly before
  any sweep is trusted.
- 1-D sweeps around production values, one knob at a time. Selection
  on pooled train-year win rate (tiebreak: mean gap).
- Two CV protocols reported per knob:
  * Walk-forward (project precedent): train years < Y, test Y, for
    Y in 2023/2024/2025.
  * Leave-one-year-out (supplementary, 6 folds -- includes the two
    weak LHP years 2021/2025 as holdouts).
- Ships nothing by itself: results are written to
  tuner_lhp_winrate_results_<date>.txt for review.

USAGE:
    python tuner_lhp_winrate.py
"""

import os
import sys
import csv
import datetime
import statistics

import backtest as bt
from backtest import (
    load_file, load_season_file, build_career_data, blended_woba,
    DECAY_CAREER_LHP, DECAY_CAREER_RHP, AGING_STRENGTH, PLAYER_AGES,
    MIN_PA_SPLITS, MIN_PA_CAREER, MIN_PA_TOTAL, MIN_PA_POST,
    TEST_YEARS, CUTOFFS,
)
import lineup_forward_test as lft

FOLDER = os.path.dirname(os.path.abspath(__file__))
today = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(FOLDER, "tuner_lhp_winrate_results_" + today + ".txt")
DETAIL_REF = os.path.join(FOLDER, "lineup_forward_test_detail_2026-07-09.csv")
BATTER_SIDES = os.path.join(bt.DATA_DIR, "batter_sides.csv")

PRIOR_SEASONS = [2016, 2017, 2018]   # strictly before every test year
PRIOR_MIN_PA_EACH_HAND = 50          # stability bar for a player's ratio to inform the prior

WALK_FORWARD_HOLDOUTS = [2023, 2024, 2025]


class Tee:
    def __init__(self, filepath):
        self.console = sys.stdout
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
# Parameters
# -------------------------------------------------------

def baseline_params():
    """Production LHP constants, plus PRIOR_PA=0 (bit-for-bit no-op)."""
    return {
        "LA":  bt.LEAGUE_AVG_WOBA_LHP,
        "WS":  bt.WEIGHT_SPLITS_LHP,  "WC": bt.WEIGHT_CAREER_LHP,
        "WT":  bt.WEIGHT_TOTAL_LHP,   "WR": bt.WEIGHT_RECENT_LHP,
        "RPS": bt.REGRESS_PA_SPLITS_LHP, "RPC": bt.REGRESS_PA_CAREER_LHP,
        "RPT": bt.REGRESS_PA_TOTAL_LHP,  "RPR": bt.REGRESS_PA_RECENT_LHP,
        "XBS": bt.XWOBA_BLEND_SPLITS_LHP, "XBC": bt.XWOBA_BLEND_CAREER_LHP,
        "XBT": bt.XWOBA_BLEND_TOTAL_LHP,  "XBR": bt.XWOBA_BLEND_RECENT_LHP,
        "PB":  bt.PLATOON_BLEND_LHP,
        "PMIN": bt.PLATOON_FACTOR_MIN_LHP, "PMAX": bt.PLATOON_FACTOR_MAX_LHP,
        "PRIOR_PA": 0.0,
    }


SWEEPS = [
    ("WC",  [0.20, 0.35, 0.45, 0.5695, 0.70, 0.85]),
    ("RPC", [100, 300, 500, 711, 900, 1200]),
    ("XB_ALL", [0.50, 0.65, 0.80, 0.9187, 1.00]),   # sets XBS=XBC=XBT=XBR together
    ("PB",  [0.31, 0.50, 0.7833, 1.00]),
    ("PRIOR_PA", [0.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]),
]


def apply_knob(P, knob, value):
    P = dict(P)
    if knob == "XB_ALL":
        P["XBS"] = P["XBC"] = P["XBT"] = P["XBR"] = value
    else:
        P[knob] = value
    return P


# -------------------------------------------------------
# Batter sides + league platoon prior (vs LHP)
# -------------------------------------------------------

def load_batter_sides():
    """player_id -> batter side vs LHP ('L'/'R'). Empty dict disables the prior."""
    if not os.path.exists(BATTER_SIDES):
        print("NOTE: " + BATTER_SIDES + " not found -- PRIOR_PA sweep will be skipped.")
        return {}
    sides = {}
    with open(BATTER_SIDES, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            side = str(row.get("side_vs_L", "")).strip()
            if side in ("L", "R"):
                sides[str(row["player_id"]).strip()] = side
    return sides


def estimate_priors(sides):
    """
    League-mean platoon ratio vs LHP by batter side, from 2016-2018 only.
    Ratio per player = pooled woba vs LHP / pooled combined woba (both
    hands, PA-weighted) -- the same 'own ratio' definition used in
    scoring. PA-weighted mean across players with >= PRIOR_MIN_PA_EACH_HAND
    PA vs each hand. Returns {'L': ratio, 'R': ratio} or {} if sides missing.
    """
    if not sides:
        return {}
    pooled = {}
    for y in PRIOR_SEASONS:
        for hand, key in (("left", "L"), ("right", "R")):
            for pid, d in load_season_file(y, hand).items():
                e = pooled.setdefault(pid, {"L": [0, 0.0], "R": [0, 0.0]})
                e[key][0] += d["pa"]
                e[key][1] += d["pa"] * d["woba"]
    acc = {"L": [0.0, 0.0], "R": [0.0, 0.0]}   # side -> [weight_sum, weighted_ratio_sum]
    for pid, e in pooled.items():
        paL, sumL = e["L"]
        paR, sumR = e["R"]
        if paL < PRIOR_MIN_PA_EACH_HAND or paR < PRIOR_MIN_PA_EACH_HAND:
            continue
        side = sides.get(pid)
        if side not in ("L", "R"):
            continue
        combined = (sumL + sumR) / (paL + paR)
        if combined <= 0:
            continue
        ratio = (sumL / paL) / combined
        w = paL + paR
        acc[side][0] += w
        acc[side][1] += w * ratio
    priors = {s: v[1] / v[0] for s, v in acc.items() if v[0] > 0}
    return priors


# -------------------------------------------------------
# Window preloading
# -------------------------------------------------------

def preload_windows():
    """
    One dict per (year, cutoff): raw per-player component entries (so the
    xwOBA blend stays parameterizable), post-cutoff vs-LHP ground truth,
    and per-team roster/real-lineup data with the real side pre-graded
    (it never depends on the LHP params being searched).
    """
    windows = []
    for year in TEST_YEARS:
        for cutoff in CUTOFFS:
            team_map = lft.load_team_map(year, cutoff)
            if not team_map:
                continue
            hand_pa_l = lft.load_hand_team_pa(year, cutoff, "vs_left")
            if hand_pa_l is None:
                continue
            pre_l  = load_file("before_" + cutoff + "_" + str(year) + "_vs_left.csv")
            pre_r  = load_file("before_" + cutoff + "_" + str(year) + "_vs_right.csv")
            car_l  = build_career_data(year, "left",  DECAY_CAREER_LHP, AGING_STRENGTH, PLAYER_AGES)
            car_r  = build_career_data(year, "right", DECAY_CAREER_RHP, AGING_STRENGTH, PLAYER_AGES)
            total  = load_file("totals_before_" + cutoff + "_" + str(year) + ".csv")
            recent = load_file("recent_before_" + cutoff + "_" + str(year) + ".csv")
            post_l = load_file("after_" + cutoff + "_" + str(year) + "_vs_left.csv")

            all_pids = set(pre_l.keys()) | set(pre_r.keys())
            players = {pid: (pre_l.get(pid), car_l.get(pid), car_r.get(pid),
                             total.get(pid), recent.get(pid))
                       for pid in all_pids}

            teams = []
            for team, pids in lft.build_rosters(team_map).items():
                if len(pids) < lft.LINEUP_SIZE:
                    continue
                real9 = lft.real_lineup_hand(team, pids, hand_pa_l)
                real_avg, n_real = lft.grade_lineup(real9, post_l)
                if real_avg is None or n_real < lft.MIN_GRADED:
                    continue
                teams.append((team, pids, real_avg, n_real))

            windows.append({"year": year, "cutoff": cutoff, "players": players,
                            "post_l": post_l, "teams": teams})
    return windows


# -------------------------------------------------------
# Parameterized LHP scoring (mirrors backtest.stabilize side='L',
# NO_CLAMP=True, plus the optional evidence-weighted platoon prior)
# -------------------------------------------------------

def stab_l(entry, P, sides, priors):
    sl, cl, cr, tot, rec = entry
    la = P["LA"]

    splits_w = blended_woba(sl,  P["XBS"])
    career_w = blended_woba(cl,  P["XBC"])
    total_w  = blended_woba(tot, P["XBT"])
    recent_w = blended_woba(rec, P["XBR"])
    cross_w  = blended_woba(cr,  P["XBC"])

    career_pa = cl["pa"] if cl else 0
    cross_pa  = cr["pa"] if cr else 0

    # Own platoon ratio (identical to production when PRIOR_PA == 0)
    ratio, own_pa = None, 0
    if (P["PB"] > 0 and career_w is not None and career_pa > 0 and
            cross_w is not None and cross_pa > 0):
        combined_pa = career_pa + cross_pa
        combined = (career_w * career_pa + cross_w * cross_pa) / combined_pa
        if combined > 0:
            ratio = career_w / combined
            own_pa = combined_pa

    platoon_factor = 1.0
    if P["PRIOR_PA"] > 0 and priors:
        # Evidence-weighted blend toward the batter-side league prior.
        pid_side = sides.get(entry_pid_holder[0]) if entry_pid_holder else None
        prior = priors.get(pid_side) if pid_side else None
        if prior is not None:
            if ratio is None:
                est = prior
            else:
                est = (ratio * own_pa + prior * P["PRIOR_PA"]) / (own_pa + P["PRIOR_PA"])
            clamped = max(P["PMIN"], min(P["PMAX"], est))
            platoon_factor = 1.0 + P["PB"] * (clamped - 1.0)
        elif ratio is not None:
            clamped = max(P["PMIN"], min(P["PMAX"], ratio))
            platoon_factor = 1.0 + P["PB"] * (clamped - 1.0)
    elif ratio is not None:
        clamped = max(P["PMIN"], min(P["PMAX"], ratio))
        platoon_factor = 1.0 + P["PB"] * (clamped - 1.0)

    components, weights = [], []
    sl_pa = sl["pa"] if sl else 0
    if splits_w is not None and sl_pa >= MIN_PA_SPLITS:
        components.append((splits_w * sl_pa + la * P["RPS"]) / (sl_pa + P["RPS"]))
        weights.append(P["WS"])
    if career_w is not None and career_pa >= MIN_PA_CAREER:
        components.append((career_w * career_pa + la * P["RPC"]) / (career_pa + P["RPC"]))
        weights.append(P["WC"])
    tot_pa = tot["pa"] if tot else 0
    if total_w is not None and tot_pa >= MIN_PA_TOTAL:
        adj = total_w * platoon_factor
        components.append((adj * tot_pa + la * P["RPT"]) / (tot_pa + P["RPT"]))
        weights.append(P["WT"])
    rec_pa = rec["pa"] if rec else 0
    if recent_w is not None and rec_pa > 0:
        adj = recent_w * platoon_factor
        components.append((adj * rec_pa + la * P["RPR"]) / (rec_pa + P["RPR"]))
        weights.append(P["WR"])

    if not components:
        return None
    total_wt = sum(weights)
    return round(sum(c * (w / total_wt) for c, w in zip(components, weights)), 4)


# The prior needs the player id inside stab_l without changing the
# entry tuples; a 1-slot holder keeps the hot loop allocation-free.
entry_pid_holder = [None]


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

def evaluate(P, windows, sides, priors):
    """
    Score every window's LHP side under params P. Returns per-year
    {year: [wins, n, gap_sum]} plus the flat per-combo list (for the
    baseline correctness gate).
    """
    by_year = {}
    combos = []
    use_prior = P["PRIOR_PA"] > 0 and priors
    for w in windows:
        scores = {}
        for pid, entry in w["players"].items():
            if use_prior:
                entry_pid_holder[0] = pid
            s = stab_l(entry, P, sides, priors)
            if s is not None:
                scores[pid] = s
        entry_pid_holder[0] = None
        post_l = w["post_l"]
        for team, pids, real_avg, n_real in w["teams"]:
            eligible = [pid for pid in pids if pid in scores]
            opt9 = sorted(eligible, key=lambda pid: scores[pid], reverse=True)[:lft.LINEUP_SIZE]
            if len(opt9) < lft.LINEUP_SIZE:
                continue
            vals = [post_l[pid]["woba"] for pid in opt9
                    if pid in post_l and post_l[pid]["pa"] >= MIN_PA_POST]
            if len(vals) < lft.MIN_GRADED:
                continue
            opt_avg = statistics.mean(vals)
            gap = opt_avg - real_avg
            e = by_year.setdefault(w["year"], [0, 0, 0.0])
            e[0] += 1 if gap > 0 else 0
            e[1] += 1
            e[2] += gap
            combos.append((w["year"], w["cutoff"], team, gap))
    return by_year, combos


def agg(by_year, years):
    wins = sum(by_year.get(y, [0, 0, 0.0])[0] for y in years)
    n    = sum(by_year.get(y, [0, 0, 0.0])[1] for y in years)
    gsum = sum(by_year.get(y, [0, 0, 0.0])[2] for y in years)
    if n == 0:
        return 0.0, 0.0, 0
    return wins / n, gsum / n, n


def fmt_pct(x):
    return format(100.0 * x, ".1f") + "%"


# -------------------------------------------------------
# Baseline correctness gate
# -------------------------------------------------------

def validate_baseline(combos):
    """Per-combo gaps must match the 2026-07-09 symmetric detail CSV's
    LHP rows (same combos, same gaps) before any sweep is trusted."""
    if not os.path.exists(DETAIL_REF):
        print("WARNING: " + DETAIL_REF + " not found -- skipping the baseline gate.")
        return True
    ref = {}
    with open(DETAIL_REF, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["hand"] != "LHP":
                continue
            ref[(int(row["year"]), row["cutoff"], row["team"])] = float(row["gap"])
    mine = {(y, c, t): g for (y, c, t, g) in combos}
    if set(ref) != set(mine):
        print("BASELINE GATE FAILED: combo sets differ (ref " + str(len(ref)) +
              ", tuner " + str(len(mine)) + ")")
        return False
    worst = max(abs(ref[k] - mine[k]) for k in ref)
    ok = worst < 1e-9
    print("Baseline gate: " + str(len(ref)) + " combos matched, max |gap diff| = " +
          format(worst, ".2e") + (" -- PASS" if ok else " -- FAIL"))
    return ok


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    print("=== LHP WIN-RATE TUNER (symmetric metric) ===")
    print("Results saved to: " + OUTPUT_FILE)
    print()

    sides = load_batter_sides()
    priors = estimate_priors(sides)
    if priors:
        print("League platoon priors vs LHP (from " +
              "/".join(str(y) for y in PRIOR_SEASONS) + " only, PA-weighted, " +
              "players with >= " + str(PRIOR_MIN_PA_EACH_HAND) + " PA each hand):")
        for side in ("R", "L"):
            if side in priors:
                print("  " + side + "HB vs LHP: ratio " + format(priors[side], ".4f"))
        print()

    print("Preloading 18 windows ...")
    windows = preload_windows()
    print("  " + str(len(windows)) + " windows loaded.")
    print()

    base = baseline_params()
    by_year_base, combos = evaluate(base, windows, sides, priors)
    if not validate_baseline(combos):
        print("Aborting: tuner does not reproduce production baseline.")
        return
    wr_all, gap_all, n_all = agg(by_year_base, TEST_YEARS)
    print("Baseline (production LHP constants): win rate " + fmt_pct(wr_all) +
          ", mean gap " + format(gap_all, "+.4f") + ", n=" + str(n_all))
    print("  by year: " + "; ".join(
        str(y) + " " + fmt_pct(agg(by_year_base, [y])[0]) for y in TEST_YEARS))
    print()

    base_by_year = by_year_base

    for knob, values in SWEEPS:
        if knob == "PRIOR_PA" and not priors:
            print("Skipping PRIOR_PA sweep (no batter_sides.csv / priors).")
            continue
        print("=" * 66)
        print("SWEEP: " + knob + "   candidates: " + ", ".join(str(v) for v in values))
        print("=" * 66)
        results = {}
        for v in values:
            P = apply_knob(base, knob, v)
            by_year, _ = evaluate(P, windows, sides, priors)
            results[v] = by_year
            wr, gap, n = agg(by_year, TEST_YEARS)
            marker = "  <- production" if by_year == base_by_year or (
                knob != "XB_ALL" and v == base.get(knob)) or (
                knob == "XB_ALL" and v == base["XBS"]) else ""
            print("  " + knob + "=" + str(v) + ": all-years win rate " + fmt_pct(wr) +
                  ", mean gap " + format(gap, "+.4f") + ", n=" + str(n) + marker)
            print("     by year: " + "; ".join(
                str(y) + " " + fmt_pct(agg(by_year, [y])[0]) for y in TEST_YEARS))

        def pick(train_years):
            best_v, best_key = None, None
            for v in values:
                wr, gap, n = agg(results[v], train_years)
                key = (round(wr, 6), round(gap, 6))
                if best_key is None or key > best_key:
                    best_key, best_v = key, v
            return best_v

        print("  Walk-forward CV (train = years before holdout):")
        wf_deltas = []
        for hold in WALK_FORWARD_HOLDOUTS:
            train = [y for y in TEST_YEARS if y < hold]
            v = pick(train)
            wr_t, gap_t, n_t = agg(results[v], [hold])
            wr_b, gap_b, _ = agg(base_by_year, [hold])
            wf_deltas.append(wr_t - wr_b)
            print("    holdout " + str(hold) + ": picked " + knob + "=" + str(v) +
                  " -> test win rate " + fmt_pct(wr_t) + " vs baseline " + fmt_pct(wr_b) +
                  " (delta " + format(100 * (wr_t - wr_b), "+.1f") + "pp)")
        print("    mean held-out win-rate delta: " +
              format(100 * statistics.mean(wf_deltas), "+.1f") + "pp")

        print("  Leave-one-year-out CV (supplementary):")
        loyo_deltas = []
        for hold in TEST_YEARS:
            train = [y for y in TEST_YEARS if y != hold]
            v = pick(train)
            wr_t, gap_t, n_t = agg(results[v], [hold])
            wr_b, gap_b, _ = agg(base_by_year, [hold])
            loyo_deltas.append(wr_t - wr_b)
            print("    holdout " + str(hold) + ": picked " + knob + "=" + str(v) +
                  " -> test " + fmt_pct(wr_t) + " vs baseline " + fmt_pct(wr_b) +
                  " (delta " + format(100 * (wr_t - wr_b), "+.1f") + "pp)")
        print("    mean held-out win-rate delta: " +
              format(100 * statistics.mean(loyo_deltas), "+.1f") + "pp")
        print()

    print("=== DONE ===")
    print("Shipping bar unchanged: repeatable, robust HELD-OUT gains only.")


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
