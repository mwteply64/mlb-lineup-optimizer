#!/usr/bin/env python3
"""
RHP Platoon Prior Tuner - Lineup Optimization Project
========================================================
Tests whether the v16 batter-hand platoon prior (shipped 2026-07-10 for
LHP only, at PLATOON_PRIOR_PA_LHP=400) would ALSO help the RHP side, on
the same symmetric win-rate metric used to validate it for LHP.

WHY THIS EXISTS
---------------
The prior was built and tuned while attention was specifically on the
LHP deficit (see tuner_lhp_winrate.py, PROJECT_HANDOFF.md section 0).
It was never tested against RHP -- not because RHP was ruled out, but
because the RHP side showed no comparable deficit at the time (68.2%
symmetric win rate, no diagnosed weakness) and effort focused on LHP.
That is a reason it wasn't tried, not evidence it wouldn't help. This
tuner closes that gap: same evidence-weighted regression mechanism
(own career platoon ratio blended toward the league-mean ratio for the
batter's side, weighted by his own career PA), same league-prior
methodology (estimated from 2016-2018 only), pointed at the RHP side.

METHOD
------
Mirrors tuner_lhp_winrate.py exactly, mirrored to the RHP side:
- Baseline correctness gate: with PLATOON_PRIOR_PA_RHP=0 (production),
  per-combo RHP results must match lineup_forward_test_detail_2026-07-10
  .csv's RHP rows exactly (the current shipped v16 baseline, which left
  RHP untouched) before any sweep is trusted.
- League prior ratios vs RHP estimated ONCE from 2016-2018 seasons only
  (strictly before every test year), by batter side (side_vs_R column in
  batter_sides.csv), same 50-PA-each-hand stability bar as the LHP prior.
- PRIOR_PA_RHP sweep with walk-forward CV (train < holdout, holdouts
  2023/2024/2025) and leave-one-year-out CV (all 6 years), same shipping
  bar as every other tuner in this project: repeatable, robust held-out
  gains only.

USAGE:
    python tuner_rhp_platoon_prior.py
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
OUTPUT_FILE = os.path.join(FOLDER, "tuner_rhp_platoon_prior_results_" + today + ".txt")
DETAIL_REF = os.path.join(FOLDER, "lineup_forward_test_detail_2026-07-10.csv")
BATTER_SIDES = os.path.join(bt.DATA_DIR, "batter_sides.csv")

PRIOR_SEASONS = [2016, 2017, 2018]
PRIOR_MIN_PA_EACH_HAND = 50
WALK_FORWARD_HOLDOUTS = [2023, 2024, 2025]
PRIOR_PA_CANDIDATES = [0.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]


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


def baseline_params():
    return {
        "LA": bt.LEAGUE_AVG_WOBA_RHP,
        "WS": bt.WEIGHT_SPLITS_RHP, "WC": bt.WEIGHT_CAREER_RHP,
        "WT": bt.WEIGHT_TOTAL_RHP, "WR": bt.WEIGHT_RECENT_RHP,
        "RPS": bt.REGRESS_PA_SPLITS_RHP, "RPC": bt.REGRESS_PA_CAREER_RHP,
        "RPT": bt.REGRESS_PA_TOTAL_RHP, "RPR": bt.REGRESS_PA_RECENT_RHP,
        "XBS": bt.XWOBA_BLEND_SPLITS_RHP, "XBC": bt.XWOBA_BLEND_CAREER_RHP,
        "XBT": bt.XWOBA_BLEND_TOTAL_RHP, "XBR": bt.XWOBA_BLEND_RECENT_RHP,
        "PB": bt.PLATOON_BLEND_RHP,
        "PMIN": bt.PLATOON_FACTOR_MIN_RHP, "PMAX": bt.PLATOON_FACTOR_MAX_RHP,
        "PRIOR_PA": 0.0,
    }


def load_batter_sides():
    if not os.path.exists(BATTER_SIDES):
        print("NOTE: " + BATTER_SIDES + " not found -- sweep will be skipped.")
        return {}
    sides = {}
    with open(BATTER_SIDES, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            side = str(row.get("side_vs_R", "")).strip()
            if side in ("L", "R"):
                sides[str(row["player_id"]).strip()] = side
    return sides


def estimate_priors(sides):
    """League-mean platoon ratio vs RHP by batter side, from 2016-2018 only."""
    if not sides:
        return {}
    pooled = {}
    for y in PRIOR_SEASONS:
        for hand, key in (("left", "L"), ("right", "R")):
            for pid, d in load_season_file(y, hand).items():
                e = pooled.setdefault(pid, {"L": [0, 0.0], "R": [0, 0.0]})
                e[key][0] += d["pa"]
                e[key][1] += d["pa"] * d["woba"]
    acc = {"L": [0.0, 0.0], "R": [0.0, 0.0]}
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
        # own ratio for the RHP-side prior = vs-RHP woba / combined woba
        ratio = (sumR / paR) / combined
        w = paL + paR
        acc[side][0] += w
        acc[side][1] += w * ratio
    return {s: v[1] / v[0] for s, v in acc.items() if v[0] > 0}


def preload_windows():
    windows = []
    for year in TEST_YEARS:
        for cutoff in CUTOFFS:
            team_map = lft.load_team_map(year, cutoff)
            if not team_map:
                continue
            hand_pa_r = lft.load_hand_team_pa(year, cutoff, "vs_right")
            if hand_pa_r is None:
                continue
            pre_l = load_file("before_" + cutoff + "_" + str(year) + "_vs_left.csv")
            pre_r = load_file("before_" + cutoff + "_" + str(year) + "_vs_right.csv")
            car_l = build_career_data(year, "left", DECAY_CAREER_LHP, AGING_STRENGTH, PLAYER_AGES)
            car_r = build_career_data(year, "right", DECAY_CAREER_RHP, AGING_STRENGTH, PLAYER_AGES)
            total = load_file("totals_before_" + cutoff + "_" + str(year) + ".csv")
            recent = load_file("recent_before_" + cutoff + "_" + str(year) + ".csv")
            post_r = load_file("after_" + cutoff + "_" + str(year) + "_vs_right.csv")

            all_pids = set(pre_l.keys()) | set(pre_r.keys())
            players = {pid: (pre_r.get(pid), car_r.get(pid), car_l.get(pid),
                             total.get(pid), recent.get(pid))
                       for pid in all_pids}

            teams = []
            for team, pids in lft.build_rosters(team_map).items():
                if len(pids) < lft.LINEUP_SIZE:
                    continue
                real9 = lft.real_lineup_hand(team, pids, hand_pa_r)
                real_avg, n_real = lft.grade_lineup(real9, post_r)
                if real_avg is None or n_real < lft.MIN_GRADED:
                    continue
                teams.append((team, pids, real_avg, n_real))

            windows.append({"year": year, "cutoff": cutoff, "players": players,
                            "post_r": post_r, "teams": teams})
    return windows


entry_pid_holder = [None]


def stab_r(entry, P, sides, priors):
    """Mirrors backtest.stabilize(side='R') exactly, with the RHP-side
    batter-hand platoon prior parameterized as P['PRIOR_PA']."""
    sr, cr, cl, tot, rec = entry
    la = P["LA"]

    splits_w = blended_woba(sr, P["XBS"])
    career_w = blended_woba(cr, P["XBC"])
    total_w = blended_woba(tot, P["XBT"])
    recent_w = blended_woba(rec, P["XBR"])
    cross_w = blended_woba(cl, P["XBC"])

    career_pa = cr["pa"] if cr else 0
    cross_pa = cl["pa"] if cl else 0

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
        pid_side = sides.get(entry_pid_holder[0]) if entry_pid_holder[0] else None
        prior = priors.get(pid_side) if pid_side else None
        if prior is not None:
            est = prior if ratio is None else (
                (ratio * own_pa + prior * P["PRIOR_PA"]) / (own_pa + P["PRIOR_PA"]))
            clamped = max(P["PMIN"], min(P["PMAX"], est))
            platoon_factor = 1.0 + P["PB"] * (clamped - 1.0)
        elif ratio is not None:
            clamped = max(P["PMIN"], min(P["PMAX"], ratio))
            platoon_factor = 1.0 + P["PB"] * (clamped - 1.0)
    elif ratio is not None:
        clamped = max(P["PMIN"], min(P["PMAX"], ratio))
        platoon_factor = 1.0 + P["PB"] * (clamped - 1.0)

    components, weights = [], []
    sr_pa = sr["pa"] if sr else 0
    if splits_w is not None and sr_pa >= MIN_PA_SPLITS:
        components.append((splits_w * sr_pa + la * P["RPS"]) / (sr_pa + P["RPS"]))
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


def evaluate(P, windows, sides, priors):
    by_year = {}
    combos = []
    use_prior = P["PRIOR_PA"] > 0 and priors
    for w in windows:
        scores = {}
        for pid, entry in w["players"].items():
            if use_prior:
                entry_pid_holder[0] = pid
            s = stab_r(entry, P, sides, priors)
            if s is not None:
                scores[pid] = s
        entry_pid_holder[0] = None
        post_r = w["post_r"]
        for team, pids, real_avg, n_real in w["teams"]:
            eligible = [pid for pid in pids if pid in scores]
            opt9 = sorted(eligible, key=lambda pid: scores[pid], reverse=True)[:lft.LINEUP_SIZE]
            if len(opt9) < lft.LINEUP_SIZE:
                continue
            vals = [post_r[pid]["woba"] for pid in opt9
                    if pid in post_r and post_r[pid]["pa"] >= MIN_PA_POST]
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
    n = sum(by_year.get(y, [0, 0, 0.0])[1] for y in years)
    gsum = sum(by_year.get(y, [0, 0, 0.0])[2] for y in years)
    if n == 0:
        return 0.0, 0.0, 0
    return wins / n, gsum / n, n


def fmt_pct(x):
    return format(100.0 * x, ".1f") + "%"


def validate_baseline(combos):
    if not os.path.exists(DETAIL_REF):
        print("WARNING: " + DETAIL_REF + " not found -- skipping the baseline gate.")
        return True
    ref = {}
    with open(DETAIL_REF, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["hand"] != "RHP":
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


def main():
    print("=== RHP PLATOON PRIOR TUNER (symmetric metric) ===")
    print("Tests whether the v16 batter-hand platoon prior (shipped LHP-only)")
    print("would also help the RHP side. Results saved to: " + OUTPUT_FILE)
    print()

    sides = load_batter_sides()
    priors = estimate_priors(sides)
    if priors:
        print("League platoon priors vs RHP (from " +
              "/".join(str(y) for y in PRIOR_SEASONS) + " only, PA-weighted, " +
              "players with >= " + str(PRIOR_MIN_PA_EACH_HAND) + " PA each hand):")
        for side in ("R", "L"):
            if side in priors:
                print("  " + side + "HB vs RHP: ratio " + format(priors[side], ".4f"))
        print()

    print("Preloading 18 windows ...")
    windows = preload_windows()
    print("  " + str(len(windows)) + " windows loaded.")
    print()

    base = baseline_params()
    by_year_base, combos = evaluate(base, windows, sides, priors)
    if not validate_baseline(combos):
        print("Aborting: tuner does not reproduce production RHP baseline.")
        return
    wr_all, gap_all, n_all = agg(by_year_base, TEST_YEARS)
    print("Baseline (production RHP constants, PRIOR_PA=0): win rate " + fmt_pct(wr_all) +
          ", mean gap " + format(gap_all, "+.4f") + ", n=" + str(n_all))
    print("  by year: " + "; ".join(
        str(y) + " " + fmt_pct(agg(by_year_base, [y])[0]) for y in TEST_YEARS))
    print()

    if not priors:
        print("No batter_sides.csv / priors -- stopping.")
        return

    print("=" * 66)
    print("SWEEP: PRIOR_PA_RHP   candidates: " + ", ".join(str(v) for v in PRIOR_PA_CANDIDATES))
    print("=" * 66)
    results = {}
    for v in PRIOR_PA_CANDIDATES:
        P = dict(base)
        P["PRIOR_PA"] = v
        by_year, _ = evaluate(P, windows, sides, priors)
        results[v] = by_year
        wr, gap, n = agg(by_year, TEST_YEARS)
        marker = "  <- production" if v == 0.0 else ""
        print("  PRIOR_PA_RHP=" + str(v) + ": all-years win rate " + fmt_pct(wr) +
              ", mean gap " + format(gap, "+.4f") + ", n=" + str(n) + marker)
        print("     by year: " + "; ".join(
            str(y) + " " + fmt_pct(agg(by_year, [y])[0]) for y in TEST_YEARS))

    def pick(train_years):
        best_v, best_key = None, None
        for v in PRIOR_PA_CANDIDATES:
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
        wr_b, gap_b, _ = agg(base_by_year := by_year_base, [hold])
        wf_deltas.append(wr_t - wr_b)
        print("    holdout " + str(hold) + ": picked PRIOR_PA_RHP=" + str(v) +
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
        wr_b, gap_b, _ = agg(by_year_base, [hold])
        loyo_deltas.append(wr_t - wr_b)
        print("    holdout " + str(hold) + ": picked PRIOR_PA_RHP=" + str(v) +
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
