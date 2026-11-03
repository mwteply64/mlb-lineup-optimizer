#!/usr/bin/env python3
"""
Tuner v2 - Full joint retune (Tasks #23, #24, #25)
====================================================
In-memory backtest engine + coordinate-ascent/random-search optimizer.
Loads every CSV exactly once, then runs thousands of trials with zero
disk I/O per trial. Mirrors backtest.py's scoring logic EXACTLY,
including v9's build_career_data() (per-season, decay-weighted) and
blend_woba() (xWOBA blending) -- this is the part v1 of the tuner did
not have, since it predated the 18-window/decay/xwoba data expansion.

TUNES JOINTLY (24 parameters):
  LEAGUE_AVG_WOBA_LHP/RHP                                   (2)
  WEIGHT_{SPLITS,CAREER,TOTAL,RECENT}_{LHP,RHP}              (8)
  REGRESS_PA_{SPLITS,CAREER,TOTAL,RECENT}_{LHP,RHP}          (8)
  PLATOON_BLEND, PLATOON_FACTOR_MIN, PLATOON_FACTOR_MAX      (3)
  DECAY_CAREER_LHP, DECAY_CAREER_RHP                         (2)  -- Task #23
  XWOBA_BLEND                                                (1)  -- Task #24

PA minimums are FIXED at 30 (not tunable, per user instruction).
NO_CLAMP is fixed True (matches current backtest.py).

TRAIN / HOLDOUT SPLIT:
  Tuning purely on all 18 windows risks overfitting to quirks of
  specific seasons now that we have enough windows to afford a real
  out-of-time check. TRAIN = 2019, 2021, 2022, 2023 (12 windows).
  HOLDOUT = 2024, 2025 (6 windows) -- the most recent seasons, held
  out entirely from the search objective. The optimizer never sees
  HOLDOUT r while searching; it's only evaluated on the final result,
  to confirm gains generalize rather than just memorizing the
  training windows.

USAGE:
  Double-click run_tuner.bat (writes tuner_results_YYYY-MM-DD.txt)
  Optional: edit N_RANDOM_ITERS_PHASE1/2 below to run longer/shorter.
"""
import os
import csv
import sys
import math
import random
import datetime
import time

FOLDER   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FOLDER, "historical testing")
today    = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(FOLDER, "tuner_results_" + today + ".txt")


class Tee:
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
# FIXED CONSTANTS (not tunable)
# -------------------------------------------------------
MIN_PA_SPLITS = 30
MIN_PA_TOTAL  = 30
MIN_PA_CAREER = 30
MIN_PA_POST   = 30
NO_CLAMP = True

TEST_YEARS = [2019, 2021, 2022, 2023, 2024, 2025]
CUTOFFS    = ["june_1", "july_1", "aug_1"]
TEST_WINDOWS = [{"year": y, "cutoff": c} for y in TEST_YEARS for c in CUTOFFS]

TRAIN_YEARS   = [2019, 2021, 2022, 2023]
HOLDOUT_YEARS = [2024, 2025]

ALL_SEASON_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]

# -------------------------------------------------------
# DATA LOADING (once)
# -------------------------------------------------------

def load_file(filename):
    path = os.path.join(DATA_DIR, filename)
    result = {}
    if not os.path.exists(path):
        print("  WARNING: File not found: " + filename)
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pid = str(row["player_id"]).strip()
                pa  = int(float(row["pa"]))
                woba_str = str(row.get("woba", "")).strip()
                if not woba_str or woba_str.lower() in ("nan", "none", ""):
                    continue
                woba = float(woba_str)
                xwoba_str = str(row.get("xwoba", "")).strip()
                xwoba = None
                if xwoba_str and xwoba_str.lower() not in ("nan", "none"):
                    try:
                        xwoba = float(xwoba_str)
                    except ValueError:
                        xwoba = None
                result[pid] = {"pa": pa, "woba": woba, "xwoba": xwoba}
            except (ValueError, KeyError, TypeError):
                continue
    return result


# Season files, cached by (hand, year)
SEASON_CACHE = {}
for hand in ("left", "right"):
    for y in ALL_SEASON_YEARS:
        SEASON_CACHE[(hand, y)] = load_file("season_" + str(y) + "_vs_" + hand + ".csv")

# Per-window files, cached by (year, cutoff)
WINDOW_CACHE = {}
for w in TEST_WINDOWS:
    year, co = w["year"], w["cutoff"]
    WINDOW_CACHE[(year, co)] = {
        "pre_l":  load_file("before_" + co + "_" + str(year) + "_vs_left.csv"),
        "pre_r":  load_file("before_" + co + "_" + str(year) + "_vs_right.csv"),
        "total":  load_file("totals_before_" + co + "_" + str(year) + ".csv"),
        "recent": load_file("recent_before_" + co + "_" + str(year) + ".csv"),
        "post_l": load_file("after_" + co + "_" + str(year) + "_vs_left.csv"),
        "post_r": load_file("after_" + co + "_" + str(year) + "_vs_right.csv"),
    }

# -------------------------------------------------------
# CAREER DATA (rebuilt per trial -- depends on decay)
# -------------------------------------------------------

def build_career_data(test_year, hand, decay):
    """Mirrors backtest.py's build_career_data() exactly, but reads
    from SEASON_CACHE (already in memory) instead of disk."""
    prior_years = [y for y in ALL_SEASON_YEARS if y < test_year]
    num, den = {}, {}
    xnum, xden = {}, {}
    pa_sum = {}
    for y in prior_years:
        season = SEASON_CACHE[(hand, y)]
        w = decay ** (test_year - y)
        for pid, d in season.items():
            wpa = w * d["pa"]
            num[pid] = num.get(pid, 0.0) + wpa * d["woba"]
            den[pid] = den.get(pid, 0.0) + wpa
            pa_sum[pid] = pa_sum.get(pid, 0) + d["pa"]
            if d.get("xwoba") is not None:
                xnum[pid] = xnum.get(pid, 0.0) + wpa * d["xwoba"]
                xden[pid] = xden.get(pid, 0.0) + wpa
    result = {}
    for pid, denom in den.items():
        if denom <= 0:
            continue
        entry = {"pa": pa_sum[pid], "woba": num[pid] / denom}
        if xden.get(pid, 0) > 0:
            entry["xwoba"] = xnum[pid] / xden[pid]
        result[pid] = entry
    return result


def blend_woba(entry, xwoba_blend):
    if entry is None:
        return None
    xw = entry.get("xwoba")
    if xw is None:
        return entry["woba"]
    return (1.0 - xwoba_blend) * entry["woba"] + xwoba_blend * xw


# -------------------------------------------------------
# SCORING MATH (mirrors backtest.py exactly)
# -------------------------------------------------------

def regress(woba, pa, regress_pa, la):
    return (woba * pa + la * regress_pa) / (pa + regress_pa)


def stabilize(p, splits_pa, splits_woba, career_pa, career_woba,
              total_pa, total_woba, recent_pa, recent_woba, side,
              career_cross_pa=0, career_cross_woba=None):
    if side == 'L':
        la, WS, WC, WT, WR = p["LA_LHP"], p["WS_L"], p["WC_L"], p["WT_L"], p["WR_L"]
        RPS, RPC, RPT, RPR = p["RPS_L"], p["RPC_L"], p["RPT_L"], p["RPR_L"]
    else:
        la, WS, WC, WT, WR = p["LA_RHP"], p["WS_R"], p["WC_R"], p["WT_R"], p["WR_R"]
        RPS, RPC, RPT, RPR = p["RPS_R"], p["RPC_R"], p["RPT_R"], p["RPR_R"]

    platoon_factor = 1.0
    pb = p["PLATOON_BLEND"]
    if (pb > 0 and career_woba is not None and career_pa > 0 and
            career_cross_woba is not None and career_cross_pa > 0):
        combined_pa = career_pa + career_cross_pa
        combined_woba = (career_woba * career_pa +
                          career_cross_woba * career_cross_pa) / combined_pa
        if combined_woba > 0:
            raw_factor = career_woba / combined_woba
            clamped_factor = max(p["PLATOON_MIN"], min(p["PLATOON_MAX"], raw_factor))
            platoon_factor = 1.0 + pb * (clamped_factor - 1.0)

    components, weights = [], []
    if splits_woba is not None and splits_pa >= MIN_PA_SPLITS:
        components.append(regress(splits_woba, splits_pa, RPS, la)); weights.append(WS)
    if career_woba is not None and career_pa >= MIN_PA_CAREER:
        components.append(regress(career_woba, career_pa, RPC, la)); weights.append(WC)
    if total_woba is not None and total_pa >= MIN_PA_TOTAL:
        adj = total_woba * platoon_factor
        components.append(regress(adj, total_pa, RPT, la)); weights.append(WT)
    if recent_woba is not None and recent_pa > 0:
        adj = recent_woba * platoon_factor
        components.append(regress(adj, recent_pa, RPR, la)); weights.append(WR)

    if not components:
        return None
    total_w = sum(weights)
    if total_w <= 0:
        return None
    return sum(c * (w / total_w) for c, w in zip(components, weights))


def rank_array(arr):
    n = len(arr)
    indexed = sorted(enumerate(arr), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j][1] == indexed[j + 1][1]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def spearman_r(x, y):
    n = len(x)
    if n < 5:
        return None
    rx, ry = rank_array(x), rank_array(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    deny = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


# -------------------------------------------------------
# WINDOW EVALUATION
# -------------------------------------------------------

def run_window(p, year, co, car_l, car_r):
    wc = WINDOW_CACHE[(year, co)]
    pre_l, pre_r = wc["pre_l"], wc["pre_r"]
    total, recent = wc["total"], wc["recent"]
    post_l, post_r = wc["post_l"], wc["post_r"]
    xb = p["XWOBA_BLEND"]

    all_pids = set(pre_l.keys()) | set(pre_r.keys())
    pred_l, act_l = [], []
    pred_r, act_r = [], []

    for pid in all_pids:
        sl, sr = pre_l.get(pid), pre_r.get(pid)
        cl, cr = car_l.get(pid), car_r.get(pid)
        tot, rec = total.get(pid), recent.get(pid)

        sl_pa = sl["pa"] if sl else 0
        sr_pa = sr["pa"] if sr else 0
        cl_pa = cl["pa"] if cl else 0
        cr_pa = cr["pa"] if cr else 0
        tot_pa = tot["pa"] if tot else 0
        rec_pa = rec["pa"] if rec else 0

        sl_w, sr_w = blend_woba(sl, xb), blend_woba(sr, xb)
        cl_w, cr_w = blend_woba(cl, xb), blend_woba(cr, xb)
        tot_w, rec_w = blend_woba(tot, xb), blend_woba(rec, xb)

        has_split_l = (sl_pa >= MIN_PA_SPLITS) or (cl_pa >= MIN_PA_CAREER)
        if has_split_l:
            post = post_l.get(pid)
            if post is not None and post["pa"] >= MIN_PA_POST:
                pred = stabilize(p, sl_pa, sl_w, cl_pa, cl_w, tot_pa, tot_w,
                                  rec_pa, rec_w, 'L',
                                  career_cross_pa=cr_pa, career_cross_woba=cr_w)
                if pred is not None:
                    pred_l.append(pred); act_l.append(post["woba"])

        has_split_r = (sr_pa >= MIN_PA_SPLITS) or (cr_pa >= MIN_PA_CAREER)
        if has_split_r:
            post = post_r.get(pid)
            if post is not None and post["pa"] >= MIN_PA_POST:
                pred = stabilize(p, sr_pa, sr_w, cr_pa, cr_w, tot_pa, tot_w,
                                  rec_pa, rec_w, 'R',
                                  career_cross_pa=cl_pa, career_cross_woba=cl_w)
                if pred is not None:
                    pred_r.append(pred); act_r.append(post["woba"])

    r_l = spearman_r(pred_l, act_l)
    r_r = spearman_r(pred_r, act_r)
    return r_l, r_r


def run_all(p, years):
    """Returns (avg_r, lhp_avg, rhp_avg) over the given subset of years,
    matching backtest.py's 'mean of all r_l/r_r values' formula."""
    car_cache = {}
    all_rs, lhp_rs, rhp_rs = [], [], []
    for year in years:
        if year not in car_cache:
            car_cache[year] = (
                build_career_data(year, "left",  p["DECAY_CAREER_LHP"]),
                build_career_data(year, "right", p["DECAY_CAREER_RHP"]),
            )
        car_l, car_r = car_cache[year]
        for co in CUTOFFS:
            r_l, r_r = run_window(p, year, co, car_l, car_r)
            if r_l is not None:
                lhp_rs.append(r_l); all_rs.append(r_l)
            if r_r is not None:
                rhp_rs.append(r_r); all_rs.append(r_r)
    if not all_rs:
        return None, None, None
    avg = sum(all_rs) / len(all_rs)
    lhp_avg = sum(lhp_rs) / len(lhp_rs) if lhp_rs else None
    rhp_avg = sum(rhp_rs) / len(rhp_rs) if rhp_rs else None
    return avg, lhp_avg, rhp_avg


def score_train(p):
    avg, _, _ = run_all(p, TRAIN_YEARS)
    return avg if avg is not None else -1.0


# -------------------------------------------------------
# PARAMETER SPACE: bounds + perturbation scale per param
# -------------------------------------------------------
# (lo, hi, step) -- step is the typical perturbation size used by the
# coordinate-ascent grid sweep and as the std-dev scale for random jumps.

BOUNDS = {
    "LA_LHP": (0.20, 0.34, 0.01),
    "LA_RHP": (0.20, 0.34, 0.01),

    "WS_L": (0.0, 1.5, 0.05), "WC_L": (0.0, 1.5, 0.05),
    "WT_L": (0.0, 1.5, 0.05), "WR_L": (0.0, 1.5, 0.05),
    "WS_R": (0.0, 1.5, 0.05), "WC_R": (0.0, 1.5, 0.05),
    "WT_R": (0.0, 1.5, 0.05), "WR_R": (0.0, 1.5, 0.05),

    "RPS_L": (1, 2500, 40), "RPC_L": (1, 2500, 40),
    "RPT_L": (1, 2500, 40), "RPR_L": (1, 2500, 40),
    "RPS_R": (1, 2500, 40), "RPC_R": (1, 2500, 40),
    "RPT_R": (1, 2500, 40), "RPR_R": (1, 2500, 40),

    "PLATOON_BLEND": (0.0, 1.0, 0.04),
    "PLATOON_MIN":   (0.30, 0.99, 0.03),
    "PLATOON_MAX":   (1.01, 2.20, 0.04),

    "DECAY_CAREER_LHP": (0.40, 1.0, 0.03),
    "DECAY_CAREER_RHP": (0.40, 1.0, 0.03),

    "XWOBA_BLEND": (0.0, 1.0, 0.05),
}

PARAM_NAMES = list(BOUNDS.keys())

# Starting point: current v9 on-disk values (backtest.py as of 2026-06-16)
START = {
    "LA_LHP": 0.2585, "LA_RHP": 0.2575,
    "WS_L": 0.1365, "WC_L": 0.6142, "WT_L": 0.1443, "WR_L": 0.1050,
    "RPS_L": 355, "RPC_L": 735, "RPT_L": 910, "RPR_L": 795,
    "WS_R": 0.2774, "WC_R": 0.2920, "WT_R": 0.2153, "WR_R": 0.2153,
    "RPS_R": 285, "RPC_R": 25, "RPT_R": 400, "RPR_R": 665,
    "PLATOON_BLEND": 0.31, "PLATOON_MIN": 0.585, "PLATOON_MAX": 1.325,
    "DECAY_CAREER_LHP": 1.0, "DECAY_CAREER_RHP": 1.0,
    "XWOBA_BLEND": 0.5,
}


def clamp(name, val):
    lo, hi, _ = BOUNDS[name]
    val = max(lo, min(hi, val))
    if name.startswith("RP"):
        val = round(val)
    return val


def fix_platoon_order(p):
    """Keep PLATOON_MIN < 1.0 < PLATOON_MAX after perturbation."""
    if p["PLATOON_MIN"] >= 1.0:
        p["PLATOON_MIN"] = 0.99
    if p["PLATOON_MAX"] <= 1.0:
        p["PLATOON_MAX"] = 1.01


# -------------------------------------------------------
# OPTIMIZER: coordinate ascent + random search
# -------------------------------------------------------

def random_search(p0, score0, n_iters, rng, log_every=250, start_time=None):
    best = dict(p0)
    best_score = score0
    n_accept = 0
    for i in range(1, n_iters + 1):
        # Random-direction step: perturb a random subset of params
        # (1 to 5 at a time), step size shrinks over the run (annealing).
        cand = dict(best)
        frac_done = i / n_iters
        temp = 1.0 - 0.85 * frac_done  # shrinks from 1.0 to 0.15
        k = rng.choice([1, 1, 1, 2, 2, 3, 5])
        names = rng.sample(PARAM_NAMES, k=min(k, len(PARAM_NAMES)))
        for name in names:
            lo, hi, step = BOUNDS[name]
            delta = rng.gauss(0, step * temp)
            cand[name] = clamp(name, cand[name] + delta)
        fix_platoon_order(cand)

        s = score_train(cand)
        if s > best_score:
            best_score = s
            best = cand
            n_accept += 1

        if log_every and i % log_every == 0:
            elapsed = time.time() - start_time if start_time else 0
            print("  [random %5d/%d] best_train_r=%.4f  accepts=%d  elapsed=%.0fs" %
                  (i, n_iters, best_score, n_accept, elapsed))
            sys.stdout.flush()
    return best, best_score


def coordinate_sweep(p0, score0, rng, max_passes=3):
    best = dict(p0)
    best_score = score0
    for pass_i in range(max_passes):
        improved_this_pass = False
        order = list(PARAM_NAMES)
        rng.shuffle(order)
        for name in order:
            lo, hi, step = BOUNDS[name]
            cur = best[name]
            for mult in (-2, -1, 1, 2):
                cand = dict(best)
                cand[name] = clamp(name, cur + mult * step)
                fix_platoon_order(cand)
                s = score_train(cand)
                if s > best_score:
                    best_score = s
                    best = cand
                    improved_this_pass = True
        print("  [coord pass %d] best_train_r=%.4f%s" %
              (pass_i + 1, best_score, "  (improved)" if improved_this_pass else "  (no change)"))
        sys.stdout.flush()
        if not improved_this_pass:
            break
    return best, best_score


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

N_RANDOM_ITERS_PHASE1 = 1500
N_RANDOM_ITERS_PHASE2 = 1500
SEED = 20260616

def main():
    rng = random.Random(SEED)
    t0 = time.time()

    print("=== TUNER v2 -- Full joint retune (Tasks #23, #24, #25) ===")
    print("Results saved to: " + OUTPUT_FILE)
    print("Train years: " + str(TRAIN_YEARS) + "  (12 windows, search objective)")
    print("Holdout years: " + str(HOLDOUT_YEARS) + "  (6 windows, evaluated only at the end)")
    print("24 tunable parameters; PA minimums fixed at 30; NO_CLAMP=True.")
    print()

    start_train_avg, start_train_lhp, start_train_rhp = run_all(START, TRAIN_YEARS)
    start_hold_avg, start_hold_lhp, start_hold_rhp = run_all(START, HOLDOUT_YEARS)
    start_all_avg, start_all_lhp, start_all_rhp = run_all(START, TEST_YEARS)
    print("BASELINE (current v9 on-disk values):")
    print("  TRAIN   (12 windows): all=%.4f  lhp=%.4f  rhp=%.4f" %
          (start_train_avg, start_train_lhp, start_train_rhp))
    print("  HOLDOUT (6 windows):  all=%.4f  lhp=%.4f  rhp=%.4f" %
          (start_hold_avg, start_hold_lhp, start_hold_rhp))
    print("  ALL 18 windows:       all=%.4f  lhp=%.4f  rhp=%.4f" %
          (start_all_avg, start_all_lhp, start_all_rhp))
    print()

    cur = dict(START)
    cur_score = score_train(cur)

    print("--- Phase 1: random search (%d iters) ---" % N_RANDOM_ITERS_PHASE1)
    cur, cur_score = random_search(cur, cur_score, N_RANDOM_ITERS_PHASE1, rng, start_time=t0)
    print()

    print("--- Phase 2: coordinate-ascent polish ---")
    cur, cur_score = coordinate_sweep(cur, cur_score, rng, max_passes=4)
    print()

    print("--- Phase 3: random search round 2 (%d iters, fine-grained) ---" % N_RANDOM_ITERS_PHASE2)
    cur, cur_score = random_search(cur, cur_score, N_RANDOM_ITERS_PHASE2, rng, start_time=t0)
    print()

    print("--- Phase 4: final coordinate-ascent polish ---")
    cur, cur_score = coordinate_sweep(cur, cur_score, rng, max_passes=4)
    print()

    final_train_avg, final_train_lhp, final_train_rhp = run_all(cur, TRAIN_YEARS)
    final_hold_avg, final_hold_lhp, final_hold_rhp = run_all(cur, HOLDOUT_YEARS)
    final_all_avg, final_all_lhp, final_all_rhp = run_all(cur, TEST_YEARS)

    elapsed = time.time() - t0
    print("=" * 60)
    print("RESULTS  (elapsed: %.0fs)" % elapsed)
    print("=" * 60)
    print()
    print("                         all      lhp      rhp")
    print("BASELINE  train        %.4f   %.4f   %.4f" % (start_train_avg, start_train_lhp, start_train_rhp))
    print("BASELINE  holdout      %.4f   %.4f   %.4f" % (start_hold_avg, start_hold_lhp, start_hold_rhp))
    print("BASELINE  all-18       %.4f   %.4f   %.4f" % (start_all_avg, start_all_lhp, start_all_rhp))
    print()
    print("TUNED     train        %.4f   %.4f   %.4f" % (final_train_avg, final_train_lhp, final_train_rhp))
    print("TUNED     holdout      %.4f   %.4f   %.4f" % (final_hold_avg, final_hold_lhp, final_hold_rhp))
    print("TUNED     all-18       %.4f   %.4f   %.4f" % (final_all_avg, final_all_lhp, final_all_rhp))
    print()
    print("Holdout delta vs baseline: %+.4f  (this is the generalization check --" %
          (final_hold_avg - start_hold_avg))
    print("if it's positive and not much smaller than the train gain, the tuned")
    print("params generalize rather than just overfitting the 12 train windows.)")
    print()
    print("=" * 60)
    print("TUNED PARAMETERS -- paste these into backtest.py (and later test.py)")
    print("=" * 60)
    print()
    print("LEAGUE_AVG_WOBA_LHP = %s" % round(cur["LA_LHP"], 4))
    print("LEAGUE_AVG_WOBA_RHP = %s" % round(cur["LA_RHP"], 4))
    print()
    print("WEIGHT_SPLITS_LHP  = %s" % round(cur["WS_L"], 4))
    print("WEIGHT_CAREER_LHP  = %s" % round(cur["WC_L"], 4))
    print("WEIGHT_TOTAL_LHP   = %s" % round(cur["WT_L"], 4))
    print("WEIGHT_RECENT_LHP  = %s" % round(cur["WR_L"], 4))
    print()
    print("REGRESS_PA_SPLITS_LHP = %s" % int(cur["RPS_L"]))
    print("REGRESS_PA_CAREER_LHP = %s" % int(cur["RPC_L"]))
    print("REGRESS_PA_TOTAL_LHP  = %s" % int(cur["RPT_L"]))
    print("REGRESS_PA_RECENT_LHP = %s" % int(cur["RPR_L"]))
    print()
    print("WEIGHT_SPLITS_RHP  = %s" % round(cur["WS_R"], 4))
    print("WEIGHT_CAREER_RHP  = %s" % round(cur["WC_R"], 4))
    print("WEIGHT_TOTAL_RHP   = %s" % round(cur["WT_R"], 4))
    print("WEIGHT_RECENT_RHP  = %s" % round(cur["WR_R"], 4))
    print()
    print("REGRESS_PA_SPLITS_RHP = %s" % int(cur["RPS_R"]))
    print("REGRESS_PA_CAREER_RHP = %s" % int(cur["RPC_R"]))
    print("REGRESS_PA_TOTAL_RHP  = %s" % int(cur["RPT_R"]))
    print("REGRESS_PA_RECENT_RHP = %s" % int(cur["RPR_R"]))
    print()
    print("PLATOON_BLEND      = %s" % round(cur["PLATOON_BLEND"], 4))
    print("PLATOON_FACTOR_MIN = %s" % round(cur["PLATOON_MIN"], 4))
    print("PLATOON_FACTOR_MAX = %s" % round(cur["PLATOON_MAX"], 4))
    print()
    print("DECAY_CAREER_LHP = %s" % round(cur["DECAY_CAREER_LHP"], 4))
    print("DECAY_CAREER_RHP = %s" % round(cur["DECAY_CAREER_RHP"], 4))
    print()
    print("XWOBA_BLEND = %s" % round(cur["XWOBA_BLEND"], 4))
    print()
    print("=== DONE ===")


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
