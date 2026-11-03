#!/usr/bin/env python3
"""
Tuner - per-component xwOBA blend split (v12), LHP and RHP independent
=====================================================================
backtest.py v12 replaced the single XWOBA_BLEND_LHP / XWOBA_BLEND_RHP
ratio with four per-component ratios per hand --
XWOBA_BLEND_{SPLITS,CAREER,TOTAL,RECENT}_{LHP,RHP}, 8 constants total --
on the theory that components with very different sample-size/noise
profiles (e.g. small-sample "recent" vs. large-sample "career") likely
have different optimal xwOBA weighting. That refactor was verified
bit-for-bit against the pre-refactor production r values (LHP 0.3299,
RHP 0.4113) before this tuner was written. This file searches the 8
new constants via the same walk-forward CV approach as
tuner_lhp.py/tuner_rhp.py.

WHY THIS TUNER IS STRUCTURALLY SIMPLER THAN tuner_lhp.py/tuner_rhp.py:
Both of those files had to solve a "cross-hand leak" problem: one
hand's tunable career-decay constant (DECAY_CAREER_LHP or _RHP) was
also used to build the career data fed into the OTHER hand's platoon
cross-hand input, so naively rebuilding career data per trial silently
changed the "frozen" hand's score too. Each file fixed this with a
second, frozen copy of career data used only for cross-hand
consumption.

That problem does not exist here. DECAY_CAREER_LHP/RHP are BOTH frozen
in this file (career decay isn't tunable here at all), so the
underlying career data (pa/woba/xwoba per player) never changes across
trials -- it's built once per year, for both hands, and reused
everywhere. The only thing that changes across trials is WHICH xwOBA
blend ratio gets applied on top of that fixed data. And per the v12
design, the cross-hand career input always uses the CONSUMING side's
own CAREER ratio (XWOBA_BLEND_CAREER_LHP when LHP consumes RHP's career
row, XWOBA_BLEND_CAREER_RHP when RHP consumes LHP's career row) -- never
the producing side's ratio. That means:
  - LHP's prediction is a pure function of the 4 LHP-xwOBA params only.
  - RHP's prediction is a pure function of the 4 RHP-xwOBA params only.
Neither side's 4 params can affect the other side's score, full stop --
no frozen-copy trick needed. This is verified directly in main() below
as a canary check (the "should be EXACTLY 0" deltas), exactly like the
cross-hand-isolation checks in tuner_lhp.py/tuner_rhp.py, just for a
different reason (no shared inputs at all, vs. a frozen duplicate).

Because the two halves are provably independent, this file runs TWO
separate 4-parameter searches (LHP-xwOBA optimizing LHP-only train r,
RHP-xwOBA optimizing RHP-only train r) rather than one blended 8-param
search -- mirroring the existing one-objective-per-tunable-group
philosophy from tuner_lhp.py/tuner_rhp.py (a blended objective can hide
one side regressing behind the other side's gain; with two truly
disjoint objectives there is no such risk, but keeping them separate
also makes each search cheaper, since it's only exploring 4 dimensions
instead of 8). The two searches' results are then merged into a single
8-key parameter dict for reporting.

TUNES (8 parameters):
  XWOBA_BLEND_{SPLITS,CAREER,TOTAL,RECENT}_LHP   (4, LHP-only objective)
  XWOBA_BLEND_{SPLITS,CAREER,TOTAL,RECENT}_RHP   (4, RHP-only objective)

FROZEN AT v12 PRODUCTION BASELINE (not tunable, by design -- everything
except the 8 xwOBA-component ratios, both hands):
  LEAGUE_AVG_WOBA_LHP=0.2943, LEAGUE_AVG_WOBA_RHP=0.2575
  WEIGHT_*_LHP / WEIGHT_*_RHP (8 weights)
  REGRESS_PA_*_LHP / REGRESS_PA_*_RHP (8 regress-PA values)
  DECAY_CAREER_LHP=0.8001, DECAY_CAREER_RHP=1.0
  PLATOON_BLEND_LHP=0.7833, PLATOON_FACTOR_MIN_LHP=0.612, PLATOON_FACTOR_MAX_LHP=1.138
  PLATOON_BLEND_RHP=0.31, PLATOON_FACTOR_MIN_RHP=0.585, PLATOON_FACTOR_MAX_RHP=1.325

PA minimums fixed at 30 (not tunable). NO_CLAMP fixed True.

STARTING POINT: all 8 params start at the v12 baseline (0.9187 for all
4 LHP components, 0.80 for all 4 RHP components) -- i.e. the search
starts exactly at "no per-component differentiation" and CV decides
whether differentiating any of the four ratios actually generalizes,
or whether the single-ratio-per-hand model was already as good as it
gets.

RUNTIME: both hands' career data is cached once per year (no tunable
decay in this file), so this should run faster per-search than
tuner_rhp.py despite running two searches per fold (4 dimensions
instead of 14 makes both random search and coordinate ascent cheaper
per pass). Expect roughly 30-60 minutes total.

USAGE:
  Double-click run_tuner_xwoba_components.bat (writes
  tuner_xwoba_components_results_YYYY-MM-DD.txt)
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
OUTPUT_FILE = os.path.join(FOLDER, "tuner_xwoba_components_results_" + today + ".txt")


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
# FIXED CONSTANTS (not tunable -- everything except the 8 xwOBA-
# component ratios, for BOTH hands, frozen at v12 production values)
# -------------------------------------------------------
MIN_PA_SPLITS = 30
MIN_PA_TOTAL  = 30
MIN_PA_CAREER = 30
MIN_PA_POST   = 30
NO_CLAMP = True

LA_LHP = 0.2943
LA_RHP = 0.2575

WS_L, WC_L, WT_L, WR_L = 0.213, 0.5695, 0.247, 0.3416
RPS_L, RPC_L, RPT_L, RPR_L = 178, 711, 1020, 757

WS_R, WC_R, WT_R, WR_R = 0.2774, 0.2920, 0.2153, 0.2153
RPS_R, RPC_R, RPT_R, RPR_R = 285, 25, 400, 665

PLATOON_BLEND_LHP, PLATOON_MIN_LHP, PLATOON_MAX_LHP = 0.7833, 0.612, 1.138
PLATOON_BLEND_RHP, PLATOON_MIN_RHP, PLATOON_MAX_RHP = 0.31, 0.585, 1.325

DECAY_CAREER_LHP = 0.8001
DECAY_CAREER_RHP = 1.0

TEST_YEARS = [2019, 2021, 2022, 2023, 2024, 2025]
CUTOFFS    = ["june_1", "july_1", "aug_1"]
TEST_WINDOWS = [{"year": y, "cutoff": c} for y in TEST_YEARS for c in CUTOFFS]

# Walk-forward (rolling-origin) CV folds -- same folds as
# tuner_lhp.py/tuner_rhp.py for direct comparability.
CV_FOLDS = [
    {"train": [2019, 2021, 2022],             "test": 2023},
    {"train": [2019, 2021, 2022, 2023],       "test": 2024},
    {"train": [2019, 2021, 2022, 2023, 2024], "test": 2025},
]
ALL_YEARS = [2019, 2021, 2022, 2023, 2024, 2025]  # final production fit, no holdout

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
# CAREER DATA
# -------------------------------------------------------
# Neither hand's decay is tunable in this file, so career data for BOTH
# hands never changes across trials -- safe to precompute once per year
# and reuse everywhere (own-hand prediction AND cross-hand platoon
# input alike). No frozen-copy trick needed here, unlike
# tuner_lhp.py/tuner_rhp.py -- see module docstring.

def build_career_data(test_year, hand, decay):
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
# SCORING MATH (mirrors backtest.py v12 exactly). Nothing inside
# stabilize() is tunable in this file -- LA/weights/regress-PA/platoon
# are all frozen module-level constants, not pulled from a params dict.
# The only thing the params dict controls is which xwOBA ratio gets
# applied when blending each component BEFORE stabilize() is called
# (see run_window below).
# -------------------------------------------------------

def regress(woba, pa, regress_pa, la):
    return (woba * pa + la * regress_pa) / (pa + regress_pa)


def stabilize(splits_pa, splits_woba, career_pa, career_woba,
              total_pa, total_woba, recent_pa, recent_woba, side,
              career_cross_pa=0, career_cross_woba=None):
    if side == 'L':
        la, WS, WC, WT, WR = LA_LHP, WS_L, WC_L, WT_L, WR_L
        RPS, RPC, RPT, RPR = RPS_L, RPC_L, RPT_L, RPR_L
        pb, pmin, pmax = PLATOON_BLEND_LHP, PLATOON_MIN_LHP, PLATOON_MAX_LHP
    else:
        la, WS, WC, WT, WR = LA_RHP, WS_R, WC_R, WT_R, WR_R
        RPS, RPC, RPT, RPR = RPS_R, RPC_R, RPT_R, RPR_R
        pb, pmin, pmax = PLATOON_BLEND_RHP, PLATOON_MIN_RHP, PLATOON_MAX_RHP

    platoon_factor = 1.0
    if (pb > 0 and career_woba is not None and career_pa > 0 and
            career_cross_woba is not None and career_cross_pa > 0):
        combined_pa = career_pa + career_cross_pa
        combined_woba = (career_woba * career_pa +
                          career_cross_woba * career_cross_pa) / combined_pa
        if combined_woba > 0:
            raw_factor = career_woba / combined_woba
            clamped_factor = max(pmin, min(pmax, raw_factor))
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
    """Returns (r_l, r_r, pa_l, pa_r). car_l/car_r are the precomputed,
    always-frozen career dicts for this year (built once in main(), never
    rebuilt per trial -- decay isn't tunable here for either hand). p
    holds only the 8 xwOBA-component ratios; LHP's prediction uses only
    the 4 *_LHP keys, RHP's uses only the 4 *_RHP keys -- including for
    the cross-hand career input, which always uses the CONSUMING side's
    own CAREER ratio (XWOBA_CAREER_LHP / XWOBA_CAREER_RHP), never the
    other hand's. That's what makes the two sides provably independent."""
    wc = WINDOW_CACHE[(year, co)]
    pre_l, pre_r = wc["pre_l"], wc["pre_r"]
    total, recent = wc["total"], wc["recent"]
    post_l, post_r = wc["post_l"], wc["post_r"]

    all_pids = set(pre_l.keys()) | set(pre_r.keys())
    pred_l, act_l = [], []
    pred_r, act_r = [], []
    pa_l_sum, pa_r_sum = 0, 0

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

        # LHP: uses only the 4 *_LHP ratios. Cross-hand input (RHP's
        # career row) is blended with LHP's own CAREER ratio.
        sl_w    = blend_woba(sl,  p["XWOBA_SPLITS_LHP"])
        cl_w_l  = blend_woba(cl,  p["XWOBA_CAREER_LHP"])
        cr_w_l  = blend_woba(cr,  p["XWOBA_CAREER_LHP"])
        tot_w_l = blend_woba(tot, p["XWOBA_TOTAL_LHP"])
        rec_w_l = blend_woba(rec, p["XWOBA_RECENT_LHP"])

        # RHP: uses only the 4 *_RHP ratios. Cross-hand input (LHP's
        # career row) is blended with RHP's own CAREER ratio.
        sr_w    = blend_woba(sr,  p["XWOBA_SPLITS_RHP"])
        cr_w_r  = blend_woba(cr,  p["XWOBA_CAREER_RHP"])
        cl_w_r  = blend_woba(cl,  p["XWOBA_CAREER_RHP"])
        tot_w_r = blend_woba(tot, p["XWOBA_TOTAL_RHP"])
        rec_w_r = blend_woba(rec, p["XWOBA_RECENT_RHP"])

        has_split_l = (sl_pa >= MIN_PA_SPLITS) or (cl_pa >= MIN_PA_CAREER)
        if has_split_l:
            post = post_l.get(pid)
            if post is not None and post["pa"] >= MIN_PA_POST:
                pred = stabilize(sl_pa, sl_w, cl_pa, cl_w_l, tot_pa, tot_w_l,
                                  rec_pa, rec_w_l, 'L',
                                  career_cross_pa=cr_pa, career_cross_woba=cr_w_l)
                if pred is not None:
                    pred_l.append(pred); act_l.append(post["woba"]); pa_l_sum += post["pa"]

        has_split_r = (sr_pa >= MIN_PA_SPLITS) or (cr_pa >= MIN_PA_CAREER)
        if has_split_r:
            post = post_r.get(pid)
            if post is not None and post["pa"] >= MIN_PA_POST:
                pred = stabilize(sr_pa, sr_w, cr_pa, cr_w_r, tot_pa, tot_w_r,
                                  rec_pa, rec_w_r, 'R',
                                  career_cross_pa=cl_pa, career_cross_woba=cl_w_r)
                if pred is not None:
                    pred_r.append(pred); act_r.append(post["woba"]); pa_r_sum += post["pa"]

    r_l = spearman_r(pred_l, act_l)
    r_r = spearman_r(pred_r, act_r)
    return r_l, r_r, pa_l_sum, pa_r_sum


def _weighted_avg(values, weights):
    total_w = sum(weights)
    if total_w <= 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total_w


def run_all(p, years, car_l_cache, car_r_cache):
    """Window-level Spearman r values combined into weighted averages,
    using sqrt(total post-cutoff PA) as the weight -- same rationale as
    tuner_lhp.py/tuner_rhp.py: raw PA would let June windows (more
    season left) outweigh August windows by ~2x for reasons having
    nothing to do with estimate reliability; sqrt damps that to ~1.4x."""
    all_rs, all_ws = [], []
    lhp_rs, lhp_ws = [], []
    rhp_rs, rhp_ws = [], []
    for year in years:
        car_l = car_l_cache[year]
        car_r = car_r_cache[year]
        for co in CUTOFFS:
            r_l, r_r, pa_l, pa_r = run_window(p, year, co, car_l, car_r)
            if r_l is not None:
                w_l = math.sqrt(pa_l)
                lhp_rs.append(r_l); lhp_ws.append(w_l)
                all_rs.append(r_l); all_ws.append(w_l)
            if r_r is not None:
                w_r = math.sqrt(pa_r)
                rhp_rs.append(r_r); rhp_ws.append(w_r)
                all_rs.append(r_r); all_ws.append(w_r)
    if not all_rs:
        return None, None, None
    avg = _weighted_avg(all_rs, all_ws)
    lhp_avg = _weighted_avg(lhp_rs, lhp_ws) if lhp_rs else None
    rhp_avg = _weighted_avg(rhp_rs, rhp_ws) if rhp_rs else None
    return avg, lhp_avg, rhp_avg


def per_cutoff_avg(p, years, car_l_cache, car_r_cache, side):
    """Diagnostic only -- NOT used in the objective. Plain (unweighted)
    average r per cutoff for one side, so a June-vs-August tradeoff from
    the sqrt-PA weighting would show up directly."""
    by_cutoff = {co: [] for co in CUTOFFS}
    for year in years:
        car_l = car_l_cache[year]
        car_r = car_r_cache[year]
        for co in CUTOFFS:
            r_l, r_r, pa_l, pa_r = run_window(p, year, co, car_l, car_r)
            v = r_l if side == 'L' else r_r
            if v is not None:
                by_cutoff[co].append(v)
    return {co: (sum(v) / len(v) if v else None) for co, v in by_cutoff.items()}


# -------------------------------------------------------
# PARAMETER SPACE
# -------------------------------------------------------

BOUNDS = {
    "XWOBA_SPLITS_LHP": (0.0, 1.0, 0.05),
    "XWOBA_CAREER_LHP": (0.0, 1.0, 0.05),
    "XWOBA_TOTAL_LHP":  (0.0, 1.0, 0.05),
    "XWOBA_RECENT_LHP": (0.0, 1.0, 0.05),

    "XWOBA_SPLITS_RHP": (0.0, 1.0, 0.05),
    "XWOBA_CAREER_RHP": (0.0, 1.0, 0.05),
    "XWOBA_TOTAL_RHP":  (0.0, 1.0, 0.05),
    "XWOBA_RECENT_RHP": (0.0, 1.0, 0.05),
}

PARAM_NAMES_LHP = ["XWOBA_SPLITS_LHP", "XWOBA_CAREER_LHP", "XWOBA_TOTAL_LHP", "XWOBA_RECENT_LHP"]
PARAM_NAMES_RHP = ["XWOBA_SPLITS_RHP", "XWOBA_CAREER_RHP", "XWOBA_TOTAL_RHP", "XWOBA_RECENT_RHP"]

# Starting point: the v12 baseline (no per-component differentiation
# yet) -- all 4 LHP components at the old single LHP ratio, all 4 RHP
# components at the old single RHP ratio. The search starts exactly at
# "all four equal" and CV decides whether differentiating helps.
START = {
    "XWOBA_SPLITS_LHP": 0.9187, "XWOBA_CAREER_LHP": 0.9187,
    "XWOBA_TOTAL_LHP":  0.9187, "XWOBA_RECENT_LHP": 0.9187,

    "XWOBA_SPLITS_RHP": 0.80, "XWOBA_CAREER_RHP": 0.80,
    "XWOBA_TOTAL_RHP":  0.80, "XWOBA_RECENT_RHP": 0.80,
}


def clamp(name, val):
    lo, hi, _ = BOUNDS[name]
    return max(lo, min(hi, val))


# -------------------------------------------------------
# OPTIMIZER -- generic over which 4-key subset is being searched, since
# the same machinery runs once for PARAM_NAMES_LHP (objective = LHP-only
# train r) and once for PARAM_NAMES_RHP (objective = RHP-only train r).
# -------------------------------------------------------

def make_score_fn(years, car_l_cache, car_r_cache, side):
    def score_train(p):
        avg, lhp_avg, rhp_avg = run_all(p, years, car_l_cache, car_r_cache)
        target = lhp_avg if side == 'L' else rhp_avg
        return target if target is not None else -1.0
    return score_train


def random_search(p0, score0, n_iters, rng, score_fn, param_names, log_every=250, start_time=None, log_label=""):
    best = dict(p0)
    best_score = score0
    n_accept = 0
    for i in range(1, n_iters + 1):
        cand = dict(best)
        frac_done = i / n_iters
        temp = 1.0 - 0.85 * frac_done
        k = rng.choice([1, 1, 1, 2, 2])
        names = rng.sample(param_names, k=min(k, len(param_names)))
        for name in names:
            lo, hi, step = BOUNDS[name]
            delta = rng.gauss(0, step * temp)
            cand[name] = clamp(name, cand[name] + delta)

        s = score_fn(cand)
        if s > best_score:
            best_score = s
            best = cand
            n_accept += 1

        if log_every and i % log_every == 0:
            elapsed = time.time() - start_time if start_time else 0
            print("    [random %5d/%d] best_%s_r=%.4f  accepts=%d  elapsed=%.0fs" %
                  (i, n_iters, log_label, best_score, n_accept, elapsed))
            sys.stdout.flush()
    return best, best_score


def coordinate_sweep(p0, score0, rng, score_fn, param_names, max_passes=4, log_label=""):
    best = dict(p0)
    best_score = score0
    for pass_i in range(max_passes):
        improved_this_pass = False
        order = list(param_names)
        rng.shuffle(order)
        for name in order:
            lo, hi, step = BOUNDS[name]
            cur = best[name]
            for mult in (-2, -1, 1, 2):
                cand = dict(best)
                cand[name] = clamp(name, cur + mult * step)
                s = score_fn(cand)
                if s > best_score:
                    best_score = s
                    best = cand
                    improved_this_pass = True
        print("    [coord pass %d] best_%s_r=%.4f%s" %
              (pass_i + 1, log_label, best_score, "  (improved)" if improved_this_pass else "  (no change)"))
        sys.stdout.flush()
        if not improved_this_pass:
            break
    return best, best_score


# -------------------------------------------------------
# ONE FULL SEARCH (random -> coord -> random -> coord), with restarts
# -------------------------------------------------------

N_RANDOM_ITERS_PHASE1 = 600
N_RANDOM_ITERS_PHASE2 = 600
SEED = 20260618
N_RESTARTS = 3  # independent random-seed attempts per search; keeps the
                # best of N as a cheap check against local optima.


def run_search(years, car_l_cache, car_r_cache, side, seed, label):
    """Runs the full 4-phase search over only the given side's 4 xwOBA
    params (the other side's 4 stay fixed at START throughout, since
    only param_names gets perturbed). Returns (tuned_params, tuned_train_score)
    as a full 8-key dict / scalar."""
    param_names = PARAM_NAMES_LHP if side == 'L' else PARAM_NAMES_RHP
    log_label = "lhp" if side == 'L' else "rhp"
    rng = random.Random(seed)
    score_fn = make_score_fn(years, car_l_cache, car_r_cache, side)
    cur = dict(START)
    cur_score = score_fn(cur)
    print("[%s] years=%s  starting %s-only train r: %.4f" % (label, years, log_label, cur_score))
    sys.stdout.flush()

    print("  --- Phase 1: random search (%d iters) ---" % N_RANDOM_ITERS_PHASE1)
    cur, cur_score = random_search(cur, cur_score, N_RANDOM_ITERS_PHASE1, rng, score_fn, param_names,
                                    start_time=time.time(), log_label=log_label)

    print("  --- Phase 2: coordinate-ascent polish ---")
    cur, cur_score = coordinate_sweep(cur, cur_score, rng, score_fn, param_names, max_passes=4, log_label=log_label)

    print("  --- Phase 3: random search round 2 (%d iters, fine-grained) ---" % N_RANDOM_ITERS_PHASE2)
    cur, cur_score = random_search(cur, cur_score, N_RANDOM_ITERS_PHASE2, rng, score_fn, param_names,
                                    start_time=time.time(), log_label=log_label)

    print("  --- Phase 4: final coordinate-ascent polish ---")
    cur, cur_score = coordinate_sweep(cur, cur_score, rng, score_fn, param_names, max_passes=4, log_label=log_label)
    print("[%s] final %s-only train r: %.4f" % (label, log_label, cur_score))
    print()
    sys.stdout.flush()
    return cur, cur_score


def run_search_with_restarts(years, car_l_cache, car_r_cache, side, base_seed, label, n_restarts=N_RESTARTS):
    best_p, best_score = None, None
    for r in range(n_restarts):
        seed = base_seed + r * 100000
        sub_label = "%s restart %d/%d" % (label, r + 1, n_restarts)
        p, score = run_search(years, car_l_cache, car_r_cache, side, seed, sub_label)
        flag = ""
        if best_score is None or score > best_score:
            best_score = score
            best_p = p
            flag = "  <- new best"
        print("[%s] restart %d/%d score=%.4f%s" % (label, r + 1, n_restarts, score, flag))
    print("[%s] kept best of %d restarts: %.4f" % (label, n_restarts, best_score))
    print()
    sys.stdout.flush()
    return best_p, best_score


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    t0 = time.time()

    print("=== TUNER XWOBA COMPONENTS -- per-component xwOBA blend split (v12) ===")
    print("Results saved to: " + OUTPUT_FILE)
    print("CV folds (rolling-origin; test year is never in that fold's training years):")
    for i, f in enumerate(CV_FOLDS, 1):
        print("  Fold %d: train=%s  test=%d" % (i, f["train"], f["test"]))
    print("Final fit: train=%s  (no holdout -- production candidate)" % ALL_YEARS)
    print("8 tunable params total, run as two independent 4-param searches:")
    print("  LHP search tunes: " + ", ".join(PARAM_NAMES_LHP) + "  (objective: LHP-only train r)")
    print("  RHP search tunes: " + ", ".join(PARAM_NAMES_RHP) + "  (objective: RHP-only train r)")
    print("Everything else (weights, regress-PA, LA, decay, platoon, both hands) frozen")
    print("at v12 production values.")
    print("Each search runs %d independent random-seed restarts, keeping the best." % N_RESTARTS)
    print("Window averages are weighted by sqrt(total post-cutoff PA).")
    print("Expect roughly 30-60 minutes total (2 searches x %d restarts each, x4 fold-equivalents)." % N_RESTARTS)
    print()

    # Neither hand's career decay is tunable here -- precompute both
    # hands' career data once per year and never rebuild it.
    car_l_cache = {}
    car_r_cache = {}
    for year in TEST_YEARS:
        car_l_cache[year] = build_career_data(year, "left", DECAY_CAREER_LHP)
        car_r_cache[year] = build_career_data(year, "right", DECAY_CAREER_RHP)

    def full_eval(p, years):
        return run_all(p, years, car_l_cache, car_r_cache)

    fold_results = []
    for i, fold in enumerate(CV_FOLDS, 1):
        train_years, test_year = fold["train"], fold["test"]
        label = "Fold %d" % i
        print("=" * 60)
        print("%s: train=%s  test=%d" % (label, train_years, test_year))
        print("=" * 60)

        base_train_all, base_train_lhp, base_train_rhp = full_eval(START, train_years)
        base_test_all, base_test_lhp, base_test_rhp = full_eval(START, [test_year])
        print("  BASELINE train (lhp/rhp/all): %.4f / %.4f / %.4f" % (base_train_lhp, base_train_rhp, base_train_all))
        print("  BASELINE test  (lhp/rhp/all): %.4f / %.4f / %.4f" % (base_test_lhp, base_test_rhp, base_test_all))
        print()

        tuned_lhp_p, _ = run_search_with_restarts(train_years, car_l_cache, car_r_cache, 'L', SEED + i, label + " LHP")
        tuned_rhp_p, _ = run_search_with_restarts(train_years, car_l_cache, car_r_cache, 'R', SEED + i + 50000, label + " RHP")

        # Canary check: each search's own result, evaluated alone, should
        # leave the OTHER side bit-for-bit at baseline.
        _, canary_lhp_after_lhp_search, canary_rhp_after_lhp_search = full_eval(tuned_lhp_p, [test_year])
        _, canary_lhp_after_rhp_search, canary_rhp_after_rhp_search = full_eval(tuned_rhp_p, [test_year])
        print("  Canary: RHP test r after LHP-only search: %.6f  (baseline %.6f, delta %+.6f, should be EXACTLY 0)" %
              (canary_rhp_after_lhp_search, base_test_rhp, canary_rhp_after_lhp_search - base_test_rhp))
        print("  Canary: LHP test r after RHP-only search: %.6f  (baseline %.6f, delta %+.6f, should be EXACTLY 0)" %
              (canary_lhp_after_rhp_search, base_test_lhp, canary_lhp_after_rhp_search - base_test_lhp))

        # Merge the two independent results into one 8-key dict.
        merged_p = dict(START)
        for name in PARAM_NAMES_LHP:
            merged_p[name] = tuned_lhp_p[name]
        for name in PARAM_NAMES_RHP:
            merged_p[name] = tuned_rhp_p[name]

        merged_train_all, merged_train_lhp, merged_train_rhp = full_eval(merged_p, train_years)
        merged_test_all, merged_test_lhp, merged_test_rhp = full_eval(merged_p, [test_year])
        print("  Merge check: merged lhp/rhp test r should exactly match the two separate searches above:")
        print("    merged test lhp=%.6f (vs %.6f)   merged test rhp=%.6f (vs %.6f)" %
              (merged_test_lhp, canary_lhp_after_lhp_search, merged_test_rhp, canary_rhp_after_rhp_search))
        print()
        print("  TUNED   train (lhp/rhp/all): %.4f / %.4f / %.4f" % (merged_train_lhp, merged_train_rhp, merged_train_all))
        print("  TUNED   test  (lhp/rhp/all): %.4f / %.4f / %.4f" % (merged_test_lhp, merged_test_rhp, merged_test_all))
        print("  LHP test delta vs baseline: %+.4f" % (merged_test_lhp - base_test_lhp))
        print("  RHP test delta vs baseline: %+.4f" % (merged_test_rhp - base_test_rhp))

        base_cutoffs_l = per_cutoff_avg(START, [test_year], car_l_cache, car_r_cache, 'L')
        tuned_cutoffs_l = per_cutoff_avg(merged_p, [test_year], car_l_cache, car_r_cache, 'L')
        base_cutoffs_r = per_cutoff_avg(START, [test_year], car_l_cache, car_r_cache, 'R')
        tuned_cutoffs_r = per_cutoff_avg(merged_p, [test_year], car_l_cache, car_r_cache, 'R')
        print("  Per-cutoff LHP test r (baseline -> tuned):")
        for co in CUTOFFS:
            bv, tv = base_cutoffs_l[co], tuned_cutoffs_l[co]
            if bv is not None and tv is not None:
                print("    %-8s  %.4f -> %.4f  (%+.4f)" % (co, bv, tv, tv - bv))
        print("  Per-cutoff RHP test r (baseline -> tuned):")
        for co in CUTOFFS:
            bv, tv = base_cutoffs_r[co], tuned_cutoffs_r[co]
            if bv is not None and tv is not None:
                print("    %-8s  %.4f -> %.4f  (%+.4f)" % (co, bv, tv, tv - bv))
        print()

        fold_results.append({
            "fold": i, "train_years": train_years, "test_year": test_year, "params": merged_p,
            "base_test_lhp": base_test_lhp, "base_test_rhp": base_test_rhp, "base_test_all": base_test_all,
            "tuned_test_lhp": merged_test_lhp, "tuned_test_rhp": merged_test_rhp, "tuned_test_all": merged_test_all,
        })

    # --- Cross-fold summary ---
    print("=" * 60)
    print("CROSS-FOLD SUMMARY -- this is the generalization evidence")
    print("=" * 60)
    print()
    print("fold  test_yr   base_lhp  tuned_lhp    delta   |  base_rhp  tuned_rhp    delta")
    for r in fold_results:
        delta_lhp = r["tuned_test_lhp"] - r["base_test_lhp"]
        delta_rhp = r["tuned_test_rhp"] - r["base_test_rhp"]
        print("%4d  %7d   %8.4f  %9.4f  %+.4f   |  %8.4f  %9.4f  %+.4f" % (
            r["fold"], r["test_year"], r["base_test_lhp"], r["tuned_test_lhp"], delta_lhp,
            r["base_test_rhp"], r["tuned_test_rhp"], delta_rhp))
    n = len(fold_results)
    avg_base_lhp  = sum(r["base_test_lhp"] for r in fold_results) / n
    avg_tuned_lhp = sum(r["tuned_test_lhp"] for r in fold_results) / n
    avg_base_rhp  = sum(r["base_test_rhp"] for r in fold_results) / n
    avg_tuned_rhp = sum(r["tuned_test_rhp"] for r in fold_results) / n
    print("-" * 78)
    print(" avg            %8.4f  %9.4f  %+.4f   |  %8.4f  %9.4f  %+.4f" % (
        avg_base_lhp, avg_tuned_lhp, avg_tuned_lhp - avg_base_lhp,
        avg_base_rhp, avg_tuned_rhp, avg_tuned_rhp - avg_base_rhp))
    print()
    print("If a side's delta is positive and similar in size across all 3 rows, that")
    print("side's per-component xwOBA differentiation generalizes across different")
    print("out-of-sample years -- not just an artifact of one holdout split. If a")
    print("side's tuned values barely move from the START baseline (0.9187 LHP /")
    print("0.80 RHP, all 4 components), that means CV found no real benefit to")
    print("differentiating that hand's components -- a legitimate negative result,")
    print("not a bug.")
    print()

    # --- Per-fold tuned params (stability check) ---
    print("=" * 60)
    print("PER-FOLD TUNED PARAMETERS (compare across folds for stability)")
    print("=" * 60)
    for r in fold_results:
        print()
        print("Fold %d (train=%s, test=%d):" % (r["fold"], r["train_years"], r["test_year"]))
        for name in PARAM_NAMES_LHP + PARAM_NAMES_RHP:
            print("  %-20s = %s" % (name, round(r["params"][name], 4)))
    print()

    # --- Final production fit on ALL years ---
    print("=" * 60)
    print("FINAL PRODUCTION FIT (trained on all %d years, no held-out year)" % len(ALL_YEARS))
    print("=" * 60)
    print("Uses every available year for max data. Trust in these values comes")
    print("from the CV evidence above, not a fresh holdout -- there isn't one")
    print("left, since all years are now in the training set.")
    print()
    final_lhp_p, _ = run_search_with_restarts(ALL_YEARS, car_l_cache, car_r_cache, 'L', SEED + 99, "Final LHP")
    final_rhp_p, _ = run_search_with_restarts(ALL_YEARS, car_l_cache, car_r_cache, 'R', SEED + 99 + 50000, "Final RHP")
    final_p = dict(START)
    for name in PARAM_NAMES_LHP:
        final_p[name] = final_lhp_p[name]
    for name in PARAM_NAMES_RHP:
        final_p[name] = final_rhp_p[name]

    final_all, final_lhp, final_rhp = full_eval(final_p, ALL_YEARS)
    base_all, base_lhp, base_rhp = full_eval(START, ALL_YEARS)
    print("  BASELINE in-sample (lhp/rhp/all): %.4f / %.4f / %.4f" % (base_lhp, base_rhp, base_all))
    print("  Final fit in-sample (lhp/rhp/all): %.4f / %.4f / %.4f" % (final_lhp, final_rhp, final_all))
    base_cutoffs_l = per_cutoff_avg(START, ALL_YEARS, car_l_cache, car_r_cache, 'L')
    final_cutoffs_l = per_cutoff_avg(final_p, ALL_YEARS, car_l_cache, car_r_cache, 'L')
    base_cutoffs_r = per_cutoff_avg(START, ALL_YEARS, car_l_cache, car_r_cache, 'R')
    final_cutoffs_r = per_cutoff_avg(final_p, ALL_YEARS, car_l_cache, car_r_cache, 'R')
    print("  Per-cutoff LHP in-sample r (baseline -> final):")
    for co in CUTOFFS:
        bv, tv = base_cutoffs_l[co], final_cutoffs_l[co]
        if bv is not None and tv is not None:
            print("    %-8s  %.4f -> %.4f  (%+.4f)" % (co, bv, tv, tv - bv))
    print("  Per-cutoff RHP in-sample r (baseline -> final):")
    for co in CUTOFFS:
        bv, tv = base_cutoffs_r[co], final_cutoffs_r[co]
        if bv is not None and tv is not None:
            print("    %-8s  %.4f -> %.4f  (%+.4f)" % (co, bv, tv, tv - bv))
    print()
    print("=" * 60)
    print("TUNED PARAMETERS -- paste these into backtest.py")
    print("=" * 60)
    print()
    print("XWOBA_BLEND_SPLITS_LHP = %s" % round(final_p["XWOBA_SPLITS_LHP"], 4))
    print("XWOBA_BLEND_CAREER_LHP = %s" % round(final_p["XWOBA_CAREER_LHP"], 4))
    print("XWOBA_BLEND_TOTAL_LHP  = %s" % round(final_p["XWOBA_TOTAL_LHP"], 4))
    print("XWOBA_BLEND_RECENT_LHP = %s" % round(final_p["XWOBA_RECENT_LHP"], 4))
    print()
    print("XWOBA_BLEND_SPLITS_RHP = %s" % round(final_p["XWOBA_SPLITS_RHP"], 4))
    print("XWOBA_BLEND_CAREER_RHP = %s" % round(final_p["XWOBA_CAREER_RHP"], 4))
    print("XWOBA_BLEND_TOTAL_RHP  = %s" % round(final_p["XWOBA_TOTAL_RHP"], 4))
    print("XWOBA_BLEND_RECENT_RHP = %s" % round(final_p["XWOBA_RECENT_RHP"], 4))
    print()
    print("--- UNCHANGED (frozen, still v12 production) ---")
    print("LEAGUE_AVG_WOBA_LHP = %s" % LA_LHP)
    print("LEAGUE_AVG_WOBA_RHP = %s" % LA_RHP)
    print("WEIGHT_SPLITS_LHP=%s  WEIGHT_CAREER_LHP=%s  WEIGHT_TOTAL_LHP=%s  WEIGHT_RECENT_LHP=%s" % (WS_L, WC_L, WT_L, WR_L))
    print("REGRESS_PA_SPLITS_LHP=%s  REGRESS_PA_CAREER_LHP=%s  REGRESS_PA_TOTAL_LHP=%s  REGRESS_PA_RECENT_LHP=%s" % (RPS_L, RPC_L, RPT_L, RPR_L))
    print("WEIGHT_SPLITS_RHP=%s  WEIGHT_CAREER_RHP=%s  WEIGHT_TOTAL_RHP=%s  WEIGHT_RECENT_RHP=%s" % (WS_R, WC_R, WT_R, WR_R))
    print("REGRESS_PA_SPLITS_RHP=%s  REGRESS_PA_CAREER_RHP=%s  REGRESS_PA_TOTAL_RHP=%s  REGRESS_PA_RECENT_RHP=%s" % (RPS_R, RPC_R, RPT_R, RPR_R))
    print("DECAY_CAREER_LHP = %s" % DECAY_CAREER_LHP)
    print("DECAY_CAREER_RHP = %s" % DECAY_CAREER_RHP)
    print("PLATOON_BLEND_LHP=%s  PLATOON_FACTOR_MIN_LHP=%s  PLATOON_FACTOR_MAX_LHP=%s" % (PLATOON_BLEND_LHP, PLATOON_MIN_LHP, PLATOON_MAX_LHP))
    print("PLATOON_BLEND_RHP=%s  PLATOON_FACTOR_MIN_RHP=%s  PLATOON_FACTOR_MAX_RHP=%s" % (PLATOON_BLEND_RHP, PLATOON_MIN_RHP, PLATOON_MAX_RHP))
    print()

    elapsed = time.time() - t0
    print("Total elapsed: %.0fs (%.1f min)" % (elapsed, elapsed / 60.0))
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
