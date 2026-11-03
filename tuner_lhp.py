#!/usr/bin/env python3
"""
Tuner v5 - LHP-focused retune, fully isolated from RHP
=====================================================================
Builds on tuner_lhp.py (v4: LHP-only objective, RHP frozen, walk-
forward CV). Five changes from v4:

1. Every knob that used to be SHARED between LHP and RHP is now split
   by hand: XWOBA_BLEND -> XWOBA_BLEND_LHP / XWOBA_BLEND_RHP, and
   PLATOON_BLEND/MIN/MAX -> *_LHP / *_RHP. In v4 these stayed shared
   because "RHP frozen" only froze the RHP-specific weight/regress/
   decay constants -- the shared platoon and xwOBA knobs still got
   tuned for LHP's benefit and that silently changed RHP's predictions
   too (the v4 output even labeled this "side effect of shared
   knobs"). Splitting them means RHP's predictions during this tuner's
   evaluation are now driven ENTIRELY by frozen baseline constants --
   true isolation, matching the original "freeze everything RHP"
   intent. The LHP side's tunable search space is unchanged in size
   (14 params, same as v4) -- this is a relabeling/isolation fix, not
   new dimensions.

2. RHP cross-hand platoon leak fixed. Even with every RHP-labeled
   constant frozen, RHP's platoon factor needs the player's LHP career
   wOBA as its "cross-hand" input -- and that LHP career data was being
   rebuilt every trial using the tunable DECAY_CAREER_LHP, so tuning
   LHP's decay was nudging RHP's predictions too (caught by the smoke
   test: a non-zero RHP delta that should have been exactly 0). Fixed
   by caching a second, frozen copy of LHP career data (built once,
   using START's DECAY_CAREER_LHP) used ONLY as RHP's cross-hand input;
   the tunable copy continues to feed LHP's own predictions. RHP's
   predictions are now bit-for-bit identical to baseline no matter what
   the LHP search does.

3. Window averages are PA-weighted, using sqrt(total post-cutoff PA)
   per window rather than raw PA. Raw PA was considered and rejected:
   earlier-season cutoffs (June 1) always have more season left than
   later ones (Aug 1), so weighting by raw future PA means June windows
   systematically outweigh August windows by about 2x in this data --
   not a precision argument, just an artifact of how much season is
   left at each cutoff. Since the model needs to perform well at every
   point in the season (not just predict best from a June vantage
   point), sqrt damps that skew to about 1.4x while still giving
   larger, more reliable windows some extra say. Per-cutoff average r
   is now also printed (baseline vs tuned, separately for June/July/
   Aug) so a real June-vs-August tradeoff would be visible directly
   instead of hidden inside one blended number.

4. Optimizer restarts: each search (3 CV folds + 1 final fit) now runs
   N_RESTARTS independent random-seed attempts and keeps the best.
   Random search + coordinate ascent can settle into a point that's
   locally unimprovable (coordinate ascent only explores one axis at a
   time from wherever it lands, and random search's cooling schedule
   shifts from exploration to exploitation as it runs) without being
   the best point reachable in the full param space. Restarting from
   different random seeds explores different regions and is a cheap
   way to check whether a meaningfully better optimum is being missed.

RE: RE-ENABLING RHP TUNING -- considered and deliberately NOT done
here. A blended LHP/RHP objective was tried and rejected: weighting
the objective toward LHP only governs which trade-offs get accepted,
it provides no protection against RHP overfitting on its own, because
9 of the 10 RHP-specific parameters have zero effect on the LHP score
-- any RHP-only "improvement" (including a degenerate one, like
REGRESS_PA_CAREER_RHP collapsing to its lower bound, which is exactly
what happened in the original joint 24-param tuner) gets accepted
regardless of the weight. It also dilutes the search: more parameter
names in the pool means each LHP parameter gets perturbed less often
for the same iteration budget. RHP should get its own dedicated
walk-forward-CV tuner later, once LHP is shipped.

TUNES (14 parameters, all LHP-specific now that platoon/xwOBA are
split):
  LEAGUE_AVG_WOBA_LHP                                        (1)
  WEIGHT_{SPLITS,CAREER,TOTAL,RECENT}_LHP                    (4)
  REGRESS_PA_{SPLITS,CAREER,TOTAL,RECENT}_LHP                (4)
  DECAY_CAREER_LHP                                           (1)
  PLATOON_BLEND_LHP, PLATOON_FACTOR_MIN_LHP, PLATOON_FACTOR_MAX_LHP (3)
  XWOBA_BLEND_LHP                                             (1)

FROZEN AT BASELINE (not tunable, by design -- everything RHP-specific,
now including its own platoon and xwOBA copies):
  LEAGUE_AVG_WOBA_RHP, WEIGHT_*_RHP, REGRESS_PA_*_RHP, DECAY_CAREER_RHP,
  PLATOON_BLEND_RHP, PLATOON_FACTOR_MIN_RHP, PLATOON_FACTOR_MAX_RHP,
  XWOBA_BLEND_RHP

PA minimums fixed at 30 (not tunable). NO_CLAMP fixed True.

RUNTIME: RHP career data is cached once per year again (DECAY_CAREER_RHP
is frozen), so the only added cost vs the original v4 is N_RESTARTS
independent searches per fold/final instead of 1. Expect roughly
45-75 minutes total (vs v4's 15-25 min) -- still much longer than
before, but well short of the RHP-enabled version's 1.5-2.5 hours.

USAGE:
  Double-click run_tuner_lhp.bat (writes tuner_lhp_results_YYYY-MM-DD.txt)
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
OUTPUT_FILE = os.path.join(FOLDER, "tuner_lhp_results_" + today + ".txt")


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

# Walk-forward (rolling-origin) CV folds. Each fold's test year is
# strictly after every year in its training set.
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
# RHP career data never changes (DECAY_CAREER_RHP is frozen), so it's
# safe to precompute once per year and reuse across every trial --
# saves real time over the search since we don't rebuild it per trial.

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
# SCORING MATH (mirrors backtest.py / tuner.py exactly)
# -------------------------------------------------------

def regress(woba, pa, regress_pa, la):
    return (woba * pa + la * regress_pa) / (pa + regress_pa)


def stabilize(p, splits_pa, splits_woba, career_pa, career_woba,
              total_pa, total_woba, recent_pa, recent_woba, side,
              career_cross_pa=0, career_cross_woba=None):
    if side == 'L':
        la, WS, WC, WT, WR = p["LA_LHP"], p["WS_L"], p["WC_L"], p["WT_L"], p["WR_L"]
        RPS, RPC, RPT, RPR = p["RPS_L"], p["RPC_L"], p["RPT_L"], p["RPR_L"]
        pb, pmin, pmax = p["PLATOON_BLEND_LHP"], p["PLATOON_MIN_LHP"], p["PLATOON_MAX_LHP"]
    else:
        la, WS, WC, WT, WR = p["LA_RHP"], p["WS_R"], p["WC_R"], p["WT_R"], p["WR_R"]
        RPS, RPC, RPT, RPR = p["RPS_R"], p["RPC_R"], p["RPT_R"], p["RPR_R"]
        pb, pmin, pmax = p["PLATOON_BLEND_RHP"], p["PLATOON_MIN_RHP"], p["PLATOON_MAX_RHP"]

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

def run_window(p, year, co, car_l, car_r, car_l_base):
    """Returns (r_l, r_r, pa_l, pa_r): the LHP/RHP Spearman r for this
    window, plus the total post-cutoff PA backing each side's r (used
    by run_all to PA-weight windows instead of averaging them equally).

    car_l is the TUNABLE LHP career data (rebuilt per trial using the
    candidate's DECAY_CAREER_LHP) -- used for LHP's own prediction.
    car_l_base is a FROZEN copy (always built with START's
    DECAY_CAREER_LHP) used only as the cross-hand input to RHP's
    platoon factor, so RHP's predictions never move when the LHP
    search perturbs DECAY_CAREER_LHP."""
    wc = WINDOW_CACHE[(year, co)]
    pre_l, pre_r = wc["pre_l"], wc["pre_r"]
    total, recent = wc["total"], wc["recent"]
    post_l, post_r = wc["post_l"], wc["post_r"]
    xb_l = p["XWOBA_BLEND_LHP"]
    xb_r = p["XWOBA_BLEND_RHP"]

    all_pids = set(pre_l.keys()) | set(pre_r.keys())
    pred_l, act_l = [], []
    pred_r, act_r = [], []
    pa_l_sum, pa_r_sum = 0, 0

    for pid in all_pids:
        sl, sr = pre_l.get(pid), pre_r.get(pid)
        cl, cr = car_l.get(pid), car_r.get(pid)
        cl_base = car_l_base.get(pid)
        tot, rec = total.get(pid), recent.get(pid)

        sl_pa = sl["pa"] if sl else 0
        sr_pa = sr["pa"] if sr else 0
        cl_pa = cl["pa"] if cl else 0
        cr_pa = cr["pa"] if cr else 0
        cl_base_pa = cl_base["pa"] if cl_base else 0
        tot_pa = tot["pa"] if tot else 0
        rec_pa = rec["pa"] if rec else 0

        # Each side blends actual/xwOBA with its own (now-frozen-for-RHP)
        # knob. The same underlying total/recent/cross-hand-career rows
        # get blended twice (once per side) since XWOBA_BLEND_LHP and
        # XWOBA_BLEND_RHP can differ.
        sl_w = blend_woba(sl, xb_l)
        cl_w_l = blend_woba(cl, xb_l)
        cr_w_l = blend_woba(cr, xb_l)
        tot_w_l = blend_woba(tot, xb_l)
        rec_w_l = blend_woba(rec, xb_l)

        sr_w = blend_woba(sr, xb_r)
        cr_w_r = blend_woba(cr, xb_r)
        cl_w_r = blend_woba(cl_base, xb_r)  # frozen cross-hand input -- see docstring
        tot_w_r = blend_woba(tot, xb_r)
        rec_w_r = blend_woba(rec, xb_r)

        has_split_l = (sl_pa >= MIN_PA_SPLITS) or (cl_pa >= MIN_PA_CAREER)
        if has_split_l:
            post = post_l.get(pid)
            if post is not None and post["pa"] >= MIN_PA_POST:
                pred = stabilize(p, sl_pa, sl_w, cl_pa, cl_w_l, tot_pa, tot_w_l,
                                  rec_pa, rec_w_l, 'L',
                                  career_cross_pa=cr_pa, career_cross_woba=cr_w_l)
                if pred is not None:
                    pred_l.append(pred); act_l.append(post["woba"]); pa_l_sum += post["pa"]

        has_split_r = (sr_pa >= MIN_PA_SPLITS) or (cr_pa >= MIN_PA_CAREER)
        if has_split_r:
            post = post_r.get(pid)
            if post is not None and post["pa"] >= MIN_PA_POST:
                pred = stabilize(p, sr_pa, sr_w, cr_pa, cr_w_r, tot_pa, tot_w_r,
                                  rec_pa, rec_w_r, 'R',
                                  career_cross_pa=cl_base_pa, career_cross_woba=cl_w_r)
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


def run_all(p, years, car_r_cache, car_l_base_cache):
    """RHP career data (car_r_cache) and the frozen LHP cross-hand
    career data (car_l_base_cache) are passed in pre-built since
    neither ever changes across trials; only the TUNABLE LHP career
    data needs rebuilding when DECAY_CAREER_LHP is perturbed. Window-
    level Spearman r values are combined into weighted averages, using
    sqrt(total post-cutoff PA) as the weight -- a window backed by more
    playing time counts for more, but raw PA was rejected because it
    would let June windows (more season left) outweigh August windows
    (less season left) by ~2x for reasons having nothing to do with
    estimate reliability; sqrt damps that to ~1.4x."""
    all_rs, all_ws = [], []
    lhp_rs, lhp_ws = [], []
    rhp_rs, rhp_ws = [], []
    for year in years:
        car_l = build_career_data(year, "left", p["DECAY_CAREER_LHP"])
        car_r = car_r_cache[year]
        car_l_base = car_l_base_cache[year]
        for co in CUTOFFS:
            r_l, r_r, pa_l, pa_r = run_window(p, year, co, car_l, car_r, car_l_base)
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


def per_cutoff_lhp_avg(p, years, car_r_cache, car_l_base_cache):
    """Diagnostic only -- NOT used in the objective. Plain (unweighted)
    average LHP r per cutoff, so a real June-vs-August tradeoff from
    the sqrt-PA weighting would show up directly instead of being
    hidden inside one blended number."""
    by_cutoff = {co: [] for co in CUTOFFS}
    for year in years:
        car_l = build_career_data(year, "left", p["DECAY_CAREER_LHP"])
        car_r = car_r_cache[year]
        car_l_base = car_l_base_cache[year]
        for co in CUTOFFS:
            r_l, r_r, pa_l, pa_r = run_window(p, year, co, car_l, car_r, car_l_base)
            if r_l is not None:
                by_cutoff[co].append(r_l)
    return {co: (sum(v) / len(v) if v else None) for co, v in by_cutoff.items()}


# -------------------------------------------------------
# PARAMETER SPACE
# -------------------------------------------------------

BOUNDS = {
    "LA_LHP": (0.20, 0.34, 0.01),
    "WS_L": (0.0, 1.5, 0.05), "WC_L": (0.0, 1.5, 0.05),
    "WT_L": (0.0, 1.5, 0.05), "WR_L": (0.0, 1.5, 0.05),
    "RPS_L": (1, 2500, 40), "RPC_L": (1, 2500, 40),
    "RPT_L": (1, 2500, 40), "RPR_L": (1, 2500, 40),
    "DECAY_CAREER_LHP": (0.40, 1.0, 0.03),

    "PLATOON_BLEND_LHP": (0.0, 1.0, 0.04),
    "PLATOON_MIN_LHP":   (0.30, 0.99, 0.03),
    "PLATOON_MAX_LHP":   (1.01, 2.20, 0.04),
    "XWOBA_BLEND_LHP":   (0.0, 1.0, 0.05),
}

PARAM_NAMES = list(BOUNDS.keys())  # 14 tunable; RHP-specific NOT included -> frozen

# Baseline = current backtest.py on-disk values (as of 2026-06-16).
# RHP-specific entries here are FROZEN -- they stay at these values for
# the entire search, never perturbed (not in PARAM_NAMES). Platoon and
# xwOBA used to be single shared values in backtest.py; split here into
# LHP/RHP copies that both start from that same original shared value,
# so the RHP copy is a faithful "what backtest.py does today" baseline.
START = {
    "LA_LHP": 0.2585, "LA_RHP": 0.2575,
    "WS_L": 0.1365, "WC_L": 0.6142, "WT_L": 0.1443, "WR_L": 0.1050,
    "RPS_L": 355, "RPC_L": 735, "RPT_L": 910, "RPR_L": 795,
    "WS_R": 0.2774, "WC_R": 0.2920, "WT_R": 0.2153, "WR_R": 0.2153,
    "RPS_R": 285, "RPC_R": 25, "RPT_R": 400, "RPR_R": 665,
    "PLATOON_BLEND_LHP": 0.31, "PLATOON_MIN_LHP": 0.585, "PLATOON_MAX_LHP": 1.325,
    "PLATOON_BLEND_RHP": 0.31, "PLATOON_MIN_RHP": 0.585, "PLATOON_MAX_RHP": 1.325,
    "DECAY_CAREER_LHP": 1.0, "DECAY_CAREER_RHP": 1.0,
    "XWOBA_BLEND_LHP": 0.5, "XWOBA_BLEND_RHP": 0.5,
}

FROZEN_RHP_KEYS = ["LA_RHP", "WS_R", "WC_R", "WT_R", "WR_R",
                    "RPS_R", "RPC_R", "RPT_R", "RPR_R", "DECAY_CAREER_RHP",
                    "PLATOON_BLEND_RHP", "PLATOON_MIN_RHP", "PLATOON_MAX_RHP",
                    "XWOBA_BLEND_RHP"]


def clamp(name, val):
    lo, hi, _ = BOUNDS[name]
    val = max(lo, min(hi, val))
    if name.startswith("RP"):
        val = round(val)
    return val


def fix_platoon_order(p):
    if p["PLATOON_MIN_LHP"] >= 1.0:
        p["PLATOON_MIN_LHP"] = 0.99
    if p["PLATOON_MAX_LHP"] <= 1.0:
        p["PLATOON_MAX_LHP"] = 1.01


# -------------------------------------------------------
# OPTIMIZER (objective = LHP-only train r, scoped to a given year list)
# -------------------------------------------------------

def make_score_fn(years, car_r_cache, car_l_base_cache):
    def score_train_lhp(p):
        avg, lhp_avg, rhp_avg = run_all(p, years, car_r_cache, car_l_base_cache)
        return lhp_avg if lhp_avg is not None else -1.0
    return score_train_lhp


def random_search(p0, score0, n_iters, rng, score_fn, log_every=250, start_time=None):
    best = dict(p0)
    best_score = score0
    n_accept = 0
    for i in range(1, n_iters + 1):
        cand = dict(best)
        frac_done = i / n_iters
        temp = 1.0 - 0.85 * frac_done
        k = rng.choice([1, 1, 1, 2, 2, 3, 5])
        names = rng.sample(PARAM_NAMES, k=min(k, len(PARAM_NAMES)))
        for name in names:
            lo, hi, step = BOUNDS[name]
            delta = rng.gauss(0, step * temp)
            cand[name] = clamp(name, cand[name] + delta)
        fix_platoon_order(cand)

        s = score_fn(cand)
        if s > best_score:
            best_score = s
            best = cand
            n_accept += 1

        if log_every and i % log_every == 0:
            elapsed = time.time() - start_time if start_time else 0
            print("    [random %5d/%d] best_lhp_r=%.4f  accepts=%d  elapsed=%.0fs" %
                  (i, n_iters, best_score, n_accept, elapsed))
            sys.stdout.flush()
    return best, best_score


def coordinate_sweep(p0, score0, rng, score_fn, max_passes=4):
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
                s = score_fn(cand)
                if s > best_score:
                    best_score = s
                    best = cand
                    improved_this_pass = True
        print("    [coord pass %d] best_lhp_r=%.4f%s" %
              (pass_i + 1, best_score, "  (improved)" if improved_this_pass else "  (no change)"))
        sys.stdout.flush()
        if not improved_this_pass:
            break
    return best, best_score


# -------------------------------------------------------
# ONE FULL SEARCH (random -> coord -> random -> coord), with restarts
# -------------------------------------------------------

N_RANDOM_ITERS_PHASE1 = 1500
N_RANDOM_ITERS_PHASE2 = 1500
SEED = 20260617
N_RESTARTS = 3  # independent random-seed attempts per search; keeps the
                # best of N as a cheap check against local optima.


def run_search(years, car_r_cache, car_l_base_cache, seed, label):
    """Runs the full 4-phase search with the LHP-only objective scoped
    to `years`. Returns (tuned_params, tuned_train_score)."""
    rng = random.Random(seed)
    score_fn = make_score_fn(years, car_r_cache, car_l_base_cache)
    cur = dict(START)
    cur_score = score_fn(cur)
    print("[%s] years=%s  starting LHP-only train r: %.4f" % (label, years, cur_score))
    sys.stdout.flush()

    print("  --- Phase 1: random search (%d iters) ---" % N_RANDOM_ITERS_PHASE1)
    cur, cur_score = random_search(cur, cur_score, N_RANDOM_ITERS_PHASE1, rng, score_fn, start_time=time.time())

    print("  --- Phase 2: coordinate-ascent polish ---")
    cur, cur_score = coordinate_sweep(cur, cur_score, rng, score_fn, max_passes=4)

    print("  --- Phase 3: random search round 2 (%d iters, fine-grained) ---" % N_RANDOM_ITERS_PHASE2)
    cur, cur_score = random_search(cur, cur_score, N_RANDOM_ITERS_PHASE2, rng, score_fn, start_time=time.time())

    print("  --- Phase 4: final coordinate-ascent polish ---")
    cur, cur_score = coordinate_sweep(cur, cur_score, rng, score_fn, max_passes=4)
    print("[%s] final LHP-only train r: %.4f" % (label, cur_score))
    print()
    sys.stdout.flush()
    return cur, cur_score


def run_search_with_restarts(years, car_r_cache, car_l_base_cache, base_seed, label, n_restarts=N_RESTARTS):
    """Runs run_search n_restarts times from different seeds and keeps
    the best. Random search + coordinate ascent can settle into a
    point that's locally unimprovable but not the best point reachable
    in the full param space; independent restarts from different
    random streams probe different regions and are a cheap check for a
    meaningfully better optimum being missed."""
    best_p, best_score = None, None
    for r in range(n_restarts):
        seed = base_seed + r * 100000
        sub_label = "%s restart %d/%d" % (label, r + 1, n_restarts)
        p, score = run_search(years, car_r_cache, car_l_base_cache, seed, sub_label)
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

    print("=== TUNER v5 -- LHP-focused retune, fully isolated from RHP ===")
    print("Results saved to: " + OUTPUT_FILE)
    print("CV folds (rolling-origin; test year is never in that fold's training years):")
    for i, f in enumerate(CV_FOLDS, 1):
        print("  Fold %d: train=%s  test=%d" % (i, f["train"], f["test"]))
    print("Final fit: train=%s  (no holdout -- production candidate)" % ALL_YEARS)
    print("%d tunable params per run (all LHP-specific; platoon & xwOBA now split by hand)." % len(PARAM_NAMES))
    print("RHP-specific params FROZEN at baseline: " + ", ".join(FROZEN_RHP_KEYS))
    print("Each search runs %d independent random-seed restarts, keeping the best." % N_RESTARTS)
    print("Window averages are weighted by sqrt(total post-cutoff PA).")
    print("Expect roughly 45-75 minutes total (4 searches x %d restarts each)." % N_RESTARTS)
    print()

    # RHP career data never changes (decay frozen) -- precompute once per year.
    # Same for the FROZEN copy of LHP career data used only as RHP's
    # cross-hand platoon input (see run_window docstring) -- it always
    # uses START's DECAY_CAREER_LHP, never the trial's tunable value.
    car_r_cache = {}
    car_l_base_cache = {}
    for year in TEST_YEARS:
        car_r_cache[year] = build_career_data(year, "right", START["DECAY_CAREER_RHP"])
        car_l_base_cache[year] = build_career_data(year, "left", START["DECAY_CAREER_LHP"])

    def full_eval(p, years):
        return run_all(p, years, car_r_cache, car_l_base_cache)

    def cutoff_eval(p, years):
        return per_cutoff_lhp_avg(p, years, car_r_cache, car_l_base_cache)

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

        tuned_p, tuned_train_score = run_search_with_restarts(train_years, car_r_cache, car_l_base_cache, SEED + i, label)

        tuned_train_all, tuned_train_lhp, tuned_train_rhp = full_eval(tuned_p, train_years)
        tuned_test_all, tuned_test_lhp, tuned_test_rhp = full_eval(tuned_p, [test_year])
        print("  TUNED   train (lhp/rhp/all): %.4f / %.4f / %.4f" % (tuned_train_lhp, tuned_train_rhp, tuned_train_all))
        print("  TUNED   test  (lhp/rhp/all): %.4f / %.4f / %.4f" % (tuned_test_lhp, tuned_test_rhp, tuned_test_all))
        print("  LHP test delta vs baseline: %+.4f" % (tuned_test_lhp - base_test_lhp))
        print("  RHP test delta vs baseline: %+.6f  (should be EXACTLY 0 now -- RHP fully frozen/isolated)" % (tuned_test_rhp - base_test_rhp))

        base_cutoffs = cutoff_eval(START, [test_year])
        tuned_cutoffs = cutoff_eval(tuned_p, [test_year])
        print("  Per-cutoff LHP test r (baseline -> tuned), watch for June gains at August's expense:")
        for co in CUTOFFS:
            bv, tv = base_cutoffs[co], tuned_cutoffs[co]
            if bv is not None and tv is not None:
                print("    %-8s  %.4f -> %.4f  (%+.4f)" % (co, bv, tv, tv - bv))
        print()

        fold_results.append({
            "fold": i, "train_years": train_years, "test_year": test_year, "params": tuned_p,
            "base_test_lhp": base_test_lhp, "base_test_rhp": base_test_rhp, "base_test_all": base_test_all,
            "tuned_test_lhp": tuned_test_lhp, "tuned_test_rhp": tuned_test_rhp, "tuned_test_all": tuned_test_all,
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
    print("If the LHP delta is positive and similar in size across all 3 rows,")
    print("the LHP retune generalizes across different out-of-sample years --")
    print("not just an artifact of one holdout split. The RHP delta should be")
    print("EXACTLY 0 in every row now (platoon/xwOBA are split AND the cross-")
    print("hand platoon input is frozen, so nothing tunable here touches RHP's")
    print("predictions at all).")
    print()

    # --- Per-fold tuned params (stability check) ---
    print("=" * 60)
    print("PER-FOLD TUNED PARAMETERS (compare across folds for stability --")
    print("big swings between folds would be a red flag, even if all 3 test")
    print("deltas above are positive)")
    print("=" * 60)
    for r in fold_results:
        print()
        print("Fold %d (train=%s, test=%d):" % (r["fold"], r["train_years"], r["test_year"]))
        for name in PARAM_NAMES:
            print("  %-18s = %s" % (name, round(r["params"][name], 4)))
    print()

    # --- Final production fit on ALL years ---
    print("=" * 60)
    print("FINAL PRODUCTION FIT (trained on all %d years, no held-out year)" % len(ALL_YEARS))
    print("=" * 60)
    print("Uses every available year for max data. Trust in these values comes")
    print("from the CV evidence above, not a fresh holdout -- there isn't one")
    print("left, since all years are now in the training set.")
    print()
    final_p, final_train_score = run_search_with_restarts(ALL_YEARS, car_r_cache, car_l_base_cache, SEED + 99, "Final")
    final_all, final_lhp, final_rhp = full_eval(final_p, ALL_YEARS)
    print("  Final fit in-sample (lhp/rhp/all): %.4f / %.4f / %.4f" % (final_lhp, final_rhp, final_all))
    base_cutoffs = cutoff_eval(START, ALL_YEARS)
    final_cutoffs = cutoff_eval(final_p, ALL_YEARS)
    print("  Per-cutoff LHP in-sample r (baseline -> final):")
    for co in CUTOFFS:
        bv, tv = base_cutoffs[co], final_cutoffs[co]
        if bv is not None and tv is not None:
            print("    %-8s  %.4f -> %.4f  (%+.4f)" % (co, bv, tv, tv - bv))
    print()
    print("=" * 60)
    print("TUNED PARAMETERS -- paste these into backtest.py (LHP-specific only)")
    print("=" * 60)
    print()
    print("LEAGUE_AVG_WOBA_LHP = %s" % round(final_p["LA_LHP"], 4))
    print()
    print("WEIGHT_SPLITS_LHP  = %s" % round(final_p["WS_L"], 4))
    print("WEIGHT_CAREER_LHP  = %s" % round(final_p["WC_L"], 4))
    print("WEIGHT_TOTAL_LHP   = %s" % round(final_p["WT_L"], 4))
    print("WEIGHT_RECENT_LHP  = %s" % round(final_p["WR_L"], 4))
    print()
    print("REGRESS_PA_SPLITS_LHP = %s" % int(final_p["RPS_L"]))
    print("REGRESS_PA_CAREER_LHP = %s" % int(final_p["RPC_L"]))
    print("REGRESS_PA_TOTAL_LHP  = %s" % int(final_p["RPT_L"]))
    print("REGRESS_PA_RECENT_LHP = %s" % int(final_p["RPR_L"]))
    print()
    print("DECAY_CAREER_LHP = %s" % round(final_p["DECAY_CAREER_LHP"], 4))
    print()
    print("PLATOON_BLEND_LHP      = %s" % round(final_p["PLATOON_BLEND_LHP"], 4))
    print("PLATOON_FACTOR_MIN_LHP = %s" % round(final_p["PLATOON_MIN_LHP"], 4))
    print("PLATOON_FACTOR_MAX_LHP = %s" % round(final_p["PLATOON_MAX_LHP"], 4))
    print()
    print("XWOBA_BLEND_LHP = %s" % round(final_p["XWOBA_BLEND_LHP"], 4))
    print()
    print("--- UNCHANGED (frozen RHP-specific, still baseline) ---")
    print("LEAGUE_AVG_WOBA_RHP = %s" % START["LA_RHP"])
    print("WEIGHT_SPLITS_RHP  = %s" % START["WS_R"])
    print("WEIGHT_CAREER_RHP  = %s" % START["WC_R"])
    print("WEIGHT_TOTAL_RHP   = %s" % START["WT_R"])
    print("WEIGHT_RECENT_RHP  = %s" % START["WR_R"])
    print("REGRESS_PA_SPLITS_RHP = %s" % START["RPS_R"])
    print("REGRESS_PA_CAREER_RHP = %s" % START["RPC_R"])
    print("REGRESS_PA_TOTAL_RHP  = %s" % START["RPT_R"])
    print("REGRESS_PA_RECENT_RHP = %s" % START["RPR_R"])
    print("DECAY_CAREER_RHP = %s" % START["DECAY_CAREER_RHP"])
    print("PLATOON_BLEND_RHP      = %s" % START["PLATOON_BLEND_RHP"])
    print("PLATOON_FACTOR_MIN_RHP = %s" % START["PLATOON_MIN_RHP"])
    print("PLATOON_FACTOR_MAX_RHP = %s" % START["PLATOON_MAX_RHP"])
    print("XWOBA_BLEND_RHP = %s" % START["XWOBA_BLEND_RHP"])
    print()
    print("Note: backtest.py currently has ONE shared PLATOON_BLEND/MIN/MAX")
    print("(and previously one shared XWOBA_BLEND) used for both hands. To")
    print("actually ship the *_LHP values above as distinct from RHP, backtest.py")
    print("needs a small change to look up these constants per side, the same")
    print("way it already does for LEAGUE_AVG_WOBA/WEIGHT_*/REGRESS_PA_*.")
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
