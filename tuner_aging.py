#!/usr/bin/env python3
"""
Tuner - aging curve strength (v13, Task #79)
=============================================
backtest.py v13 added an additive, per-prior-season age correction to the
CAREER component only (build_career_data()): each prior season's woba/xwoba
is shifted by AGING_STRENGTH * aging_delta(age that season, age in
test_year), using a delta-method curve estimated from 2,446 adjacent-season
player pairs (build_aging_curve.py). That wiring was verified bit-for-bit
against the pre-v13 production r values (LHP 0.3299, RHP 0.4113, all 0.3706)
at the neutral placeholder AGING_STRENGTH=0.0. This file searches for the
single AGING_STRENGTH value that generalizes best out-of-sample.

WHY A 1-D GRID SEARCH INSTEAD OF tuner_xwoba_components.py'S RANDOM-SEARCH/
COORDINATE-ASCENT MACHINERY:
There is exactly one tunable parameter here, shared by both hands (the
curve itself isn't hand-specific -- it was estimated from full-season data
pooled across all PA, not split by opposing pitcher hand). A 1-D objective
in a bounded range is cheap and safe to search exhaustively with a grid;
the multi-restart random-search + coordinate-ascent machinery in
tuner_lhp.py/tuner_rhp.py/tuner_xwoba_components.py exists to handle 8-14
dimensions where local optima are a real risk. That risk doesn't really
apply to a single dimension, so a coarse grid (step 0.1) followed by a
fine grid around the coarse optimum (step 0.01) is simpler, fully
deterministic (no RNG, no restarts needed), and just as reliable here.

WHY CAREER DATA CANNOT BE CACHED ACROSS TRIALS (unlike tuner_xwoba_
components.py, where DECAY_CAREER_LHP/RHP were both frozen so career data
never changed and was built once per year):
AGING_STRENGTH directly changes build_career_data()'s output -- it shifts
the per-season woba/xwoba BEFORE the PA/decay-weighted average is taken.
So career data must be rebuilt for every candidate value tried. The raw
per-season files themselves (SEASON_CACHE) and the raw per-window files
(WINDOW_CACHE) don't depend on AGING_STRENGTH at all, so those are still
cached once up front -- only the build_career_data() call itself re-runs
per candidate (cheap: it's an in-memory loop over already-loaded season
dicts, no disk I/O per trial).

OBJECTIVE: a single combined score (LHP+RHP pooled), computed as the
SIMPLE (unweighted) mean of every window's Spearman r -- this is exactly
backtest.py's own "Overall average r" formula. Verified directly: taking
an unweighted mean of the 36 production window r-values (18 windows x
LHP/RHP) reproduces KNOWN_BASELINE_ALL/LHP/RHP (0.3706/0.3299/0.4113)
bit-for-bit, while a sqrt(PA)-weighted mean of the same values does not
(0.3742/0.3309/0.4115 -- close, but not an exact match). Earlier tuners in
this project (tuner_xwoba_components.py etc.) used sqrt(PA)-weighting as
their search objective, but that convention doesn't carry over here: this
tuner's canary requires bit-for-bit reproduction of backtest.py's literal
printed metric, and searching on a differently-weighted proxy would risk
shipping a value that doesn't actually optimize what backtest.py reports.
LHP-only and RHP-only averages (also simple means) are still reported
alongside the combined number for diagnostic purposes.

TUNES (1 parameter): AGING_STRENGTH

FROZEN AT v13 PRODUCTION BASELINE (not tunable -- everything else):
  LEAGUE_AVG_WOBA_LHP=0.2943, LEAGUE_AVG_WOBA_RHP=0.2575
  WEIGHT_*_LHP / WEIGHT_*_RHP (8 weights), REGRESS_PA_*_LHP/_RHP (8 values)
  DECAY_CAREER_LHP=0.8001, DECAY_CAREER_RHP=1.0
  PLATOON_BLEND_LHP=0.7833 (clamp [0.612, 1.138])
  PLATOON_BLEND_RHP=0.31   (clamp [0.585, 1.325])
  XWOBA_BLEND_{SPLITS,CAREER,TOTAL,RECENT}_LHP = 0.9187 (all 4)
  XWOBA_BLEND_{SPLITS,CAREER,TOTAL,RECENT}_RHP = 0.80   (all 4)
  AGING_CURVE_CUM (the estimated curve itself, ages 21-36), aging_cum()'s
  edge-clamping behavior outside that range -- not tunable, only the
  STRENGTH multiplier applied on top of it is.
PA minimums fixed at 30 (splits/career/total/post). NO_CLAMP fixed True.

GRID: coarse pass over [-1.0, 2.0] step 0.1 (31 points, includes 0.0 and
1.0 -- "no adjustment" and "apply the curve exactly as estimated"), then a
fine pass over [best-0.1, best+0.1] step 0.01 (21 points) around whichever
coarse point scored best.

USAGE:
  Double-click run_tuner_aging.bat (writes tuner_aging_results_YYYY-MM-DD.txt)
"""
import os
import csv
import sys
import datetime
import time

FOLDER   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FOLDER, "historical testing")
today    = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(FOLDER, "tuner_aging_results_" + today + ".txt")


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
# FIXED CONSTANTS (not tunable -- everything except AGING_STRENGTH,
# frozen at v13 production values)
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

XWOBA_SPLITS_LHP, XWOBA_CAREER_LHP, XWOBA_TOTAL_LHP, XWOBA_RECENT_LHP = 0.9187, 0.9187, 0.9187, 0.9187
XWOBA_SPLITS_RHP, XWOBA_CAREER_RHP, XWOBA_TOTAL_RHP, XWOBA_RECENT_RHP = 0.80, 0.80, 0.80, 0.80

TEST_YEARS = [2019, 2021, 2022, 2023, 2024, 2025]
CUTOFFS    = ["june_1", "july_1", "aug_1"]
TEST_WINDOWS = [{"year": y, "cutoff": c} for y in TEST_YEARS for c in CUTOFFS]

# Walk-forward (rolling-origin) CV folds -- same folds as every other
# tuner in this project, for direct comparability.
CV_FOLDS = [
    {"train": [2019, 2021, 2022],             "test": 2023},
    {"train": [2019, 2021, 2022, 2023],       "test": 2024},
    {"train": [2019, 2021, 2022, 2023, 2024], "test": 2025},
]
ALL_YEARS = [2019, 2021, 2022, 2023, 2024, 2025]  # final production fit, no holdout

ALL_SEASON_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]

# Known v12/v13-baseline production r values (AGING_STRENGTH=0.0), from
# backtest.py's own docstring and the last verified production run.
# Used below as a canary: this tuner's scoring math must reproduce these
# exactly before any search results can be trusted.
KNOWN_BASELINE_ALL = 0.3706
KNOWN_BASELINE_LHP = 0.3299
KNOWN_BASELINE_RHP = 0.4113

# -------------------------------------------------------
# AGING CURVE (frozen -- same table/clamp logic as backtest.py v13;
# only the STRENGTH multiplier applied on top of this is tunable)
# -------------------------------------------------------
AGING_CURVE_CUM = {
    21: 0.0046, 22: 0.0040, 23: 0.0056, 24: 0.0062, 25: 0.0041, 26: 0.0007,
    27: -0.0065, 28: -0.0120, 29: -0.0171, 30: -0.0308, 31: -0.0409,
    32: -0.0494, 33: -0.0641, 34: -0.0714, 35: -0.0854, 36: -0.1010,
}
AGING_CURVE_MIN_AGE = min(AGING_CURVE_CUM)
AGING_CURVE_MAX_AGE = max(AGING_CURVE_CUM)


def aging_cum(age):
    clamped = max(AGING_CURVE_MIN_AGE, min(AGING_CURVE_MAX_AGE, age))
    return AGING_CURVE_CUM[clamped]


def aging_delta(age_from, age_to):
    return aging_cum(age_to) - aging_cum(age_from)


def load_player_ages():
    path = os.path.join(FOLDER, "player_ages.csv")
    result = {}
    if not os.path.exists(path):
        print("  WARNING: player_ages.csv not found -- aging adjustment will be a no-op for every player.")
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pid = str(int(float(row["mlbam_id"])))
            except (ValueError, KeyError, TypeError):
                continue
            ages = {}
            for col, val in row.items():
                if not col.startswith("age_"):
                    continue
                try:
                    season = int(col[len("age_"):])
                except ValueError:
                    continue
                v = str(val).strip()
                if not v or v.lower() in ("nan", "none"):
                    continue
                try:
                    ages[season] = int(float(v))
                except ValueError:
                    continue
            result[pid] = ages
    return result


PLAYER_AGES = load_player_ages()

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


# Season files, cached by (hand, year). Independent of AGING_STRENGTH --
# safe to load once and reuse across every trial.
SEASON_CACHE = {}
for hand in ("left", "right"):
    for y in ALL_SEASON_YEARS:
        SEASON_CACHE[(hand, y)] = load_file("season_" + str(y) + "_vs_" + hand + ".csv")

# Per-window files, cached by (year, cutoff). Also independent of
# AGING_STRENGTH -- only the career inputs change per trial, not these.
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
# CAREER DATA -- NOT cacheable across trials (see module docstring).
# Rebuilt fresh for every (test_year, hand, aging_strength) combination.
# -------------------------------------------------------

def build_career_data(test_year, hand, decay, aging_strength):
    prior_years = [y for y in ALL_SEASON_YEARS if y < test_year]
    num, den = {}, {}
    xnum, xden = {}, {}
    pa_sum = {}
    for y in prior_years:
        season = SEASON_CACHE[(hand, y)]
        w = decay ** (test_year - y)
        for pid, d in season.items():
            delta = 0.0
            if aging_strength != 0.0:
                player_ages = PLAYER_AGES.get(pid)
                if player_ages:
                    age_y = player_ages.get(y)
                    age_t = player_ages.get(test_year)
                    if age_y is not None and age_t is not None:
                        delta = aging_strength * aging_delta(age_y, age_t)
            adj_woba = d["woba"] + delta
            wpa = w * d["pa"]
            num[pid] = num.get(pid, 0.0) + wpa * adj_woba
            den[pid] = den.get(pid, 0.0) + wpa
            pa_sum[pid] = pa_sum.get(pid, 0) + d["pa"]
            if d.get("xwoba") is not None:
                adj_xwoba = d["xwoba"] + delta
                xnum[pid] = xnum.get(pid, 0.0) + wpa * adj_xwoba
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
# SCORING MATH (mirrors backtest.py v13 exactly). AGING_STRENGTH is the
# only thing that varies between calls -- it only ever affects which
# car_l/car_r dict gets passed in, never anything inside stabilize().
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
    # backtest.py rounds this to 4dp before ranking -- must match exactly
    # so tie-breaking in rank_array() lines up bit-for-bit.
    return round(sum(c * (w / total_w) for c, w in zip(components, weights)), 4)


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
    return round(num / (denx * deny), 4)


# -------------------------------------------------------
# WINDOW EVALUATION
# -------------------------------------------------------

def run_window(year, co, car_l, car_r):
    """Returns (r_l, r_r, pa_l, pa_r). car_l/car_r are this trial's
    aging-adjusted career dicts for `year` (rebuilt by the caller for
    the candidate AGING_STRENGTH being scored)."""
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

        sl_w   = blend_woba(sl,  XWOBA_SPLITS_LHP)
        cl_w_l = blend_woba(cl,  XWOBA_CAREER_LHP)
        cr_w_l = blend_woba(cr,  XWOBA_CAREER_LHP)
        tot_w_l = blend_woba(tot, XWOBA_TOTAL_LHP)
        rec_w_l = blend_woba(rec, XWOBA_RECENT_LHP)

        sr_w   = blend_woba(sr,  XWOBA_SPLITS_RHP)
        cr_w_r = blend_woba(cr,  XWOBA_CAREER_RHP)
        cl_w_r = blend_woba(cl,  XWOBA_CAREER_RHP)
        tot_w_r = blend_woba(tot, XWOBA_TOTAL_RHP)
        rec_w_r = blend_woba(rec, XWOBA_RECENT_RHP)

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


# Per-(year, aging_strength) career-data memo, cleared between distinct
# grid searches via a plain dict keyed on (year, aging_strength) -- a
# single search tries the same candidate strength across every cutoff in
# a year 3 times (once per cutoff), so this still saves 2/3 of the
# rebuild work within one run_all() call, even though nothing survives
# across different candidate values (by design -- see module docstring).
def run_all(aging_strength, years):
    """Window-level Spearman r values combined into a SIMPLE (unweighted)
    mean across every window/hand -- exactly backtest.py's own "Overall
    average r" formula (see module docstring OBJECTIVE section for the
    verification that motivated this, instead of sqrt(PA)-weighting)."""
    car_cache = {}
    all_rs, lhp_rs, rhp_rs = [], [], []
    for year in years:
        if year not in car_cache:
            car_l = build_career_data(year, "left",  DECAY_CAREER_LHP, aging_strength)
            car_r = build_career_data(year, "right", DECAY_CAREER_RHP, aging_strength)
            car_cache[year] = (car_l, car_r)
        car_l, car_r = car_cache[year]
        for co in CUTOFFS:
            r_l, r_r, pa_l, pa_r = run_window(year, co, car_l, car_r)
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


def per_cutoff_avg(aging_strength, years, side):
    """Diagnostic only -- NOT used in the objective."""
    by_cutoff = {co: [] for co in CUTOFFS}
    car_cache = {}
    for year in years:
        if year not in car_cache:
            car_l = build_career_data(year, "left",  DECAY_CAREER_LHP, aging_strength)
            car_r = build_career_data(year, "right", DECAY_CAREER_RHP, aging_strength)
            car_cache[year] = (car_l, car_r)
        car_l, car_r = car_cache[year]
        for co in CUTOFFS:
            r_l, r_r, _, _ = run_window(year, co, car_l, car_r)
            v = r_l if side == 'L' else r_r
            if v is not None:
                by_cutoff[co].append(v)
    return {co: (sum(v) / len(v) if v else None) for co, v in by_cutoff.items()}


# -------------------------------------------------------
# GRID SEARCH
# -------------------------------------------------------

COARSE_LO, COARSE_HI, COARSE_STEP = -1.0, 2.0, 0.1
FINE_RADIUS, FINE_STEP = 0.1, 0.01


def _frange(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 10) for i in range(n + 1)]


def grid_search(years, label):
    """Coarse grid over [COARSE_LO, COARSE_HI], then a fine grid around
    the coarse optimum. Returns (best_value, best_avg, best_lhp, best_rhp,
    all_results) where all_results is a list of (value, avg, lhp, rhp)
    for every point evaluated (coarse + fine), sorted by value."""
    t0 = time.time()
    coarse_candidates = _frange(COARSE_LO, COARSE_HI, COARSE_STEP)
    results = {}
    for c in coarse_candidates:
        avg, lhp, rhp = run_all(c, years)
        results[c] = (avg, lhp, rhp)
    best_c = max(coarse_candidates, key=lambda c: results[c][0] if results[c][0] is not None else -1.0)
    print("  [%s] coarse grid: %d points, best so far AGING_STRENGTH=%.2f (avg r=%.4f)  elapsed=%.0fs" %
          (label, len(coarse_candidates), best_c, results[best_c][0], time.time() - t0))
    sys.stdout.flush()

    fine_candidates = _frange(round(best_c - FINE_RADIUS, 10), round(best_c + FINE_RADIUS, 10), FINE_STEP)
    for c in fine_candidates:
        if c in results:
            continue
        avg, lhp, rhp = run_all(c, years)
        results[c] = (avg, lhp, rhp)
    best_c = max(results.keys(), key=lambda c: results[c][0] if results[c][0] is not None else -1.0)
    best_avg, best_lhp, best_rhp = results[best_c]
    print("  [%s] fine grid: %d additional points, final best AGING_STRENGTH=%.3f (avg r=%.4f)  elapsed=%.0fs" %
          (label, len(fine_candidates), best_c, best_avg, time.time() - t0))
    sys.stdout.flush()

    all_results = sorted([(c, v[0], v[1], v[2]) for c, v in results.items()], key=lambda t: t[0])
    return best_c, best_avg, best_lhp, best_rhp, all_results


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    t0 = time.time()

    print("=== TUNER AGING -- aging curve strength (v13, Task #79) ===")
    print("Results saved to: " + OUTPUT_FILE)
    print("CV folds (rolling-origin; test year is never in that fold's training years):")
    for i, f in enumerate(CV_FOLDS, 1):
        print("  Fold %d: train=%s  test=%d" % (i, f["train"], f["test"]))
    print("Final fit: train=%s  (no holdout -- production candidate)" % ALL_YEARS)
    print("1 tunable param (AGING_STRENGTH), searched via coarse grid [%.1f, %.1f] step %.1f," %
          (COARSE_LO, COARSE_HI, COARSE_STEP))
    print("then a fine grid +/-%.2f step %.2f around the coarse optimum." % (FINE_RADIUS, FINE_STEP))
    print("Objective is the COMBINED (LHP+RHP pooled) SIMPLE average r across all")
    print("windows -- matches backtest.py's own \"Overall average r\" formula exactly.")
    print("AGING_STRENGTH is one knob shared by both hands, not split.")
    print("Everything else (weights, regress-PA, LA, decay, platoon, xwOBA blend,")
    print("both hands) frozen at v13 production values.")
    print()

    # --- Canary check: AGING_STRENGTH=0.0 must reproduce the known
    # production baseline bit-for-bit (to 4 decimals) before any search
    # result here can be trusted. ---
    base_all, base_lhp, base_rhp = run_all(0.0, ALL_YEARS)
    print("CANARY CHECK (AGING_STRENGTH=0.0 vs known v12/v13 production baseline):")
    print("  all: %.4f  (expected %.4f, delta %+.6f)" % (base_all, KNOWN_BASELINE_ALL, base_all - KNOWN_BASELINE_ALL))
    print("  lhp: %.4f  (expected %.4f, delta %+.6f)" % (base_lhp, KNOWN_BASELINE_LHP, base_lhp - KNOWN_BASELINE_LHP))
    print("  rhp: %.4f  (expected %.4f, delta %+.6f)" % (base_rhp, KNOWN_BASELINE_RHP, base_rhp - KNOWN_BASELINE_RHP))
    canary_ok = (round(base_all, 4) == KNOWN_BASELINE_ALL and
                 round(base_lhp, 4) == KNOWN_BASELINE_LHP and
                 round(base_rhp, 4) == KNOWN_BASELINE_RHP)
    print("  CANARY " + ("PASSED -- proceeding with search." if canary_ok else
                          "FAILED -- this tuner's scoring math does NOT match production. STOPPING."))
    print()
    if not canary_ok:
        print("=== ABORTED (canary check failed) ===")
        return

    fold_results = []
    for i, fold in enumerate(CV_FOLDS, 1):
        train_years, test_year = fold["train"], fold["test"]
        label = "Fold %d" % i
        print("=" * 60)
        print("%s: train=%s  test=%d" % (label, train_years, test_year))
        print("=" * 60)

        base_train_all, base_train_lhp, base_train_rhp = run_all(0.0, train_years)
        base_test_all, base_test_lhp, base_test_rhp = run_all(0.0, [test_year])
        print("  BASELINE (AGING_STRENGTH=0.0) train (all/lhp/rhp): %.4f / %.4f / %.4f" %
              (base_train_all, base_train_lhp, base_train_rhp))
        print("  BASELINE (AGING_STRENGTH=0.0) test  (all/lhp/rhp): %.4f / %.4f / %.4f" %
              (base_test_all, base_test_lhp, base_test_rhp))
        print()

        best_c, best_train_all, best_train_lhp, best_train_rhp, _ = grid_search(train_years, label)
        test_all, test_lhp, test_rhp = run_all(best_c, [test_year])
        print()
        print("  TUNED AGING_STRENGTH=%.3f" % best_c)
        print("  TUNED train (all/lhp/rhp): %.4f / %.4f / %.4f" % (best_train_all, best_train_lhp, best_train_rhp))
        print("  TUNED test  (all/lhp/rhp): %.4f / %.4f / %.4f" % (test_all, test_lhp, test_rhp))
        print("  Test delta vs baseline (all): %+.4f" % (test_all - base_test_all))
        print("  Test delta vs baseline (lhp): %+.4f" % (test_lhp - base_test_lhp))
        print("  Test delta vs baseline (rhp): %+.4f" % (test_rhp - base_test_rhp))

        base_cutoffs_l = per_cutoff_avg(0.0, [test_year], 'L')
        tuned_cutoffs_l = per_cutoff_avg(best_c, [test_year], 'L')
        base_cutoffs_r = per_cutoff_avg(0.0, [test_year], 'R')
        tuned_cutoffs_r = per_cutoff_avg(best_c, [test_year], 'R')
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
            "fold": i, "train_years": train_years, "test_year": test_year,
            "aging_strength": best_c,
            "base_test_all": base_test_all, "base_test_lhp": base_test_lhp, "base_test_rhp": base_test_rhp,
            "tuned_test_all": test_all, "tuned_test_lhp": test_lhp, "tuned_test_rhp": test_rhp,
        })

    # --- Cross-fold summary ---
    print("=" * 60)
    print("CROSS-FOLD SUMMARY -- this is the generalization evidence")
    print("=" * 60)
    print()
    print("fold  test_yr  AGING_STR   base_all  tuned_all    delta  |  base_lhp tuned_lhp  base_rhp tuned_rhp")
    for r in fold_results:
        d_all = r["tuned_test_all"] - r["base_test_all"]
        print("%4d  %7d   %7.3f   %8.4f  %9.4f  %+.4f  |  %8.4f %9.4f  %8.4f %9.4f" % (
            r["fold"], r["test_year"], r["aging_strength"],
            r["base_test_all"], r["tuned_test_all"], d_all,
            r["base_test_lhp"], r["tuned_test_lhp"], r["base_test_rhp"], r["tuned_test_rhp"]))
    n = len(fold_results)
    avg_base_all  = sum(r["base_test_all"] for r in fold_results) / n
    avg_tuned_all = sum(r["tuned_test_all"] for r in fold_results) / n
    avg_strength  = sum(r["aging_strength"] for r in fold_results) / n
    print("-" * 60)
    print(" avg                %7.3f   %8.4f  %9.4f  %+.4f" % (
        avg_strength, avg_base_all, avg_tuned_all, avg_tuned_all - avg_base_all))
    print()
    print("If the per-fold AGING_STRENGTH values land in a similar place and the test")
    print("delta is consistently positive, the aging correction generalizes out-of-")
    print("sample. If the per-fold values swing wildly (e.g. one fold strongly")
    print("negative, another strongly positive) or deltas are flat-to-negative, that's")
    print("a legitimate signal the correction doesn't generalize, or that 0.0 (current")
    print("production) is already about as good as it gets -- not a bug.")
    print()

    # --- Final production fit on ALL years ---
    print("=" * 60)
    print("FINAL PRODUCTION FIT (trained on all %d years, no held-out year)" % len(ALL_YEARS))
    print("=" * 60)
    print("Uses every available year for max data. Trust in this value comes from")
    print("the CV evidence above, not a fresh holdout -- there isn't one left, since")
    print("all years are now in the training set.")
    print()
    final_c, final_all, final_lhp, final_rhp, all_results = grid_search(ALL_YEARS, "Final")
    print()
    print("  BASELINE in-sample (all/lhp/rhp): %.4f / %.4f / %.4f" % (base_all, base_lhp, base_rhp))
    print("  Final fit in-sample (all/lhp/rhp): %.4f / %.4f / %.4f" % (final_all, final_lhp, final_rhp))
    print("  Delta vs baseline (all): %+.4f" % (final_all - base_all))

    base_cutoffs_l = per_cutoff_avg(0.0, ALL_YEARS, 'L')
    final_cutoffs_l = per_cutoff_avg(final_c, ALL_YEARS, 'L')
    base_cutoffs_r = per_cutoff_avg(0.0, ALL_YEARS, 'R')
    final_cutoffs_r = per_cutoff_avg(final_c, ALL_YEARS, 'R')
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

    print("  Full grid (value -> all/lhp/rhp avg r), sorted by AGING_STRENGTH:")
    for c, avg, lhp, rhp in all_results:
        marker = "  <- final" if c == final_c else ""
        lhp_s = "%.4f" % lhp if lhp is not None else " N/A "
        rhp_s = "%.4f" % rhp if rhp is not None else " N/A "
        avg_s = "%.4f" % avg if avg is not None else " N/A "
        print("    %+6.2f   all=%s  lhp=%s  rhp=%s%s" % (c, avg_s, lhp_s, rhp_s, marker))
    print()

    print("=" * 60)
    print("TUNED PARAMETER -- paste this into backtest.py")
    print("=" * 60)
    print()
    print("AGING_STRENGTH = %s" % round(final_c, 3))
    print()
    print("--- UNCHANGED (frozen, still v13 production) ---")
    print("AGING_CURVE_CUM = (unchanged -- only the strength multiplier was tuned here)")
    print("LEAGUE_AVG_WOBA_LHP=%s  LEAGUE_AVG_WOBA_RHP=%s" % (LA_LHP, LA_RHP))
    print("WEIGHT_SPLITS_LHP=%s  WEIGHT_CAREER_LHP=%s  WEIGHT_TOTAL_LHP=%s  WEIGHT_RECENT_LHP=%s" % (WS_L, WC_L, WT_L, WR_L))
    print("REGRESS_PA_SPLITS_LHP=%s  REGRESS_PA_CAREER_LHP=%s  REGRESS_PA_TOTAL_LHP=%s  REGRESS_PA_RECENT_LHP=%s" % (RPS_L, RPC_L, RPT_L, RPR_L))
    print("WEIGHT_SPLITS_RHP=%s  WEIGHT_CAREER_RHP=%s  WEIGHT_TOTAL_RHP=%s  WEIGHT_RECENT_RHP=%s" % (WS_R, WC_R, WT_R, WR_R))
    print("REGRESS_PA_SPLITS_RHP=%s  REGRESS_PA_CAREER_RHP=%s  REGRESS_PA_TOTAL_RHP=%s  REGRESS_PA_RECENT_RHP=%s" % (RPS_R, RPC_R, RPT_R, RPR_R))
    print("DECAY_CAREER_LHP=%s  DECAY_CAREER_RHP=%s" % (DECAY_CAREER_LHP, DECAY_CAREER_RHP))
    print("PLATOON_BLEND_LHP=%s  PLATOON_FACTOR_MIN_LHP=%s  PLATOON_FACTOR_MAX_LHP=%s" % (PLATOON_BLEND_LHP, PLATOON_MIN_LHP, PLATOON_MAX_LHP))
    print("PLATOON_BLEND_RHP=%s  PLATOON_FACTOR_MIN_RHP=%s  PLATOON_FACTOR_MAX_RHP=%s" % (PLATOON_BLEND_RHP, PLATOON_MIN_RHP, PLATOON_MAX_RHP))
    print("XWOBA_BLEND_{SPLITS,CAREER,TOTAL,RECENT}_LHP = 0.9187 (all 4, unchanged)")
    print("XWOBA_BLEND_{SPLITS,CAREER,TOTAL,RECENT}_RHP = 0.80   (all 4, unchanged)")
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
