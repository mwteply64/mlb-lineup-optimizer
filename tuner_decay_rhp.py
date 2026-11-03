#!/usr/bin/env python3
"""
Tuner - DECAY_CAREER_RHP (Task #80)
====================================
backtest.py has carried DECAY_CAREER_RHP=1.0 (no recency weighting -- every
prior season counts equally, regardless of how long ago it was) as an
untuned placeholder since v9. v10 gave the LHP analog its own dedicated
single-knob search (tuner_lhp.py -> DECAY_CAREER_LHP=0.8001) but RHP never
got the same treatment: v11's tuner_rhp.py only tested DECAY_CAREER_RHP
inside a 14-parameter joint search (which failed out-of-sample on every
fold) and a follow-up one-at-a-time isolation test, where it showed "an
interesting but inconclusive partial signal" that was never followed up
with a focused search of its own. DECAY_CAREER_RHP=1.0 not making much
baseball sense (a player's 2016 season counting exactly as much as their
2024 season) is the whole reason this tuner exists. This file gives it
that dedicated walk-forward CV search.

WHY A 1-D GRID SEARCH INSTEAD OF tuner_rhp.py'S RANDOM-SEARCH/COORDINATE-
ASCENT MACHINERY:
Unlike that 14-parameter joint search, there is exactly one free parameter
here -- DECAY_CAREER_LHP is frozen at its already-tuned v10 value (0.8001)
and every other RHP/LHP constant is frozen at current production. A 1-D
objective in a bounded range is cheap and safe to search exhaustively with
a grid; the multi-restart machinery in tuner_lhp.py/tuner_rhp.py/
tuner_xwoba_components.py exists to handle 8-14 dimensions where local
optima are a real risk. That risk doesn't apply to a single dimension, so
a coarse grid (step 0.03) followed by a fine grid around the coarse
optimum (step 0.003) is simpler, fully deterministic (no RNG, no restarts
needed), and just as reliable here. Same pattern as tuner_aging.py.

WHY CAREER DATA CANNOT BE CACHED ACROSS TRIALS:
DECAY_CAREER_RHP directly changes build_career_data()'s output for the
"right" hand -- it's the per-season recency weight (decay ** (test_year -
season_year)) applied BEFORE the PA-weighted average is taken. So RHP
career data must be rebuilt for every candidate value tried. LHP career
data never changes (DECAY_CAREER_LHP is frozen), so it's built once per
year and reused across every candidate. The raw per-season files
(SEASON_CACHE) and per-window files (WINDOW_CACHE) don't depend on decay
at all, so those are cached once up front regardless of hand.

AGING_STRENGTH is omitted entirely rather than threaded through as a
parameter: backtest.py v13 short-circuits the age correction with
`if aging_strength != 0.0` (a true skip, not a zero-multiply -- verified
in tuner_aging.py's own canary check), and AGING_STRENGTH stays at its
frozen production value of 0.0 throughout this search. Leaving it out of
build_career_data() here is mathematically identical to passing 0.0, and
this tuner's own canary check below confirms that.

OBJECTIVE: a single combined score (LHP+RHP pooled), computed as the
SIMPLE (unweighted) mean of every window's Spearman r -- exactly
backtest.py's own "Overall average r" formula. Same convention as
tuner_aging.py (verified there to reproduce KNOWN_BASELINE_ALL/LHP/RHP
bit-for-bit; a sqrt(PA)-weighted mean does not). LHP-only and RHP-only
averages (also simple means) are reported alongside for diagnostics.

TUNES (1 parameter): DECAY_CAREER_RHP

FROZEN AT CURRENT PRODUCTION (not tunable -- everything else):
  LEAGUE_AVG_WOBA_LHP=0.2943, LEAGUE_AVG_WOBA_RHP=0.2575
  WEIGHT_*_LHP / WEIGHT_*_RHP (8 weights), REGRESS_PA_*_LHP/_RHP (8 values)
  DECAY_CAREER_LHP=0.8001 (already tuned, v10)
  AGING_STRENGTH=0.0 (omitted entirely -- true no-op, see above)
  PLATOON_BLEND_LHP=0.7833 (clamp [0.612, 1.138])
  PLATOON_BLEND_RHP=0.31   (clamp [0.585, 1.325])
  XWOBA_BLEND_{SPLITS,CAREER,TOTAL,RECENT}_LHP = 0.9187 (all 4)
  XWOBA_BLEND_{SPLITS,CAREER,TOTAL,RECENT}_RHP = 0.80   (all 4)
PA minimums fixed at 30 (splits/career/total/post). NO_CLAMP fixed True.

GRID: coarse pass over [0.40, 1.00] step 0.03 (21 points, includes 1.00 --
"no decay", the current production value). Bounds match the range
tuner_rhp.py's joint search already used for this same parameter
(BOUNDS["DECAY_CAREER_RHP"] = (0.40, 1.0, 0.03)), so this is a refinement
of previously-vetted territory, not a new guess. Decay > 1.0 isn't
searched -- it would mean an older season counts MORE than a newer one,
which doesn't make baseball sense. Then a fine pass over
[best-0.03, best+0.03] step 0.003 (21 points) around whichever coarse
point scored best.

USAGE:
  Double-click run_tuner_decay_rhp.bat (writes
  tuner_decay_rhp_results_YYYY-MM-DD.txt)
"""
import os
import csv
import sys
import datetime
import time

FOLDER   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FOLDER, "historical testing")
today    = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(FOLDER, "tuner_decay_rhp_results_" + today + ".txt")


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
# FIXED CONSTANTS (not tunable -- everything except DECAY_CAREER_RHP,
# frozen at current production values)
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

DECAY_CAREER_LHP = 0.8001  # frozen -- already tuned (v10)

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

# Known production r values (DECAY_CAREER_RHP=1.0, AGING_STRENGTH=0.0),
# from backtest.py's own docstring and the last verified production run.
# Used below as a canary: this tuner's scoring math must reproduce these
# exactly before any search results can be trusted.
KNOWN_BASELINE_ALL = 0.3706
KNOWN_BASELINE_LHP = 0.3299
KNOWN_BASELINE_RHP = 0.4113

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


# Season files, cached by (hand, year). Independent of decay -- safe to
# load once and reuse across every trial.
SEASON_CACHE = {}
for hand in ("left", "right"):
    for y in ALL_SEASON_YEARS:
        SEASON_CACHE[(hand, y)] = load_file("season_" + str(y) + "_vs_" + hand + ".csv")

# Per-window files, cached by (year, cutoff). Also independent of decay --
# only the career inputs change per trial, not these.
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
# CAREER DATA -- NOT cacheable across trials for the "right" hand (see
# module docstring). LHP career data IS cacheable since DECAY_CAREER_LHP
# never changes, but for simplicity this function is called fresh for
# both hands per year inside run_all()'s own per-year cache.
# -------------------------------------------------------

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
# SCORING MATH (mirrors backtest.py production exactly). DECAY_CAREER_RHP
# is the only thing that varies between calls -- it only ever affects
# which car_r dict gets passed in, never anything inside stabilize().
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
    """Returns (r_l, r_r, pa_l, pa_r). car_l/car_r are this trial's career
    dicts for `year` (car_r rebuilt by the caller for the candidate
    DECAY_CAREER_RHP being scored; car_l always built with the frozen
    DECAY_CAREER_LHP)."""
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


# Per-year career-data memo, cleared between distinct grid searches via a
# plain dict keyed on year -- a single search tries the same candidate
# decay across every cutoff in a year 3 times (once per cutoff), so this
# still saves 2/3 of the rebuild work within one run_all() call, even
# though nothing survives across different candidate values (by design --
# see module docstring).
def run_all(decay_rhp, years):
    """Window-level Spearman r values combined into a SIMPLE (unweighted)
    mean across every window/hand -- exactly backtest.py's own "Overall
    average r" formula (see module docstring OBJECTIVE section)."""
    car_cache = {}
    all_rs, lhp_rs, rhp_rs = [], [], []
    for year in years:
        if year not in car_cache:
            car_l = build_career_data(year, "left",  DECAY_CAREER_LHP)
            car_r = build_career_data(year, "right", decay_rhp)
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


def per_cutoff_avg(decay_rhp, years, side):
    """Diagnostic only -- NOT used in the objective."""
    by_cutoff = {co: [] for co in CUTOFFS}
    car_cache = {}
    for year in years:
        if year not in car_cache:
            car_l = build_career_data(year, "left",  DECAY_CAREER_LHP)
            car_r = build_career_data(year, "right", decay_rhp)
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

COARSE_LO, COARSE_HI, COARSE_STEP = 0.40, 1.00, 0.03
FINE_RADIUS, FINE_STEP = 0.03, 0.003


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
    print("  [%s] coarse grid: %d points, best so far DECAY_CAREER_RHP=%.2f (avg r=%.4f)  elapsed=%.0fs" %
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
    print("  [%s] fine grid: %d additional points, final best DECAY_CAREER_RHP=%.3f (avg r=%.4f)  elapsed=%.0fs" %
          (label, len(fine_candidates), best_c, best_avg, time.time() - t0))
    sys.stdout.flush()

    all_results = sorted([(c, v[0], v[1], v[2]) for c, v in results.items()], key=lambda t: t[0])
    return best_c, best_avg, best_lhp, best_rhp, all_results


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    t0 = time.time()

    print("=== TUNER DECAY RHP -- DECAY_CAREER_RHP (Task #80) ===")
    print("Results saved to: " + OUTPUT_FILE)
    print("CV folds (rolling-origin; test year is never in that fold's training years):")
    for i, f in enumerate(CV_FOLDS, 1):
        print("  Fold %d: train=%s  test=%d" % (i, f["train"], f["test"]))
    print("Final fit: train=%s  (no holdout -- production candidate)" % ALL_YEARS)
    print("1 tunable param (DECAY_CAREER_RHP), searched via coarse grid [%.2f, %.2f] step %.2f," %
          (COARSE_LO, COARSE_HI, COARSE_STEP))
    print("then a fine grid +/-%.3f step %.3f around the coarse optimum." % (FINE_RADIUS, FINE_STEP))
    print("Objective is the COMBINED (LHP+RHP pooled) SIMPLE average r across all")
    print("windows -- matches backtest.py's own \"Overall average r\" formula exactly.")
    print("DECAY_CAREER_RHP only affects the RHP career build -- DECAY_CAREER_LHP stays")
    print("frozen at 0.8001 throughout. Everything else (weights, regress-PA, LA,")
    print("aging, platoon, xwOBA blend, both hands) frozen at current production values.")
    print()

    # --- Canary check: DECAY_CAREER_RHP=1.0 (current production, "no
    # decay") must reproduce the known production baseline bit-for-bit (to
    # 4 decimals) before any search result here can be trusted. ---
    base_all, base_lhp, base_rhp = run_all(1.0, ALL_YEARS)
    print("CANARY CHECK (DECAY_CAREER_RHP=1.0 vs known production baseline):")
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

        base_train_all, base_train_lhp, base_train_rhp = run_all(1.0, train_years)
        base_test_all, base_test_lhp, base_test_rhp = run_all(1.0, [test_year])
        print("  BASELINE (DECAY_CAREER_RHP=1.0) train (all/lhp/rhp): %.4f / %.4f / %.4f" %
              (base_train_all, base_train_lhp, base_train_rhp))
        print("  BASELINE (DECAY_CAREER_RHP=1.0) test  (all/lhp/rhp): %.4f / %.4f / %.4f" %
              (base_test_all, base_test_lhp, base_test_rhp))
        print()

        best_c, best_train_all, best_train_lhp, best_train_rhp, _ = grid_search(train_years, label)
        test_all, test_lhp, test_rhp = run_all(best_c, [test_year])
        print()
        print("  TUNED DECAY_CAREER_RHP=%.3f" % best_c)
        print("  TUNED train (all/lhp/rhp): %.4f / %.4f / %.4f" % (best_train_all, best_train_lhp, best_train_rhp))
        print("  TUNED test  (all/lhp/rhp): %.4f / %.4f / %.4f" % (test_all, test_lhp, test_rhp))
        print("  Test delta vs baseline (all): %+.4f" % (test_all - base_test_all))
        print("  Test delta vs baseline (lhp): %+.4f" % (test_lhp - base_test_lhp))
        print("  Test delta vs baseline (rhp): %+.4f" % (test_rhp - base_test_rhp))

        base_cutoffs_l = per_cutoff_avg(1.0, [test_year], 'L')
        tuned_cutoffs_l = per_cutoff_avg(best_c, [test_year], 'L')
        base_cutoffs_r = per_cutoff_avg(1.0, [test_year], 'R')
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
            "decay_rhp": best_c,
            "base_test_all": base_test_all, "base_test_lhp": base_test_lhp, "base_test_rhp": base_test_rhp,
            "tuned_test_all": test_all, "tuned_test_lhp": test_lhp, "tuned_test_rhp": test_rhp,
        })

    # --- Cross-fold summary ---
    print("=" * 60)
    print("CROSS-FOLD SUMMARY -- this is the generalization evidence")
    print("=" * 60)
    print()
    print("fold  test_yr  DECAY_RHP   base_all  tuned_all    delta  |  base_lhp tuned_lhp  base_rhp tuned_rhp")
    for r in fold_results:
        d_all = r["tuned_test_all"] - r["base_test_all"]
        print("%4d  %7d   %7.3f   %8.4f  %9.4f  %+.4f  |  %8.4f %9.4f  %8.4f %9.4f" % (
            r["fold"], r["test_year"], r["decay_rhp"],
            r["base_test_all"], r["tuned_test_all"], d_all,
            r["base_test_lhp"], r["tuned_test_lhp"], r["base_test_rhp"], r["tuned_test_rhp"]))
    n = len(fold_results)
    avg_base_all  = sum(r["base_test_all"] for r in fold_results) / n
    avg_tuned_all = sum(r["tuned_test_all"] for r in fold_results) / n
    avg_decay     = sum(r["decay_rhp"] for r in fold_results) / n
    print("-" * 60)
    print(" avg                %7.3f   %8.4f  %9.4f  %+.4f" % (
        avg_decay, avg_base_all, avg_tuned_all, avg_tuned_all - avg_base_all))
    print()
    print("If the per-fold DECAY_CAREER_RHP values land in a similar place and the test")
    print("delta is consistently positive, the recency weighting generalizes out-of-")
    print("sample. If the per-fold values swing wildly (e.g. one fold near 1.0, another")
    print("near 0.4) or deltas are flat-to-negative, that's a legitimate signal the")
    print("correction doesn't generalize, or that 1.0 (current production) is already")
    print("about as good as it gets -- not a bug.")
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

    base_cutoffs_l = per_cutoff_avg(1.0, ALL_YEARS, 'L')
    final_cutoffs_l = per_cutoff_avg(final_c, ALL_YEARS, 'L')
    base_cutoffs_r = per_cutoff_avg(1.0, ALL_YEARS, 'R')
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

    print("  Full grid (value -> all/lhp/rhp avg r), sorted by DECAY_CAREER_RHP:")
    for c, avg, lhp, rhp in all_results:
        marker = "  <- final" if c == final_c else ""
        lhp_s = "%.4f" % lhp if lhp is not None else " N/A "
        rhp_s = "%.4f" % rhp if rhp is not None else " N/A "
        avg_s = "%.4f" % avg if avg is not None else " N/A "
        print("    %.3f   all=%s  lhp=%s  rhp=%s%s" % (c, avg_s, lhp_s, rhp_s, marker))
    print()

    print("=" * 60)
    print("TUNED PARAMETER -- paste this into backtest.py")
    print("=" * 60)
    print()
    print("DECAY_CAREER_RHP = %s" % round(final_c, 3))
    print()
    print("--- UNCHANGED (frozen, still current production) ---")
    print("AGING_STRENGTH = 0.0 (unchanged)")
    print("LEAGUE_AVG_WOBA_LHP=%s  LEAGUE_AVG_WOBA_RHP=%s" % (LA_LHP, LA_RHP))
    print("WEIGHT_SPLITS_LHP=%s  WEIGHT_CAREER_LHP=%s  WEIGHT_TOTAL_LHP=%s  WEIGHT_RECENT_LHP=%s" % (WS_L, WC_L, WT_L, WR_L))
    print("REGRESS_PA_SPLITS_LHP=%s  REGRESS_PA_CAREER_LHP=%s  REGRESS_PA_TOTAL_LHP=%s  REGRESS_PA_RECENT_LHP=%s" % (RPS_L, RPC_L, RPT_L, RPR_L))
    print("WEIGHT_SPLITS_RHP=%s  WEIGHT_CAREER_RHP=%s  WEIGHT_TOTAL_RHP=%s  WEIGHT_RECENT_RHP=%s" % (WS_R, WC_R, WT_R, WR_R))
    print("REGRESS_PA_SPLITS_RHP=%s  REGRESS_PA_CAREER_RHP=%s  REGRESS_PA_TOTAL_RHP=%s  REGRESS_PA_RECENT_RHP=%s" % (RPS_R, RPC_R, RPT_R, RPR_R))
    print("DECAY_CAREER_LHP=%s (unchanged)" % DECAY_CAREER_LHP)
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
