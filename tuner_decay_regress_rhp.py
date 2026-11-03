#!/usr/bin/env python3
"""
Tuner - DECAY_CAREER_RHP x REGRESS_PA_CAREER_RHP joint search (Task #81)
=========================================================================
Task #80 (tuner_decay_rhp.py) gave DECAY_CAREER_RHP its own dedicated 1-D
walk-forward CV search, holding REGRESS_PA_CAREER_RHP fixed at 25 (its v9
placeholder value). That search was rejected: per-fold optimal decay values
drifted instead of converging (0.409 -> 0.574 -> 0.622 across the three
folds), which is exactly the "unstable, not generalizing" pattern this
project's ship bar is designed to catch.

This tuner asks whether REGRESS_PA_CAREER_RHP=25 itself is the problem.
Per backtest.py's own v11 docstring, REGRESS_PA_CAREER_RHP "remain[s] at
[its] original v9 neutral placeholder, pending stronger evidence or a
future dedicated re-tune" -- it has NEVER been validated by walk-forward
CV. Task #54 only confirmed it traces to v8's coordinate-ascent + random-
search (1500+ trials, in-sample only, no train/test split -- predates this
project's v10+ walk-forward-CV standard) and is deliberate, not a typo.
It's also an outlier vs. its sibling RHP regress constants: RPS_R=285,
RPT_R=400, RPR_R=665 (a 285-665 band) vs. RPC_R=25 -- more than 10x smaller
than any of them.

WHY JOINT, NOT TWO SEPARATE 1-D SEARCHES:
DECAY_CAREER_RHP and REGRESS_PA_CAREER_RHP both act on the same underlying
career-average number. DECAY_CAREER_RHP sets the per-season-weighted career
average (inside build_career_data()); REGRESS_PA_CAREER_RHP then shrinks
that average toward league mean before it's blended with splits/total/
recent (inside stabilize()'s regress() call). A miscalibrated regress
constant can distort the apparent payoff of any given decay value -- which
is a plausible mechanical explanation for Task #80's drift, distinct from
"there's no real recency effect." Searching them together is the only way
to tell those two explanations apart.

WHY COORDINATE ASCENT + LOCAL REFINEMENT, NOT A FULL 2-D GRID:
A full grid over DECAY_CAREER_RHP's 21-point coarse range x
REGRESS_PA_CAREER_RHP's 21-point coarse range is 441 evaluations before any
fine pass -- too expensive to run inside this sandbox's per-call time
budget, even phased. Instead: (1) grid-search DECAY_CAREER_RHP alone with
REGRESS_PA_CAREER_RHP held at its current production value of 25 (~40
evals); (2) grid-search REGRESS_PA_CAREER_RHP alone with DECAY_CAREER_RHP
held at step 1's result (~40 evals, and cheap -- see below); (3) a small
local 2-D refinement grid (7 decay points x 9 regress points = 63 evals)
around step 2's result, to catch any interaction the sequential passes
missed. ~143 evals/year-set total. This mirrors v8's own precedent
("coordinate-ascent + random-search re-optimization... confirmed via 1500+
random trials") at a much smaller, walk-forward-CV-appropriate scale.

WHY REGRESS_PA_CAREER_RHP SWEEPS DON'T REQUIRE REBUILDING CAREER DATA:
REGRESS_PA_CAREER_RHP is applied inside stabilize()'s regress() call, never
inside build_career_data(). So for a FIXED DECAY_CAREER_RHP, every
REGRESS_PA_CAREER_RHP candidate can reuse the same career dict -- only the
per-player regress()/stabilize() math changes. Step 2's grid search (and
the inner loop of step 3's refinement grid) exploit this: career data is
built once per decay value and reused across every regress_pa candidate
tried with that decay. Step 1's grid search gets no such shortcut --
DECAY_CAREER_RHP changes the career dict itself, so it must be rebuilt for
every candidate (same constraint as tuner_decay_rhp.py).

PHASED CLI (for sandbox execution under a ~45s per-call budget):
  python3 tuner_decay_regress_rhp.py            -> full unphased run
                                                    (for the local .bat file,
                                                    no time limit there)
  python3 tuner_decay_regress_rhp.py fold1      -> canary + CV fold 1
  python3 tuner_decay_regress_rhp.py fold2      -> CV fold 2
  python3 tuner_decay_regress_rhp.py fold3      -> CV fold 3
  python3 tuner_decay_regress_rhp.py final      -> cross-fold summary +
                                                    final production fit
Each fold phase persists its result to a JSON checkpoint file so later
phases (run as separate processes) can pick up where the previous one left
off. `final` requires fold1/fold2/fold3 to already be checkpointed. The
unphased path runs all four stages in one process, writing/reading the
same checkpoint file as the phased path (so the code is identical either
way -- no separate "unphased" logic branch to maintain).

OBJECTIVE: SIMPLE (unweighted) mean of every window's Spearman r --
backtest.py's own "Overall average r" formula, same convention as every
other tuner in this project.

TUNES (2 parameters, RHP only): DECAY_CAREER_RHP, REGRESS_PA_CAREER_RHP

FROZEN AT CURRENT PRODUCTION (not tunable -- everything else, identical
list to tuner_decay_rhp.py): LEAGUE_AVG_WOBA_LHP/RHP, all 8 WEIGHT_* and
the 7 other REGRESS_PA_* values, DECAY_CAREER_LHP=0.8001, AGING_STRENGTH=0.0
(omitted entirely -- true no-op, same as tuner_decay_rhp.py), both
PLATOON_BLEND_* + clamps, all 8 XWOBA_BLEND_* values. PA minimums fixed at
30. NO_CLAMP fixed True.

GRID:
  DECAY_CAREER_RHP:        coarse [0.40, 1.00] step 0.03 (21 pts),
                           fine +/-0.03 step 0.003 (21 pts) -- unchanged
                           bounds from tuner_decay_rhp.py.
  REGRESS_PA_CAREER_RHP:   coarse [0, 800] step 40 (21 pts) -- upper bound
                           matches the 285-665 band of its siblings with
                           headroom; 0 is included (no regression at all)
                           since 25 is already so close to that extreme,
                           fine +/-40 step 4 (21 pts).
  Local refinement:        decay +/-0.03 step 0.01 (7 pts) x
                           regress_pa +/-40 step 10 (9 pts) around the
                           coordinate-ascent result (63 pts).

USAGE:
  Double-click run_tuner_decay_regress_rhp.bat (writes
  tuner_decay_regress_rhp_results_YYYY-MM-DD.txt)
"""
import os
import csv
import sys
import json
import datetime
import time

FOLDER   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FOLDER, "historical testing")
today    = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_FILE     = os.path.join(FOLDER, "tuner_decay_regress_rhp_results_" + today + ".txt")
CHECKPOINT_FILE = os.path.join(FOLDER, "tuner_decay_regress_rhp_checkpoint.json")


class Tee:
    def __init__(self, filepath, mode="w"):
        self.console = sys.stdout
        self.file    = open(filepath, mode, encoding="utf-8")
    def write(self, text):
        self.console.write(text)
        self.file.write(text)
    def flush(self):
        self.console.flush()
        self.file.flush()
    def close(self):
        self.file.close()


# -------------------------------------------------------
# FIXED CONSTANTS (not tunable -- everything except DECAY_CAREER_RHP and
# REGRESS_PA_CAREER_RHP, frozen at current production values)
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
RPS_R, RPT_R, RPR_R = 285, 400, 665
RPC_R_PROD = 25  # current production REGRESS_PA_CAREER_RHP -- never CV-validated (see module docstring); used as the canary value and as the fixed value during step 1 (decay-only) of the coordinate-ascent search.

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

# Known production r values (DECAY_CAREER_RHP=1.0, REGRESS_PA_CAREER_RHP=25,
# AGING_STRENGTH=0.0), from backtest.py's own docstring and the last
# verified production run. Used below as a canary: this tuner's scoring
# math must reproduce these exactly before any search result can be trusted.
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


# Season files, cached by (hand, year). Independent of decay/regress -- safe
# to load once and reuse across every trial.
SEASON_CACHE = {}
for hand in ("left", "right"):
    for y in ALL_SEASON_YEARS:
        SEASON_CACHE[(hand, y)] = load_file("season_" + str(y) + "_vs_" + hand + ".csv")

# Per-window files, cached by (year, cutoff). Also independent of decay/
# regress -- only the career inputs and the stabilize() math change.
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
# CAREER DATA -- depends on DECAY_CAREER_RHP only (not on
# REGRESS_PA_CAREER_RHP, which is applied later inside stabilize()). LHP
# career data never changes (DECAY_CAREER_LHP is frozen).
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


def build_car_cache(decay_rhp, years):
    """Builds {year: (car_l, car_r)} once for a fixed decay value, so every
    REGRESS_PA_CAREER_RHP candidate tried with that decay can reuse it
    instead of rebuilding career data (see module docstring)."""
    car_cache = {}
    for year in years:
        car_l = build_career_data(year, "left",  DECAY_CAREER_LHP)
        car_r = build_career_data(year, "right", decay_rhp)
        car_cache[year] = (car_l, car_r)
    return car_cache


def blend_woba(entry, xwoba_blend):
    if entry is None:
        return None
    xw = entry.get("xwoba")
    if xw is None:
        return entry["woba"]
    return (1.0 - xwoba_blend) * entry["woba"] + xwoba_blend * xw


# -------------------------------------------------------
# SCORING MATH (mirrors backtest.py production exactly). DECAY_CAREER_RHP
# (via which car_r dict is passed in) and REGRESS_PA_CAREER_RHP (via the
# rpc_r argument to stabilize()) are the only two things that vary between
# calls.
# -------------------------------------------------------

def regress(woba, pa, regress_pa, la):
    return (woba * pa + la * regress_pa) / (pa + regress_pa)


def stabilize(splits_pa, splits_woba, career_pa, career_woba,
              total_pa, total_woba, recent_pa, recent_woba, side,
              career_cross_pa=0, career_cross_woba=None, rpc_r=RPC_R_PROD):
    if side == 'L':
        la, WS, WC, WT, WR = LA_LHP, WS_L, WC_L, WT_L, WR_L
        RPS, RPC, RPT, RPR = RPS_L, RPC_L, RPT_L, RPR_L
        pb, pmin, pmax = PLATOON_BLEND_LHP, PLATOON_MIN_LHP, PLATOON_MAX_LHP
    else:
        la, WS, WC, WT, WR = LA_RHP, WS_R, WC_R, WT_R, WR_R
        RPS, RPC, RPT, RPR = RPS_R, rpc_r, RPT_R, RPR_R
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

def run_window(year, co, car_l, car_r, rpc_r):
    """Returns (r_l, r_r, pa_l, pa_r). car_l/car_r are this trial's career
    dicts for `year` (car_r built for the candidate DECAY_CAREER_RHP being
    scored; car_l always built with the frozen DECAY_CAREER_LHP). rpc_r is
    the candidate REGRESS_PA_CAREER_RHP, threaded into the 'R'-side
    stabilize() calls only."""
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
                                  career_cross_pa=cl_pa, career_cross_woba=cl_w_r,
                                  rpc_r=rpc_r)
                if pred is not None:
                    pred_r.append(pred); act_r.append(post["woba"]); pa_r_sum += post["pa"]

    r_l = spearman_r(pred_l, act_l)
    r_r = spearman_r(pred_r, act_r)
    return r_l, r_r, pa_l_sum, pa_r_sum


def run_all(decay_rhp, regress_pa_rhp, years, car_cache=None):
    """Window-level Spearman r values combined into a SIMPLE (unweighted)
    mean across every window/hand -- exactly backtest.py's own "Overall
    average r" formula. If car_cache is provided (built via
    build_car_cache() for this same decay_rhp), reused as-is -- callers
    sweeping REGRESS_PA_CAREER_RHP at a fixed decay use this to skip
    rebuilding career data on every trial."""
    if car_cache is None:
        car_cache = {}
    all_rs, lhp_rs, rhp_rs = [], [], []
    for year in years:
        if year not in car_cache:
            car_l = build_career_data(year, "left",  DECAY_CAREER_LHP)
            car_r = build_career_data(year, "right", decay_rhp)
            car_cache[year] = (car_l, car_r)
        car_l, car_r = car_cache[year]
        for co in CUTOFFS:
            r_l, r_r, pa_l, pa_r = run_window(year, co, car_l, car_r, regress_pa_rhp)
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


def per_cutoff_avg(decay_rhp, regress_pa_rhp, years, side, car_cache=None):
    """Diagnostic only -- NOT used in the objective."""
    by_cutoff = {co: [] for co in CUTOFFS}
    if car_cache is None:
        car_cache = {}
    for year in years:
        if year not in car_cache:
            car_l = build_career_data(year, "left",  DECAY_CAREER_LHP)
            car_r = build_career_data(year, "right", decay_rhp)
            car_cache[year] = (car_l, car_r)
        car_l, car_r = car_cache[year]
        for co in CUTOFFS:
            r_l, r_r, _, _ = run_window(year, co, car_l, car_r, regress_pa_rhp)
            v = r_l if side == 'L' else r_r
            if v is not None:
                by_cutoff[co].append(v)
    return {co: (sum(v) / len(v) if v else None) for co, v in by_cutoff.items()}


# -------------------------------------------------------
# GRID SEARCH -- step 1 (decay alone), step 2 (regress_pa alone), step 3
# (local 2-D refinement). See module docstring for the full rationale.
# -------------------------------------------------------

DECAY_COARSE_LO, DECAY_COARSE_HI, DECAY_COARSE_STEP = 0.40, 1.00, 0.03
DECAY_FINE_RADIUS, DECAY_FINE_STEP = 0.03, 0.003

REGRESS_COARSE_LO, REGRESS_COARSE_HI, REGRESS_COARSE_STEP = 0, 800, 40
REGRESS_FINE_RADIUS, REGRESS_FINE_STEP = 40, 4

REFINE_DECAY_RADIUS, REFINE_DECAY_STEP = 0.03, 0.01
REFINE_REGRESS_RADIUS, REFINE_REGRESS_STEP = 40, 10


def _frange(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 10) for i in range(n + 1)]


def _irange(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [int(round(lo + i * step)) for i in range(n + 1)]


def grid_search_decay(regress_pa_fixed, years, label):
    """Step 1: search DECAY_CAREER_RHP alone, REGRESS_PA_CAREER_RHP held
    fixed (at RPC_R_PROD=25 for the first coordinate-ascent pass). Career
    data must be rebuilt every trial (decay changes it) -- no caching
    possible here, same as tuner_decay_rhp.py."""
    t0 = time.time()
    coarse = _frange(DECAY_COARSE_LO, DECAY_COARSE_HI, DECAY_COARSE_STEP)
    results = {}
    for c in coarse:
        results[c] = run_all(c, regress_pa_fixed, years)
    best_c = max(coarse, key=lambda c: results[c][0] if results[c][0] is not None else -1.0)
    print("  [%s] decay coarse grid (regress_pa=%d fixed): %d pts, best DECAY_CAREER_RHP=%.2f (avg r=%.4f)  elapsed=%.0fs" %
          (label, regress_pa_fixed, len(coarse), best_c, results[best_c][0], time.time() - t0))
    sys.stdout.flush()

    fine = _frange(round(best_c - DECAY_FINE_RADIUS, 10), round(best_c + DECAY_FINE_RADIUS, 10), DECAY_FINE_STEP)
    fine = [c for c in fine if DECAY_COARSE_LO <= c <= DECAY_COARSE_HI]
    for c in fine:
        if c not in results:
            results[c] = run_all(c, regress_pa_fixed, years)
    best_c = max(results.keys(), key=lambda c: results[c][0] if results[c][0] is not None else -1.0)
    best_avg, best_lhp, best_rhp = results[best_c]
    print("  [%s] decay fine grid: %d additional pts, final best DECAY_CAREER_RHP=%.3f (avg r=%.4f)  elapsed=%.0fs" %
          (label, len(fine), best_c, best_avg, time.time() - t0))
    sys.stdout.flush()
    return best_c, best_avg, best_lhp, best_rhp, results


def grid_search_regress(decay_fixed, years, label):
    """Step 2: search REGRESS_PA_CAREER_RHP alone, DECAY_CAREER_RHP held
    fixed (at step 1's result). Career data is built ONCE for this decay
    value and reused across every regress_pa candidate -- see module
    docstring."""
    t0 = time.time()
    car_cache = build_car_cache(decay_fixed, years)
    coarse = _irange(REGRESS_COARSE_LO, REGRESS_COARSE_HI, REGRESS_COARSE_STEP)
    results = {}
    for rpc in coarse:
        results[rpc] = run_all(decay_fixed, rpc, years, car_cache=car_cache)
    best_rpc = max(coarse, key=lambda rpc: results[rpc][0] if results[rpc][0] is not None else -1.0)
    print("  [%s] regress_pa coarse grid (decay=%.3f fixed): %d pts, best REGRESS_PA_CAREER_RHP=%d (avg r=%.4f)  elapsed=%.0fs" %
          (label, decay_fixed, len(coarse), best_rpc, results[best_rpc][0], time.time() - t0))
    sys.stdout.flush()

    fine = _irange(max(REGRESS_COARSE_LO, best_rpc - REGRESS_FINE_RADIUS),
                    min(REGRESS_COARSE_HI, best_rpc + REGRESS_FINE_RADIUS), REGRESS_FINE_STEP)
    for rpc in fine:
        if rpc not in results:
            results[rpc] = run_all(decay_fixed, rpc, years, car_cache=car_cache)
    best_rpc = max(results.keys(), key=lambda rpc: results[rpc][0] if results[rpc][0] is not None else -1.0)
    best_avg, best_lhp, best_rhp = results[best_rpc]
    print("  [%s] regress_pa fine grid: %d additional pts, final best REGRESS_PA_CAREER_RHP=%d (avg r=%.4f)  elapsed=%.0fs" %
          (label, len(fine), best_rpc, best_avg, time.time() - t0))
    sys.stdout.flush()
    return best_rpc, best_avg, best_lhp, best_rhp, results


def local_refine(decay_center, regress_center, years, label):
    """Step 3: small joint 2-D grid around the coordinate-ascent result, to
    catch any decay x regress_pa interaction the sequential 1-D passes
    might have missed."""
    t0 = time.time()
    decay_cands = [round(decay_center + d, 10) for d in
                   _frange(-REFINE_DECAY_RADIUS, REFINE_DECAY_RADIUS, REFINE_DECAY_STEP)]
    decay_cands = [d for d in decay_cands if DECAY_COARSE_LO <= d <= DECAY_COARSE_HI]
    regress_cands = [regress_center + d for d in
                      _irange(-REFINE_REGRESS_RADIUS, REFINE_REGRESS_RADIUS, REFINE_REGRESS_STEP)]
    regress_cands = [r for r in regress_cands if REGRESS_COARSE_LO <= r <= REGRESS_COARSE_HI]

    results = {}  # (decay, regress) -> (avg, lhp, rhp)
    for d in decay_cands:
        car_cache = build_car_cache(d, years)
        for r in regress_cands:
            results[(d, r)] = run_all(d, r, years, car_cache=car_cache)
    best_key = max(results.keys(), key=lambda k: results[k][0] if results[k][0] is not None else -1.0)
    best_avg, best_lhp, best_rhp = results[best_key]
    print("  [%s] local refinement: %d x %d = %d pts, best (decay=%.3f, regress_pa=%d) avg r=%.4f  elapsed=%.0fs" %
          (label, len(decay_cands), len(regress_cands), len(results), best_key[0], best_key[1], best_avg, time.time() - t0))
    sys.stdout.flush()
    return best_key[0], best_key[1], best_avg, best_lhp, best_rhp, results


def joint_search(years, label):
    """Full coordinate-ascent + local-refinement joint search. Returns
    (decay_best, regress_best, avg_best, lhp_best, rhp_best, diag) where
    diag captures the intermediate steps for reporting."""
    decay1, avg1, lhp1, rhp1, _ = grid_search_decay(RPC_R_PROD, years, label + " step1")
    regress1, avg2, lhp2, rhp2, _ = grid_search_regress(decay1, years, label + " step2")
    decay3, regress3, avg3, lhp3, rhp3, _ = local_refine(decay1, regress1, years, label + " step3")

    candidates = [
        (decay1, RPC_R_PROD, avg1, lhp1, rhp1),
        (decay1, regress1, avg2, lhp2, rhp2),
        (decay3, regress3, avg3, lhp3, rhp3),
    ]
    best = max(candidates, key=lambda t: t[2] if t[2] is not None else -1.0)
    diag = {
        "step1_decay": decay1, "step1_avg": avg1,
        "step2_regress": regress1, "step2_avg": avg2,
        "step3_decay": decay3, "step3_regress": regress3, "step3_avg": avg3,
    }
    return best[0], best[1], best[2], best[3], best[4], diag


# -------------------------------------------------------
# CHECKPOINT HELPERS
# -------------------------------------------------------

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -------------------------------------------------------
# CANARY
# -------------------------------------------------------

def run_canary():
    base_all, base_lhp, base_rhp = run_all(1.0, RPC_R_PROD, ALL_YEARS)
    print("CANARY CHECK (DECAY_CAREER_RHP=1.0, REGRESS_PA_CAREER_RHP=%d vs known production baseline):" % RPC_R_PROD)
    print("  all: %.4f  (expected %.4f, delta %+.6f)" % (base_all, KNOWN_BASELINE_ALL, base_all - KNOWN_BASELINE_ALL))
    print("  lhp: %.4f  (expected %.4f, delta %+.6f)" % (base_lhp, KNOWN_BASELINE_LHP, base_lhp - KNOWN_BASELINE_LHP))
    print("  rhp: %.4f  (expected %.4f, delta %+.6f)" % (base_rhp, KNOWN_BASELINE_RHP, base_rhp - KNOWN_BASELINE_RHP))
    ok = (round(base_all, 4) == KNOWN_BASELINE_ALL and
          round(base_lhp, 4) == KNOWN_BASELINE_LHP and
          round(base_rhp, 4) == KNOWN_BASELINE_RHP)
    print("  CANARY " + ("PASSED -- proceeding." if ok else
                          "FAILED -- this tuner's scoring math does NOT match production. STOPPING."))
    print()
    sys.stdout.flush()
    return ok, base_all, base_lhp, base_rhp


# -------------------------------------------------------
# PER-FOLD PHASE
# -------------------------------------------------------

def run_fold_phase(fold_idx):
    ok, _, _, _ = run_canary()
    if not ok:
        print("=== ABORTED (canary check failed) ===")
        return None

    fold = CV_FOLDS[fold_idx - 1]
    train_years, test_year = fold["train"], fold["test"]
    label = "Fold %d" % fold_idx
    print("=" * 60)
    print("%s: train=%s  test=%d" % (label, train_years, test_year))
    print("=" * 60)

    base_train_all, base_train_lhp, base_train_rhp = run_all(1.0, RPC_R_PROD, train_years)
    base_test_all, base_test_lhp, base_test_rhp = run_all(1.0, RPC_R_PROD, [test_year])
    print("  BASELINE (decay=1.0, regress_pa=%d) train (all/lhp/rhp): %.4f / %.4f / %.4f" %
          (RPC_R_PROD, base_train_all, base_train_lhp, base_train_rhp))
    print("  BASELINE (decay=1.0, regress_pa=%d) test  (all/lhp/rhp): %.4f / %.4f / %.4f" %
          (RPC_R_PROD, base_test_all, base_test_lhp, base_test_rhp))
    print()

    decay_best, regress_best, train_avg, train_lhp, train_rhp, diag = joint_search(train_years, label)
    test_all, test_lhp, test_rhp = run_all(decay_best, regress_best, [test_year])
    print()
    print("  TUNED DECAY_CAREER_RHP=%.3f  REGRESS_PA_CAREER_RHP=%d" % (decay_best, regress_best))
    print("  TUNED train (all/lhp/rhp): %.4f / %.4f / %.4f" % (train_avg, train_lhp, train_rhp))
    print("  TUNED test  (all/lhp/rhp): %.4f / %.4f / %.4f" % (test_all, test_lhp, test_rhp))
    print("  Test delta vs baseline (all): %+.4f" % (test_all - base_test_all))
    print("  Test delta vs baseline (lhp): %+.4f" % (test_lhp - base_test_lhp))
    print("  Test delta vs baseline (rhp): %+.4f" % (test_rhp - base_test_rhp))

    base_cutoffs_r = per_cutoff_avg(1.0, RPC_R_PROD, [test_year], 'R')
    tuned_cutoffs_r = per_cutoff_avg(decay_best, regress_best, [test_year], 'R')
    print("  Per-cutoff RHP test r (baseline -> tuned):")
    for co in CUTOFFS:
        bv, tv = base_cutoffs_r[co], tuned_cutoffs_r[co]
        if bv is not None and tv is not None:
            print("    %-8s  %.4f -> %.4f  (%+.4f)" % (co, bv, tv, tv - bv))
    print()

    result = {
        "fold": fold_idx, "train_years": train_years, "test_year": test_year,
        "decay_rhp": decay_best, "regress_pa_rhp": regress_best,
        "step1_decay": diag["step1_decay"], "step2_regress": diag["step2_regress"],
        "step3_decay": diag["step3_decay"], "step3_regress": diag["step3_regress"],
        "base_test_all": base_test_all, "base_test_lhp": base_test_lhp, "base_test_rhp": base_test_rhp,
        "tuned_test_all": test_all, "tuned_test_lhp": test_lhp, "tuned_test_rhp": test_rhp,
    }

    checkpoint = load_checkpoint()
    checkpoint.setdefault("folds", {})[str(fold_idx)] = result
    save_checkpoint(checkpoint)
    print("Checkpoint saved (fold %d)." % fold_idx)
    return result


# -------------------------------------------------------
# FINAL PHASE
# -------------------------------------------------------

def run_final_phase():
    ok, base_all, base_lhp, base_rhp = run_canary()
    if not ok:
        print("=== ABORTED (canary check failed) ===")
        return

    checkpoint = load_checkpoint()
    folds = checkpoint.get("folds", {})
    missing = [i for i in (1, 2, 3) if str(i) not in folds]
    if missing:
        print("ERROR: final phase requires fold(s) %s to be checkpointed first. Run those phases before 'final'." % missing)
        return

    fold_results = [folds[str(i)] for i in (1, 2, 3)]

    print("=" * 60)
    print("CROSS-FOLD SUMMARY -- this is the generalization evidence")
    print("=" * 60)
    print()
    print("fold  test_yr  DECAY_RHP  RPC_RHP   base_all  tuned_all    delta  |  base_rhp tuned_rhp")
    for r in fold_results:
        d_all = r["tuned_test_all"] - r["base_test_all"]
        print("%4d  %7d   %7.3f  %7d   %8.4f  %9.4f  %+.4f  |  %8.4f %9.4f" % (
            r["fold"], r["test_year"], r["decay_rhp"], r["regress_pa_rhp"],
            r["base_test_all"], r["tuned_test_all"], d_all,
            r["base_test_rhp"], r["tuned_test_rhp"]))
    n = len(fold_results)
    avg_base_all  = sum(r["base_test_all"] for r in fold_results) / n
    avg_tuned_all = sum(r["tuned_test_all"] for r in fold_results) / n
    avg_decay     = sum(r["decay_rhp"] for r in fold_results) / n
    avg_regress   = sum(r["regress_pa_rhp"] for r in fold_results) / n
    print("-" * 60)
    print(" avg                %7.3f  %7.1f   %8.4f  %9.4f  %+.4f" % (
        avg_decay, avg_regress, avg_base_all, avg_tuned_all, avg_tuned_all - avg_base_all))
    print()
    print("Per-fold (decay, regress_pa) pairs, for stability/convergence assessment:")
    for r in fold_results:
        print("  Fold %d: (decay=%.3f, regress_pa=%d)  [step1 decay@regress=25: %.3f -> step2 regress@that decay: %d -> step3 refine: (%.3f, %d)]" % (
            r["fold"], r["decay_rhp"], r["regress_pa_rhp"],
            r["step1_decay"], r["step2_regress"], r["step3_decay"], r["step3_regress"]))
    print()
    print("Ship bar (same as every other tuner in this project): only ship if the test")
    print("delta is consistently positive AND the per-fold (decay, regress_pa) pairs land")
    print("in a similar place. Drifting/non-converging pairs or a flat-to-negative average")
    print("delta means this doesn't generalize -- not a bug.")
    print()

    print("=" * 60)
    print("FINAL PRODUCTION FIT (trained on all %d years, no held-out year)" % len(ALL_YEARS))
    print("=" * 60)
    print("Uses every available year for max data. Trust in this value comes from the CV")
    print("evidence above, not a fresh holdout -- there isn't one left.")
    print()
    final_decay, final_regress, final_avg, final_lhp, final_rhp, diag = joint_search(ALL_YEARS, "Final")
    print()
    print("  BASELINE in-sample (all/lhp/rhp): %.4f / %.4f / %.4f" % (base_all, base_lhp, base_rhp))
    print("  Final fit in-sample (all/lhp/rhp): %.4f / %.4f / %.4f" % (final_avg, final_lhp, final_rhp))
    print("  Delta vs baseline (all): %+.4f" % (final_avg - base_all))
    print()

    base_cutoffs_r = per_cutoff_avg(1.0, RPC_R_PROD, ALL_YEARS, 'R')
    final_cutoffs_r = per_cutoff_avg(final_decay, final_regress, ALL_YEARS, 'R')
    print("  Per-cutoff RHP in-sample r (baseline -> final):")
    for co in CUTOFFS:
        bv, tv = base_cutoffs_r[co], final_cutoffs_r[co]
        if bv is not None and tv is not None:
            print("    %-8s  %.4f -> %.4f  (%+.4f)" % (co, bv, tv, tv - bv))
    print()

    print("=" * 60)
    print("TUNED PARAMETERS -- paste this into backtest.py")
    print("=" * 60)
    print()
    print("DECAY_CAREER_RHP = %s" % round(final_decay, 3))
    print("REGRESS_PA_CAREER_RHP = %s" % round(final_regress))
    print()
    print("--- UNCHANGED (frozen, still current production) ---")
    print("AGING_STRENGTH = 0.0 (unchanged)")
    print("LEAGUE_AVG_WOBA_LHP=%s  LEAGUE_AVG_WOBA_RHP=%s" % (LA_LHP, LA_RHP))
    print("WEIGHT_SPLITS_LHP=%s  WEIGHT_CAREER_LHP=%s  WEIGHT_TOTAL_LHP=%s  WEIGHT_RECENT_LHP=%s" % (WS_L, WC_L, WT_L, WR_L))
    print("REGRESS_PA_SPLITS_LHP=%s  REGRESS_PA_CAREER_LHP=%s  REGRESS_PA_TOTAL_LHP=%s  REGRESS_PA_RECENT_LHP=%s" % (RPS_L, RPC_L, RPT_L, RPR_L))
    print("WEIGHT_SPLITS_RHP=%s  WEIGHT_CAREER_RHP=%s  WEIGHT_TOTAL_RHP=%s  WEIGHT_RECENT_RHP=%s" % (WS_R, WC_R, WT_R, WR_R))
    print("REGRESS_PA_SPLITS_RHP=%s  REGRESS_PA_TOTAL_RHP=%s  REGRESS_PA_RECENT_RHP=%s (unchanged)" % (RPS_R, RPT_R, RPR_R))
    print("DECAY_CAREER_LHP=%s (unchanged)" % DECAY_CAREER_LHP)
    print("PLATOON_BLEND_LHP=%s  PLATOON_FACTOR_MIN_LHP=%s  PLATOON_FACTOR_MAX_LHP=%s" % (PLATOON_BLEND_LHP, PLATOON_MIN_LHP, PLATOON_MAX_LHP))
    print("PLATOON_BLEND_RHP=%s  PLATOON_FACTOR_MIN_RHP=%s  PLATOON_FACTOR_MAX_RHP=%s" % (PLATOON_BLEND_RHP, PLATOON_MIN_RHP, PLATOON_MAX_RHP))
    print("XWOBA_BLEND_{SPLITS,CAREER,TOTAL,RECENT}_LHP = 0.9187 (all 4, unchanged)")
    print("XWOBA_BLEND_{SPLITS,CAREER,TOTAL,RECENT}_RHP = 0.80   (all 4, unchanged)")
    print()

    checkpoint["final"] = {
        "decay_rhp": final_decay, "regress_pa_rhp": final_regress,
        "final_avg": final_avg, "final_lhp": final_lhp, "final_rhp": final_rhp,
        "base_all": base_all, "base_lhp": base_lhp, "base_rhp": base_rhp,
    }
    save_checkpoint(checkpoint)
    print("=== DONE ===")


# -------------------------------------------------------
# MAIN / CLI DISPATCH
# -------------------------------------------------------

def print_header():
    print("=== TUNER DECAY x REGRESS RHP -- DECAY_CAREER_RHP x REGRESS_PA_CAREER_RHP (Task #81) ===")
    print("Results saved to: " + OUTPUT_FILE)
    print("CV folds (rolling-origin; test year is never in that fold's training years):")
    for i, f in enumerate(CV_FOLDS, 1):
        print("  Fold %d: train=%s  test=%d" % (i, f["train"], f["test"]))
    print("Final fit: train=%s  (no holdout -- production candidate)" % ALL_YEARS)
    print("2 tunable params (DECAY_CAREER_RHP, REGRESS_PA_CAREER_RHP), searched via")
    print("coordinate ascent (decay grid @ regress=25 -> regress grid @ tuned decay)")
    print("plus a small local 2-D refinement grid. See module docstring for the full")
    print("rationale and bounds.")
    print("Objective is the COMBINED (LHP+RHP pooled) SIMPLE average r across all")
    print("windows -- matches backtest.py's own \"Overall average r\" formula exactly.")
    print("Only the RHP career build/regression changes -- LHP side frozen throughout.")
    print()
    sys.stdout.flush()


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else None
    t0 = time.time()

    if phase is None:
        print_header()
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        for i in (1, 2, 3):
            if run_fold_phase(i) is None:
                return
        run_final_phase()
    elif phase == "fold1":
        print_header()
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        run_fold_phase(1)
    elif phase == "fold2":
        run_fold_phase(2)
    elif phase == "fold3":
        run_fold_phase(3)
    elif phase == "final":
        run_final_phase()
    elif phase == "canary":
        # Ad hoc smoke-test hook: canary check only, no fold work, no
        # checkpoint writes. Not part of the documented CLI surface.
        run_canary()
    else:
        print("Unknown phase: '%s'. Valid: (none), fold1, fold2, fold3, final" % phase)
        return

    elapsed = time.time() - t0
    print("Phase elapsed: %.0fs (%.1f min)" % (elapsed, elapsed / 60.0))


_phase_arg = sys.argv[1] if len(sys.argv) > 1 else None
_tee_mode = "a" if _phase_arg in ("fold2", "fold3", "final") else "w"
tee = Tee(OUTPUT_FILE, mode=_tee_mode)
sys.stdout = tee
try:
    main()
except Exception as e:
    print("ERROR: " + str(e))
    import traceback
    traceback.print_exc()
tee.close()
sys.stdout = tee.console
