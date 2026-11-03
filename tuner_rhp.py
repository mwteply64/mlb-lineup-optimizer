#!/usr/bin/env python3
"""
Tuner - RHP-focused retune, fully isolated from LHP
=====================================================================
Mirror image of tuner_lhp.py (v5), with every role swapped: RHP is now
the tunable side, LHP is now fully frozen. Built after backtest.py was
updated to v10 (LHP retuned + per-hand platoon/xwOBA split shipped),
which is what makes this possible -- RHP can now be searched without
touching a single LHP-labeled constant.

WHY THIS IS A NEW FILE, NOT A FLAG ON tuner_lhp.py:
tuner_lhp.py's docstring explains why a blended LHP+RHP objective was
tried and rejected during the LHP retune: weighting the objective
toward one side doesn't protect the OTHER side from overfitting on its
own, because most of that side's parameters have zero effect on the
side being protected -- any improvement on the unprotected side gets
accepted regardless of the weight. The same problem applies in
reverse here, so RHP gets its own dedicated walk-forward-CV tuner with
an RHP-only objective, exactly like LHP got, rather than re-enabling
RHP inside tuner_lhp.py.

FROZEN BASELINE IS v10, NOT THE OLD v8/v9 DEFAULTS (Task #2 from the
pre-RHP-tuning punch list):
tuner_lhp.py's own START dict freezes LHP at the OLD pre-retune values
(LA_LHP=0.2585, DECAY_CAREER_LHP=1.0, etc.) because that tuner's job
was to FIND the new LHP values. Now that backtest.py ships those new
values, this tuner's frozen LHP baseline uses them instead:
LA_LHP=0.2943, DECAY_CAREER_LHP=0.8001, XWOBA_BLEND_LHP=0.9187, and
the rest of the v10 LHP weight/regress-PA/platoon block. This matters
beyond cosmetics: RHP's platoon factor needs LHP's career wOBA as its
cross-hand input, and that career data is built using
DECAY_CAREER_LHP. Freezing it at the stale 1.0 instead of the real
0.8001 would mean every RHP prediction in this search is computed
against a cross-hand input that doesn't match what backtest.py
actually ships -- silently invalid baseline and search results.

CROSS-HAND LEAK FIX, MIRRORED (Task #3):
tuner_lhp.py discovered that even with every RHP-labeled constant
frozen, RHP's platoon factor still needed LHP's career wOBA, and that
LHP career data was being rebuilt every trial using the tunable
DECAY_CAREER_LHP -- so tuning LHP's decay was silently nudging RHP's
"frozen" predictions too. Fixed there with a second, frozen copy of
LHP career data used only as RHP's cross-hand input. The mirror image
applies here: LHP's platoon factor needs RHP's career wOBA as ITS
cross-hand input, and RHP's career data is now being rebuilt every
trial using the tunable DECAY_CAREER_RHP. Fixed the same way: car_r_base
is a second, frozen copy of RHP career data (built once, using START's
DECAY_CAREER_RHP -- the current backtest.py production value, 1.0)
used ONLY as LHP's cross-hand input. The tunable copy (car_r) continues
to feed RHP's own predictions. LHP's predictions are now bit-for-bit
identical to baseline no matter what the RHP search does -- the same
canary property tuner_lhp.py validated for RHP, now checked for LHP.

Conversely, LHP's OWN career data (car_l) needs no such split here:
DECAY_CAREER_LHP is frozen (not tunable in this file at all), so car_l
never changes across trials. It's computed once per year and reused
both for LHP's own prediction and as RHP's cross-hand input -- exactly
how tuner_lhp.py treated car_r, since DECAY_CAREER_RHP was frozen there.

SANITY-CHECKED STARTING POINT (Task #4):
REGRESS_PA_CAREER_RHP=25 looked far out of line with every other
regress-PA value in START (178-1020) before trusting it as this
search's starting point. A direct empirical check (sweeping RPC_R from
1 to 2000 through backtest.py's actual scoring, in-sample across all
18 windows) found weighted-avg RHP r moving monotonically from 0.4078
(RPC_R=1) down to 0.3929 (RPC_R=2000) -- i.e. the in-sample metric
genuinely prefers less regression, all the way to the search floor.
That is exactly the failure mode tuner_lhp.py's docstring warns about:
in an earlier joint 24-param tuner (predating both LHP- and RHP-only
tuners), REGRESS_PA_CAREER_RHP collapsed to its lower bound -- a
degenerate, likely-overfit solution that an in-sample-only objective
can't distinguish from genuine signal. RPC_R=25 is not a corrupted
value (the regression math behaves sanely there, and it sits on a
fairly flat part of the curve, not a cliff edge) but the in-sample
preference for "less regression" is real and unverified out-of-sample.
This is precisely why this tuner uses walk-forward CV with a true
held-out test year, same as tuner_lhp.py -- if RPC_R genuinely wants to
shrink, the CV test-year deltas will show it generalizing; if it's
overfit noise, CV will catch a fold where train improves and test
doesn't (the same diagnostic that caught LHP's fold-1 issue during the
LHP retune). The search bounds below intentionally still allow RPC_R
down to 1 -- the point is to let CV decide, not to manually clamp away
the failure mode before measuring it.

TUNES (14 parameters, all RHP-specific):
  LEAGUE_AVG_WOBA_RHP                                        (1)
  WEIGHT_{SPLITS,CAREER,TOTAL,RECENT}_RHP                    (4)
  REGRESS_PA_{SPLITS,CAREER,TOTAL,RECENT}_RHP                (4)
  DECAY_CAREER_RHP                                           (1)
  PLATOON_BLEND_RHP, PLATOON_FACTOR_MIN_RHP, PLATOON_FACTOR_MAX_RHP (3)
  XWOBA_BLEND_RHP                                             (1)

FROZEN AT v10 PRODUCTION BASELINE (not tunable, by design -- everything
LHP-specific):
  LEAGUE_AVG_WOBA_LHP=0.2943, WEIGHT_*_LHP, REGRESS_PA_*_LHP,
  DECAY_CAREER_LHP=0.8001, PLATOON_BLEND_LHP=0.7833,
  PLATOON_FACTOR_MIN_LHP=0.612, PLATOON_FACTOR_MAX_LHP=1.138,
  XWOBA_BLEND_LHP=0.9187

PA minimums fixed at 30 (not tunable). NO_CLAMP fixed True.

RUNTIME: LHP career data is cached once per year (DECAY_CAREER_LHP is
frozen), so the cost shape matches tuner_lhp.py: N_RESTARTS independent
searches per fold/final. Expect roughly 45-75 minutes total.

USAGE:
  Double-click run_tuner_rhp.bat (writes tuner_rhp_results_YYYY-MM-DD.txt)
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
OUTPUT_FILE = os.path.join(FOLDER, "tuner_rhp_results_" + today + ".txt")


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
# strictly after every year in its training set. Same folds as
# tuner_lhp.py for direct comparability.
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
# LHP career data never changes here (DECAY_CAREER_LHP is frozen at the
# v10 production value), so it's safe to precompute once per year and
# reuse across every trial -- saves real time over the search since we
# don't rebuild it per trial. Mirrors how tuner_lhp.py treated RHP
# career data (frozen there since DECAY_CAREER_RHP wasn't tunable).

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
# SCORING MATH (mirrors backtest.py / tuner_lhp.py exactly)
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

def run_window(p, year, co, car_l, car_r, car_r_base):
    """Returns (r_l, r_r, pa_l, pa_r): the LHP/RHP Spearman r for this
    window, plus the total post-cutoff PA backing each side's r (used
    by run_all to PA-weight windows instead of averaging them equally).

    car_r is the TUNABLE RHP career data (rebuilt per trial using the
    candidate's DECAY_CAREER_RHP) -- used for RHP's own prediction.
    car_r_base is a FROZEN copy (always built with START's
    DECAY_CAREER_RHP) used only as the cross-hand input to LHP's
    platoon factor, so LHP's predictions never move when the RHP
    search perturbs DECAY_CAREER_RHP. car_l is fully frozen (LHP isn't
    tunable in this file at all, including its decay) and is used both
    for LHP's own prediction and as RHP's cross-hand input."""
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
        cr_base = car_r_base.get(pid)
        tot, rec = total.get(pid), recent.get(pid)

        sl_pa = sl["pa"] if sl else 0
        sr_pa = sr["pa"] if sr else 0
        cl_pa = cl["pa"] if cl else 0
        cr_pa = cr["pa"] if cr else 0
        cr_base_pa = cr_base["pa"] if cr_base else 0
        tot_pa = tot["pa"] if tot else 0
        rec_pa = rec["pa"] if rec else 0

        # Each side blends actual/xwOBA with its own knob. The same
        # underlying total/recent/cross-hand-career rows get blended
        # twice (once per side) since XWOBA_BLEND_LHP and
        # XWOBA_BLEND_RHP can differ.

        # LHP (frozen canary): own component = car_l (always frozen);
        # cross-hand input = car_r_base (frozen baseline, NOT the
        # tunable car_r) -- this is what keeps LHP's score bit-for-bit
        # identical no matter what the RHP search does.
        sl_w   = blend_woba(sl, xb_l)
        cl_w_l = blend_woba(cl, xb_l)
        cr_w_l = blend_woba(cr_base, xb_l)
        tot_w_l = blend_woba(tot, xb_l)
        rec_w_l = blend_woba(rec, xb_l)

        # RHP (tunable): own component = car_r (this trial's candidate
        # DECAY_CAREER_RHP); cross-hand input = car_l (fully frozen,
        # since LHP isn't tunable here at all).
        sr_w   = blend_woba(sr, xb_r)
        cr_w_r = blend_woba(cr, xb_r)
        cl_w_r = blend_woba(cl, xb_r)
        tot_w_r = blend_woba(tot, xb_r)
        rec_w_r = blend_woba(rec, xb_r)

        has_split_l = (sl_pa >= MIN_PA_SPLITS) or (cl_pa >= MIN_PA_CAREER)
        if has_split_l:
            post = post_l.get(pid)
            if post is not None and post["pa"] >= MIN_PA_POST:
                pred = stabilize(p, sl_pa, sl_w, cl_pa, cl_w_l, tot_pa, tot_w_l,
                                  rec_pa, rec_w_l, 'L',
                                  career_cross_pa=cr_base_pa, career_cross_woba=cr_w_l)
                if pred is not None:
                    pred_l.append(pred); act_l.append(post["woba"]); pa_l_sum += post["pa"]

        has_split_r = (sr_pa >= MIN_PA_SPLITS) or (cr_pa >= MIN_PA_CAREER)
        if has_split_r:
            post = post_r.get(pid)
            if post is not None and post["pa"] >= MIN_PA_POST:
                pred = stabilize(p, sr_pa, sr_w, cr_pa, cr_w_r, tot_pa, tot_w_r,
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


def run_all(p, years, car_l_cache, car_r_base_cache):
    """LHP career data (car_l_cache) and the frozen RHP cross-hand
    career data (car_r_base_cache) are passed in pre-built since
    neither ever changes across trials; only the TUNABLE RHP career
    data needs rebuilding when DECAY_CAREER_RHP is perturbed. Window-
    level Spearman r values are combined into weighted averages, using
    sqrt(total post-cutoff PA) as the weight -- same rationale as
    tuner_lhp.py: raw PA would let June windows (more season left)
    outweigh August windows by ~2x for reasons having nothing to do
    with estimate reliability; sqrt damps that to ~1.4x."""
    all_rs, all_ws = [], []
    lhp_rs, lhp_ws = [], []
    rhp_rs, rhp_ws = [], []
    for year in years:
        car_r = build_career_data(year, "right", p["DECAY_CAREER_RHP"])
        car_l = car_l_cache[year]
        car_r_base = car_r_base_cache[year]
        for co in CUTOFFS:
            r_l, r_r, pa_l, pa_r = run_window(p, year, co, car_l, car_r, car_r_base)
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


def per_cutoff_rhp_avg(p, years, car_l_cache, car_r_base_cache):
    """Diagnostic only -- NOT used in the objective. Plain (unweighted)
    average RHP r per cutoff, so a real June-vs-August tradeoff from
    the sqrt-PA weighting would show up directly instead of being
    hidden inside one blended number."""
    by_cutoff = {co: [] for co in CUTOFFS}
    for year in years:
        car_r = build_career_data(year, "right", p["DECAY_CAREER_RHP"])
        car_l = car_l_cache[year]
        car_r_base = car_r_base_cache[year]
        for co in CUTOFFS:
            r_l, r_r, pa_l, pa_r = run_window(p, year, co, car_l, car_r, car_r_base)
            if r_r is not None:
                by_cutoff[co].append(r_r)
    return {co: (sum(v) / len(v) if v else None) for co, v in by_cutoff.items()}


# -------------------------------------------------------
# PARAMETER SPACE
# -------------------------------------------------------

BOUNDS = {
    "LA_RHP": (0.20, 0.34, 0.01),
    "WS_R": (0.0, 1.5, 0.05), "WC_R": (0.0, 1.5, 0.05),
    "WT_R": (0.0, 1.5, 0.05), "WR_R": (0.0, 1.5, 0.05),
    "RPS_R": (1, 2500, 40), "RPC_R": (1, 2500, 40),
    "RPT_R": (1, 2500, 40), "RPR_R": (1, 2500, 40),
    "DECAY_CAREER_RHP": (0.40, 1.0, 0.03),

    "PLATOON_BLEND_RHP": (0.0, 1.0, 0.04),
    "PLATOON_MIN_RHP":   (0.30, 0.99, 0.03),
    "PLATOON_MAX_RHP":   (1.01, 2.20, 0.04),
    "XWOBA_BLEND_RHP":   (0.0, 1.0, 0.05),
}

PARAM_NAMES = list(BOUNDS.keys())  # 14 tunable; LHP-specific NOT included -> frozen

# Baseline: LHP side frozen at v10 PRODUCTION values (the ones now
# shipped in backtest.py after the LHP retune) -- NOT the stale
# pre-retune defaults tuner_lhp.py started from. RHP side starts from
# backtest.py's current (still-untuned) values, same as before.
START = {
    "LA_LHP": 0.2943, "LA_RHP": 0.2575,
    "WS_L": 0.213, "WC_L": 0.5695, "WT_L": 0.247, "WR_L": 0.3416,
    "RPS_L": 178, "RPC_L": 711, "RPT_L": 1020, "RPR_L": 757,
    "WS_R": 0.2774, "WC_R": 0.2920, "WT_R": 0.2153, "WR_R": 0.2153,
    "RPS_R": 285, "RPC_R": 25, "RPT_R": 400, "RPR_R": 665,
    "PLATOON_BLEND_LHP": 0.7833, "PLATOON_MIN_LHP": 0.612, "PLATOON_MAX_LHP": 1.138,
    "PLATOON_BLEND_RHP": 0.31, "PLATOON_MIN_RHP": 0.585, "PLATOON_MAX_RHP": 1.325,
    "DECAY_CAREER_LHP": 0.8001, "DECAY_CAREER_RHP": 1.0,
    "XWOBA_BLEND_LHP": 0.9187, "XWOBA_BLEND_RHP": 0.5,
}

FROZEN_LHP_KEYS = ["LA_LHP", "WS_L", "WC_L", "WT_L", "WR_L",
                    "RPS_L", "RPC_L", "RPT_L", "RPR_L", "DECAY_CAREER_LHP",
                    "PLATOON_BLEND_LHP", "PLATOON_MIN_LHP", "PLATOON_MAX_LHP",
                    "XWOBA_BLEND_LHP"]


def clamp(name, val):
    lo, hi, _ = BOUNDS[name]
    val = max(lo, min(hi, val))
    if name.startswith("RP"):
        val = round(val)
    return val


def fix_platoon_order(p):
    if p["PLATOON_MIN_RHP"] >= 1.0:
        p["PLATOON_MIN_RHP"] = 0.99
    if p["PLATOON_MAX_RHP"] <= 1.0:
        p["PLATOON_MAX_RHP"] = 1.01


# -------------------------------------------------------
# OPTIMIZER (objective = RHP-only train r, scoped to a given year list)
# -------------------------------------------------------

def make_score_fn(years, car_l_cache, car_r_base_cache):
    def score_train_rhp(p):
        avg, lhp_avg, rhp_avg = run_all(p, years, car_l_cache, car_r_base_cache)
        return rhp_avg if rhp_avg is not None else -1.0
    return score_train_rhp


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
            print("    [random %5d/%d] best_rhp_r=%.4f  accepts=%d  elapsed=%.0fs" %
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
        print("    [coord pass %d] best_rhp_r=%.4f%s" %
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
SEED = 20260618
N_RESTARTS = 3  # independent random-seed attempts per search; keeps the
                # best of N as a cheap check against local optima.


def run_search(years, car_l_cache, car_r_base_cache, seed, label):
    """Runs the full 4-phase search with the RHP-only objective scoped
    to `years`. Returns (tuned_params, tuned_train_score)."""
    rng = random.Random(seed)
    score_fn = make_score_fn(years, car_l_cache, car_r_base_cache)
    cur = dict(START)
    cur_score = score_fn(cur)
    print("[%s] years=%s  starting RHP-only train r: %.4f" % (label, years, cur_score))
    sys.stdout.flush()

    print("  --- Phase 1: random search (%d iters) ---" % N_RANDOM_ITERS_PHASE1)
    cur, cur_score = random_search(cur, cur_score, N_RANDOM_ITERS_PHASE1, rng, score_fn, start_time=time.time())

    print("  --- Phase 2: coordinate-ascent polish ---")
    cur, cur_score = coordinate_sweep(cur, cur_score, rng, score_fn, max_passes=4)

    print("  --- Phase 3: random search round 2 (%d iters, fine-grained) ---" % N_RANDOM_ITERS_PHASE2)
    cur, cur_score = random_search(cur, cur_score, N_RANDOM_ITERS_PHASE2, rng, score_fn, start_time=time.time())

    print("  --- Phase 4: final coordinate-ascent polish ---")
    cur, cur_score = coordinate_sweep(cur, cur_score, rng, score_fn, max_passes=4)
    print("[%s] final RHP-only train r: %.4f" % (label, cur_score))
    print()
    sys.stdout.flush()
    return cur, cur_score


def run_search_with_restarts(years, car_l_cache, car_r_base_cache, base_seed, label, n_restarts=N_RESTARTS):
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
        p, score = run_search(years, car_l_cache, car_r_base_cache, seed, sub_label)
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

    print("=== TUNER RHP -- RHP-focused retune, fully isolated from LHP ===")
    print("Results saved to: " + OUTPUT_FILE)
    print("CV folds (rolling-origin; test year is never in that fold's training years):")
    for i, f in enumerate(CV_FOLDS, 1):
        print("  Fold %d: train=%s  test=%d" % (i, f["train"], f["test"]))
    print("Final fit: train=%s  (no holdout -- production candidate)" % ALL_YEARS)
    print("%d tunable params per run (all RHP-specific)." % len(PARAM_NAMES))
    print("LHP-specific params FROZEN at v10 production baseline: " + ", ".join(FROZEN_LHP_KEYS))
    print("Each search runs %d independent random-seed restarts, keeping the best." % N_RESTARTS)
    print("Window averages are weighted by sqrt(total post-cutoff PA).")
    print("Expect roughly 45-75 minutes total (4 searches x %d restarts each)." % N_RESTARTS)
    print()

    # LHP career data never changes (decay frozen at v10's 0.8001) --
    # precompute once per year. Same for the FROZEN copy of RHP career
    # data used only as LHP's cross-hand platoon input (see run_window
    # docstring) -- it always uses START's DECAY_CAREER_RHP, never the
    # trial's tunable value.
    car_l_cache = {}
    car_r_base_cache = {}
    for year in TEST_YEARS:
        car_l_cache[year] = build_career_data(year, "left", START["DECAY_CAREER_LHP"])
        car_r_base_cache[year] = build_career_data(year, "right", START["DECAY_CAREER_RHP"])

    def full_eval(p, years):
        return run_all(p, years, car_l_cache, car_r_base_cache)

    def cutoff_eval(p, years):
        return per_cutoff_rhp_avg(p, years, car_l_cache, car_r_base_cache)

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

        tuned_p, tuned_train_score = run_search_with_restarts(train_years, car_l_cache, car_r_base_cache, SEED + i, label)

        tuned_train_all, tuned_train_lhp, tuned_train_rhp = full_eval(tuned_p, train_years)
        tuned_test_all, tuned_test_lhp, tuned_test_rhp = full_eval(tuned_p, [test_year])
        print("  TUNED   train (lhp/rhp/all): %.4f / %.4f / %.4f" % (tuned_train_lhp, tuned_train_rhp, tuned_train_all))
        print("  TUNED   test  (lhp/rhp/all): %.4f / %.4f / %.4f" % (tuned_test_lhp, tuned_test_rhp, tuned_test_all))
        print("  RHP test delta vs baseline: %+.4f" % (tuned_test_rhp - base_test_rhp))
        print("  LHP test delta vs baseline: %+.6f  (should be EXACTLY 0 now -- LHP fully frozen/isolated)" % (tuned_test_lhp - base_test_lhp))

        base_cutoffs = cutoff_eval(START, [test_year])
        tuned_cutoffs = cutoff_eval(tuned_p, [test_year])
        print("  Per-cutoff RHP test r (baseline -> tuned), watch for June gains at August's expense:")
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
    print("fold  test_yr   base_rhp  tuned_rhp    delta   |  base_lhp  tuned_lhp    delta")
    for r in fold_results:
        delta_rhp = r["tuned_test_rhp"] - r["base_test_rhp"]
        delta_lhp = r["tuned_test_lhp"] - r["base_test_lhp"]
        print("%4d  %7d   %8.4f  %9.4f  %+.4f   |  %8.4f  %9.4f  %+.4f" % (
            r["fold"], r["test_year"], r["base_test_rhp"], r["tuned_test_rhp"], delta_rhp,
            r["base_test_lhp"], r["tuned_test_lhp"], delta_lhp))
    n = len(fold_results)
    avg_base_rhp  = sum(r["base_test_rhp"] for r in fold_results) / n
    avg_tuned_rhp = sum(r["tuned_test_rhp"] for r in fold_results) / n
    avg_base_lhp  = sum(r["base_test_lhp"] for r in fold_results) / n
    avg_tuned_lhp = sum(r["tuned_test_lhp"] for r in fold_results) / n
    print("-" * 78)
    print(" avg            %8.4f  %9.4f  %+.4f   |  %8.4f  %9.4f  %+.4f" % (
        avg_base_rhp, avg_tuned_rhp, avg_tuned_rhp - avg_base_rhp,
        avg_base_lhp, avg_tuned_lhp, avg_tuned_lhp - avg_base_lhp))
    print()
    print("If the RHP delta is positive and similar in size across all 3 rows,")
    print("the RHP retune generalizes across different out-of-sample years --")
    print("not just an artifact of one holdout split. Watch REGRESS_PA_CAREER_RHP")
    print("specifically across folds (see per-fold params below): if it collapses")
    print("toward the search floor (1) AND the test-year delta is flat or negative")
    print("in any fold, that's the degenerate overfit pattern flagged in this")
    print("file's docstring, not genuine signal. The LHP delta should be EXACTLY 0")
    print("in every row now (platoon/xwOBA are split AND the cross-hand platoon")
    print("input is frozen, so nothing tunable here touches LHP's predictions).")
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
    final_p, final_train_score = run_search_with_restarts(ALL_YEARS, car_l_cache, car_r_base_cache, SEED + 99, "Final")
    final_all, final_lhp, final_rhp = full_eval(final_p, ALL_YEARS)
    print("  Final fit in-sample (lhp/rhp/all): %.4f / %.4f / %.4f" % (final_lhp, final_rhp, final_all))
    base_cutoffs = cutoff_eval(START, ALL_YEARS)
    final_cutoffs = cutoff_eval(final_p, ALL_YEARS)
    print("  Per-cutoff RHP in-sample r (baseline -> final):")
    for co in CUTOFFS:
        bv, tv = base_cutoffs[co], final_cutoffs[co]
        if bv is not None and tv is not None:
            print("    %-8s  %.4f -> %.4f  (%+.4f)" % (co, bv, tv, tv - bv))
    print()
    print("=" * 60)
    print("TUNED PARAMETERS -- paste these into backtest.py (RHP-specific only)")
    print("=" * 60)
    print()
    print("LEAGUE_AVG_WOBA_RHP = %s" % round(final_p["LA_RHP"], 4))
    print()
    print("WEIGHT_SPLITS_RHP  = %s" % round(final_p["WS_R"], 4))
    print("WEIGHT_CAREER_RHP  = %s" % round(final_p["WC_R"], 4))
    print("WEIGHT_TOTAL_RHP   = %s" % round(final_p["WT_R"], 4))
    print("WEIGHT_RECENT_RHP  = %s" % round(final_p["WR_R"], 4))
    print()
    print("REGRESS_PA_SPLITS_RHP = %s" % int(final_p["RPS_R"]))
    print("REGRESS_PA_CAREER_RHP = %s" % int(final_p["RPC_R"]))
    print("REGRESS_PA_TOTAL_RHP  = %s" % int(final_p["RPT_R"]))
    print("REGRESS_PA_RECENT_RHP = %s" % int(final_p["RPR_R"]))
    print()
    print("DECAY_CAREER_RHP = %s" % round(final_p["DECAY_CAREER_RHP"], 4))
    print()
    print("PLATOON_BLEND_RHP      = %s" % round(final_p["PLATOON_BLEND_RHP"], 4))
    print("PLATOON_FACTOR_MIN_RHP = %s" % round(final_p["PLATOON_MIN_RHP"], 4))
    print("PLATOON_FACTOR_MAX_RHP = %s" % round(final_p["PLATOON_MAX_RHP"], 4))
    print()
    print("XWOBA_BLEND_RHP = %s" % round(final_p["XWOBA_BLEND_RHP"], 4))
    print()
    print("--- UNCHANGED (frozen LHP-specific, still v10 production) ---")
    print("LEAGUE_AVG_WOBA_LHP = %s" % START["LA_LHP"])
    print("WEIGHT_SPLITS_LHP  = %s" % START["WS_L"])
    print("WEIGHT_CAREER_LHP  = %s" % START["WC_L"])
    print("WEIGHT_TOTAL_LHP   = %s" % START["WT_L"])
    print("WEIGHT_RECENT_LHP  = %s" % START["WR_L"])
    print("REGRESS_PA_SPLITS_LHP = %s" % START["RPS_L"])
    print("REGRESS_PA_CAREER_LHP = %s" % START["RPC_L"])
    print("REGRESS_PA_TOTAL_LHP  = %s" % START["RPT_L"])
    print("REGRESS_PA_RECENT_LHP = %s" % START["RPR_L"])
    print("DECAY_CAREER_LHP = %s" % START["DECAY_CAREER_LHP"])
    print("PLATOON_BLEND_LHP      = %s" % START["PLATOON_BLEND_LHP"])
    print("PLATOON_FACTOR_MIN_LHP = %s" % START["PLATOON_MIN_LHP"])
    print("PLATOON_FACTOR_MAX_LHP = %s" % START["PLATOON_MAX_LHP"])
    print("XWOBA_BLEND_LHP = %s" % START["XWOBA_BLEND_LHP"])
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
