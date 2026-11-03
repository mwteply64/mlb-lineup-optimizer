#!/usr/bin/env python3
"""
Backtest - Lineup Optimization Historical Validation
=====================================================
Tests how well the stabilized wOBA model predicts actual future
performance across 18 historical test windows (6 seasons x 3 cutoffs).

For each window:
  Input:  stats UP TO the cutoff date  (before_X_YEAR files)
  Target: stats AFTER the cutoff date  (after_X_YEAR files)
  Metric: Spearman rank correlation between predicted and actual wOBA

SCORING LOGIC mirrors test.py, adapted for wOBA instead of wRC+:
  - Same weights, regression PA constants, and PA-based clamping
  - Clamp proportionally rescaled: wRC+ range [0,200] around 100
    becomes wOBA range [0, 0.630] around 0.315
  - Same weighted-average formula with weight renormalization when
    a component is missing (insufficient PA or no data)
  - Lineup ordering logic (slump, handedness, position) excluded --
    not needed for ranking correlation testing

IMPROVEMENTS (v8):
  #2: Separate league-average anchors for LHP and RHP models.
      These are tuned for ranking accuracy, not empirical averages
      (see CONFIGURATION section below for current values).
  #4: Platoon tendency prior -- scales total/recent wOBA by the
      player's career platoon ratio so those components better
      reflect expected performance vs that pitcher handedness.
      Currently enabled (blend > 0) -- re-tuning found a small gain.
  #5: Separate weight and regression-PA sets for LHP vs RHP models,
      allowing independent tuning of each side.

v9 DATA EXPANSION (this revision):
  - Test set expanded from 9 windows (2023-2025) to 18 windows
    (2019, 2021, 2022, 2023, 2024, 2025 -- skips COVID-shortened 2020),
    using the corrected wOBA data produced by download_v3_expansion.py.
  - Career data is no longer a fixed pre-combined 3-year file. It is
    now built at runtime from ALL available per-season files strictly
    before the test year (build_career_data()), weighted by recency via
    DECAY_CAREER_LHP/RHP. Those constants are set to 1.0 (no decay --
    plain PA-weighted average across all prior seasons) as a neutral
    placeholder; tuning them is Task #23, not done in this revision.
  - load_file() now also reads an "xwoba" column where present (season,
    before, totals_before, and recent_before files all have it; after_*
    ground-truth files do not). It is loaded and carried through
    build_career_data() for future use, but NOT YET blended into the
    stabilize() scoring formula -- that blend is Task #24.
  - The old combined_2019_2022 / 2021_2023 / 2022_2024 files are no
    longer read by this script. They are kept on disk only because
    nothing else has been confirmed to still need them.

USAGE:
  Double-click run_backtest.bat
  Results saved to backtest_results_YYYY-MM-DD.txt

TUNING:
  Change the weights or regression constants below, re-run, and
  compare the new correlation numbers to the previous baseline.
  Change ONE parameter at a time.
"""

import os
import csv
import sys
import datetime

# -------------------------------------------------------
# OUTPUT SETUP
# -------------------------------------------------------

FOLDER   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FOLDER, "historical testing")
today    = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(FOLDER, "backtest_results_" + today + ".txt")


class Tee:
    """Write to both console and file simultaneously."""
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
# CONFIGURATION
# Mirrors test.py -- change these to tune the model.
# -------------------------------------------------------

# League-average wOBA by pitcher handedness (improvement #2).
# These are regression anchors tuned for ranking accuracy, NOT
# simple empirical averages.  The optimal values for Spearman r
# turned out lower than the true MLB average (0.315) for both sides,
# suggesting the model benefits from conservative anchors that don't
# over-inflate predictions for small-sample players.
LEAGUE_AVG_WOBA_LHP = 0.2585
LEAGUE_AVG_WOBA_RHP = 0.2575

# -------------------------------------------------------
# LHP model -- weights and regression PA constants (#5)
# These apply when predicting performance vs left-handed pitchers.
# -------------------------------------------------------
# TUNING LOG (Spearman r, avg across all 9 windows):
#   v1 baseline   -- splits=0.40 career=0.225 total=0.25 recent=0.125 -> all=0.2809
#   v2 optimized  -- career weight raised to 1.00, recent cut to 0.05  -> all=0.3122
#   v3 tuned      -- NC=True, LA=0.308, WC=1.20, MPP=44               -> all=0.3225
#   v4 tuned      -- MS=10, MC=1, MPP=46, LA~0.299                    -> all=0.3288
#   v5 tuned      -- LA=0.315, all minimums=30, WC=0.90, WR=0.15,
#                    all regression PA raised to 400                   -> all=0.3151
#   v6 tuned      -- separate LA_LHP=0.275 LA_RHP=0.290, separate
#                    LHP/RHP weight and regress PA sets (#2 and #5)   -> all=0.3257
#   v7 (hand-tuned by user)                                            -> all=0.3253
#   v8 tuned      -- coordinate-ascent + random-search re-optimization
#                    of all weights/regress-PA/LA values starting from
#                    v7, plus re-enabling the platoon prior (#4) with
#                    blend=0.31 and a wider clamp range -- both turned
#                    out to help once paired with the new weight shape.
#                    Confirmed via 1500+ random trials with no further
#                    improvement found nearby.                        -> all=0.3311
#   Key insight: The two sides still benefit from very different model
#   shapes. LHP leans on career data heavily (~61% weight) with high
#   regression (735-910 PA) to fight noise. RHP spreads weight across
#   splits/career/total/recent more evenly but barely regresses career
#   data at all (25 PA) because real RHP career numbers are reliable.
#   Weights below are proportionally renormalized so each side's four
#   weights sum to 1.00 -- purely cosmetic, doesn't change predictions
#   (stabilize() already renormalizes by sum(weights) internally, so
#   scaling all four weights by a constant is a no-op on the result).
WEIGHT_SPLITS_LHP  = 0.1365
WEIGHT_CAREER_LHP  = 0.6142
WEIGHT_TOTAL_LHP   = 0.1443
WEIGHT_RECENT_LHP  = 0.1050

REGRESS_PA_SPLITS_LHP = 355
REGRESS_PA_CAREER_LHP = 735
REGRESS_PA_TOTAL_LHP  = 910
REGRESS_PA_RECENT_LHP = 795

# -------------------------------------------------------
# RHP model -- weights and regression PA constants (#5)
# These apply when predicting performance vs right-handed pitchers.
# Weights below are proportionally renormalized to sum to 1.00 (see
# note above the LHP weights -- this is cosmetic and r-neutral).
# -------------------------------------------------------
WEIGHT_SPLITS_RHP  = 0.2774
WEIGHT_CAREER_RHP  = 0.2920
WEIGHT_TOTAL_RHP   = 0.2153
WEIGHT_RECENT_RHP  = 0.2153

REGRESS_PA_SPLITS_RHP = 285
REGRESS_PA_CAREER_RHP = 25
REGRESS_PA_TOTAL_RHP  = 400
REGRESS_PA_RECENT_RHP = 665

# -------------------------------------------------------
# PA minimums -- FIXED, do not tune.
# -------------------------------------------------------
# Minimum pre-cutoff PA to include a component
MIN_PA_SPLITS = 30
MIN_PA_TOTAL  = 30
MIN_PA_CAREER = 30
# No minimum for recent (any data used -- same as test.py)

# Minimum post-cutoff PA to use as ground truth
# (filters out players who barely played after the cutoff date)
MIN_PA_POST = 30

# Disable PA-based clamping (improves correlation by allowing extreme platoon splits)
NO_CLAMP = True

# Players to show in top/bottom/miss lists per window
DISPLAY_N = 5

# Platoon tendency blend (#4).
# Scales total/recent wOBA by a player's career platoon ratio.
# Originally tested in isolation against the v6 weight shape and found
# to have ~zero effect, so it was left off in v6/v7. After re-tuning all
# weights and regression constants (v8), re-sweeping this blend found a
# small but real gain (+0.0007) at blend=0.31 with a wider factor clamp.
PLATOON_BLEND = 0.31
PLATOON_FACTOR_MIN = 0.585
PLATOON_FACTOR_MAX = 1.325

# -------------------------------------------------------
# Career recency decay -- NOT YET TUNED (Task #23).
# -------------------------------------------------------
# Career data is now built from every per-season file strictly before
# the test year (see build_career_data() below). Each season's
# contribution is weighted by decay ** (test_year - season_year), so
# decay < 1.0 downweights seasons further in the past relative to the
# test year. Set to 1.0 for now -- i.e. no recency weighting, every
# prior season counted equally by its own PA -- as a neutral starting
# point. Separate LHP/RHP constants follow the existing pattern (#5).
DECAY_CAREER_LHP = 1.0
DECAY_CAREER_RHP = 1.0

# -------------------------------------------------------
# TEST WINDOWS
# -------------------------------------------------------
# Every season with a complete before_X/after_X/totals_before_X/
# recent_before_X file set, x every cutoff. COVID-shortened 2020 has
# no data and is skipped (matches the old career_group skip pattern).

TEST_YEARS = [2019, 2021, 2022, 2023, 2024, 2025]
CUTOFFS    = ["june_1", "july_1", "aug_1"]

TEST_WINDOWS = [
    {"year": year, "cutoff": cutoff}
    for year in TEST_YEARS
    for cutoff in CUTOFFS
]

# Seasons with a season_<year>_vs_left/right.csv file available, used
# as the raw material for build_career_data(). 2020 has no file (COVID).
ALL_SEASON_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]

# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------

def load_file(filename):
    """
    Load a wOBA CSV from the historical testing folder.
    Returns dict: player_id (str) -> {name, pa, woba, xwoba}
    "xwoba" is None when the column is absent or blank (e.g. the
    after_* ground-truth files don't have one) -- not yet used in
    scoring (see v9 note in the module docstring; Task #24).
    Silently skips rows with missing/null wOBA.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print("  WARNING: File not found: " + filename)
        return {}
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pid      = str(row["player_id"]).strip()
                pa       = int(float(row["pa"]))
                woba_str = str(row.get("woba", "")).strip()
                if not woba_str or woba_str.lower() in ("nan", "none", ""):
                    continue
                woba  = float(woba_str)
                first = str(row.get("first_name", "")).strip()
                last  = str(row.get("last_name",  "")).strip()
                name  = (last + ", " + first).strip(", ")
                xwoba_str = str(row.get("xwoba", "")).strip()
                xwoba = None
                if xwoba_str and xwoba_str.lower() not in ("nan", "none"):
                    try:
                        xwoba = float(xwoba_str)
                    except ValueError:
                        xwoba = None
                result[pid] = {"name": name, "pa": pa, "woba": woba, "xwoba": xwoba}
            except (ValueError, KeyError):
                continue
    return result


def load_season_file(year, hand):
    """Load season_<year>_vs_<hand>.csv (hand: 'left' or 'right')."""
    return load_file("season_" + str(year) + "_vs_" + hand + ".csv")


def build_career_data(test_year, hand, decay):
    """
    Build career wOBA (and, where available, xwOBA) for every player by
    combining every season_<year>_vs_<hand>.csv file strictly before
    test_year -- i.e. only seasons that would actually have been known
    at prediction time, no data leakage from the test year itself.

    Each season's contribution is weighted by its own PA times
    decay ** (test_year - season_year), so seasons further in the past
    count for less when decay < 1.0. decay=1.0 (current default,
    pending Task #23) means plain PA-weighting with no recency tilt.

    Returns dict: player_id (str) -> {name, pa, woba[, xwoba]}
    "pa" is the player's RAW (unweighted) total PA across all prior
    seasons used -- this is what MIN_PA_CAREER gates against, so the
    decay only affects the wOBA value itself, not whether there's
    enough career data to trust it.
    """
    prior_years = [y for y in ALL_SEASON_YEARS if y < test_year]
    num, den   = {}, {}
    xnum, xden = {}, {}
    pa_sum, names = {}, {}

    for y in prior_years:
        season = load_season_file(y, hand)
        w = decay ** (test_year - y)
        for pid, d in season.items():
            wpa = w * d["pa"]
            num[pid]    = num.get(pid, 0.0) + wpa * d["woba"]
            den[pid]    = den.get(pid, 0.0) + wpa
            pa_sum[pid] = pa_sum.get(pid, 0) + d["pa"]
            names[pid]  = d["name"]
            if d.get("xwoba") is not None:
                xnum[pid] = xnum.get(pid, 0.0) + wpa * d["xwoba"]
                xden[pid] = xden.get(pid, 0.0) + wpa

    result = {}
    for pid, denom in den.items():
        if denom <= 0:
            continue
        entry = {"name": names[pid], "pa": pa_sum[pid], "woba": num[pid] / denom}
        if xden.get(pid, 0) > 0:
            entry["xwoba"] = xnum[pid] / xden[pid]
        result[pid] = entry
    return result


# -------------------------------------------------------
# SCORING MATH
# Mirrors test.py; all functions adapted proportionally for wOBA.
# -------------------------------------------------------

def regress(woba, pa, regress_pa, la):
    """
    Regress wOBA toward the side-appropriate league average.
    Mirrors regress_to_mean() in test.py.
    la: league-average wOBA for this pitcher handedness.
    """
    return (woba * pa + la * regress_pa) / (pa + regress_pa)


def clamp_woba_by_pa(woba, pa, la):
    """
    PA-based clamp. Mirrors clamp_wrc_by_pa() in test.py, scaled
    proportionally to the wOBA range.
    la: league-average wOBA for this pitcher handedness.
    """
    s  = la / 100.0
    lo = max(0.0,      la - pa * s)
    hi = min(la * 2.0, la + pa * s)
    return max(lo, min(hi, float(woba)))


def stabilize(splits_pa, splits_woba,
              career_pa, career_woba,
              total_pa,  total_woba,
              recent_pa, recent_woba,
              side='R',
              career_cross_pa=0, career_cross_woba=None):
    """
    Compute stabilized wOBA from up to 4 components.
    Mirrors build_stabilized_score() in test.py exactly.

    Components with insufficient PA are dropped; remaining weights
    are renormalized so they always sum to 1.0.

    No PA clamp on recent (matches test.py: small sample,
    clamp would mute its signal).

    side: 'L' or 'R' -- selects LHP or RHP constants (#2, #5).
    career_cross_pa, career_cross_woba: career stats vs the OPPOSITE
    side, used to compute the platoon tendency prior (#4).
    """
    # -- Select side-appropriate constants (#2, #5) --
    if side == 'L':
        la  = LEAGUE_AVG_WOBA_LHP
        WS  = WEIGHT_SPLITS_LHP
        WC  = WEIGHT_CAREER_LHP
        WT  = WEIGHT_TOTAL_LHP
        WR  = WEIGHT_RECENT_LHP
        RPS = REGRESS_PA_SPLITS_LHP
        RPC = REGRESS_PA_CAREER_LHP
        RPT = REGRESS_PA_TOTAL_LHP
        RPR = REGRESS_PA_RECENT_LHP
    else:
        la  = LEAGUE_AVG_WOBA_RHP
        WS  = WEIGHT_SPLITS_RHP
        WC  = WEIGHT_CAREER_RHP
        WT  = WEIGHT_TOTAL_RHP
        WR  = WEIGHT_RECENT_RHP
        RPS = REGRESS_PA_SPLITS_RHP
        RPC = REGRESS_PA_CAREER_RHP
        RPT = REGRESS_PA_TOTAL_RHP
        RPR = REGRESS_PA_RECENT_RHP

    # -- Platoon tendency prior (#4) --
    # Scales total/recent wOBA by the player's career platoon ratio so
    # those non-split components better reflect the expected performance
    # vs this specific pitcher handedness.  Controlled by PLATOON_BLEND:
    # 0.0 = no adjustment (current setting), 1.0 = full adjustment.
    platoon_factor = 1.0
    if (PLATOON_BLEND > 0 and
            career_woba is not None and career_pa > 0 and
            career_cross_woba is not None and career_cross_pa > 0):
        combined_pa = career_pa + career_cross_pa
        combined_woba = (career_woba * career_pa +
                         career_cross_woba * career_cross_pa) / combined_pa
        if combined_woba > 0:
            raw_factor = career_woba / combined_woba
            clamped_factor = max(PLATOON_FACTOR_MIN,
                                 min(PLATOON_FACTOR_MAX, raw_factor))
            platoon_factor = 1.0 + PLATOON_BLEND * (clamped_factor - 1.0)

    components, weights = [], []

    if splits_woba is not None and splits_pa >= MIN_PA_SPLITS:
        clamped = splits_woba if NO_CLAMP else clamp_woba_by_pa(splits_woba, splits_pa, la)
        components.append(regress(clamped, splits_pa, RPS, la))
        weights.append(WS)

    if career_woba is not None and career_pa >= MIN_PA_CAREER:
        clamped = career_woba if NO_CLAMP else clamp_woba_by_pa(career_woba, career_pa, la)
        components.append(regress(clamped, career_pa, RPC, la))
        weights.append(WC)

    if total_woba is not None and total_pa >= MIN_PA_TOTAL:
        adj    = total_woba * platoon_factor
        clamped = adj if NO_CLAMP else clamp_woba_by_pa(adj, total_pa, la)
        components.append(regress(clamped, total_pa, RPT, la))
        weights.append(WT)

    if recent_woba is not None and recent_pa > 0:
        # No clamp on recent -- mirrors test.py
        adj = recent_woba * platoon_factor
        components.append(regress(adj, recent_pa, RPR, la))
        weights.append(WR)

    if not components:
        return None

    total_w = sum(weights)
    return round(sum(c * (w / total_w) for c, w in zip(components, weights)), 4)


# -------------------------------------------------------
# STATISTICS
# -------------------------------------------------------

def rank_array(arr):
    """Return 1-based ranks (averaged for ties) for a list of values."""
    n = len(arr)
    indexed = sorted(enumerate(arr), key=lambda t: t[1])
    ranks   = [0.0] * n
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
    """
    Spearman rank correlation between two equal-length lists.
    Returns None if fewer than 5 paired values.
    """
    n = len(x)
    if n < 5:
        return None
    rx   = rank_array(x)
    ry   = rank_array(y)
    mx   = sum(rx) / n
    my   = sum(ry) / n
    num  = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    deny = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if denx == 0 or deny == 0:
        return None
    return round(num / (denx * deny), 4)


# -------------------------------------------------------
# REPORTING HELPERS
# -------------------------------------------------------

def fmt_r(r):
    return str(r) if r is not None else "N/A"


def print_player_list(pairs, label, key, n, ascending=False, total=None):
    """Print top-N players from pairs sorted by key."""
    sorted_pairs = sorted(pairs, key=lambda p: p[key], reverse=not ascending)[:n]
    total_str    = ("/" + str(total)) if total else ""
    print("  " + label)
    for p in sorted_pairs:
        print("    " + p["name"].ljust(25) +
              "  predicted: " + str(round(p["predicted"], 3)) +
              "  actual: "    + str(round(p["actual"],    3)) +
              "  post-PA: "   + str(p["actual_pa"]))


def print_misses(pairs, hand, n=DISPLAY_N):
    """Show the n biggest model overestimates and underestimates."""
    if len(pairs) < 5:
        return
    total  = len(pairs)
    pred_r = rank_array([p["predicted"] for p in pairs])
    act_r  = rank_array([p["actual"]    for p in pairs])
    diffs  = []
    for i, p in enumerate(pairs):
        # Positive diff = model underrated (actual rank higher than predicted)
        # Negative diff = model overrated (actual rank lower than predicted)
        diff = act_r[i] - pred_r[i]
        diffs.append({**p, "diff": diff,
                      "pred_rank": int(pred_r[i]),
                      "act_rank":  int(act_r[i]),
                      "total":     total})
    overrated  = sorted(diffs, key=lambda x: x["diff"])[:n]
    underrated = sorted(diffs, key=lambda x: x["diff"], reverse=True)[:n]
    print("  Biggest OVERESTIMATES vs " + hand + ":")
    for d in overrated:
        print("    " + d["name"].ljust(25) +
              "  pred " + str(round(d["predicted"], 3)) +
              " (rank " + str(d["pred_rank"]) + "/" + str(total) + ")" +
              "  actual " + str(round(d["actual"], 3)) +
              " (rank " + str(d["act_rank"]) + "/" + str(total) + ")")
    print("  Biggest UNDERESTIMATES vs " + hand + ":")
    for d in underrated:
        print("    " + d["name"].ljust(25) +
              "  pred " + str(round(d["predicted"], 3)) +
              " (rank " + str(d["pred_rank"]) + "/" + str(total) + ")" +
              "  actual " + str(round(d["actual"], 3)) +
              " (rank " + str(d["act_rank"]) + "/" + str(total) + ")")


# -------------------------------------------------------
# WINDOW RUNNER
# -------------------------------------------------------

def run_window(window):
    year  = window["year"]
    co    = window["cutoff"]
    label = str(year) + " before " + co.replace("_", " ")

    # -- Load pre-cutoff data --
    pre_l  = load_file("before_" + co + "_" + str(year) + "_vs_left.csv")
    pre_r  = load_file("before_" + co + "_" + str(year) + "_vs_right.csv")
    car_l  = build_career_data(year, "left",  DECAY_CAREER_LHP)
    car_r  = build_career_data(year, "right", DECAY_CAREER_RHP)
    total  = load_file("totals_before_" + co + "_" + str(year) + ".csv")
    recent = load_file("recent_before_" + co + "_" + str(year) + ".csv")

    # -- Load post-cutoff ground truth --
    post_l = load_file("after_" + co + "_" + str(year) + "_vs_left.csv")
    post_r = load_file("after_" + co + "_" + str(year) + "_vs_right.csv")

    # -- Compute stabilized wOBA for every player seen in splits data --
    all_pids = set(pre_l.keys()) | set(pre_r.keys())

    scored = {}
    for pid in all_pids:
        sl  = pre_l.get(pid)
        sr  = pre_r.get(pid)
        cl  = car_l.get(pid)
        cr  = car_r.get(pid)
        tot = total.get(pid)
        rec = recent.get(pid)
        name = (sl or sr or tot or {}).get("name", pid)

        # For the LHP prediction, career_cross is the vs-RHP career data (#4)
        stab_l = stabilize(
            sl["pa"]  if sl  else 0, sl["woba"]  if sl  else None,
            cl["pa"]  if cl  else 0, cl["woba"]  if cl  else None,
            tot["pa"] if tot else 0, tot["woba"] if tot else None,
            rec["pa"] if rec else 0, rec["woba"] if rec else None,
            side='L',
            career_cross_pa=cr["pa"]   if cr else 0,
            career_cross_woba=cr["woba"] if cr else None,
        )
        # For the RHP prediction, career_cross is the vs-LHP career data (#4)
        stab_r = stabilize(
            sr["pa"]  if sr  else 0, sr["woba"]  if sr  else None,
            cr["pa"]  if cr  else 0, cr["woba"]  if cr  else None,
            tot["pa"] if tot else 0, tot["woba"] if tot else None,
            rec["pa"] if rec else 0, rec["woba"] if rec else None,
            side='R',
            career_cross_pa=cl["pa"]   if cl else 0,
            career_cross_woba=cl["woba"] if cl else None,
        )
        scored[pid] = {"name": name, "stab_l": stab_l, "stab_r": stab_r}

    # -- Build paired (predicted, actual) lists --
    # Require the player has at least some handedness-specific data
    # (current-year splits OR career splits) for that side -- otherwise
    # we're predicting LHP performance purely from non-split data, which
    # the model wasn't designed to do.

    def build_pairs(stab_key, pre_dict, career_dict, post_dict):
        pairs = []
        for pid, s in scored.items():
            pred = s[stab_key]
            if pred is None:
                continue
            has_split = (pid in pre_dict    and pre_dict[pid]["pa"]    >= MIN_PA_SPLITS) or \
                        (pid in career_dict and career_dict[pid]["pa"] >= MIN_PA_CAREER)
            if not has_split:
                continue
            post = post_dict.get(pid)
            if post is None or post["pa"] < MIN_PA_POST:
                continue
            pairs.append({
                "name":      s["name"],
                "predicted": pred,
                "actual":    post["woba"],
                "actual_pa": post["pa"],
            })
        return pairs

    pairs_l = build_pairs("stab_l", pre_l, car_l, post_l)
    pairs_r = build_pairs("stab_r", pre_r, car_r, post_r)

    r_l = spearman_r([p["predicted"] for p in pairs_l],
                     [p["actual"]    for p in pairs_l])
    r_r = spearman_r([p["predicted"] for p in pairs_r],
                     [p["actual"]    for p in pairs_r])

    return {
        "label":   label,
        "r_l":     r_l,
        "r_r":     r_r,
        "n_l":     len(pairs_l),
        "n_r":     len(pairs_r),
        "pairs_l": pairs_l,
        "pairs_r": pairs_r,
    }


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    print("=== BACKTEST v8 -- Lineup Optimization Historical Validation ===")
    print("Results saved to: " + OUTPUT_FILE)
    print("Configuration (v8 tuned: separate LHP/RHP constants + platoon prior):")
    print("  League avg wOBA:  LHP=" + str(LEAGUE_AVG_WOBA_LHP) +
          "  RHP=" + str(LEAGUE_AVG_WOBA_RHP))
    print("  LHP Weights:  splits=" + str(WEIGHT_SPLITS_LHP) +
          "  career=" + str(WEIGHT_CAREER_LHP) +
          "  total=" + str(WEIGHT_TOTAL_LHP) +
          "  recent=" + str(WEIGHT_RECENT_LHP))
    print("  RHP Weights:  splits=" + str(WEIGHT_SPLITS_RHP) +
          "  career=" + str(WEIGHT_CAREER_RHP) +
          "  total=" + str(WEIGHT_TOTAL_RHP) +
          "  recent=" + str(WEIGHT_RECENT_RHP))
    print("  LHP Regress PA:  splits=" + str(REGRESS_PA_SPLITS_LHP) +
          "  career=" + str(REGRESS_PA_CAREER_LHP) +
          "  total=" + str(REGRESS_PA_TOTAL_LHP) +
          "  recent=" + str(REGRESS_PA_RECENT_LHP))
    print("  RHP Regress PA:  splits=" + str(REGRESS_PA_SPLITS_RHP) +
          "  career=" + str(REGRESS_PA_CAREER_RHP) +
          "  total=" + str(REGRESS_PA_TOTAL_RHP) +
          "  recent=" + str(REGRESS_PA_RECENT_RHP))
    print("  Min PA (pre): splits=" + str(MIN_PA_SPLITS) +
          "  career=" + str(MIN_PA_CAREER) +
          "  total=" + str(MIN_PA_TOTAL))
    print("  Min PA (post-cutoff ground truth): " + str(MIN_PA_POST))
    print("  Platoon blend: " + str(PLATOON_BLEND) +
          "  factor clamp: [" + str(PLATOON_FACTOR_MIN) +
          ", " + str(PLATOON_FACTOR_MAX) + "]")
    print("  Career recency decay (not yet tuned): LHP=" + str(DECAY_CAREER_LHP) +
          "  RHP=" + str(DECAY_CAREER_RHP))
    print("  Test years: " + ", ".join(str(y) for y in TEST_YEARS) +
          "  (" + str(len(TEST_WINDOWS)) + " windows total)")
    print()

    all_results = []

    for window in TEST_WINDOWS:
        label = str(window["year"]) + " before " + window["cutoff"].replace("_", " ")
        print("=" * 60)
        print("WINDOW: " + label)
        print("=" * 60)

        res = run_window(window)
        all_results.append(res)

        print()
        print("  vs LHP:  n=" + str(res["n_l"]) +
              "  Spearman r=" + fmt_r(res["r_l"]))
        print("  vs RHP:  n=" + str(res["n_r"]) +
              "  Spearman r=" + fmt_r(res["r_r"]))
        print()

        # Top and bottom predicted players with actual outcomes
        if res["pairs_l"]:
            print_player_list(res["pairs_l"], "Top " + str(DISPLAY_N) + " predicted vs LHP (actual outcome):",
                              "predicted", DISPLAY_N, ascending=False)
            print_player_list(res["pairs_l"], "Bottom " + str(DISPLAY_N) + " predicted vs LHP (actual outcome):",
                              "predicted", DISPLAY_N, ascending=True)
        if res["pairs_r"]:
            print_player_list(res["pairs_r"], "Top " + str(DISPLAY_N) + " predicted vs RHP (actual outcome):",
                              "predicted", DISPLAY_N, ascending=False)
            print_player_list(res["pairs_r"], "Bottom " + str(DISPLAY_N) + " predicted vs RHP (actual outcome):",
                              "predicted", DISPLAY_N, ascending=True)
        print()

        # Biggest misses
        if res["pairs_l"]:
            print_misses(res["pairs_l"], "LHP")
        if res["pairs_r"]:
            print_misses(res["pairs_r"], "RHP")
        print()

    # -------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------
    print("=" * 60)
    print("SUMMARY: Spearman Rank Correlation Across All Windows")
    print("=" * 60)
    print()
    print("Window                    N(LHP)  r(LHP)  N(RHP)  r(RHP)  Avg r")
    print("------------------------  ------  ------  ------  ------  -----")

    valid_r = []
    for res in all_results:
        label = res["label"].ljust(24)
        n_l   = str(res["n_l"]).rjust(6)
        r_l   = (str(res["r_l"]) if res["r_l"] is not None else "  N/A").rjust(6)
        n_r   = str(res["n_r"]).rjust(6)
        r_r   = (str(res["r_r"]) if res["r_r"] is not None else "  N/A").rjust(6)

        rs = [v for v in [res["r_l"], res["r_r"]] if v is not None]
        avg_r = round(sum(rs) / len(rs), 4) if rs else None
        avg_str = (str(avg_r) if avg_r is not None else "N/A").rjust(5)
        if avg_r is not None:
            valid_r.append(avg_r)

        print(label + "  " + n_l + "  " + r_l + "  " + n_r + "  " + r_r + "  " + avg_str)

    print()
    if valid_r:
        all_rs = [v for res in all_results for v in [res["r_l"], res["r_r"]] if v is not None]
        lhp_rs = [res["r_l"] for res in all_results if res["r_l"] is not None]
        rhp_rs = [res["r_r"] for res in all_results if res["r_r"] is not None]
        if lhp_rs:
            print("Overall average r (LHP): " + str(round(sum(lhp_rs) / len(lhp_rs), 4)))
        if rhp_rs:
            print("Overall average r (RHP): " + str(round(sum(rhp_rs) / len(rhp_rs), 4)))
        print("Overall average r (all): " + str(round(sum(all_rs) / len(all_rs), 4)))
    print()
    print("Interpretation:")
    print("  r > 0.60  -- strong predictive power")
    print("  r 0.45-0.60 -- good for player data at this sample size")
    print("  r 0.30-0.45 -- moderate; model capturing real signal")
    print("  r < 0.30  -- weak; consider revisiting weights")
    print()
    print("To tune: change one parameter above, re-run, compare avg r.")
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
