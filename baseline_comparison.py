#!/usr/bin/env python3
"""
Baseline Comparison -- "Compared To What?"
============================================
Answers the question every tuning result has been dodging: is the
production wOBA model (overall Spearman r = 0.3706, LHP 0.3299, RHP
0.4113, per backtest.py v15) actually doing meaningful work, or would
something dramatically simpler get most of the way there?

This script runs the SAME 18 historical windows (6 test seasons x 3
cutoffs) through FOUR different predictors, with the SAME
player-inclusion rules and SAME post-cutoff ground truth as
backtest.py, so the only thing that differs between methods is the
prediction formula itself:

  1. PRODUCTION
     backtest.py's full v15 model, mirrored here exactly: weights,
     PA-regression, platoon prior, xWOBA blend, career decay, aging
     curve -- every current shipped constant. This is the thing being
     tested.

  2. NAIVE LAST SEASON
     That player's single most recent prior full season wOBA vs. this
     hand, completely unregressed. No averaging across seasons, no
     shrinkage toward league mean. The simplest "what have you done
     lately" guess.

  3. NAIVE CAREER AVG
     PA-weighted average wOBA across every prior season on file. No
     recency decay, no aging adjustment, no regression toward league
     average, no xWOBA blend, no platoon prior, no current-season
     splits/recent data at all. The simplest "what have you done,
     period" guess -- the zero-sophistication floor.

  4. NAIVE CAREER REGRESSED
     The same career average as #3, shrunk toward league average with
     a single round, deliberately UNTUNED regression constant
     (NAIVE_REGRESS_PA = 400 PA, shared by both hands). Roughly what a
     careful analyst would build in an afternoon with no CV search at
     all -- the "cheap, statistically sound" floor, as opposed to
     #2/#3's "zero sophistication" floor.

All three naive predictors fall back to the side's league-average wOBA
(LEAGUE_AVG_WOBA_LHP/RHP) for any player missing the relevant data, so
every player who clears the model's own inclusion bar (current-season
splits PA or career PA above the production minimums) gets SOME
prediction under every method -- nobody is silently dropped from one
method's pool and not another's, which would bias the comparison.
(In fact, because PA totals don't depend on decay/aging, the eligible
player pool is provably identical across all four methods in every
window -- see build_pairs() below.)

CANARY: before trusting any baseline numbers, this script first
recomputes the PRODUCTION method across all 18 windows and checks it
reproduces backtest.py's own KNOWN_PRODUCTION_R_* values. If that
check fails, the mirrored code below has drifted from backtest.py and
nothing else in this report should be trusted.

Note: this script does NOT modify backtest.py or any of its shipped
constants. It is a read-only diagnostic / comparison tool, not a
tuning run.

USAGE:
  Double-click run_baseline_comparison.bat
  Results saved to baseline_comparison_results_YYYY-MM-DD.txt
"""

import os
import csv
import sys
import datetime

FOLDER   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FOLDER, "historical testing")
today    = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(FOLDER, "baseline_comparison_results_" + today + ".txt")


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


# -------------------------------------------------------------------
# PRODUCTION CONFIGURATION -- mirrored exactly from backtest.py v15.
# Do not "improve" these here -- this script's job is to faithfully
# reproduce the shipped model, not to tune it further.
# -------------------------------------------------------------------

LEAGUE_AVG_WOBA_LHP = 0.2943
LEAGUE_AVG_WOBA_RHP = 0.2575

WEIGHT_SPLITS_LHP = 0.213
WEIGHT_CAREER_LHP = 0.5695
WEIGHT_TOTAL_LHP  = 0.247
WEIGHT_RECENT_LHP = 0.3416

REGRESS_PA_SPLITS_LHP = 178
REGRESS_PA_CAREER_LHP = 711
REGRESS_PA_TOTAL_LHP  = 1020
REGRESS_PA_RECENT_LHP = 757

WEIGHT_SPLITS_RHP = 0.2774
WEIGHT_CAREER_RHP = 0.2920
WEIGHT_TOTAL_RHP  = 0.2153
WEIGHT_RECENT_RHP = 0.2153

REGRESS_PA_SPLITS_RHP = 285
REGRESS_PA_CAREER_RHP = 25
REGRESS_PA_TOTAL_RHP  = 400
REGRESS_PA_RECENT_RHP = 665

MIN_PA_SPLITS = 30
MIN_PA_TOTAL  = 30
MIN_PA_CAREER = 30
MIN_PA_POST   = 30
NO_CLAMP = True  # current production value -- this script assumes it
                 # and never calls a clamp function, matching backtest.py

PLATOON_BLEND_LHP      = 0.7833
PLATOON_FACTOR_MIN_LHP = 0.612
PLATOON_FACTOR_MAX_LHP = 1.138

PLATOON_BLEND_RHP      = 0.31
PLATOON_FACTOR_MIN_RHP = 0.585
PLATOON_FACTOR_MAX_RHP = 1.325

DECAY_CAREER_LHP = 0.8001
DECAY_CAREER_RHP = 1.0

XWOBA_BLEND_SPLITS_LHP = 0.9187
XWOBA_BLEND_CAREER_LHP = 0.9187
XWOBA_BLEND_TOTAL_LHP  = 0.9187
XWOBA_BLEND_RECENT_LHP = 0.9187

XWOBA_BLEND_SPLITS_RHP = 0.80
XWOBA_BLEND_CAREER_RHP = 0.80
XWOBA_BLEND_TOTAL_RHP  = 0.80
XWOBA_BLEND_RECENT_RHP = 0.80

AGING_CURVE_CUM = {
    21: 0.0046, 22: 0.0040, 23: 0.0056, 24: 0.0062, 25: 0.0041, 26: 0.0007,
    27: -0.0065, 28: -0.0120, 29: -0.0171, 30: -0.0308, 31: -0.0409,
    32: -0.0494, 33: -0.0641, 34: -0.0714, 35: -0.0854, 36: -0.1010,
}
AGING_CURVE_MIN_AGE = min(AGING_CURVE_CUM)
AGING_CURVE_MAX_AGE = max(AGING_CURVE_CUM)
AGING_STRENGTH = 0.0

# This script's own naive-baseline knob -- deliberately a round,
# UNTUNED number, shared by both hands. Changing this to "optimize" it
# would defeat the purpose (see predictor #4 above): it's meant to be
# a reasonable choice a careful analyst makes without running any
# search, not the best possible value.
NAIVE_REGRESS_PA = 400

KNOWN_PRODUCTION_R_ALL = 0.3706
KNOWN_PRODUCTION_R_LHP = 0.3299
KNOWN_PRODUCTION_R_RHP = 0.4113


def aging_cum(age):
    clamped = max(AGING_CURVE_MIN_AGE, min(AGING_CURVE_MAX_AGE, age))
    return AGING_CURVE_CUM[clamped]


def aging_delta(age_from, age_to):
    return aging_cum(age_to) - aging_cum(age_from)


def load_player_ages():
    path = os.path.join(FOLDER, "player_ages.csv")
    result = {}
    if not os.path.exists(path):
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

TEST_YEARS = [2019, 2021, 2022, 2023, 2024, 2025]
CUTOFFS = ["june_1", "july_1", "aug_1"]
TEST_WINDOWS = [
    {"year": year, "cutoff": cutoff}
    for year in TEST_YEARS
    for cutoff in CUTOFFS
]
ALL_SEASON_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]


def load_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print("  WARNING: File not found: " + filename)
        return {}
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pid = str(row["player_id"]).strip()
                pa = int(float(row["pa"]))
                woba_str = str(row.get("woba", "")).strip()
                if not woba_str or woba_str.lower() in ("nan", "none"):
                    continue
                woba = float(woba_str)
                first = str(row.get("first_name", "")).strip()
                last = str(row.get("last_name", "")).strip()
                name = (last + ", " + first).strip(", ")
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
    return load_file("season_" + str(year) + "_vs_" + hand + ".csv")


def build_career_data(test_year, hand, decay, aging_strength=0.0, ages=None):
    prior_years = [y for y in ALL_SEASON_YEARS if y < test_year]
    num, den = {}, {}
    xnum, xden = {}, {}
    pa_sum, names = {}, {}

    for y in prior_years:
        season = load_season_file(y, hand)
        w = decay ** (test_year - y)
        for pid, d in season.items():
            delta = 0.0
            if aging_strength != 0.0 and ages:
                player_ages = ages.get(pid)
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
            names[pid] = d["name"]
            if d.get("xwoba") is not None:
                adj_xwoba = d["xwoba"] + delta
                xnum[pid] = xnum.get(pid, 0.0) + wpa * adj_xwoba
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


def blend_woba(woba, xwoba, xwoba_blend):
    if xwoba is None:
        return woba
    return (1.0 - xwoba_blend) * woba + xwoba_blend * xwoba


def blended_woba(entry, xwoba_blend):
    if entry is None:
        return None
    return blend_woba(entry["woba"], entry.get("xwoba"), xwoba_blend)


def regress(woba, pa, regress_pa, la):
    return (woba * pa + la * regress_pa) / (pa + regress_pa)


def stabilize(splits_pa, splits_woba,
              career_pa, career_woba,
              total_pa, total_woba,
              recent_pa, recent_woba,
              side='R',
              career_cross_pa=0, career_cross_woba=None):
    if side == 'L':
        la = LEAGUE_AVG_WOBA_LHP
        WS, WC, WT, WR = WEIGHT_SPLITS_LHP, WEIGHT_CAREER_LHP, WEIGHT_TOTAL_LHP, WEIGHT_RECENT_LHP
        RPS, RPC, RPT, RPR = REGRESS_PA_SPLITS_LHP, REGRESS_PA_CAREER_LHP, REGRESS_PA_TOTAL_LHP, REGRESS_PA_RECENT_LHP
        PB, PMIN, PMAX = PLATOON_BLEND_LHP, PLATOON_FACTOR_MIN_LHP, PLATOON_FACTOR_MAX_LHP
    else:
        la = LEAGUE_AVG_WOBA_RHP
        WS, WC, WT, WR = WEIGHT_SPLITS_RHP, WEIGHT_CAREER_RHP, WEIGHT_TOTAL_RHP, WEIGHT_RECENT_RHP
        RPS, RPC, RPT, RPR = REGRESS_PA_SPLITS_RHP, REGRESS_PA_CAREER_RHP, REGRESS_PA_TOTAL_RHP, REGRESS_PA_RECENT_RHP
        PB, PMIN, PMAX = PLATOON_BLEND_RHP, PLATOON_FACTOR_MIN_RHP, PLATOON_FACTOR_MAX_RHP

    platoon_factor = 1.0
    if (PB > 0 and career_woba is not None and career_pa > 0 and
            career_cross_woba is not None and career_cross_pa > 0):
        combined_pa = career_pa + career_cross_pa
        combined_woba = (career_woba * career_pa +
                          career_cross_woba * career_cross_pa) / combined_pa
        if combined_woba > 0:
            raw_factor = career_woba / combined_woba
            clamped_factor = max(PMIN, min(PMAX, raw_factor))
            platoon_factor = 1.0 + PB * (clamped_factor - 1.0)

    components, weights = [], []

    if splits_woba is not None and splits_pa >= MIN_PA_SPLITS:
        components.append(regress(splits_woba, splits_pa, RPS, la))
        weights.append(WS)

    if career_woba is not None and career_pa >= MIN_PA_CAREER:
        components.append(regress(career_woba, career_pa, RPC, la))
        weights.append(WC)

    if total_woba is not None and total_pa >= MIN_PA_TOTAL:
        adj = total_woba * platoon_factor
        components.append(regress(adj, total_pa, RPT, la))
        weights.append(WT)

    if recent_woba is not None and recent_pa > 0:
        adj = recent_woba * platoon_factor
        components.append(regress(adj, recent_pa, RPR, la))
        weights.append(WR)

    if not components:
        return None

    total_w = sum(weights)
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
    rx = rank_array(x)
    ry = rank_array(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    deny = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if denx == 0 or deny == 0:
        return None
    return round(num / (denx * deny), 4)


def fmt_r(r):
    return str(r) if r is not None else "N/A"


# -------------------------------------------------------------------
# NAIVE PREDICTORS
# -------------------------------------------------------------------

def naive_last_season(pid, last_season_dict, la):
    """Unregressed wOBA from the single most recent prior season.
    Falls back to league average if the player has no row in that
    season's file for this hand."""
    entry = last_season_dict.get(pid)
    return la if entry is None else entry["woba"]


def naive_career_avg(pid, career_naive_dict, la):
    """PA-weighted average wOBA across every prior season, no decay,
    no aging, no regression. Falls back to league average if the
    player has no career data at all (e.g. a rookie)."""
    entry = career_naive_dict.get(pid)
    return la if entry is None else entry["woba"]


def naive_career_regressed(pid, career_naive_dict, la):
    """Same career average as naive_career_avg(), shrunk toward league
    average with a single untuned regression constant."""
    entry = career_naive_dict.get(pid)
    if entry is None:
        return la
    return regress(entry["woba"], entry["pa"], NAIVE_REGRESS_PA, la)


# -------------------------------------------------------------------
# WINDOW RUNNER
# -------------------------------------------------------------------

METHODS = ["production", "naive_last_season", "naive_career_avg", "naive_career_regressed"]
METHOD_LABELS = {
    "production": "Production",
    "naive_last_season": "NaiveLastSeason",
    "naive_career_avg": "NaiveCareerAvg",
    "naive_career_regressed": "NaiveCareerRegr",
}


def run_window(window):
    year = window["year"]
    co = window["cutoff"]
    label = str(year) + " before " + co.replace("_", " ")

    pre_l = load_file("before_" + co + "_" + str(year) + "_vs_left.csv")
    pre_r = load_file("before_" + co + "_" + str(year) + "_vs_right.csv")
    total = load_file("totals_before_" + co + "_" + str(year) + ".csv")
    recent = load_file("recent_before_" + co + "_" + str(year) + ".csv")
    post_l = load_file("after_" + co + "_" + str(year) + "_vs_left.csv")
    post_r = load_file("after_" + co + "_" + str(year) + "_vs_right.csv")

    # Production career data (current shipped decay/aging).
    car_l_prod = build_career_data(year, "left", DECAY_CAREER_LHP, AGING_STRENGTH, PLAYER_AGES)
    car_r_prod = build_career_data(year, "right", DECAY_CAREER_RHP, AGING_STRENGTH, PLAYER_AGES)

    # Naive career data (decay=1.0, no aging) -- used by predictors
    # #3/#4. PA totals are decay-independent, so car_*_naive's "pa"
    # field is identical to car_*_prod's; only the "woba" value
    # differs. (RHP's production decay is already 1.0, so car_r_naive
    # and car_r_prod are numerically identical for RHP -- expected,
    # not a bug.) This identity is what guarantees the same player
    # pool gets scored under every method below.
    car_l_naive = build_career_data(year, "left", 1.0, 0.0, None)
    car_r_naive = build_career_data(year, "right", 1.0, 0.0, None)

    prior_years = [y for y in ALL_SEASON_YEARS if y < year]
    last_year = max(prior_years) if prior_years else None
    last_l = load_season_file(last_year, "left") if last_year else {}
    last_r = load_season_file(last_year, "right") if last_year else {}

    all_pids = set(pre_l.keys()) | set(pre_r.keys())

    scored = {}
    for pid in all_pids:
        sl = pre_l.get(pid)
        sr = pre_r.get(pid)
        clp = car_l_prod.get(pid)
        crp = car_r_prod.get(pid)
        tot = total.get(pid)
        rec = recent.get(pid)

        prod_l = stabilize(
            sl["pa"] if sl else 0, blended_woba(sl, XWOBA_BLEND_SPLITS_LHP),
            clp["pa"] if clp else 0, blended_woba(clp, XWOBA_BLEND_CAREER_LHP),
            tot["pa"] if tot else 0, blended_woba(tot, XWOBA_BLEND_TOTAL_LHP),
            rec["pa"] if rec else 0, blended_woba(rec, XWOBA_BLEND_RECENT_LHP),
            side='L',
            career_cross_pa=crp["pa"] if crp else 0,
            career_cross_woba=blended_woba(crp, XWOBA_BLEND_CAREER_LHP),
        )
        prod_r = stabilize(
            sr["pa"] if sr else 0, blended_woba(sr, XWOBA_BLEND_SPLITS_RHP),
            crp["pa"] if crp else 0, blended_woba(crp, XWOBA_BLEND_CAREER_RHP),
            tot["pa"] if tot else 0, blended_woba(tot, XWOBA_BLEND_TOTAL_RHP),
            rec["pa"] if rec else 0, blended_woba(rec, XWOBA_BLEND_RECENT_RHP),
            side='R',
            career_cross_pa=clp["pa"] if clp else 0,
            career_cross_woba=blended_woba(clp, XWOBA_BLEND_CAREER_RHP),
        )

        scored[pid] = {
            "prod_l": prod_l, "prod_r": prod_r,
            "naive_last_l": naive_last_season(pid, last_l, LEAGUE_AVG_WOBA_LHP),
            "naive_last_r": naive_last_season(pid, last_r, LEAGUE_AVG_WOBA_RHP),
            "naive_avg_l": naive_career_avg(pid, car_l_naive, LEAGUE_AVG_WOBA_LHP),
            "naive_avg_r": naive_career_avg(pid, car_r_naive, LEAGUE_AVG_WOBA_RHP),
            "naive_reg_l": naive_career_regressed(pid, car_l_naive, LEAGUE_AVG_WOBA_LHP),
            "naive_reg_r": naive_career_regressed(pid, car_r_naive, LEAGUE_AVG_WOBA_RHP),
        }

    def build_pairs(pred_key, pre_dict, career_dict, post_dict):
        pairs_pred, pairs_actual = [], []
        for pid, s in scored.items():
            pred = s[pred_key]
            if pred is None:
                continue
            has_split = (pid in pre_dict and pre_dict[pid]["pa"] >= MIN_PA_SPLITS) or \
                        (pid in career_dict and career_dict[pid]["pa"] >= MIN_PA_CAREER)
            if not has_split:
                continue
            post = post_dict.get(pid)
            if post is None or post["pa"] < MIN_PA_POST:
                continue
            pairs_pred.append(pred)
            pairs_actual.append(post["woba"])
        return pairs_pred, pairs_actual

    results = {}
    key_map = {
        "production": ("prod_l", "prod_r"),
        "naive_last_season": ("naive_last_l", "naive_last_r"),
        "naive_career_avg": ("naive_avg_l", "naive_avg_r"),
        "naive_career_regressed": ("naive_reg_l", "naive_reg_r"),
    }
    for method, (kl, kr) in key_map.items():
        pl, al = build_pairs(kl, pre_l, car_l_naive, post_l)
        pr, ar = build_pairs(kr, pre_r, car_r_naive, post_r)
        results[method] = {
            "r_l": spearman_r(pl, al), "n_l": len(pl),
            "r_r": spearman_r(pr, ar), "n_r": len(pr),
        }

    return {"label": label, "results": results}


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    print("=== BASELINE COMPARISON -- \"Compared To What?\" ===")
    print("Results saved to: " + OUTPUT_FILE)
    print()

    window_results = [run_window(w) for w in TEST_WINDOWS]

    # -- Canary: recompute PRODUCTION across all 18 windows and check
    # it reproduces backtest.py's own published numbers. --
    prod_lhp = [wr["results"]["production"]["r_l"] for wr in window_results
                if wr["results"]["production"]["r_l"] is not None]
    prod_rhp = [wr["results"]["production"]["r_r"] for wr in window_results
                if wr["results"]["production"]["r_r"] is not None]
    avg_lhp = round(sum(prod_lhp) / len(prod_lhp), 4)
    avg_rhp = round(sum(prod_rhp) / len(prod_rhp), 4)
    avg_all = round(sum(prod_lhp + prod_rhp) / len(prod_lhp + prod_rhp), 4)
    canary_ok = (abs(avg_all - KNOWN_PRODUCTION_R_ALL) < 0.0005 and
                 abs(avg_lhp - KNOWN_PRODUCTION_R_LHP) < 0.0005 and
                 abs(avg_rhp - KNOWN_PRODUCTION_R_RHP) < 0.0005)
    print("CANARY CHECK (mirrored production code vs. backtest.py's own numbers):")
    print("  all=" + str(avg_all) + " (expected " + str(KNOWN_PRODUCTION_R_ALL) + ")" +
          "  LHP=" + str(avg_lhp) + " (expected " + str(KNOWN_PRODUCTION_R_LHP) + ")" +
          "  RHP=" + str(avg_rhp) + " (expected " + str(KNOWN_PRODUCTION_R_RHP) + ")")
    print("  " + ("PASS" if canary_ok else "*** FAIL -- do not trust the comparison below ***"))
    print()

    # -- Per-window table --
    print("-" * 100)
    print("PER-WINDOW SPEARMAN r  (L = vs LHP pitchers, R = vs RHP pitchers)")
    print("-" * 100)
    header = "Window".ljust(20) + "n(L,R)".ljust(12)
    for m in METHODS:
        header += METHOD_LABELS[m].rjust(20)
    print(header)
    for wr in window_results:
        prod = wr["results"]["production"]
        row = wr["label"].ljust(20)
        row += ("(" + str(prod["n_l"]) + "," + str(prod["n_r"]) + ")").ljust(12)
        for m in METHODS:
            r = wr["results"][m]
            cell = "L=" + fmt_r(r["r_l"]) + " R=" + fmt_r(r["r_r"])
            row += cell.rjust(20)
        print(row)
    print()

    # -- Summary --
    print("=" * 100)
    print("SUMMARY -- average Spearman r across all 18 windows (simple mean, matches backtest.py's metric)")
    print("=" * 100)
    summary = {}
    for m in METHODS:
        lhp_rs = [wr["results"][m]["r_l"] for wr in window_results if wr["results"][m]["r_l"] is not None]
        rhp_rs = [wr["results"][m]["r_r"] for wr in window_results if wr["results"][m]["r_r"] is not None]
        all_rs = lhp_rs + rhp_rs
        summary[m] = {
            "lhp": round(sum(lhp_rs) / len(lhp_rs), 4),
            "rhp": round(sum(rhp_rs) / len(rhp_rs), 4),
            "all": round(sum(all_rs) / len(all_rs), 4),
        }

    prod_all = summary["production"]["all"]
    print("Method".ljust(22) + "All".rjust(8) + "LHP".rjust(8) + "RHP".rjust(8) + "  Delta vs Production (all)")
    for m in METHODS:
        s = summary[m]
        if m == "production":
            delta_str = "--"
        else:
            delta = round(prod_all - s["all"], 4)
            delta_str = ("+" if delta >= 0 else "") + str(delta)
        print(METHOD_LABELS[m].ljust(22) + str(s["all"]).rjust(8) + str(s["lhp"]).rjust(8) +
              str(s["rhp"]).rjust(8) + "  " + delta_str)
    print()


if __name__ == "__main__":
    tee = Tee(OUTPUT_FILE)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee.console
        tee.close()
