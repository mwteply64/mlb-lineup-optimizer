import os
import sys
import json
import datetime
import pandas as pd

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
FOLDER = os.path.dirname(os.path.abspath(__file__))

FILE_2026_VS_RIGHT   = os.path.join(FOLDER, "2026_vs_right.csv")
FILE_2026_VS_LEFT    = os.path.join(FOLDER, "2026_vs_left.csv")
FILE_2026_TOTAL      = os.path.join(FOLDER, "2026_total.csv")
FILE_CAREER_VS_RIGHT = os.path.join(FOLDER, "career_vs_right.csv")
FILE_CAREER_VS_LEFT  = os.path.join(FOLDER, "career_vs_left.csv")
FILE_ROSTER          = os.path.join(FOLDER, "roster_resource.xlsx")
FILE_RECENT          = os.path.join(FOLDER, "recent.csv")

# Prior full-season files -- used only by the recency-restriction exemption
# check below (RECENCY_EXEMPT_*). Static/complete, no leakage risk.
FILE_SEASON_2025_VS_RIGHT = os.path.join(FOLDER, "historical testing", "season_2025_vs_right.csv")
FILE_SEASON_2025_VS_LEFT  = os.path.join(FOLDER, "historical testing", "season_2025_vs_left.csv")

PLAYER_COL  = "Name"
TEAM_COL    = "Team"

ROSTER_ID_COL    = "PlayerId"   # now holds MLBAM id, not FanGraphs id -- see load_roster()
ROSTER_MLBAM_COL = "MLBAMID"    # raw roster_resource.xlsx column sourcing that id
ROSTER_NAME_COL  = "Name"
ROSTER_TEAM_COL  = "Team"
ROSTER_POS_COL   = "Position"
ROSTER_BT_COL    = "B/T"
ROSTER_SECTION_COL = "Section"
IL_SECTION_PATTERN  = r"(?:Injured|Restricted|Bereavement)\s+List"   # matches roster_resource.xlsx Section values like "Injured List (Position Players)", "Restricted List", "Bereavement List"

PITCHER_POSITIONS = {"SP", "RP", "P", "CL"}

POSITION_EXPANSIONS = {
    "INF":  ["1B", "2B", "3B", "SS"],
    "OF":   ["LF", "CF", "RF"],
    "UTIL": ["1B", "2B", "3B", "SS", "LF", "CF", "RF", "C"],
    "IF":   ["1B", "2B", "3B", "SS"],
    "MI":   ["2B", "SS"],
    "CI":   ["1B", "3B"],
}

REQUIRED_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]

# -------------------------------------------------------
# wOBA SCORING ENGINE
# Ported from backtest.py's validated, CV-tested stabilization model
# (18-window walk-forward backtest against real Statcast outcomes).
# Constants below are copied verbatim from backtest.py's current
# production values. See backtest.py's module docstring for the full
# tuning history behind each one.
# -------------------------------------------------------

# PA minimums -- FIXED, do not tune (backtest.py's own description).
MIN_PA_SPLITS = 30
MIN_PA_TOTAL  = 30
MIN_PA_CAREER = 30
# No minimum for recent (any data used).

# League-average wOBA by pitcher handedness -- regression anchors tuned
# for ranking accuracy, NOT empirical averages. Deliberately suppressed
# below the true league average (see woba_to_wrcplus() below for the
# true-average constants used at the wRC+ conversion boundary instead).
LEAGUE_AVG_WOBA_LHP = 0.2943
LEAGUE_AVG_WOBA_RHP = 0.2575

# LHP model -- weights and regression PA constants
WEIGHT_SPLITS_LHP  = 0.213
WEIGHT_CAREER_LHP  = 0.5695
WEIGHT_TOTAL_LHP   = 0.247
WEIGHT_RECENT_LHP  = 0.3416

REGRESS_PA_SPLITS_LHP = 178
REGRESS_PA_CAREER_LHP = 711
REGRESS_PA_TOTAL_LHP  = 1020
REGRESS_PA_RECENT_LHP = 757

# RHP model -- weights and regression PA constants
WEIGHT_SPLITS_RHP  = 0.2774
WEIGHT_CAREER_RHP  = 0.2920
WEIGHT_TOTAL_RHP   = 0.2153
WEIGHT_RECENT_RHP  = 0.2153

REGRESS_PA_SPLITS_RHP = 285
REGRESS_PA_CAREER_RHP = 25
REGRESS_PA_TOTAL_RHP  = 400
REGRESS_PA_RECENT_RHP = 665

# Platoon tendency prior: scales total/recent wOBA by the player's career
# platoon ratio. 0.0 = no adjustment, 1.0 = full adjustment.
PLATOON_BLEND_LHP      = 0.7833
PLATOON_FACTOR_MIN_LHP = 0.612
PLATOON_FACTOR_MAX_LHP = 1.138

PLATOON_BLEND_RHP      = 0.31
PLATOON_FACTOR_MIN_RHP = 0.585
PLATOON_FACTOR_MAX_RHP = 1.325

# Batter-hand platoon prior (v16, shipped 2026-07-10 -- copied verbatim
# from backtest.py; see its v16 docstring section for the full CV
# evidence). The player's own career platoon ratio is regressed toward
# the league-mean ratio for the side he bats from vs LHP by
# PLATOON_PRIOR_PA_LHP pseudo-PA of prior evidence. Own career PA keeps
# full weight, so established/reverse-split hitters keep their own
# ratios. RHP ships at 0 (bit-for-bit no-op). Batter sides come from
# "historical testing/batter_sides.csv" (covers everyone with any
# 2016-2026 Statcast PA), with the roster's B/T column as fallback for
# brand-new players (switch hitters bat R vs LHP).
PLATOON_PRIOR_PA_LHP = 400
PLATOON_PRIOR_PA_RHP = 0
PLATOON_PRIOR_RATIO_VS_LHP = {
    "L": 0.9036224553071249,
    "R": 1.031091013338655,
}
FILE_BATTER_SIDES = os.path.join(FOLDER, "historical testing", "batter_sides.csv")

# xWOBA blend per component: 0.0 = pure wOBA, 1.0 = pure xWOBA. Falls
# back to pure wOBA when a player/file has no xWOBA value.
XWOBA_BLEND_SPLITS_LHP = 0.9187
XWOBA_BLEND_CAREER_LHP = 0.9187
XWOBA_BLEND_TOTAL_LHP  = 0.9187
XWOBA_BLEND_RECENT_LHP = 0.9187

XWOBA_BLEND_SPLITS_RHP = 0.80
XWOBA_BLEND_CAREER_RHP = 0.80
XWOBA_BLEND_TOTAL_RHP  = 0.80
XWOBA_BLEND_RECENT_RHP = 0.80

# Note: backtest.py's PA-based clamp_woba_by_pa() is intentionally NOT
# ported here -- backtest.py's validated production config runs with
# NO_CLAMP=True (disabling that clamp improves ranking accuracy), so
# stabilize() below skips it unconditionally on splits/career/total.
# clamp_wrc_by_pa() further below is a separate, independent mechanism
# used only by the fallback pool (not part of the ported/validated engine).

# wOBA -> wRC+ conversion anchor (boundary only -- see woba_to_wrcplus()).
# True 2026 league constants (FanGraphs Guts!, sourced 2026-06-19): these
# are live in-season estimates that will drift slightly as the season
# progresses -- recheck https://www.fangraphs.com/tools/guts periodically
# (e.g. monthly), same as WOBA_WEIGHTS[2026] in download_v3_expansion.py.
LEAGUE_AVG_WOBA_2026 = 0.318
WOBA_SCALE_2026      = 1.244
R_PER_PA_2026        = 0.118

LINEUP_SIZE          = 9
MAX_SAME_HAND_IN_ROW = 2
FALLBACK_MIN_SLOT    = 6
FALLBACK_MIN_PA_FOR_SCORE = 10

# Current-season PA floor: a main-pool player with fewer than
# RECENCY_PA_FLOOR total 2026 PA is barred from batting in slots
# 1..RECENCY_MIN_SLOT-1, UNLESS they're exempt -- i.e. they logged more
# than RECENCY_EXEMPT_MIN_PA_LAST_SEASON PA last season (2025) at better
# than RECENCY_EXEMPT_MIN_WRC_LAST_SEASON wRC+ (combined across both
# pitcher hands). Independent of, and stacks with, the fallback-pool
# mechanism above -- this one targets thin CURRENT-SEASON playing time
# on an otherwise normal main-pool player (e.g. an established veteran
# who's only had a handful of PA so far this year), not thin/missing
# data forcing fallback-pool sourcing.
RECENCY_PA_FLOOR                   = 30
RECENCY_MIN_SLOT                   = 6
RECENCY_EXEMPT_MIN_PA_LAST_SEASON  = 100
RECENCY_EXEMPT_MIN_WRC_LAST_SEASON = 100

RECENT_SLUMP_THRESHOLD = 70
SLUMP_TOP_SLOTS        = 5
SLUMP_CAREER_EXEMPTION = 95

HANDEDNESS_SWAP_MAX_COST = 7   # strict less-than used in comparison

# Career splits tiebreaker: when two adjacent players have 2026 split wRC+
# within SPLITS_TIEBREAK_WINDOW of each other, the lower-ranked player moves
# up if their career split wRC+ exceeds the higher-ranked player's by at least
# CAREER_TIEBREAK_ADVANTAGE points.
SPLITS_TIEBREAK_WINDOW    = 7
CAREER_TIEBREAK_ADVANTAGE = 7

# -------------------------------------------------------
# FORWARD-VALIDATION SUMMARY (written by live_forward_validation.py)
# -------------------------------------------------------
# run_comparison() reads this cached file, if present, instead of
# self-grading the optimizer's own picks against its own lookup table.
# See live_forward_validation.py's module docstring for the full
# non-circular rolling-cutoff methodology.
LIVE_FORWARD_VALIDATION_SUMMARY = os.path.join(FOLDER, "live_forward_validation_summary.json")
# -------------------------------------------------------

today = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(FOLDER, "results_" + today + ".txt")
# -------------------------------------------------------


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


# -------------------------------------------------------
# Flag helpers — robust across all pandas versions/dtypes
# -------------------------------------------------------

def _flag(player, key):
    """Read a boolean flag from a player Series safely."""
    try:
        val = player[key]
        if val is None:
            return False
        if isinstance(val, float) and val != val:
            return False
        return bool(val)
    except (KeyError, TypeError):
        return False

def is_fallback(player): return _flag(player, "_fallback")
def is_tiny(player):     return _flag(player, "_tiny_sample")
def is_recency_restricted(player): return _flag(player, "_recency_restricted")
def is_zero_data_fallback(player): return _flag(player, "_zero_data")


# -------------------------------------------------------
# Data loading
# -------------------------------------------------------

def load_csv(filepath, label):
    df = pd.read_csv(filepath)
    print("Loaded " + label + ": " + str(len(df)) + " rows.")
    return df


def expand_positions(pos_str):
    """Expand a position string into a list of individual position codes."""
    pos = str(pos_str).strip()
    if pos in POSITION_EXPANSIONS:
        return POSITION_EXPANSIONS[pos]
    parts = [p.strip() for p in pos.split("/")]
    expanded = []
    for p in parts:
        if p in POSITION_EXPANSIONS:
            expanded.extend(POSITION_EXPANSIONS[p])
        elif p and p != "nan":
            expanded.append(p)
    return list(set(expanded))


def load_roster(filepath, exclude_unavailable=True):
    """
    Load both sheets from the roster Excel file (vs RHP and vs LHP).

    exclude_unavailable (2026-07-09): when True (default -- correct for
    building TODAY's lineup), players whose Section marks them
    IL/restricted/bereavement are dropped. Pass False when the roster is
    being used to evaluate PAST decision points (live_forward_validation.py):
    a player on the IL today may have been fully available at an earlier
    cutoff, and excluding him there leaks today's injury knowledge into a
    past lineup decision. Post-cutoff PA gates handle true unavailability
    naturally on the grading side.
    Combines position eligibility from both sheets per player so that
    a player listed at different positions in each sheet gets the full
    union of eligible positions.
    Also stores the real-life platoon lineup slot (Role column, 1-9)
    from each sheet for use in the phase 2 comparison.

    ROSTER_ID_COL ("PlayerId") is populated from the MLBAMID column, not
    FanGraphs' own PlayerId column -- the live stats pipeline keys players
    by MLBAM id (player_id), so the join key has to match. The column name
    "PlayerId" is kept as-is downstream to avoid touching every call site
    that reads it.
    """
    xl = pd.ExcelFile(filepath)
    print("Roster file sheets found: " + str(xl.sheet_names))

    # Try to find the two platoon sheets regardless of exact capitalisation
    sheet_names = xl.sheet_names
    rhp_sheet = next((s for s in sheet_names if "rhp" in s.lower()), None)
    lhp_sheet = next((s for s in sheet_names if "lhp" in s.lower()), None)

    if rhp_sheet is None or lhp_sheet is None:
        print("  WARNING: Could not identify vs RHP / vs LHP sheets. Using first two sheets.")
        rhp_sheet = sheet_names[0]
        lhp_sheet = sheet_names[1] if len(sheet_names) > 1 else sheet_names[0]

    df_rhp = xl.parse(rhp_sheet)
    df_lhp = xl.parse(lhp_sheet)
    print("Loaded " + rhp_sheet + ": " + str(len(df_rhp)) + " rows.")
    print("Loaded " + lhp_sheet + ": " + str(len(df_lhp)) + " rows.")

    # Combine both sheets
    df_rhp["_sheet"] = "RHP"
    df_lhp["_sheet"] = "LHP"
    df_all = pd.concat([df_rhp, df_lhp], ignore_index=True)

    # Parse handedness from B/T column (take left side of slash)
    df_all["Handedness"] = df_all[ROSTER_BT_COL].astype(str).str.split("/").str[0].str.strip()

    # Exclude pitchers
    before = len(df_all)
    df_all = df_all[~df_all[ROSTER_POS_COL].astype(str).str.strip().isin(PITCHER_POSITIONS)].copy()
    removed = before - len(df_all)
    if removed > 0:
        print("  Excluded " + str(removed) + " pitcher row(s) from roster.")

    # Exclude unavailable players (Section column, e.g. "Injured List (Position Players)",
    # "Injured List (Starting Pitchers)", "Restricted List", "Bereavement List") so they
    # aren't put in the lineup just because they have recent stats on file.
    # Skipped when exclude_unavailable=False (retrospective validation use).
    if exclude_unavailable and ROSTER_SECTION_COL in df_all.columns:
        before = len(df_all)
        on_il = df_all[ROSTER_SECTION_COL].astype(str).str.contains(IL_SECTION_PATTERN, regex=True, case=False, na=False)
        df_all = df_all[~on_il].copy()
        removed = before - len(df_all)
        if removed > 0:
            print("  Excluded " + str(removed) + " unavailable (IL/restricted/bereavement) row(s) from roster.")

    # MLBAMID arrives as a float (e.g. 621493.0) -- clean to a plain int
    # string so it matches the stats files' player_id column exactly.
    df_all[ROSTER_ID_COL] = pd.to_numeric(df_all[ROSTER_MLBAM_COL], errors="coerce")
    before = len(df_all)
    df_all = df_all[df_all[ROSTER_ID_COL].notna()].copy()
    removed = before - len(df_all)
    if removed > 0:
        print("  Excluded " + str(removed) + " roster row(s) with no MLBAMID.")
    df_all[ROSTER_ID_COL] = df_all[ROSTER_ID_COL].astype(int).astype(str)

    # Build composite position eligibility per player across both sheets
    pos_by_player = {}
    hand_by_player = {}
    name_by_player = {}
    team_by_player = {}

    for _, row in df_all.iterrows():
        pid  = str(row[ROSTER_ID_COL])
        pos  = expand_positions(row[ROSTER_POS_COL])
        hand = str(row["Handedness"])
        name = str(row[ROSTER_NAME_COL])
        team = str(row[ROSTER_TEAM_COL])
        if pid not in pos_by_player:
            pos_by_player[pid]  = set()
            hand_by_player[pid] = hand
            name_by_player[pid] = name
            team_by_player[pid] = team
        pos_by_player[pid].update(pos)

    # Build the composite roster DataFrame
    records = []
    for pid, positions in pos_by_player.items():
        positions.add("DH")   # every position player is DH eligible
        records.append({
            ROSTER_ID_COL:   pid,
            ROSTER_NAME_COL: name_by_player[pid],
            ROSTER_TEAM_COL: team_by_player[pid],
            "Handedness":    hand_by_player[pid],
            "Positions":     list(positions),
        })
    roster = pd.DataFrame(records)
    print("  Roster loaded with " + str(len(roster)) + " unique position players.")

    # Also build and return the real-life platoon lineups for phase 2 comparison
    # Role column (1-9) gives the batting order slot. Built from the
    # already-cleaned df_all so the MLBAMID fix-up above applies here too.
    platoon_rhp = df_all[df_all["_sheet"] == "RHP"][[ROSTER_ID_COL, ROSTER_NAME_COL, ROSTER_TEAM_COL, "Role"]].copy()
    platoon_rhp = platoon_rhp[pd.to_numeric(platoon_rhp["Role"], errors="coerce").between(1, 9)]
    platoon_rhp["Role"] = platoon_rhp["Role"].astype(int)
    platoon_rhp = platoon_rhp.rename(columns={"Role": "Lineup_Slot"})

    platoon_lhp = df_all[df_all["_sheet"] == "LHP"][[ROSTER_ID_COL, ROSTER_NAME_COL, ROSTER_TEAM_COL, "Role"]].copy()
    platoon_lhp = platoon_lhp[pd.to_numeric(platoon_lhp["Role"], errors="coerce").between(1, 9)]
    platoon_lhp["Role"] = platoon_lhp["Role"].astype(int)
    platoon_lhp = platoon_lhp.rename(columns={"Role": "Lineup_Slot"})

    return roster, platoon_rhp, platoon_lhp


def apply_pa_filter(df, pa_col, min_pa, label):
    before = len(df)
    df = df[df[pa_col] >= min_pa].copy()
    removed = before - len(df)
    if removed > 0:
        print("  Filtered " + str(removed) + " player(s) from " + label +
              " with fewer than " + str(min_pa) + " PA.")
    return df


# -------------------------------------------------------
# Scoring
# Ported from backtest.py: regress() / blend_woba() / stabilize() carry
# over the same math and the same CV-validated constants. woba_to_wrcplus()
# is the one new piece -- a boundary conversion so every wOBA-space score
# still lands on the familiar wRC+ scale (100 = league average) that the
# rest of this file's thresholds (RECENT_SLUMP_THRESHOLD, etc.) assume.
# -------------------------------------------------------

def regress(woba, pa, regress_pa, league_avg):
    """Shrink a wOBA value toward a league-average anchor by PA sample size."""
    if pd.isna(woba) or pd.isna(pa):
        return woba
    return (woba * pa + league_avg * regress_pa) / (pa + regress_pa)


def blend_woba(woba, xwoba, blend):
    """Blend actual wOBA with expected wOBA (xwOBA). Falls back to pure
    actual wOBA (or pure xWOBA) if the other is missing."""
    if xwoba is None or pd.isna(xwoba):
        return woba
    if woba is None or pd.isna(woba):
        return xwoba
    return woba * (1 - blend) + xwoba * blend


def woba_to_wrcplus(woba):
    """
    Boundary conversion: wOBA -> wRC+ (100 = true 2026 league average).
    Standard formula: wRC+ = 100 + 100 * (wOBA - lgwOBA) / (wOBAScale * R/PA)
    Uses the TRUE league constants (LEAGUE_AVG_WOBA_2026 etc.), not the
    suppressed LEAGUE_AVG_WOBA_LHP/RHP regression anchors used inside
    stabilize() -- those exist purely to bias the internal regression step
    for ranking accuracy and would distort the "100 = average" meaning of
    every wRC+-scale number downstream.
    """
    if woba is None or pd.isna(woba):
        return float("nan")
    return round(100.0 + 100.0 * (woba - LEAGUE_AVG_WOBA_2026) /
                 (WOBA_SCALE_2026 * R_PER_PA_2026), 1)


def clamp_wrc_by_pa(wrc, pa):
    """
    PA-dependent wRC+ clamp: for every 10 PA, the allowed range expands
    by 10 in each direction from 100 (league average).
      10 PA  -> [90, 110]
      50 PA  -> [50, 150]
      100 PA -> [0, 200]  (fully unconstrained beyond this)
    Independent of the main wOBA engine below, which runs with
    backtest.py's validated NO_CLAMP=True (no PA clamp at all). This is
    test.py's own, separate safety margin, used only by the fallback pool.
    """
    if pd.isna(wrc) or pd.isna(pa):
        return wrc
    pa = float(pa)
    lo = max(0.0,   100.0 - pa)
    hi = min(200.0, 100.0 + pa)
    return max(lo, min(hi, float(wrc)))


def stabilize(splits_pa, splits_woba, career_pa, career_woba,
              total_pa, total_woba, recent_pa, recent_woba, side,
              career_cross_pa=0, career_cross_woba=None, anchor_override=None,
              prior_side=None):
    """
    Combine all four wOBA components (current-season splits, career
    splits, current-season total, recent) into one regressed, weighted
    wOBA. Ported from backtest.py's stabilize(): same math, same
    validated hand-specific constants, NO_CLAMP=True (PA-based wOBA
    clamp permanently disabled per backtest.py's CV finding).

    Platoon tendency prior: total/recent wOBA get scaled by the player's
    own career platoon ratio -- this hand's career wOBA divided by the
    PA-weighted COMBINED career wOBA across both hands (career_cross_pa/
    career_cross_woba supply the opposite hand's side of that blend).
    The raw ratio is clipped to [PLATOON_FACTOR_MIN, PLATOON_FACTOR_MAX]
    first, then blended toward 1.0 by PLATOON_BLEND -- identical to
    backtest.py's validated block (2026-07-08 fix; see comment at the
    platoon_factor computation below).

    anchor_override: if given, replaces the hand-specific regression
    anchor (LEAGUE_AVG_WOBA_LHP/RHP) with this value everywhere `la` is
    used below. The default anchors are deliberately suppressed below
    the true league average for ranking accuracy (see the constants'
    comments) -- great for relative ordering, but it means a thin-sample
    player gets shrunk toward a below-100 wRC+ point even with zero
    actual signal of being a below-average hitter. That's fine for
    ranking/ordering (this function's normal use), but it corrupts any
    ABSOLUTE-scale comparison (e.g. the runs-left-on-table math in
    run_comparison()), since real lineups are full of exactly the kind
    of thin-sample players this drags down for no talent-based reason.
    Pass anchor_override=LEAGUE_AVG_WOBA_2026 to get a valuation-fair
    version of the same blend, anchored at the true 100-wRC+ point.
    """
    if side == "L":
        la = LEAGUE_AVG_WOBA_LHP
        WS, WC, WT, WR = WEIGHT_SPLITS_LHP, WEIGHT_CAREER_LHP, WEIGHT_TOTAL_LHP, WEIGHT_RECENT_LHP
        RPS, RPC, RPT, RPR = (REGRESS_PA_SPLITS_LHP, REGRESS_PA_CAREER_LHP,
                              REGRESS_PA_TOTAL_LHP, REGRESS_PA_RECENT_LHP)
        platoon_blend  = PLATOON_BLEND_LHP
        pf_min, pf_max = PLATOON_FACTOR_MIN_LHP, PLATOON_FACTOR_MAX_LHP
        prior_pa, prior_table = PLATOON_PRIOR_PA_LHP, PLATOON_PRIOR_RATIO_VS_LHP
    else:
        la = LEAGUE_AVG_WOBA_RHP
        WS, WC, WT, WR = WEIGHT_SPLITS_RHP, WEIGHT_CAREER_RHP, WEIGHT_TOTAL_RHP, WEIGHT_RECENT_RHP
        RPS, RPC, RPT, RPR = (REGRESS_PA_SPLITS_RHP, REGRESS_PA_CAREER_RHP,
                              REGRESS_PA_TOTAL_RHP, REGRESS_PA_RECENT_RHP)
        platoon_blend  = PLATOON_BLEND_RHP
        pf_min, pf_max = PLATOON_FACTOR_MIN_RHP, PLATOON_FACTOR_MAX_RHP
        prior_pa, prior_table = PLATOON_PRIOR_PA_RHP, {}   # no vs-RHP priors (knob ships at 0)

    if anchor_override is not None:
        la = anchor_override

    # Platoon factor -- EXACT port of backtest.py's validated block
    # (2026-07-08 fix, extended by v16 on 2026-07-10). Ratio of this-hand
    # career wOBA to the PA-weighted COMBINED career wOBA (both hands),
    # regressed toward the batter-side league prior by prior_pa pseudo-PA
    # (v16 -- a no-op when prior_pa == 0 or prior_side is unknown), then
    # ratio clamped FIRST, then blended toward 1.0. Mirrors backtest.py's
    # stabilize() verbatim; every tuned constant belongs to that math.
    platoon_factor = 1.0
    ratio, own_pa = None, 0
    if (platoon_blend > 0 and
            career_woba is not None and not pd.isna(career_woba) and
            career_pa is not None and not pd.isna(career_pa) and career_pa > 0 and
            career_cross_woba is not None and not pd.isna(career_cross_woba) and
            career_cross_pa is not None and not pd.isna(career_cross_pa) and career_cross_pa > 0):
        combined_pa = career_pa + career_cross_pa
        combined_woba = (career_woba * career_pa +
                         career_cross_woba * career_cross_pa) / combined_pa
        if combined_woba > 0:
            ratio = career_woba / combined_woba
            own_pa = combined_pa

    prior = prior_table.get(prior_side) if (prior_pa > 0 and prior_side) else None
    if prior is not None:
        est = prior if ratio is None else (
            (ratio * own_pa + prior * prior_pa) / (own_pa + prior_pa))
    else:
        est = ratio
    if est is not None:
        clamped_factor = max(pf_min, min(pf_max, est))
        platoon_factor = 1.0 + platoon_blend * (clamped_factor - 1.0)

    components, weights = [], []

    if splits_woba is not None and not pd.isna(splits_woba) and splits_pa >= MIN_PA_SPLITS:
        components.append(regress(splits_woba, splits_pa, RPS, la))
        weights.append(WS)

    if career_woba is not None and not pd.isna(career_woba) and career_pa >= MIN_PA_CAREER:
        components.append(regress(career_woba, career_pa, RPC, la))
        weights.append(WC)

    if total_woba is not None and not pd.isna(total_woba) and total_pa >= MIN_PA_TOTAL:
        adj = total_woba * platoon_factor
        components.append(regress(adj, total_pa, RPT, la))
        weights.append(WT)

    if recent_woba is not None and not pd.isna(recent_woba) and recent_pa > 0:
        adj = recent_woba * platoon_factor
        components.append(regress(adj, recent_pa, RPR, la))
        weights.append(WR)

    if not components:
        return None

    total_weight = sum(weights)
    return sum(c * (w / total_weight) for c, w in zip(components, weights))


# -------------------------------------------------------
# Recency restriction -- prior-season exemption lookup
# -------------------------------------------------------

_BATTER_SIDES_CACHE = None


def load_batter_sides_lookup():
    """
    Build {player_id: {"L": side_vs_LHP, "R": side_vs_RHP}} from
    batter_sides.csv (download_batter_sides.py; sides derived from
    Statcast's per-pitch `stand`, so switch hitters resolve correctly
    per matchup). Used by build_master() to feed the v16 batter-hand
    platoon prior. Cached after first load; a missing file degrades to
    an empty lookup (the prior falls back to the roster B/T column, and
    to a no-op where that's missing too) rather than crashing.
    """
    global _BATTER_SIDES_CACHE
    if _BATTER_SIDES_CACHE is not None:
        return _BATTER_SIDES_CACHE
    lookup = {}
    try:
        df = pd.read_csv(FILE_BATTER_SIDES)
        for row in df.itertuples(index=False):
            pid = str(row.player_id).strip()
            entry = {}
            for hand, col in (("L", "side_vs_L"), ("R", "side_vs_R")):
                v = str(getattr(row, col, "")).strip()
                if v in ("L", "R"):
                    entry[hand] = v
            if entry:
                lookup[pid] = entry
    except (FileNotFoundError, OSError, KeyError) as e:
        print("  WARNING: could not load batter_sides.csv (" + str(e) +
              "). Platoon prior will rely on the roster B/T column only.")
    _BATTER_SIDES_CACHE = lookup
    return lookup


_LAST_SEASON_LOOKUP_CACHE = None


def load_last_season_lookup():
    """
    Build a {PlayerId: {"pa": <2025 PA, both hands combined>,
    "wrc_plus": <2025 wRC+, PA-weighted blend of both hands>}} lookup from
    the prior full season's vs-RHP/vs-LHP files. Used only by the
    RECENCY_PA_FLOOR exemption check in build_master() below. Cached after
    first load (module-level, lazy -- never runs unless build_master() is
    actually called, so importing this module as a library, e.g. from
    live_forward_validation.py, carries no extra cost or failure risk).
    Missing/unreadable files degrade to an empty lookup (nobody exempted)
    instead of crashing -- this is a secondary gate, not core scoring.
    """
    global _LAST_SEASON_LOOKUP_CACHE
    if _LAST_SEASON_LOOKUP_CACHE is not None:
        return _LAST_SEASON_LOOKUP_CACHE

    combined = {}
    try:
        for filepath in (FILE_SEASON_2025_VS_RIGHT, FILE_SEASON_2025_VS_LEFT):
            df = pd.read_csv(filepath)
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
            df = df[df["player_id"].notna()]
            df["player_id"] = df["player_id"].astype(int).astype(str)
            for _, row in df.iterrows():
                pid  = row["player_id"]
                pa   = float(row["pa"]) if pd.notna(row["pa"]) else 0.0
                woba = float(row["woba"]) if pd.notna(row["woba"]) else None
                entry = combined.setdefault(pid, {"pa": 0.0, "pa_woba_sum": 0.0})
                entry["pa"] += pa
                if woba is not None:
                    entry["pa_woba_sum"] += pa * woba
    except (FileNotFoundError, OSError, KeyError) as e:
        print("  WARNING: could not load 2025 season file(s) for the recency-restriction "
              "exemption check (" + str(e) + "). No players will be exempted.")
        _LAST_SEASON_LOOKUP_CACHE = {}
        return _LAST_SEASON_LOOKUP_CACHE

    lookup = {}
    for pid, entry in combined.items():
        pa = entry["pa"]
        blended_woba = (entry["pa_woba_sum"] / pa) if pa > 0 else None
        lookup[pid] = {
            "pa":       pa,
            "wrc_plus": woba_to_wrcplus(blended_woba) if blended_woba is not None else float("nan"),
        }

    _LAST_SEASON_LOOKUP_CACHE = lookup
    return lookup


def build_master(splits_df, career_same_df, career_cross_df, total_df, recent_df, roster, side):
    """
    Build the scored player pool for one pitcher-handedness side ('L' or
    'R'). Every component is xWOBA-blended, then run through stabilize()
    (regression + platoon prior), then converted to wRC+ at the boundary
    via woba_to_wrcplus() so every downstream column keeps its old
    wRC+-scale name and meaning.

    Base population = the union of every player appearing in ANY of the
    four source tables (2026 splits, career splits, 2026 total, recent) --
    NOT just players who clear MIN_PA_SPLITS in this hand's 2026 splits
    file. MIN_PA_SPLITS/MIN_PA_CAREER/MIN_PA_TOTAL still apply exactly as
    before, but as per-component gates *inside* stabilize() (it already
    drops any single component that's missing or too thin and renormalizes
    over whatever's left -- that's the same code path every master player
    already goes through whenever their career/total/recent PA is too
    low). The only change is that a player thin on current-season splits
    specifically no longer gets discarded before stabilize() ever sees
    them. Previously they fell out of this PA-weighted blend entirely and
    into build_fallback_pool()'s cruder, hard-capped (FALLBACK_WRC_CAP)
    formula -- an inconsistent two-tier system that was the root cause of
    the real-vs-optimized scoring asymmetry fixed in run_comparison().
    Tiny samples are still protected: stabilize()'s regress() shrinks hard
    toward league average as PA shrinks, so a 1-PA player still lands
    close to 100, not at some wild extreme.

    Also sets _recency_restricted (see RECENCY_PA_FLOOR) on every row:
    True if the player's current-season (2026) total PA is below
    RECENCY_PA_FLOOR and they don't clear the prior-season exemption
    (RECENCY_EXEMPT_MIN_PA_LAST_SEASON PA at better than
    RECENCY_EXEMPT_MIN_WRC_LAST_SEASON wRC+ in 2025). balance_handedness()
    uses this to bar the player from slots 1..RECENCY_MIN_SLOT-1 --
    independent of, and stacked on top of, the separate is_fallback()
    pool-membership restriction.

    Stabilized_wRC+ is the only score this function produces; it's used
    both for ranking/ordering and as the optimizer's player-quality
    estimate. (Earlier versions also computed a second, true-average-
    anchored Valuation_wRC+ column purely to self-grade the optimizer's
    own picks in run_comparison() -- removed when that self-grading
    comparison was replaced with live_forward_validation.py's
    non-circular forward-validation, which grades against real outcomes
    instead of this table.)
    """
    if side == "L":
        xb_splits, xb_career = XWOBA_BLEND_SPLITS_LHP, XWOBA_BLEND_CAREER_LHP
        xb_total,  xb_recent = XWOBA_BLEND_TOTAL_LHP,  XWOBA_BLEND_RECENT_LHP
    else:
        xb_splits, xb_career = XWOBA_BLEND_SPLITS_RHP, XWOBA_BLEND_CAREER_RHP
        xb_total,  xb_recent = XWOBA_BLEND_TOTAL_RHP,  XWOBA_BLEND_RECENT_RHP

    def prep(df):
        # player_id arrives as a float (e.g. 621493.0) -- cast through
        # int first so it matches the roster's cleaned MLBAMID-based
        # PlayerId strings ("621493"), not "621493.0".
        df = df.copy()
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
        df = df[df["player_id"].notna()]
        df["player_id"] = df["player_id"].astype(int).astype(str)
        return df.drop_duplicates(subset=["player_id"]).set_index("player_id")

    splits_df       = prep(splits_df)
    career_same_df   = prep(career_same_df)
    career_cross_df  = prep(career_cross_df)
    total_df         = prep(total_df)
    recent_df        = prep(recent_df)

    # Diagnostic only -- no longer used to restrict population. Reports
    # how many players are thin on current-season splits specifically
    # and will be scored off career/total/recent instead.
    thin_splits = int((splits_df["pa"] < MIN_PA_SPLITS).sum())
    if thin_splits > 0:
        print("  " + str(thin_splits) + " player(s) have fewer than " + str(MIN_PA_SPLITS) +
              " PA in 2026 splits; scored off career/total/recent instead (no longer excluded).")

    # Population = union of every source relevant to scoring this hand.
    # career_cross_df is deliberately excluded -- it's a platoon-factor
    # input, not a scoring component, so it shouldn't gate inclusion.
    all_pids = set(splits_df.index) | set(career_same_df.index) | set(total_df.index) | set(recent_df.index)

    last_season_lookup = load_last_season_lookup()

    # v16 batter-hand platoon prior inputs: primary source is
    # batter_sides.csv (actual Statcast stand per matchup); fallback for
    # brand-new players is the roster's B/T bats letter, resolved per
    # matchup the standard way (a switch hitter bats R vs LHP, L vs RHP).
    batter_sides = load_batter_sides_lookup()
    roster_bats = dict(zip(roster[ROSTER_ID_COL].astype(str),
                           roster["Handedness"].astype(str).str.strip().str.upper()))

    def prior_side_for(pid):
        entry = batter_sides.get(pid)
        if entry and side in entry:
            return entry[side]
        bats = roster_bats.get(pid, "")
        if bats == "S":
            return "R" if side == "L" else "L"
        if bats in ("L", "R"):
            return bats
        return None

    rows = []
    for pid in sorted(all_pids):
        srow = splits_df.loc[pid]       if pid in splits_df.index       else None
        crow = career_same_df.loc[pid]  if pid in career_same_df.index  else None
        xrow = career_cross_df.loc[pid] if pid in career_cross_df.index else None
        trow = total_df.loc[pid]        if pid in total_df.index        else None
        rrow = recent_df.loc[pid]       if pid in recent_df.index       else None

        splits_woba = blend_woba(srow["woba"], srow.get("xwoba"), xb_splits) if srow is not None else None
        career_woba = blend_woba(crow["woba"], crow.get("xwoba"), xb_career) if crow is not None else None
        total_woba  = blend_woba(trow["woba"], trow.get("xwoba"), xb_total)  if trow is not None else None
        recent_woba = blend_woba(rrow["woba"], rrow.get("xwoba"), xb_recent) if rrow is not None else None
        cross_woba  = blend_woba(xrow["woba"], xrow.get("xwoba"), xb_career) if xrow is not None else None

        stab_woba = stabilize(
            srow["pa"] if srow is not None else 0, splits_woba,
            crow["pa"] if crow is not None else 0, career_woba,
            trow["pa"] if trow is not None else 0, total_woba,
            rrow["pa"] if rrow is not None else 0, recent_woba,
            side,
            career_cross_pa=(xrow["pa"] if xrow is not None else 0),
            career_cross_woba=cross_woba,
            prior_side=prior_side_for(pid),
        )
        if stab_woba is None:
            continue

        # Recency restriction: thin current-season (2026) PA bars a player
        # from the top of the order unless last season's (2025) performance
        # exempts them. Missing total-PA data counts as 0 PA here (not NaN)
        # -- a player we have no current-season record for at all is, if
        # anything, MORE in need of the restriction, not less.
        current_pa_2026 = trow["pa"] if trow is not None else 0.0
        if pd.isna(current_pa_2026):
            current_pa_2026 = 0.0
        last_season = last_season_lookup.get(pid)
        recency_exempt = bool(
            last_season is not None and
            last_season["pa"] > RECENCY_EXEMPT_MIN_PA_LAST_SEASON and
            pd.notna(last_season["wrc_plus"]) and
            last_season["wrc_plus"] > RECENCY_EXEMPT_MIN_WRC_LAST_SEASON
        )
        recency_restricted = bool(current_pa_2026 < RECENCY_PA_FLOOR and not recency_exempt)

        rows.append({
            "PlayerId":        pid,
            "PA_2026splits":   (srow["pa"] if srow is not None else float("nan")),
            "wRC+_2026splits": woba_to_wrcplus(splits_woba),
            "PA_career":       (crow["pa"] if crow is not None else float("nan")),
            "wRC+_career":     (woba_to_wrcplus(career_woba) if crow is not None else float("nan")),
            "PA_2026total":    (trow["pa"] if trow is not None else float("nan")),
            "wRC+_2026total":  (woba_to_wrcplus(total_woba) if trow is not None else float("nan")),
            "PA_recent":       (rrow["pa"] if rrow is not None else float("nan")),
            "wRC+_recent":     (woba_to_wrcplus(recent_woba) if rrow is not None else float("nan")),
            "Stabilized_wRC+": woba_to_wrcplus(stab_woba),
            "_recency_restricted": recency_restricted,
        })

    master = pd.DataFrame(rows)
    if len(master) == 0:
        return master

    roster_lookup = roster[[ROSTER_ID_COL, PLAYER_COL, TEAM_COL, "Handedness", "Positions"]].copy()
    roster_lookup[ROSTER_ID_COL] = roster_lookup[ROSTER_ID_COL].astype(str)
    master = master.merge(roster_lookup, on="PlayerId", how="left")
    return master.dropna(subset=[PLAYER_COL]).reset_index(drop=True)


# -------------------------------------------------------
# Fallback pool
# -------------------------------------------------------

FALLBACK_WEIGHT_CAREER = 0.60
FALLBACK_WEIGHT_TOTAL  = 0.40
FALLBACK_WRC_CAP       = 120   # Hard ceiling on any fallback player's stabilized wRC+


def fallback_score(wrc_total, wrc_career):
    has_total  = pd.notna(wrc_total)
    has_career = pd.notna(wrc_career)
    if has_total and has_career:
        return round(wrc_total * FALLBACK_WEIGHT_TOTAL + wrc_career * FALLBACK_WEIGHT_CAREER, 1)
    elif has_total:
        return round(float(     wrc_total), 1)
    elif has_career:
        return round(float(wrc_career), 1)
    return None


def build_fallback_pool(team, total_df, career_df, roster, league_avg_wrc=100.0):
    """
    Last-resort pool for filling lineup slots the main pool can't cover.
    Uses season-total PA/wOBA plus same-hand career wOBA (NOT handedness-
    specific 2026 splits, by definition -- that's what makes this the
    fallback tier). Pure actual wOBA, no xWOBA blend and no platoon prior
    -- this is test.py's own, deliberately simple safety-net tier,
    independent of backtest.py's validated engine, and keeps its own
    PA-based clamp_wrc_by_pa() plus a hard wRC+ cap as a conservative
    margin for the riskiest data tier.

    Built from the full active roster for this team, NOT from total_df --
    total_df only contains players who've recorded at least one PA this
    season, so starting from it silently excludes a true zero-PA callup
    or debut from ever being a fallback candidate, even though the roster
    knows exactly what position they play. Starting from the roster and
    left-joining total/career data means a genuine zero-data player is
    still a candidate: fallback_score() returns None for them (no total
    AND no career record), and instead of being dropped, they're scored
    at league_avg_wrc -- the calling code's own empirical mean
    Stabilized_wRC+ for this hand, not a hardcoded constant, since the
    model's stabilization/shrinkage already pulls the "average" player
    somewhere other than a flat 100.

    No position-count requirement anymore either: a single-position
    specialist is just as valid a last-resort fill as a flexible one.
    Requiring 2+ eligible positions never gated anything meaningful here
    -- any real, eligible candidate (single- or multi-position) would
    already have been picked up by assign_lineup_positions()'s own DP
    matching before this pool is ever consulted; the filter only ever
    risked excluding a legitimate last-resort single-position fill.
    """
    pool = roster[roster[TEAM_COL] == team][
        [ROSTER_ID_COL, PLAYER_COL, TEAM_COL, "Handedness", "Positions"]].copy()
    pool = pool.rename(columns={ROSTER_ID_COL: "PlayerId"})
    pool["PlayerId"] = pool["PlayerId"].astype(str)

    total = total_df.rename(columns={"player_id": "PlayerId"}).copy()
    total["PlayerId"] = pd.to_numeric(total["PlayerId"], errors="coerce")
    total = total[total["PlayerId"].notna()]
    total["PlayerId"] = total["PlayerId"].astype(int).astype(str)
    total_info = total[["PlayerId", "woba", "pa"]].rename(columns={"woba": "total_woba_raw"})
    pool = pool.merge(total_info, on="PlayerId", how="left")

    career = career_df.rename(columns={"player_id": "PlayerId"}).copy()
    career["PlayerId"] = pd.to_numeric(career["PlayerId"], errors="coerce")
    career = career[career["PlayerId"].notna()]
    career["PlayerId"] = career["PlayerId"].astype(int).astype(str)
    career_info = career[["PlayerId", "woba", "pa"]].rename(
        columns={"woba": "career_woba_raw", "pa": "PA_career_raw"})
    pool = pool.merge(career_info, on="PlayerId", how="left")

    pool["_fallback"]       = True
    # NaN-safe: a player with no total_df row at all has pa == NaN, which
    # must count as tiny (it's the thinnest possible sample), not silently
    # fail the "< FALLBACK_MIN_PA_FOR_SCORE" comparison the way NaN < n does.
    pool["_tiny_sample"]    = pool["pa"].apply(lambda x: bool(pd.isna(x) or x < FALLBACK_MIN_PA_FOR_SCORE))
    pool["wRC+_2026splits"] = float("nan")
    pool["PA_2026splits"]   = float("nan")
    pool["PA_career"]       = float("nan")   # kept NaN so display shows N/A (below PA threshold)
    pool["wRC+_2026total"]  = pool["total_woba_raw"].apply(woba_to_wrcplus)
    pool["PA_2026total"]    = pool["pa"]
    pool["wRC+_recent"]     = float("nan")
    pool["PA_recent"]       = float("nan")
    pool["Stabilized_wRC+"] = pool.apply(
        lambda r: fallback_score(
            clamp_wrc_by_pa(woba_to_wrcplus(r["total_woba_raw"]), r["pa"]),
            # Clamp career by its own PA so tiny-sample outliers don't dominate
            clamp_wrc_by_pa(woba_to_wrcplus(r["career_woba_raw"]), r["PA_career_raw"])), axis=1)

    # True zero-data players (no total AND no career record anywhere) get
    # fallback_score() == None. Mark them before filling so print_lineup()
    # can flag them distinctly from a thin-but-real-data fallback, then
    # give them the model's own league-average score instead of dropping
    # them -- a real roster-tagged player should still be able to occupy
    # an otherwise-empty slot, even with zero history to score off of.
    pool["_zero_data"] = pool["Stabilized_wRC+"].isna()
    pool["Stabilized_wRC+"] = pool["Stabilized_wRC+"].fillna(league_avg_wrc)
    # Hard cap: fallback players lack handedness splits by definition, so cap confidence
    pool["Stabilized_wRC+"] = pool["Stabilized_wRC+"].clip(upper=FALLBACK_WRC_CAP)
    return pool.dropna(subset=["Positions"]).reset_index(drop=True)


# -------------------------------------------------------
# Lineup assignment
# -------------------------------------------------------

def _optimal_assignment(candidates, slots):
    """Maximum-weight bipartite matching of players to required lineup slots.

    Replaces the old greedy slot-by-slot fill, which could lock a positionally-
    flexible player (e.g. INF/OF) into an early slot in fixed order, excluding a
    less-flexible but higher-scoring player from a slot only they could fill.
    This DP considers all 9 slots simultaneously and always finds the globally
    optimal assignment.

    Uses bitmask DP (no external deps):
      dp[k][mask] = best total value achievable by assigning the first k
                    candidates to exactly the slots indicated by mask.
      value = BIG + score per covered slot, so slot-coverage is optimized
              before raw score (mirrors "fill all slots, then maximize score").
    """
    n_slots  = len(slots)
    n_states = 1 << n_slots
    n_cand   = len(candidates)
    NEG_INF  = float("-inf")
    BIG      = 1_000_000.0

    cand_scores = [get_score(c) for c in candidates]
    cand_elig   = [set(c["Positions"]) if isinstance(c["Positions"], list) else set()
                   for c in candidates]

    dp = [[NEG_INF] * n_states for _ in range(n_cand + 1)]
    dp[0][0] = 0.0

    for k in range(1, n_cand + 1):
        c_idx = k - 1
        score = cand_scores[c_idx]
        elig  = cand_elig[c_idx]
        prev  = dp[k - 1]
        cur   = dp[k]
        for mask in range(n_states):
            best = prev[mask]
            for s_idx, slot in enumerate(slots):
                if not (mask & (1 << s_idx)):
                    continue
                if slot not in elig:
                    continue
                prev_mask = mask ^ (1 << s_idx)
                pv = prev[prev_mask]
                if pv == NEG_INF:
                    continue
                val = pv + BIG + score
                if val > best:
                    best = val
            cur[mask] = best

    best_mask = max(range(n_states), key=lambda m: dp[n_cand][m])

    assignment = {slot: None for slot in slots}
    mask = best_mask
    k = n_cand
    while k > 0 and mask != 0:
        c_idx = k - 1
        score = cand_scores[c_idx]
        elig  = cand_elig[c_idx]
        if dp[k][mask] == dp[k - 1][mask]:
            k -= 1
            continue
        placed = False
        for s_idx, slot in enumerate(slots):
            if not (mask & (1 << s_idx)):
                continue
            if slot not in elig:
                continue
            prev_mask = mask ^ (1 << s_idx)
            pv = dp[k - 1][prev_mask]
            if pv == NEG_INF:
                continue
            if abs(pv + BIG + score - dp[k][mask]) < 1e-6:
                assignment[slot] = candidates[c_idx]
                mask = prev_mask
                k -= 1
                placed = True
                break
        if not placed:
            k -= 1

    return assignment


def assign_lineup_positions(team_data, fallback_pool=None):
    available = team_data.copy().reset_index(drop=True)
    if "_fallback"    not in available.columns: available["_fallback"]    = False
    if "_tiny_sample" not in available.columns: available["_tiny_sample"] = False
    slot_order = ["C"] + [p for p in REQUIRED_POSITIONS if p != "C"]

    candidates = [row for _, row in available.iterrows()]
    optimal = _optimal_assignment(candidates, slot_order)

    assigned = {}
    used_ids = set()
    for slot in slot_order:
        best = optimal.get(slot)
        if best is not None:
            best = best.copy()
            best["_slot"] = slot
            assigned[slot] = best
            used_ids.add(best["PlayerId"])
        else:
            assigned[slot] = None

    if fallback_pool is not None:
        for slot in slot_order:
            if assigned[slot] is not None:
                continue
            fb_eligible = fallback_pool[
                fallback_pool["PlayerId"].apply(lambda pid: pid not in used_ids) &
                fallback_pool["Positions"].apply(lambda pos: slot in pos if isinstance(pos, list) else False)
            ].sort_values("Stabilized_wRC+", ascending=False)
            if len(fb_eligible) > 0:
                best = fb_eligible.iloc[0].copy()
                best["_slot"]     = slot
                best["_fallback"] = True
                assigned[slot]    = best
                used_ids.add(best["PlayerId"])
                note = " [tiny sample — will bat last]" if is_tiny(best) else ""
                print("  Fallback used for slot " + slot + ": " + str(best[PLAYER_COL]) + note)

    for slot in slot_order:
        if assigned[slot] is None:
            print("  WARNING: Could not fill slot " + slot + " from main pool or fallback pool.")

    return assigned


# -------------------------------------------------------
# Ordering helpers
# -------------------------------------------------------

def get_score(player):
    val = player.get("Stabilized_wRC+", None)
    if val is None or (isinstance(val, float) and val != val):
        return 0.0
    return float(val)

def get_recent_wrc(player):
    val = player.get("wRC+_recent", None)
    if val is None or (isinstance(val, float) and val != val):
        return None
    return float(val)

def get_career_wrc(player):
    val = player.get("wRC+_career", None)
    if val is None or (isinstance(val, float) and val != val):
        return None
    return float(val)

def is_slumping(player):
    recent = get_recent_wrc(player)
    if recent is None or recent >= RECENT_SLUMP_THRESHOLD:
        return False
    career = get_career_wrc(player)
    return career is None or career <= SLUMP_CAREER_EXEMPTION


# -------------------------------------------------------
# Main ordering function
# -------------------------------------------------------

def balance_handedness(assigned):
    """
    Steps:
    1   Sort by Stabilized_wRC+ descending. This is the primary and only
        ranking criterion — career data is already embedded in the score.
    2   Fallback zone: regular fallbacks compete by score in slots
        FALLBACK_MIN_SLOT+. Tiny-sample fallbacks always last.
    2b  Recency-restriction zone: main-pool players thin on current-season
        PA (and not exempted by last season's performance) are barred
        from slots 1..RECENCY_MIN_SLOT-1, competing by score below that
        line alongside fallbacks. Independent of step 2's pool-membership
        gate -- see RECENCY_PA_FLOOR.
    3   Slump restriction: slumping non-exempt players removed from top
        SLUMP_TOP_SLOTS. Fallbacks and recency-restricted players never
        used as replacements.
    4   Re-sort top zone by score (corrects slump-swap inversions).
    4.5 Re-sort bottom zone by score (tiny-sample last).
    5   Handedness streak balancing. Fallbacks, recency-restricted, and
        slumping players never promoted into protected zones. Swaps use
        strict less-than threshold.
    6   Post-swap score-order correction. Fallback and recency-restricted
        players are never moved by this step (insulated both ways).
    """

    players = [p for p in assigned.values() if p is not None]
    order   = sorted(players, key=get_score, reverse=True)

    # --- Step 2: fallback zone ---
    cutoff   = FALLBACK_MIN_SLOT - 1
    non_fb   = [p for p in order if not is_fallback(p)]
    reg_fb   = sorted([p for p in order if is_fallback(p) and not is_tiny(p)],
                      key=get_score, reverse=True)
    tiny_fb  = sorted([p for p in order if is_tiny(p)],
                      key=get_score, reverse=True)
    bot_norm = sorted(reg_fb + non_fb[cutoff:], key=get_score, reverse=True)
    order    = (non_fb[:cutoff] + bot_norm + tiny_fb)[:LINEUP_SIZE]

    # --- Step 2b: recency-restriction zone ---
    # Same shape as step 2, but keyed on RECENCY_PA_FLOOR/is_recency_restricted()
    # instead of pool membership. Tiny-sample fallbacks still always bat last.
    # IMPORTANT: filter restricted players OUT of the candidate pool BEFORE
    # slicing the protected zone (mirrors step 2's non_fb[:cutoff] pattern) --
    # slicing order[:rec_cutoff] first and removing restricted players from
    # inside that slice (the original, buggy approach) shrinks the protected
    # zone below rec_cutoff players, which then lets a high-scoring demoted
    # player resurface right at the boundary and get pulled back to slot 1
    # by step 4's top-zone resort.
    rec_cutoff = RECENCY_MIN_SLOT - 1
    non_rec_all = [p for p in order if not is_recency_restricted(p)]
    reg_rec     = sorted([p for p in order if is_recency_restricted(p) and not is_tiny(p)],
                         key=get_score, reverse=True)
    tiny_tail   = sorted([p for p in non_rec_all if is_tiny(p)], key=get_score, reverse=True)
    non_rec     = [p for p in non_rec_all if not is_tiny(p)]
    bot_zone    = sorted(reg_rec + non_rec[rec_cutoff:], key=get_score, reverse=True)
    order       = (non_rec[:rec_cutoff] + bot_zone + tiny_tail)[:LINEUP_SIZE]

    # --- Step 2.5: career splits tiebreaker ---
    # For adjacent players whose 2026 split wRC+ values are within
    # SPLITS_TIEBREAK_WINDOW of each other, promote the lower-ranked player
    # if their career split wRC+ advantage meets CAREER_TIEBREAK_ADVANTAGE.
    # Skips fallback, recency-restricted, and tiny-sample players. Repeats
    # until stable.
    changed = True
    passes  = 0
    while changed and passes < 20:
        changed = False
        passes += 1
        for i in range(len(order) - 1):
            p1 = order[i]
            p2 = order[i + 1]
            if (is_fallback(p1) or is_fallback(p2) or is_tiny(p1) or is_tiny(p2) or
                    is_recency_restricted(p1) or is_recency_restricted(p2)):
                continue
            s1 = p1.get("wRC+_2026splits", None)
            s2 = p2.get("wRC+_2026splits", None)
            if s1 is None or s2 is None:
                continue
            try:
                s1, s2 = float(s1), float(s2)
            except (TypeError, ValueError):
                continue
            if (s1 != s1) or (s2 != s2):  # NaN check
                continue
            if abs(s1 - s2) <= SPLITS_TIEBREAK_WINDOW:
                c1 = p1.get("wRC+_career", None)
                c2 = p2.get("wRC+_career", None)
                if c2 is None:
                    continue
                try:
                    c2 = float(c2)
                    c1 = float(c1) if c1 is not None else None
                except (TypeError, ValueError):
                    continue
                if (c1 is None or c2 - c1 >= CAREER_TIEBREAK_ADVANTAGE):
                    order[i], order[i + 1] = order[i + 1], order[i]
                    changed = True

    # --- Step 3: slump restriction ---
    slump_cutoff = SLUMP_TOP_SLOTS
    moved = True
    while moved:
        moved = False
        for i in range(min(slump_cutoff, len(order))):
            if is_slumping(order[i]):
                best_j, best_val = None, -1
                for j in range(slump_cutoff, len(order)):
                    cand = order[j]
                    if (not is_fallback(cand) and not is_slumping(cand) and
                            not is_recency_restricted(cand) and get_score(cand) > best_val):
                        best_val = get_score(cand)
                        best_j   = j
                if best_j is not None:
                    order[i], order[best_j] = order[best_j], order[i]
                    moved = True
                    break

    # --- Step 4: re-sort top zone ---
    top   = sorted(order[:slump_cutoff], key=get_score, reverse=True)
    order = top + order[slump_cutoff:]

    # --- Step 4.5: re-sort bottom zone ---
    bot      = order[slump_cutoff:]
    bot_main = sorted([p for p in bot if not is_tiny(p)], key=get_score, reverse=True)
    bot_tiny = sorted([p for p in bot if is_tiny(p)],     key=get_score, reverse=True)
    order    = order[:slump_cutoff] + bot_main + bot_tiny

    # --- Step 5: handedness streak balancing ---
    def hand(p):
        return str(p.get("Handedness", "?")) if p is not None else "?"

    changed = True
    passes  = 0
    while changed and passes < 20:
        changed = False
        passes += 1
        for i in range(len(order) - MAX_SAME_HAND_IN_ROW):
            streak = [hand(order[i + k]) for k in range(MAX_SAME_HAND_IN_ROW + 1)]
            if len(set(streak)) == 1 and streak[0] != "?":
                swap_pos     = i + MAX_SAME_HAND_IN_ROW
                bumped_score = get_score(order[swap_pos])
                for j in range(swap_pos + 1, len(order)):
                    candidate = order[j]
                    if hand(candidate) != streak[0]:
                        if is_fallback(candidate) and swap_pos < cutoff:
                            continue
                        if is_recency_restricted(candidate) and swap_pos < rec_cutoff:
                            continue
                        if swap_pos < slump_cutoff and is_slumping(candidate):
                            continue
                        cost = bumped_score - get_score(candidate)
                        if cost < HANDEDNESS_SWAP_MAX_COST:
                            order[swap_pos], order[j] = order[j], order[swap_pos]
                            changed = True
                        break

    # --- Step 6: post-swap score-order correction ---
    # Handedness swaps (step 5) can displace higher-scored non-fallback players
    # behind lower-scored ones. Correct inversions via non-adjacent swaps,
    # subject to: (a) no handedness streak created, (b) slumping players stay
    # out of the protected top zone (positions < slump_cutoff). Fallback and
    # recency-restricted players are fully insulated from this step (never
    # the promoted nor the demoted party) so it can't undo their restriction.
    corrected = True
    while corrected:
        corrected = False
        for i in range(len(order)):
            p_i = order[i]
            if p_i is None or is_fallback(p_i) or is_tiny(p_i) or is_recency_restricted(p_i):
                continue
            for j in range(i + 1, len(order)):
                p_j = order[j]
                if p_j is None or is_fallback(p_j) or is_tiny(p_j) or is_recency_restricted(p_j):
                    continue
                # Never promote a slumping player back into the protected top zone
                if is_slumping(p_j) and i < slump_cutoff:
                    continue
                if get_score(p_j) <= get_score(p_i) + 0.001:
                    continue
                # Tentatively swap and check for handedness violations
                order[i], order[j] = order[j], order[i]
                violation = False
                for k in range(max(0, i - MAX_SAME_HAND_IN_ROW + 1),
                               min(len(order) - MAX_SAME_HAND_IN_ROW, j + 1)):
                    streak = [hand(order[k + m]) for m in range(MAX_SAME_HAND_IN_ROW + 1)]
                    if len(set(streak)) == 1 and streak[0] != "?":
                        violation = True
                        break
                if violation:
                    order[i], order[j] = order[j], order[i]  # revert
                else:
                    corrected = True
                    break
            if corrected:
                break

    return order


# -------------------------------------------------------
# Output
# -------------------------------------------------------

def print_lineup(ordered_players, matchup_label):
    print("=== " + matchup_label + " ===")
    for i, player in enumerate(ordered_players):
        if player is None:
            print(str(i + 1) + ". (No qualifying player found for this slot)")
            continue
        fallback   = is_fallback(player)
        tiny       = is_tiny(player)
        recency    = is_recency_restricted(player)
        zero_data  = is_zero_data_fallback(player)
        splits_26 = (str(int(player["wRC+_2026splits"])) + " in " + str(int(player["PA_2026splits"])) + " PA"
                     if pd.notna(player.get("wRC+_2026splits")) else "N/A")
        career    = (str(int(player["wRC+_career"])) + " in " + str(int(player["PA_career"])) + " PA"
                     if pd.notna(player.get("wRC+_career")) and pd.notna(player.get("PA_career")) else "N/A")
        total_26  = (str(int(player["wRC+_2026total"])) + " in " + str(int(player["PA_2026total"])) + " PA"
                     if pd.notna(player.get("wRC+_2026total")) and pd.notna(player.get("PA_2026total")) else "N/A")
        recent    = (str(int(player["wRC+_recent"])) + " in " + str(int(player["PA_recent"])) + " PA"
                     if pd.notna(player.get("wRC+_recent")) and pd.notna(player.get("PA_recent")) else "N/A")
        if zero_data:
            fb_note = "  [FALLBACK - no performance data anywhere, league-average placeholder]"
        elif tiny:
            fb_note = "  [FALLBACK - tiny sample, batting last]"
        elif fallback:
            fb_note = "  [FALLBACK - no handedness splits, score capped at " + str(FALLBACK_WRC_CAP) + "]"
        elif recency:
            fb_note = ("  [RECENCY - under " + str(RECENCY_PA_FLOOR) + " 2026 PA, no " +
                        str(RECENCY_EXEMPT_MIN_PA_LAST_SEASON) + "+ PA at " +
                        str(RECENCY_EXEMPT_MIN_WRC_LAST_SEASON) + "+ wRC+ last season; restricted to slot " +
                        str(RECENCY_MIN_SLOT) + "+]")
        else:
            fb_note = ""
        print(
            str(i + 1) + ". " + str(player[PLAYER_COL]) +
            ", " + str(player.get("_slot", "?")) +
            ", " + str(player.get("Handedness", "?")) +
            "  Stabilized wRC+: " + str(player.get("Stabilized_wRC+", "N/A")) +
            fb_note +
            "  (2026 splits: " + splits_26 +
            ", Career splits: " + career +
            ", 2026 total: " + total_26 +
            ", Recent: " + recent + ")"
        )
    print()


# -------------------------------------------------------
# Runner
# -------------------------------------------------------

def run_comparison():
    """
    Reports forward-validated evidence on whether the optimizer's picks
    actually outperform real lineup decisions -- WITHOUT building or
    grading anything itself.

    This replaces the old self-graded "runs left on table" comparison,
    which built the "optimal" lineup by maximizing Valuation_wRC+ and
    then graded BOTH lineups using that same lookup table -- circular by
    construction, since a lineup picked to maximize a metric will almost
    always score highest on that exact metric. (User decision 2026-06-23:
    replace entirely rather than keep both -- "the less clutter the
    better.")

    The actual comparison now lives in live_forward_validation.py, a
    separate, periodically-run script that tests the optimizer's picks
    against REAL, now-known outcomes at past in-season decision points
    (data through date D builds the lineup; data AFTER D grades it --
    so the lineup can never have seen what it's graded against). That
    script writes LIVE_FORWARD_VALIDATION_SUMMARY; this function just
    displays it.

    Deliberately NOT run automatically as part of run_optimizer() --
    it requires pulling several weeks of full-league Statcast data per
    cutoff, which would make every daily run slow and network-dependent.
    Re-run live_forward_validation.py periodically (e.g. weekly) to
    refresh the numbers shown here.
    """
    print("==================================================")
    print("FORWARD-VALIDATED LINEUP COMPARISON")
    print("==================================================")
    print()

    if not os.path.exists(LIVE_FORWARD_VALIDATION_SUMMARY):
        print("No forward-validation data yet.")
        print("Run live_forward_validation.py to generate it (see that script's")
        print("docstring -- tests the optimizer against real, now-known outcomes")
        print("at past in-season decision points, not against its own lookup table).")
        print()
        return

    try:
        with open(LIVE_FORWARD_VALIDATION_SUMMARY, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print("Could not read " + LIVE_FORWARD_VALIDATION_SUMMARY + ": " + str(e))
        print("Re-run live_forward_validation.py to regenerate it.")
        print()
        return

    generated     = summary.get("generated", "unknown date")
    cutoffs       = summary.get("cutoffs_tested", [])
    n_total       = summary.get("n_windows_overall") or 0
    win_rate      = summary.get("win_rate_overall")
    mean_gap      = summary.get("mean_gap_overall")
    by_team       = summary.get("by_team", {})

    print("Generated: " + str(generated))
    print("Rolling cutoffs tested: " + (", ".join(cutoffs) if cutoffs else "none"))
    print("Team-hand-cutoff combinations graded: " + str(n_total))
    print()

    if n_total == 0 or not by_team:
        print("Not enough graded combinations yet to report. Re-run")
        print("live_forward_validation.py later in the season once more")
        print("post-cutoff data has accumulated.")
        print()
        return

    print("Overall: optimal beat real in " + str(round(100 * win_rate, 1)) + "% of graded windows")
    print("Overall mean wOBA gap (optimal - real): " + str(round(mean_gap, 5)))
    print()
    print("This is plain average ACTUAL wOBA (not a slot-weighted run value --")
    print("the real lineup here is a PA-share proxy with no genuine batting-order")
    print("information, so weighting by slot would assume an order we don't have).")
    print()

    rows = sorted(by_team.items(), key=lambda kv: kv[1].get("mean_gap", 0), reverse=True)
    print("Per-team rollup (sorted by mean wOBA gap, descending):")
    print("Team    N   Wins  Win%    Mean Gap")
    print("----  ---  -----  ------  --------")
    for team, r in rows:
        n      = r.get("n_windows", 0)
        wins   = r.get("wins", 0)
        wr     = r.get("win_rate", 0)
        gap    = r.get("mean_gap", 0)
        print(str(team).ljust(6) + str(n).rjust(3) + "  " +
              str(wins).rjust(5) + "  " +
              (str(round(100 * wr, 1)) + "%").rjust(6) + "  " +
              str(round(gap, 5)).rjust(8))
    n_favoring = sum(1 for _, r in rows if r.get("mean_gap", 0) > 0)
    print()
    print(str(n_favoring) + "/" + str(len(rows)) + " teams favor the optimal lineup on average.")
    print("Positive gap = optimal lineup's real-outcome wOBA beat the real lineup's.")
    print("Negative gap = real lineup outperformed the optimal pick over that span.")
    print()


def run_optimizer():
    print("Loading files...\n")
    raw_2026_right   = load_csv(FILE_2026_VS_RIGHT,   "2026 vs RHP")
    raw_2026_left    = load_csv(FILE_2026_VS_LEFT,    "2026 vs LHP")
    raw_2026_total   = load_csv(FILE_2026_TOTAL,      "2026 total")
    raw_career_right = load_csv(FILE_CAREER_VS_RIGHT, "career vs RHP")
    raw_career_left  = load_csv(FILE_CAREER_VS_LEFT,  "career vs LHP")
    raw_recent       = load_csv(FILE_RECENT,           "recent 14 days total")
    roster, _, _ = load_roster(FILE_ROSTER)
    print()

    print("Building master tables...\n")
    print("--- vs RHP ---")
    master_right = build_master(
        raw_2026_right.copy(), raw_career_right.copy(), raw_career_left.copy(),
        raw_2026_total.copy(), raw_recent.copy(), roster, side="R")
    print()
    print("--- vs LHP ---")
    master_left = build_master(
        raw_2026_left.copy(), raw_career_left.copy(), raw_career_right.copy(),
        raw_2026_total.copy(), raw_recent.copy(), roster, side="L")
    print()

    teams = sorted(master_right[TEAM_COL].dropna().unique())
    print("Teams found: " + ", ".join(teams) + "\n")

    # League-average Stabilized_wRC+ per hand, computed once from the full
    # master population (not hardcoded 100) -- used by build_fallback_pool()
    # as the placeholder score for a true zero-data roster player, since the
    # model's own stabilization/shrinkage doesn't necessarily center exactly
    # on 100.
    league_avg_right = master_right["Stabilized_wRC+"].mean()
    league_avg_left  = master_left["Stabilized_wRC+"].mean()

    for team in teams:
        print("##################################################")
        print("TEAM: " + team)
        print("##################################################\n")

        team_right = master_right[master_right[TEAM_COL] == team]
        team_left  = master_left[master_left[TEAM_COL] == team]

        fb_right = build_fallback_pool(team, raw_2026_total.copy(), raw_career_right.copy(), roster,
                                        league_avg_wrc=league_avg_right)
        fb_left  = build_fallback_pool(team, raw_2026_total.copy(), raw_career_left.copy(),  roster,
                                        league_avg_wrc=league_avg_left)

        if len(team_right) == 0:
            print("No qualifying vs RHP data for " + team + ". Skipping.\n")
        else:
            ordered_right = balance_handedness(assign_lineup_positions(team_right, fb_right))
            print_lineup(ordered_right, "Optimal Lineup vs Right-Handed Pitcher")

        if len(team_left) == 0:
            print("No qualifying vs LHP data for " + team + ". Skipping.\n")
        else:
            ordered_left = balance_handedness(assign_lineup_positions(team_left, fb_left))
            print_lineup(ordered_left, "Optimal Lineup vs Left-Handed Pitcher")

    # Phase 2: forward-validated comparison (reads live_forward_validation.py's
    # cached results; does not build or grade anything itself -- see
    # run_comparison()'s docstring).
    print()
    run_comparison()


# -------------------------------------------------------
# Entry point
# -------------------------------------------------------

if __name__ == "__main__":
    tee = Tee(OUTPUT_FILE)
    sys.stdout = tee

    print("=== LINEUP OPTIMIZER v23 STARTED ===")
    print("Results will be saved to: " + OUTPUT_FILE + "\n")

    try:
        run_optimizer()
    except Exception as e:
        print("ERROR: Something went wrong.")
        print(str(e))
        import traceback
        traceback.print_exc()

    print("=== DONE ===")
    print("Open this file in Notepad to read results with JAWS: " + OUTPUT_FILE)

    tee.close()
    sys.stdout = tee.console