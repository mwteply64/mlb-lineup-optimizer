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
    build_career_data(), and is now blended into each of the four
    predictor components (splits/career/total/recent) at a starting
    XWOBA_BLEND = 0.5 (even split between actual wOBA and xwOBA). The
    blend falls back to pure wOBA when xwOBA isn't available for a
    player/file. Ground-truth after_* files are never blended -- the
    actual outcome stays actual wOBA. Tuning the blend ratio (and
    whether it should differ per component) is Task #24.
  - The old combined_2019_2022 / 2021_2023 / 2022_2024 files are no
    longer read by this script. They are kept on disk only because
    nothing else has been confirmed to still need them.

v10 LHP RETUNE (this revision):
  - tuner_lhp.py ran a walk-forward-CV search over all 14 LHP-specific
    knobs (weights, regress-PA, LA, DECAY_CAREER_LHP, the platoon
    blend/clamp, and XWOBA_BLEND) while RHP stayed frozen at its old
    v8/v9 baseline. Out-of-sample avg Spearman r (LHP): 0.2999 -> 0.3129.
    Final production fit (all 6 years, no holdout): r=0.3335 in-sample.
  - PLATOON_BLEND/PLATOON_FACTOR_MIN/PLATOON_FACTOR_MAX and XWOBA_BLEND
    were previously single values shared by both hands. They are now
    split into _LHP/_RHP pairs (Task #24/#25 follow-on) so the new LHP
    values can be used without changing RHP's behavior at all. RHP's
    copies keep the exact pre-existing shared values (PLATOON_BLEND=0.31,
    PLATOON_FACTOR_MIN=0.585, PLATOON_FACTOR_MAX=1.325, XWOBA_BLEND=0.5)
    until RHP gets its own dedicated tuning pass.
  - blend_woba()/blended_woba() now take the blend ratio as an explicit
    argument instead of reading a module-level constant, and run_window()
    blends each cross-hand career value twice -- once per side, using
    that side's own XWOBA_BLEND -- since LHP and RHP can now disagree on
    how much xwOBA to mix in. This mirrors tuner_lhp.py's scoring math
    exactly (it was already built this way to keep both files in sync).
  - DECAY_CAREER_RHP and XWOBA_BLEND_RHP are still NOT tuned (Task #23/
    #24 for RHP, pending the RHP-side tuner). Only the LHP=0.8001 /
    LHP=0.9187 values reflect a real search; the RHP figures are the
    original neutral placeholders.

v11 RHP PARTIAL RETUNE (this revision):
  - tuner_rhp.py ran the same walk-forward-CV search over all 14
    RHP-specific knobs, mirroring tuner_lhp.py with LHP frozen at its
    v10 values. The full 14-parameter jointly-tuned set was NOT shipped:
    per-fold tuned values for most params were unstable across the 3
    CV folds (e.g. PLATOON_BLEND_RHP and DECAY_CAREER_RHP swung across
    large fractions of their bounds depending on training years), and
    the joint set scored worse than baseline out-of-sample in all 3
    folds.
  - Before discarding the run entirely, each of the 14 params was
    isolation-tested one at a time (only that param changed, the other
    13 left at baseline) across the same 3 walk-forward folds. Only
    XWOBA_BLEND_RHP showed a robust, repeatable standalone gain (2 of 3
    folds positive in two independently-built tests); every other
    param was flat-to-negative on average when isolated, including
    DECAY_CAREER_RHP despite an interesting but inconclusive partial
    signal. WS_R (splits weight) showed a faint positive average too,
    but well under the ~0.043 noise floor of year-to-year baseline r,
    so it was left untouched.
  - XWOBA_BLEND_RHP alone was then validated with a proper per-fold
    walk-forward grid search (train years pick the candidate value,
    held-out test year scores it): optimal value converged to ~0.80 in
    2 of 3 folds and 0.7333 in the third, test-year delta +0.0045 avg
    (2/3 folds positive). Final all-years fit: r 0.4103 -> 0.4147 at
    XWOBA_BLEND_RHP=0.80. Shipped. DECAY_CAREER_RHP and the other 12
    RHP params remain at their original v9 neutral placeholders,
    pending stronger evidence or a future dedicated re-tune.

v12 XWOBA PER-COMPONENT SPLIT (this revision):
  - XWOBA_BLEND_LHP/RHP were each a single ratio applied identically to
    all four predictor components (splits/career/total/recent). Split
    each into 4 components -- XWOBA_BLEND_SPLITS/CAREER/TOTAL/RECENT_
    LHP and the RHP equivalents, 8 constants total -- on the theory
    that components with different sample-size/noise profiles (e.g.
    small-sample "recent" vs large-sample "career") likely have
    different optimal xwOBA weighting. This is a pure refactor in this
    revision: all 8 new constants start equal to the old single value
    for their hand (0.9187 for all 4 LHP components, 0.80 for all 4
    RHP components), verified to reproduce the pre-refactor r values
    bit-for-bit (LHP 0.3299, RHP 0.4113) before any new tuning. The
    cross-hand platoon input (career_cross) reuses its hand's CAREER
    ratio, since it's still career-type data, just sourced from the
    opposite hand's column -- this keeps the count at 4 per hand
    instead of 5. tuner_xwoba_components.py searches these 8 values via
    the same walk-forward CV approach as tuner_lhp.py/tuner_rhp.py,
    with every other parameter (including the two hands' non-xwOBA
    knobs) frozen at current production values.

v13 AGING CURVE INTEGRATION (this revision):
  - New pipeline (build_age_data.py -> player_ages.csv, then
    build_aging_curve.py -> aging_curve_results_*.txt /
    aging_curve_pairs.csv) estimated a delta-method aging curve from
    every available adjacent-season player pair (2,446 pairs, "baseball
    age" as of June 30, PA-weighted). Result: flat/peak ages 21-26,
    then a steady decline through the late 20s and 30s -- cumulative
    wOBA delta from age 21 is roughly +0.006 at the 24 peak and -0.101
    by age 36 (the oldest bin with N>=20 pairs; everything outside
    21-36 is too thin to trust on its own).
  - That cumulative curve is embedded below as AGING_CURVE_CUM (one
    entry per reliable age, 21-36). aging_cum() clamps any age outside
    that range to its nearer edge (a 19-year-old is scored like a
    21-year-old, a 40-year-old like a 36-year-old) rather than
    extrapolating into the noisy thin tails. aging_delta(age_from,
    age_to) is just the difference of two cum() lookups, so the
    arbitrary "vs youngest" reference point the curve was reported
    against cancels out.
  - Only the CAREER component is touched. splits/total/recent are all
    built from data inside the same season being predicted, so there's
    no age gap to correct for -- only career data spans multiple PRIOR
    seasons at different ages from the target year. build_career_data()
    now shifts each prior season's woba (and xwoba, kept consistent
    since XWOBA_BLEND_*_LHP=0.9187 means LHP's career component is
    almost entirely xwoba) by AGING_STRENGTH * aging_delta(age that
    season, age in test_year), using player_ages.csv (joined on the
    same MLBAM id already used as player_id everywhere else in this
    project), BEFORE that season's value is folded into the
    decay/PA-weighted career average. A player missing from
    player_ages.csv, or missing an age for either year, gets delta=0
    for that season (graceful no-op fallback, not an error).
  - AGING_STRENGTH is a new single knob shared by both hands (the curve
    itself isn't hand-specific -- it was estimated from full-season
    data pooled across all PA, not split by opposing pitcher hand).
    Ships at the neutral placeholder 0.0 (pure no-op: with
    aging_strength == 0.0, build_career_data() skips the age lookup
    entirely and adds exactly 0.0 to each woba/xwoba, so v13 reproduces
    v12's r values bit-for-bit).
  - Task #79 (tuner_aging.py) ran the same walk-forward-CV 1-D grid
    search used to validate every other knob in this file. Final
    all-years fit picked AGING_STRENGTH=0.86 (in-sample r 0.3706 ->
    0.3789, +0.0083, on a smooth plateau from ~0.7-1.1 -- not a single
    noisy spike). But the 3-fold walk-forward CV evidence was much
    weaker and, by this project's own bar (see v11's XWOBA_BLEND_RHP
    precedent: only ship sub-results with repeatable, robust per-fold
    held-out gains), did not clear it: test-year r deltas were +0.0053
    (2023), -0.0093 (2024), +0.0071 (2025) -- average +0.0011, inside
    the noise floor, with one fold a clear regression. Worse, the
    combined metric masked offsetting hand-level swings: fold 2023 was
    LHP -0.0330 / RHP +0.0436, fold 2024 was LHP +0.0020 / RHP -0.0206 --
    a single shared knob helping one hand and hurting the other by
    similar magnitudes in different years is a sign of fold-specific
    noise, not a stable global aging effect. Per-fold optimal strengths
    (1.37, 1.09, 0.87) never flipped sign and trended toward the
    final-fit value as more training data was used, which is the one
    encouraging signal -- but not enough on its own. Decision: held at
    AGING_STRENGTH=0.0 (no change). Full grid results are in
    tuner_aging_results_2026-06-18.txt for anyone who wants to revisit
    this with more data in a future season.

v14 DECAY_CAREER_RHP DEDICATED RETUNE (this revision):
  - DECAY_CAREER_RHP had never gotten its own focused search -- v11's
    tuner_rhp.py only tried it inside a 14-parameter joint search (which
    failed out-of-sample on every fold) and a one-at-a-time isolation
    test, where it showed "an interesting but inconclusive partial
    signal" that was never followed up. Task #80 (tuner_decay_rhp.py)
    gave it the same dedicated 1-D walk-forward-CV grid search used for
    DECAY_CAREER_LHP (v10) and AGING_STRENGTH (v13), searching [0.40,
    1.00] (decay > 1.0 would mean older seasons count MORE than newer
    ones, which doesn't make baseball sense), with DECAY_CAREER_LHP
    frozen at 0.8001 and every other knob frozen at current production.
  - Final all-years fit picked DECAY_CAREER_RHP=0.646 (in-sample r
    0.3706 -> 0.3713, only +0.0007 -- much smaller than DECAY_CAREER_LHP's
    own v10 gain, and on a wide, flat plateau from ~0.55-0.70 rather than
    a sharp optimum). The 3-fold walk-forward CV evidence was negative:
    test-year r deltas were -0.0010 (2023), -0.0041 (2024), +0.0020
    (2025) -- average -0.0041 on the RHP-only metric (-0.0010 combined),
    two of three folds a clear regression. Per-fold optimal decay values
    also drifted with more training data (0.409, 0.574, 0.622) rather
    than settling near a single value, suggesting the in-sample optimum
    is fold-size-dependent rather than a stable global effect. By this
    project's bar (same one applied to v11's XWOBA_BLEND_RHP and v13's
    AGING_STRENGTH: only ship repeatable, robust held-out gains), this
    does not clear it. Decision: held at DECAY_CAREER_RHP=1.0 (no
    change, no decay -- plain PA-weighted average across all prior
    seasons). Full grid results are in
    tuner_decay_rhp_results_2026-06-19.txt for anyone who wants to
    revisit this with more data in a future season.
  - This closes out Task #23 (recency-weight decay constants) for both
    hands: LHP was tuned and shipped in v10 (0.8001); RHP was searched
    with the same rigor in this revision and the evidence says to leave
    it alone. Both are now backed by a dedicated CV search either way --
    the placeholder-by-default state RHP was in since v9 is gone.

v15 DECAY x REGRESS_PA_CAREER_RHP JOINT RETUNE (this revision):
  - v14 left open whether DECAY_CAREER_RHP's per-fold drift (0.409, 0.574,
    0.622) was a real-but-noisy recency signal or an artifact of
    REGRESS_PA_CAREER_RHP=25 -- a value that, per v11's own note, was
    never individually CV-validated (it traces to v8's in-sample-only
    1500+ trial search) and is a 10x+ outlier vs. its RHP siblings
    (REGRESS_PA_SPLITS_RHP=285, _TOTAL_RHP=400, _RECENT_RHP=665). Task #81
    (tuner_decay_regress_rhp.py) tested this directly: a joint walk-
    forward-CV search over both knobs together (coordinate ascent -- decay
    grid at regress_pa=25, then regress_pa grid at the tuned decay -- plus
    a small local 2-D refinement grid), same 3 folds as every other tuner
    here.
  - Freeing REGRESS_PA_CAREER_RHP did not fix the instability -- it added
    a second one. Per-fold (decay, regress_pa) pairs: (0.409, 788),
    (0.594, 10), (0.652, 12). DECAY_CAREER_RHP drifted to nearly the same
    values as v14's decay-only search, so regress_pa=25 wasn't the cause
    of that drift. Meanwhile REGRESS_PA_CAREER_RHP itself swung between
    the two ends of its [0, 800] search range across folds -- fold 1
    wanted heavy regression (788, near the max), folds 2-3 wanted almost
    none (10-12, near the min). Test-year r deltas (combined): -0.0210
    (2023), -0.0043 (2024), +0.0018 (2025) -- average -0.0078, with fold 1
    a clear regression. Final all-years fit: decay=0.676, regress_pa=12,
    in-sample r 0.3706 -> 0.3715 (+0.0009) -- a small in-sample gain
    consistent with overfitting noise rather than a real signal, given the
    cross-fold picture.
  - By this project's bar (repeatable, robust held-out gains required),
    this does not clear it. Decision: held at DECAY_CAREER_RHP=1.0,
    REGRESS_PA_CAREER_RHP=25 (no change to either). Full results are in
    tuner_decay_regress_rhp_results_2026-06-19.txt.
  - This closes out the aging/decay/regression line of investigation for
    RHP (Tasks #23, #80, #81): three independent angles -- decay alone,
    decay+regress jointly, and (separately, v13) the aging curve -- all
    failed to clear the held-out-CV bar. The RHP career component stays a
    plain, undecayed, lightly-regressed PA-weighted average across all
    prior seasons, as it's been since v9.

v16 BATTER-HAND PLATOON PRIOR (this revision, shipped 2026-07-10):
  - First change tuned and validated on the SYMMETRIC WIN-RATE metric
    (beat the manager's actual top-9 by PA vs the specific hand) rather
    than Spearman r -- see tuner_lhp_winrate.py and
    tuner_lhp_winrate_results_2026-07-10.txt.
  - stabilize() gains an optional pid argument. For the LHP side, the
    player's own career platoon ratio is regressed toward the league-
    mean ratio for the side he bats from vs LHP (batter_sides.csv,
    built by download_batter_sides.py from Statcast's per-pitch stand
    column) by PLATOON_PRIOR_PA_LHP = 400 pseudo-PA. Own career PA
    keeps full weight, so established/reverse-split hitters keep their
    own ratios; only thin-evidence players move toward the prior
    (RHB-vs-LHP ~1.031, LHB-vs-LHP ~0.904, estimated from 2016-2018
    only). RHP ships at PLATOON_PRIOR_PA_RHP = 0 (bit-for-bit no-op;
    the RHP model showed no comparable deficit).
  - CV evidence (walk-forward, symmetric LHP win rate): 400 picked in
    all 3 folds, mean held-out +1.8pp, weak years 2021/2025 improved
    most. Two alternative candidates (XWOBA_BLEND->0.65; lower
    REGRESS_PA_CAREER_LHP) were tested and do NOT stack with this
    change (joint grids scored worse than either knob alone) -- do not
    ship them on top of it.
  - Shipped symmetric LHP baseline moves 56.7% -> 58.3% (all-years,
    n=509); see PROJECT_HANDOFF.md section 0 for the full re-baseline.

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
# LHP value updated by the v10 walk-forward-CV retune (was 0.2585).
LEAGUE_AVG_WOBA_LHP = 0.2943
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
#   v9->v10 LHP retune -- walk-forward CV over all 14 LHP knobs at once
#                    (weights, regress-PA, LA, career decay, platoon
#                    blend/clamp, xwOBA blend), RHP held frozen. Out-of-
#                    sample avg r: 0.2999 -> 0.3129. Career weight still
#                    dominates (~57%) but with much lower regression PA
#                    on splits (178, was 355) and a recency decay on
#                    career data (0.80) that v8/v9 didn't have at all.
#   Key insight: The two sides still benefit from very different model
#   shapes. LHP leans on career data heavily (~57% weight) with high
#   regression (711-1020 PA) to fight noise. RHP spreads weight across
#   splits/career/total/recent more evenly but barely regresses career
#   data at all (25 PA) because real RHP career numbers are reliable.
#   Weights below are proportionally renormalized so each side's four
#   weights sum to 1.00 -- purely cosmetic, doesn't change predictions
#   (stabilize() already renormalizes by sum(weights) internally, so
#   scaling all four weights by a constant is a no-op on the result).
WEIGHT_SPLITS_LHP  = 0.213
WEIGHT_CAREER_LHP  = 0.5695
WEIGHT_TOTAL_LHP   = 0.247
WEIGHT_RECENT_LHP  = 0.3416

REGRESS_PA_SPLITS_LHP = 178
REGRESS_PA_CAREER_LHP = 711
REGRESS_PA_TOTAL_LHP  = 1020
REGRESS_PA_RECENT_LHP = 757

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
# v10: split into LHP/RHP copies. The LHP retune found a much heavier
# blend helps that side (0.78 vs the old shared 0.31), though a
# sensitivity sweep around the final fit showed the score is genuinely
# flat across most of [0.5, 1.0] -- the fold-to-fold swings seen during
# tuning (0.52/0.85/0.97) are different points on the same plateau, not
# a sign the search failed to converge. RHP keeps the original shared
# value until it gets its own tuning pass.
PLATOON_BLEND_LHP = 0.7833
PLATOON_FACTOR_MIN_LHP = 0.612
PLATOON_FACTOR_MAX_LHP = 1.138

PLATOON_BLEND_RHP = 0.31
PLATOON_FACTOR_MIN_RHP = 0.585
PLATOON_FACTOR_MAX_RHP = 1.325

# -------------------------------------------------------
# Batter-hand platoon prior (v16, shipped 2026-07-10).
# -------------------------------------------------------
# The platoon factor above is built purely from the player's OWN career
# splits -- deliberately, because reverse-split hitters exist. But the
# 2026-07-10 LHP diagnostics showed the model's marginal vs-LHP picks
# average only ~27 pre-cutoff PA vs LHP, i.e. the platoon ratio is
# estimated from very thin evidence exactly where lineup decisions are
# decided. This prior is a REGRESSION TARGET, not an override: the
# player's own career platoon ratio is blended with the league-mean
# ratio for the side he actually bats from vs this pitcher hand,
# weighted by his own career evidence:
#
#   est_ratio = (own_ratio * own_career_pa + prior_ratio * PRIOR_PA)
#               / (own_career_pa + PRIOR_PA)
#
# so an established (including reverse-split) hitter keeps his own
# ratio nearly untouched, while a thin-sample hitter is pulled toward
# the typical same-side/opposite-side factor. PRIOR_PA = 0 disables the
# prior entirely (bit-for-bit pre-v16 behavior).
#
# LHP value (400) from tuner_lhp_winrate.py's walk-forward CV on the
# symmetric win-rate metric (2026-07-10): picked 400 in all 3
# walk-forward folds (+1.8pp mean held-out win rate; the two weak LHP
# years improved most: 2021 +3.5pp, 2025 +4.7pp at the LOYO picks), on
# a smooth plateau from ~100-800. RHP ships at 0 (no-op): the RHP model
# showed no comparable deficit (68.2% symmetric win rate) and no
# vs-RHP prior ratios have been estimated/validated.
#
# Prior ratios estimated from the 2016-2018 seasons ONLY (strictly
# before every test year, so no CV fold ever saw a prior fitted on its
# own data): PA-weighted mean own-ratio of players with >= 50 PA vs
# each hand, grouped by the side they bat from vs LHP (from
# batter_sides.csv, built by download_batter_sides.py from Statcast's
# per-pitch `stand` column -- switch hitters resolve per matchup, e.g.
# "R" vs LHP). Full precision kept so shipped results reproduce the
# tuner's CV rows exactly.
PLATOON_PRIOR_PA_LHP = 400
PLATOON_PRIOR_PA_RHP = 0
PLATOON_PRIOR_RATIO_VS_LHP = {
    "L": 0.9036224553071249,   # LHB vs LHP: ~10% below own combined average
    "R": 1.031091013338655,    # RHB vs LHP: ~3% above own combined average
}


def load_batter_sides():
    """
    Load batter_sides.csv (built by download_batter_sides.py): one row
    per batter with the side he actually bats from vs each pitcher hand
    (majority Statcast `stand` per matchup, so switch hitters resolve
    correctly). Returns dict: player_id (str) -> {"side_vs_L": 'L'/'R',
    "side_vs_R": 'L'/'R'}. Missing file degrades gracefully (the
    platoon prior becomes a no-op for every player) rather than
    erroring, same policy as load_player_ages().
    """
    path = os.path.join(DATA_DIR, "batter_sides.csv")
    result = {}
    if not os.path.exists(path):
        print("  WARNING: batter_sides.csv not found -- the batter-hand "
              "platoon prior will be a no-op for every player.")
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = str(row.get("player_id", "")).strip()
            if not pid:
                continue
            entry = {}
            for key in ("side_vs_L", "side_vs_R"):
                v = str(row.get(key, "")).strip()
                if v in ("L", "R"):
                    entry[key] = v
            if entry:
                result[pid] = entry
    return result


BATTER_SIDES = load_batter_sides()

# -------------------------------------------------------
# Career recency decay (Task #23 -- LHP tuned in v10, RHP still pending).
# -------------------------------------------------------
# Career data is now built from every per-season file strictly before
# the test year (see build_career_data() below). Each season's
# contribution is weighted by decay ** (test_year - season_year), so
# decay < 1.0 downweights seasons further in the past relative to the
# test year. 1.0 = no recency weighting, every prior season counted
# equally by its own PA. LHP's v10 retune found 0.80 (some recency
# tilt) beats the flat 1.0 baseline; RHP keeps the original 1.0 neutral
# placeholder until it gets its own tuning pass.
DECAY_CAREER_LHP = 0.8001
DECAY_CAREER_RHP = 1.0

# -------------------------------------------------------
# xWOBA blend (Task #24 -- now split per-component, see v12 note up top).
# -------------------------------------------------------
# Each of the four predictor components (splits/career/total/recent)
# blends its actual wOBA with its xWOBA (contact-quality-based expected
# wOBA, strips out some luck/defense noise) before clamping/regression.
# 0.0 = pure wOBA, 1.0 = pure xWOBA. Falls back to pure wOBA when a
# player/file has no xWOBA value. Ground-truth (after_* files) is never
# blended -- actual outcomes stay actual wOBA regardless of this
# setting.
#
# v10: split into a single LHP/RHP pair (one ratio per hand, shared
# across all 4 components). LHP search found 0.9187; RHP kept the
# neutral 0.5 placeholder.
# v11: RHP's single shared ratio retuned to 0.80 (see v11 note above).
# v12: each hand's single ratio split further into 4 per-component
# constants (splits/career/total/recent). All 8 currently still equal
# their hand's old shared value -- this is the pre-tuning starting
# point for tuner_xwoba_components.py, not yet a result of a new
# search. Update this comment once that tuner produces a shipped set.
XWOBA_BLEND_SPLITS_LHP = 0.9187
XWOBA_BLEND_CAREER_LHP = 0.9187
XWOBA_BLEND_TOTAL_LHP  = 0.9187
XWOBA_BLEND_RECENT_LHP = 0.9187

XWOBA_BLEND_SPLITS_RHP = 0.80
XWOBA_BLEND_CAREER_RHP = 0.80
XWOBA_BLEND_TOTAL_RHP  = 0.80
XWOBA_BLEND_RECENT_RHP = 0.80

# -------------------------------------------------------
# Aging curve (v13, Task #79 -- not yet tuned).
# -------------------------------------------------------
# Delta-method aging curve estimated by build_aging_curve.py from 2,446
# adjacent-season player pairs (see aging_curve_results_*.txt). Values
# are the cumulative PA-weighted mean wOBA delta by age, relative to an
# arbitrary reference age -- only DIFFERENCES between two ages are ever
# used (aging_delta() below), so that reference point cancels out and
# doesn't matter. Only ages 21-36 are included; those are the bins with
# N >= 20 player-pairs. Ages 19-20 and 37-42 had real bins in the raw
# output but were too thin (N < 20, one age had N=1) to trust, so they
# are intentionally left out here and handled by clamping instead (see
# aging_cum() below).
AGING_CURVE_CUM = {
    21: 0.0046, 22: 0.0040, 23: 0.0056, 24: 0.0062, 25: 0.0041, 26: 0.0007,
    27: -0.0065, 28: -0.0120, 29: -0.0171, 30: -0.0308, 31: -0.0409,
    32: -0.0494, 33: -0.0641, 34: -0.0714, 35: -0.0854, 36: -0.1010,
}
AGING_CURVE_MIN_AGE = min(AGING_CURVE_CUM)
AGING_CURVE_MAX_AGE = max(AGING_CURVE_CUM)

# Single knob shared by both hands -- the curve itself isn't hand-
# specific. 0.0 = no adjustment at all (current production behavior,
# and the only value verified bit-for-bit against v12 so far). 1.0 =
# apply the curve exactly as estimated. Pending tuner_aging.py (Task
# #79); this neutral placeholder ships until that tuner produces a
# validated value.
AGING_STRENGTH = 0.0


def aging_cum(age):
    """Cumulative aging-curve value at `age`, clamping to the nearest
    reliable edge (21 or 36) outside the trustworthy range instead of
    extrapolating into the thin tail bins."""
    clamped = max(AGING_CURVE_MIN_AGE, min(AGING_CURVE_MAX_AGE, age))
    return AGING_CURVE_CUM[clamped]


def aging_delta(age_from, age_to):
    """Expected wOBA shift moving from age_from to age_to, per the
    aging curve. The arbitrary reference point each cum() value is
    measured against cancels out in this subtraction."""
    return aging_cum(age_to) - aging_cum(age_from)


def load_player_ages():
    """
    Load player_ages.csv (built by build_age_data.py): one row per
    player with mlbam_id and one age_<season> column per season.
    Returns dict: player_id (str, same join key as load_file() uses --
    MLBAM id) -> {season (int): age (int)}.
    Missing file or missing/blank age cells degrade gracefully (that
    player/season just won't get an age adjustment) rather than erroring,
    since this is reference data, not required for the model to run.
    """
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


def build_career_data(test_year, hand, decay, aging_strength=0.0, ages=None):
    """
    Build career wOBA (and, where available, xwOBA) for every player by
    combining every season_<year>_vs_<hand>.csv file strictly before
    test_year -- i.e. only seasons that would actually have been known
    at prediction time, no data leakage from the test year itself.

    Each season's contribution is weighted by its own PA times
    decay ** (test_year - season_year), so seasons further in the past
    count for less when decay < 1.0. decay=1.0 (current default,
    pending Task #23) means plain PA-weighting with no recency tilt.

    v13: before that PA-weighting, each season's woba/xwoba is shifted
    by aging_strength * aging_delta(age that player had in season `y`,
    age they have in test_year), using `ages` (PLAYER_AGES). This
    corrects each prior season for how much aging/decline would be
    expected between when it happened and the season being predicted --
    decay controls how much a season counts at all, this controls what
    value it should count AS. aging_strength=0.0 (current production
    default) skips the lookup entirely and adds exactly 0.0, so this is
    a bit-for-bit no-op versus the pre-v13 behavior. A player missing
    from `ages`, or missing an age for either year, also falls back to
    delta=0.0 for that season.

    Returns dict: player_id (str) -> {name, pa, woba[, xwoba]}
    "pa" is the player's RAW (unweighted) total PA across all prior
    seasons used -- this is what MIN_PA_CAREER gates against, so the
    decay/aging adjustment only affect the wOBA value itself, not
    whether there's enough career data to trust it.
    """
    prior_years = [y for y in ALL_SEASON_YEARS if y < test_year]
    num, den   = {}, {}
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
            num[pid]    = num.get(pid, 0.0) + wpa * adj_woba
            den[pid]    = den.get(pid, 0.0) + wpa
            pa_sum[pid] = pa_sum.get(pid, 0) + d["pa"]
            names[pid]  = d["name"]
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


# -------------------------------------------------------
# XWOBA BLEND (v9, Task #24 starting point)
# -------------------------------------------------------

def blend_woba(woba, xwoba, xwoba_blend):
    """
    Blend actual wOBA with xWOBA per the given blend ratio. Falls back
    to pure wOBA when xWOBA isn't available (e.g. small-sample rows
    where Baseball Savant doesn't report an xwOBA).

    v10: xwoba_blend is now passed in explicitly (XWOBA_BLEND_LHP or
    XWOBA_BLEND_RHP) instead of read from a single shared module-level
    constant, since the two sides can now disagree on the ratio.
    """
    if xwoba is None:
        return woba
    return (1.0 - xwoba_blend) * woba + xwoba_blend * xwoba


def blended_woba(entry, xwoba_blend):
    """
    Given a load_file()/build_career_data() entry (or None), return its
    blended wOBA, or None if there's no entry at all. Used at the
    run_window() call sites so every predictor component (splits,
    career, total, recent) is blended consistently, including the
    career_cross values used by the platoon tendency prior (#4).

    v10: callers must pass the blend ratio for the side that is
    CONSUMING this value, not the side the data describes -- e.g. when
    LHP's career_cross uses RHP's career data, it's blended with this
    side's CAREER ratio (LHP is doing the predicting), and the same RHP
    career row gets blended again with RHP's own CAREER ratio when used
    as RHP's own career component. v12: each side now has 4 separate
    per-component ratios instead of 1 shared one; the cross-hand value
    always uses the consuming side's CAREER ratio (see v12 note up
    top). This mirrors tuner_xwoba_components.py exactly.
    """
    if entry is None:
        return None
    return blend_woba(entry["woba"], entry.get("xwoba"), xwoba_blend)


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
              career_cross_pa=0, career_cross_woba=None,
              pid=None):
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
    pid: player's MLBAM id (str), used only to look up his batter side
    for the v16 batter-hand platoon prior. Omitting it (None) makes the
    prior a no-op for this call -- pre-v16 behavior -- so legacy callers
    that don't pass it keep their old results.
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
        PB   = PLATOON_BLEND_LHP
        PMIN = PLATOON_FACTOR_MIN_LHP
        PMAX = PLATOON_FACTOR_MAX_LHP
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
        PB   = PLATOON_BLEND_RHP
        PMIN = PLATOON_FACTOR_MIN_RHP
        PMAX = PLATOON_FACTOR_MAX_RHP

    # -- Platoon tendency prior (#4, extended by v16) --
    # Scales total/recent wOBA by the player's career platoon ratio so
    # those non-split components better reflect the expected performance
    # vs this specific pitcher handedness. Controlled by PB (this side's
    # PLATOON_BLEND_LHP/RHP): 0.0 = no adjustment, 1.0 = full adjustment.
    # v16: the player's own ratio is regressed toward the league-mean
    # ratio for his batter side (see PLATOON_PRIOR_PA_* constants) by
    # PPA "pseudo-PA" of prior evidence -- own career PA keeps its full
    # weight, so established/reverse-split ratios survive. With PPA=0
    # (RHP side, or pid/side unknown) this reduces bit-for-bit to the
    # pre-v16 formula.
    if side == 'L':
        PPA = PLATOON_PRIOR_PA_LHP
        prior_table = PLATOON_PRIOR_RATIO_VS_LHP
        side_key = "side_vs_L"
    else:
        PPA = PLATOON_PRIOR_PA_RHP
        prior_table = {}   # no vs-RHP prior ratios estimated (knob ships at 0)
        side_key = "side_vs_R"

    platoon_factor = 1.0
    ratio, own_pa = None, 0
    if (PB > 0 and
            career_woba is not None and career_pa > 0 and
            career_cross_woba is not None and career_cross_pa > 0):
        combined_pa = career_pa + career_cross_pa
        combined_woba = (career_woba * career_pa +
                         career_cross_woba * career_cross_pa) / combined_pa
        if combined_woba > 0:
            ratio = career_woba / combined_woba
            own_pa = combined_pa

    prior = None
    if PPA > 0 and pid is not None:
        bs = BATTER_SIDES.get(str(pid))
        if bs:
            prior = prior_table.get(bs.get(side_key))

    if prior is not None:
        est = prior if ratio is None else (
            (ratio * own_pa + prior * PPA) / (own_pa + PPA))
    else:
        est = ratio
    if est is not None:
        clamped_factor = max(PMIN, min(PMAX, est))
        platoon_factor = 1.0 + PB * (clamped_factor - 1.0)

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
    car_l  = build_career_data(year, "left",  DECAY_CAREER_LHP, AGING_STRENGTH, PLAYER_AGES)
    car_r  = build_career_data(year, "right", DECAY_CAREER_RHP, AGING_STRENGTH, PLAYER_AGES)
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

        # v12: each side now has 4 separate per-component xWOBA blend
        # ratios instead of 1 shared one. The cross-hand career value
        # (career_cross) reuses its consuming side's CAREER ratio.
        # Mirrors tuner_xwoba_components.py.

        # For the LHP prediction, career_cross is the vs-RHP career data (#4)
        stab_l = stabilize(
            sl["pa"]  if sl  else 0, blended_woba(sl,  XWOBA_BLEND_SPLITS_LHP),
            cl["pa"]  if cl  else 0, blended_woba(cl,  XWOBA_BLEND_CAREER_LHP),
            tot["pa"] if tot else 0, blended_woba(tot, XWOBA_BLEND_TOTAL_LHP),
            rec["pa"] if rec else 0, blended_woba(rec, XWOBA_BLEND_RECENT_LHP),
            side='L',
            career_cross_pa=cr["pa"]   if cr else 0,
            career_cross_woba=blended_woba(cr, XWOBA_BLEND_CAREER_LHP),
            pid=pid,
        )
        # For the RHP prediction, career_cross is the vs-LHP career data (#4)
        stab_r = stabilize(
            sr["pa"]  if sr  else 0, blended_woba(sr,  XWOBA_BLEND_SPLITS_RHP),
            cr["pa"]  if cr  else 0, blended_woba(cr,  XWOBA_BLEND_CAREER_RHP),
            tot["pa"] if tot else 0, blended_woba(tot, XWOBA_BLEND_TOTAL_RHP),
            rec["pa"] if rec else 0, blended_woba(rec, XWOBA_BLEND_RECENT_RHP),
            side='R',
            career_cross_pa=cl["pa"]   if cl else 0,
            career_cross_woba=blended_woba(cl, XWOBA_BLEND_CAREER_RHP),
            pid=pid,
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
    print("=== BACKTEST v13 -- Lineup Optimization Historical Validation ===")
    print("Results saved to: " + OUTPUT_FILE)
    print("Configuration (v10 LHP retune + v11 RHP xwOBA retune + v12 per-component xwOBA split")
    print("+ v13 aging curve, not yet tuned):")
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
    print("  LHP Platoon blend: " + str(PLATOON_BLEND_LHP) +
          "  factor clamp: [" + str(PLATOON_FACTOR_MIN_LHP) +
          ", " + str(PLATOON_FACTOR_MAX_LHP) + "]  (tuned)")
    print("  RHP Platoon blend: " + str(PLATOON_BLEND_RHP) +
          "  factor clamp: [" + str(PLATOON_FACTOR_MIN_RHP) +
          ", " + str(PLATOON_FACTOR_MAX_RHP) + "]  (not yet tuned)")
    print("  Career recency decay:  LHP=" + str(DECAY_CAREER_LHP) + " (tuned)" +
          "  RHP=" + str(DECAY_CAREER_RHP) + " (not yet tuned)")
    print("  xWOBA blend LHP:  splits=" + str(XWOBA_BLEND_SPLITS_LHP) +
          "  career=" + str(XWOBA_BLEND_CAREER_LHP) +
          "  total=" + str(XWOBA_BLEND_TOTAL_LHP) +
          "  recent=" + str(XWOBA_BLEND_RECENT_LHP))
    print("  xWOBA blend RHP:  splits=" + str(XWOBA_BLEND_SPLITS_RHP) +
          "  career=" + str(XWOBA_BLEND_CAREER_RHP) +
          "  total=" + str(XWOBA_BLEND_TOTAL_RHP) +
          "  recent=" + str(XWOBA_BLEND_RECENT_RHP))
    print("  Aging strength: " + str(AGING_STRENGTH) +
          "  (0.0 = no-op; players with ages loaded: " + str(len(PLAYER_AGES)) + ")")
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


if __name__ == "__main__":
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
