# Lineup Optimization Project — Handoff Document
*Generated 2026-07-03. Read this file at the start of a new session to resume without context loss.*

---

## 0. UPDATE — 2026-07-08/09/10 sessions (Claude Code). Read this first; it supersedes stale numbers below.

### 2026-07-10: LHP DIAGNOSTICS (no constants changed)

Diagnostic pass on the symmetric-framework detail CSVs plus a swap-level
re-scoring of all 18 historical windows. Findings:

1. **The LHP deficit is now statistically solid.** Under the symmetric
   framework the historical RHP-LHP win-rate difference is +11.5pp
   (68.2% vs 56.7%), year-clustered 95% CI [+6.9%, +16.7%] — excludes zero.
   (Under the old hand-agnostic framework it had straddled zero.)
2. **The model's LHP swaps still ADD value — just less than RHP swaps.**
   Swap-level test (players the model promotes into the top-9 vs the ones it
   benches, graded on post-cutoff wOBA vs that hand): LHP swap-ins beat
   swap-outs by +0.0140 pooled (59.4% of 360 paired combos); RHP +0.0251
   (67.1% of 422). The model is not wrong vs LHP, it's less right.
3. **Two structural reasons the LHP edge shrinks to near-coin-flip:**
   (a) The benchmark is stronger vs LHP — switching the real lineup from
   hand-agnostic to hand-specific raised its graded wOBA by +0.0020 vs LHP
   but only +0.0002 vs RHP (manager platooning adds real value almost
   exclusively vs LHP); (b) the marginal LHP pick is priced off very thin
   direct evidence — swap-ins average only 26.8 pre-cutoff PA vs LHP
   (swap-outs 64.2), so LHP predictions lean on the career/xwOBA prior.
4. **Where LHP wins and loses:** improves with later cutoffs (52.2% June ->
   61.1% Aug — more vs-LHP data helps); does BETTER on bigger lineup changes
   (61.5% at n_diff 3-4) than small tweaks (54.4% at n_diff 0-2); grading
   thinness does NOT explain it (well-graded-only combos: 54.1%). Year
   pattern is bimodal: ~61% in 2019/2022/2023/2024 but 47.1% in BOTH 2021
   and 2025. Worst LHP teams are platoon-forward orgs (LAD, MIL, ATL, TB).
5. **Live 2026 vs LHP (42.5%, gap +0.0011) is statistically flat** (p=0.54,
   n=113) — consistent with either the 2021/2025-style bad-year pattern or
   simply noise at this sample size.

Implication for tuning: the target is the LHP model's behavior on
thin-split players (career weight 0.5695 / regress 711, xwOBA blend 0.9187,
platoon blend 0.7833 — the constants that dominate when direct vs-LHP PA is
~27). Any retune should use the SYMMETRIC win-rate metric (not Spearman r,
which is what v10 optimized) and hold out 2021/2025-style years.

### 2026-07-10 (later): LHP WIN-RATE TUNER RESULTS — two shippable candidates, mutually exclusive

New tools: `tuner_lhp_winrate.py` (+ run_tuner_lhp_winrate.bat) — first tuner
scored on the symmetric win-rate metric. Preloads all 18 windows; baseline
gate reproduces lineup_forward_test_detail_2026-07-09.csv's 510 LHP combos
bit-for-bit before any sweep. `download_batter_sides.py` — builds
"historical testing/batter_sides.csv" (2,579 batters; side actually hit
from vs each pitcher hand, from Statcast `stand`; 200 switch hitters).
Full sweep results: tuner_lhp_winrate_results_2026-07-10.txt.

Baseline: LHP win rate 56.7%, gap +0.0040, n=510 (by year: 60.9 / 47.1 /
61.8 / 62.4 / 60.8 / 47.1 — 2021 and 2025 are the weak years).

Sweep verdicts (walk-forward = train years before holdout, 2023/24/25;
LOYO = leave-one-year-out, all 6 years):
- WEIGHT_CAREER_LHP: flat, noise (WF +0.3pp, LOYO +0.1pp). NO SHIP.
- REGRESS_PA_CAREER_LHP: lower helps (100-300 -> ~59.2% all-years) but
  per-fold picks drift; WF +1.1pp, LOYO +1.9pp. Suggestive, NO SHIP alone.
- PLATOON_BLEND_LHP: flat (WF -0.8pp, LOYO -0.1pp). NO SHIP.
- XWOBA_BLEND (all 4 LHP components -> 0.65): picked in ALL 9 CV folds;
  WF +1.3pp, LOYO +1.7pp; gains concentrate in the weak years (2021 +4.7,
  2025 +7.1). CLEARS the v11 shipping bar.
- PLATOON_PRIOR_PA (NEW knob -- evidence-weighted batter-side platoon
  prior; priors from 2016-18 only: RHB-vs-LHP 1.0311, LHB-vs-LHP 0.9036;
  own career ratio keeps its weight, so reverse-split hitters are
  preserved; PRIOR_PA=0 == production bit-for-bit): smooth plateau
  100-800, WF picked 400 in all 3 folds (+1.8pp mean, 2025 +4.7),
  LOYO picks 200/400 (+1.3pp mean). CLEARS the bar.

INTERACTION TESTS (both negative — ship at most ONE change):
- XB_ALL x RPC joint grid: does not stack (joint CV worse than 1-D XB).
- XB_ALL x PRIOR_PA joint grid: does not stack (XB=0.65 + prior scores
  worse than either alone; one WF fold -9.4pp). The two knobs fix the
  same players — both temper aggressive extrapolation on thin vs-LHP
  samples — so they are substitutes, not complements.

DECISION (user, 2026-07-10): ship PLATOON_PRIOR_PA_LHP = 400. SHIPPED
same day as backtest.py v16 + test.py port:
- backtest.py: PLATOON_PRIOR_PA_LHP/RHP + PLATOON_PRIOR_RATIO_VS_LHP
  constants (full precision), BATTER_SIDES loader, stabilize() gains an
  optional pid arg (None = pre-v16 behavior for legacy callers).
  run_window() passes pid. RHP knob = 0 (bit-for-bit no-op).
- test.py: same constants; stabilize() gains prior_side; build_master()
  resolves each player's side from batter_sides.csv with roster B/T
  fallback (S bats R vs LHP).
- lineup_forward_test.py passes pid; lineup_ablation_test.py passes pid
  and gains a "no_prior" ablation config.
- VERIFIED: 3-way parity (tuner / backtest.stabilize / test.stabilize)
  exact on sample cases incl. the no-career pure-prior path; full
  lineup_forward_test re-run reproduces the tuner's PRIOR-400 CV row
  digit-for-digit and leaves RHP untouched.
- NEW CANONICAL HISTORICAL BASELINE (symmetric, 2026-07-10, n=1,047):
  overall 63.4% win rate, gap +0.0049; vs LHP 58.3% / +0.0043 (weak
  years now 2021 50.6%, 2025 51.8%); vs RHP unchanged 68.2% / +0.0055.
- XB_ALL=0.65 was the not-shipped alternative (equal evidence, mutually
  exclusive). Do NOT ship it on top of the prior. tuner_lhp_winrate.py's
  baseline gate still validates against the PRE-ship 2026-07-09 detail
  CSV with PRIOR_PA=0 baseline params -- both intentionally pinned.

### 2026-07-09: SYMMETRIC MEASUREMENT FRAMEWORK — the new canonical numbers

Per the user's direction, past and present now measure the exact same question:
"does the model's unconstrained top-9 vs a specific pitcher hand beat the
manager's actual top-9 by PA vs that hand?" Three changes:

1. **lineup_forward_test.py** — real lineup is now top-9 by PA **vs the specific
   hand** (was top-9 by total PA). Requires the 36 new
   `before_<cutoff>_<year>_vs_<hand>_with_team.csv` files built by
   `download_team_tags_by_hand.py` / `run_team_tags_by_hand.bat`. Legacy
   benchmark behind `--hand-agnostic-real` (suffixed outputs).
2. **live_forward_validation.py** — optimal lineup is now the plain
   position-agnostic top-9 by model score vs the hand (was test.py's
   position-DP/fallback/ordering pipeline), mirroring the historical optimal
   exactly, full-9 required. Legacy machinery test behind `--full-pipeline`
   (suffixed outputs).
3. **IL leakage fixed** — `test.py`'s `load_roster()` gained
   `exclude_unavailable` (default True, production unchanged); validation
   passes False so current IL status no longer removes players from past-cutoff
   pools.

**Canonical baselines under the symmetric framework (both from 2026-07-09 runs):**
- HISTORICAL (2019–2025, n=1,048): **62.6% win rate, mean gap +0.0047**,
  year-clustered 95% CI on win rate [59.1%, 65.7%], gap CI [+0.0036, +0.0060],
  permutation p=0.0000. By hand: 56.7% vs LHP, 68.2% vs RHP.
- LIVE 2026 (5 cutoffs through 06-10, n=262): **50.4% win rate, mean gap
  +0.0023**, cutoff-clustered gap CI [+0.0009, +0.0036], permutation p=0.029 —
  the live edge is now POSITIVE and statistically significant. By hand:
  42.5% vs LHP (gap +0.0011, n.s.), 56.4% vs RHP (gap +0.0033, p=0.0096).

Interpretation: the hand-specific benchmark costs ~3pp historically (65.8% →
62.6%) because beating real platoon deployment is harder — almost entirely on
the LHP side. The big live jump (41% → 50.4%) came from removing the
position-feasibility constraint and the IL leakage, i.e. measurement artifacts,
not model changes. The remaining live-vs-historical difference (50.4% vs 62.6%,
non-overlapping clustered CIs) is real but the live gap itself is positive —
2026 is a below-average but genuinely positive year for the model so far, worst
vs LHP. Candidates for the residual: one season of correlated data (5 clusters
understate uncertainty), players absent from today's roster file still being
excluded from past optimal pools (DFA'd/demoted players — roster_resource.xlsx
has no historical archive), and possibly sharper league-wide platooning in 2026.

Note the by-cutoff trend under the symmetric framework is much flatter (58.3 /
52.5 / 47.4 / 47.1 / 42.9 with positive gaps at both ends) — the earlier
"model decays over the season" narrative was mostly the position/IL artifacts
compounding, not model decay.

### 2026-07-08: platoon fix, baseline backfill, first experiments

**Platoon-formula bug found and FIXED in test.py.** test.py's stabilize() had never
actually matched backtest.py's validated platoon block, despite comments claiming a
verbatim port: it divided this-hand career wOBA by the CROSS-hand wOBA (instead of the
PA-weighted combined wOBA), clamped AFTER blending (instead of before), and ignored
career_cross_pa entirely — roughly 2x the tuned adjustment strength with no
sample-size protection. Fixed 2026-07-08 to an exact port; verified bit-for-bit
against backtest.stabilize() on 7 test cases including clamp-engaging extremes.
Effect on live validation was negligible (41.4% -> 41.1%, 10/263 combos changed,
1 win/loss flip) — the fix is correct but was not the live-underperformance driver.

**Historical baseline updated (complete 18-window set).** The missing
totals_before_june_1_2019_with_team.csv was backfilled (2019 June 1 had been silently
skipped). New baseline from lineup_forward_test_detail_2026-07-08.csv:
**65.8% win rate, +0.0058 mean gap, n=1,048** (was 66.1% / +0.0059 / n=988).
Year-clustered 95% CI [63.2%, 69.1%]; permutation p=0.0000.

**Live 2026 validation refreshed (5 cutoffs through 2026-06-10, post-fix):**
41.1% win rate, mean gap -0.0010 — but the cutoff-clustered CI on the gap is
[-0.0025, +0.0006] and permutation p=0.36, so the live result is statistically
FLAT, not significantly negative.

**Why live looks so far below 65.8% — benchmark asymmetry (CONFIRMED MAJOR).**
The live script's "real lineup" is top-9 by PA vs the SPECIFIC pitcher hand (it
encodes managers' actual platoon deployment); the historical baseline's real lineup
is top-9 by TOTAL PA (hand-agnostic, same 9 vs both hands) — a much weaker opponent.
Running the live validation with the new `--hand-agnostic-real` flag (mirrors the
historical methodology exactly): **46.3% win rate, mean gap flips POSITIVE
(+0.00019)**; vs LHP goes 36.3% -> 47.7% (gap +0.00284); the first three cutoffs sit
at ~50%. Only hand-agnostic live numbers are comparable to the 65.8% baseline.

**Ruled out:** the 2026-into-career fold-in (update_live_data.py's 2026-06-24
change). Tested via the new `--history-only-career` flag: slightly WORSE (40.5%,
gap -0.00171), decline by cutoff unchanged. Keep the fold-in.

**Still open on the remaining gap:** position-constraint asymmetry (live optimal
must field a position-feasible 9; historical optimal is unconstrained top-9),
today's-roster/IL leakage into past cutoffs (170 current-IL players excluded
retroactively; 34/300 builds had an unfilled slot), and single-season noise
(only 5 highly-correlated cutoff clusters).

**Tooling notes:** live_forward_validation.py now takes `--history-only-career`
and `--hand-agnostic-real` (combinable); experiment outputs carry `_histcareer` /
`_handagnostic` filename suffixes and NEVER overwrite the production summary JSON
or same-day production results. live_uncertainty_analysis.py and
lineup_change_magnitude_analysis.py glob date-only detail filenames so they skip
experiment outputs. Pre-fix live results archived in
old_results_archive/pre_platoon_fix_2026-07-08/. Section 5's sandbox quirks
(bash stale-file cache, rm blocked) were Cowork-specific and do NOT apply in
Claude Code sessions.

---

## 1. What This Project Does

Builds a **stabilized wOBA model** that predicts each MLB batter's future performance
vs left-handed (LHP) and right-handed (RHP) pitchers, then uses those predictions to
construct an **optimal team lineup** (top-9 by predicted wOBA). The model has been
validated via a **team-constrained, non-circular forward test**: the "real" lineup is
the team's top-9 players ranked by actual pre-cutoff PA (a proxy for who actually
played), the "optimal" lineup is the team's top-9 ranked by the model's predictions —
both graded on **actual post-cutoff wOBA**. No future information leaks into any prediction.

The primary metric is **win rate** (fraction of team-window-hand combos where the
model's optimal lineup outperformed the real lineup) and **mean wOBA gap** (optimal
minus real). Historical validated baseline: **66.1% win rate, +0.0059 mean gap** across
988 combos, 6 seasons (2019, 2021–2025), 3 cutoffs each (June 1, July 1, Aug 1).

---

## 2. File Architecture

### Core engine (do not modify without re-running validation)
| File | Role |
|---|---|
| `backtest.py` | Scoring engine — all weights, regression PA, stabilize(), blend_woba(), build_career_data(). Imported read-only by every validation/ablation script. |
| `test.py` | Live production optimizer for the **current 2026 season**. Reads from live data files (2026_*.csv, career_vs_*.csv, recent.csv). Outputs predicted lineups for any given date. |
| `update_live_data.py` | Builds the live 2026 data files used by test.py and live_forward_validation.py. Run this first any time you want fresh 2026 data. Run via `run_update_live_data.bat`. |

### Validation scripts
| File | Role |
|---|---|
| `lineup_forward_test.py` | Historical (2019–2025) team-constrained forward test. Outputs `lineup_forward_test_detail_YYYY-MM-DD.csv` (one row per combo: year, cutoff, team, hand, real_avg, opt_avg, gap, n_real, n_opt, win, n_diff). |
| `live_forward_validation.py` | Non-circular live forward validation for the **current 2026 season**. Outputs `live_forward_validation_detail_YYYY-MM-DD.csv` + `live_forward_validation_summary.json`. Also writes `live_forward_validation_build_log.csv` for diagnostics. |
| `historical_uncertainty_analysis.py` | Bootstrap CI + sign-flip permutation test on the historical detail CSV. Reads the latest `lineup_forward_test_detail_*.csv`. |
| `live_uncertainty_analysis.py` | Same as above but for the live detail CSV. |
| `lineup_change_magnitude_analysis.py` | Buckets results by `n_diff` (how many of the model's 9 picks differed from the real lineup: 1-swap, 2-swap, 3+swap). Reads both historical and live detail CSVs. Requires n_diff column — live CSV didn't have it before the July 3 data refresh; historical does. |
| `lineup_ablation_test.py` | Turns off one model component at a time (splits/career/total/recent/platoon/xwoba/decay/clamp) and reruns the forward-test metric. Imports only from backtest.py (does NOT import lineup_forward_test.py — see Known Bugs). Results saved to `lineup_ablation_test_results.txt`. |
| `baseline_comparison.py` | Compares model to a simple PA-rank baseline. |

### Tuner scripts (historical CV searches — for reference, not daily use)
| File | Role |
|---|---|
| `tuner.py` | General-purpose coordinate-ascent + random-search optimizer |
| `tuner_lhp.py` | LHP-specific walk-forward CV (produced v10 LHP values) |
| `tuner_rhp.py` | RHP-specific walk-forward CV (only XWOBA_BLEND_RHP was shipped) |
| `tuner_xwoba_components.py` | Per-component xWOBA blend tuner (8 constants, LHP+RHP × 4) |
| `tuner_aging.py` | AGING_STRENGTH CV search (result: not shipped, held at 0.0) |
| `tuner_decay_rhp.py` | DECAY_CAREER_RHP 1-D CV search (result: not shipped, held at 1.0) |
| `tuner_decay_regress_rhp.py` | Joint DECAY_CAREER_RHP × REGRESS_PA_CAREER_RHP CV search (result: not shipped) |

### Data download/build scripts
| File | Role |
|---|---|
| `download_woba_splits.py` | Pulls original wOBA split data |
| `download_v3_expansion.py` | Expanded pull: xwOBA, per-season career, 2019/2021/2022 years |
| `download_additional_data.py` | Additional data pulls |
| `download_team_tags.py` | Team assignments for players (produces `*_with_team.csv` files) |
| `build_age_data.py` | Builds `player_ages.csv` from Chadwick register |
| `build_aging_curve.py` | Estimates delta-method aging curve; produces `aging_curve_pairs.csv` |

### Batch launchers (.bat files)
All `.bat` files use `cd /d "%~dp0"` then call their Python counterpart directly,
so they **must stay in the same folder as the .py files**. Moving any .bat into a
subfolder would break it. Each has a corresponding `run_*_log.txt` that gets
overwritten on each run.

### Live data files (root of project folder)
`2026_total.csv`, `2026_vs_left.csv`, `2026_vs_right.csv`, `career_vs_left.csv`,
`career_vs_right.csv`, `recent.csv` — built by `update_live_data.py`.

### Static data files
`player_ages.csv` (Chadwick MLBAM age lookup), `aging_curve_pairs.csv`,
`roster_resource.xlsx`, `tuner_decay_regress_rhp_checkpoint.json`.

### Historical data folder
`historical testing/` — DATA_DIR for backtest.py and all validation scripts.
Contains the 18-window set of `before_*/after_*` CSV files by year and cutoff.
**Do not move or rename this folder.**

---

## 3. Production Model Constants (backtest.py v15)

```
LEAGUE_AVG_WOBA_LHP = 0.2943        LEAGUE_AVG_WOBA_RHP = 0.2575

# LHP weights / regress PA
WEIGHT_SPLITS_LHP  = 0.213          REGRESS_PA_SPLITS_LHP = 178
WEIGHT_CAREER_LHP  = 0.5695         REGRESS_PA_CAREER_LHP = 711
WEIGHT_TOTAL_LHP   = 0.247          REGRESS_PA_TOTAL_LHP  = 1020
WEIGHT_RECENT_LHP  = 0.3416         REGRESS_PA_RECENT_LHP = 757

# RHP weights / regress PA
WEIGHT_SPLITS_RHP  = 0.2774         REGRESS_PA_SPLITS_RHP = 285
WEIGHT_CAREER_RHP  = 0.2920         REGRESS_PA_CAREER_RHP = 25
WEIGHT_TOTAL_RHP   = 0.2153         REGRESS_PA_TOTAL_RHP  = 400
WEIGHT_RECENT_RHP  = 0.2153         REGRESS_PA_RECENT_RHP = 665

# PA minimums (not tunable)
MIN_PA_SPLITS = MIN_PA_TOTAL = MIN_PA_CAREER = 30
MIN_PA_POST   = 30
NO_CLAMP      = True   # clamping disabled — improves Spearman-r; confirmed beneficial in win-rate metric too

# Platoon blend
PLATOON_BLEND_LHP = 0.7833     PLATOON_FACTOR_MIN/MAX_LHP = 0.612 / 1.138
PLATOON_BLEND_RHP = 0.31       PLATOON_FACTOR_MIN/MAX_RHP = 0.585 / 1.325

# Career recency decay (LHP tuned v10; RHP 3 CV searches all failed, stays 1.0)
DECAY_CAREER_LHP = 0.8001      DECAY_CAREER_RHP = 1.0

# xWOBA blend per component (v12 split; not yet individually tuned — all still at hand-level starting point)
XWOBA_BLEND_SPLITS_LHP = XWOBA_BLEND_CAREER_LHP = XWOBA_BLEND_TOTAL_LHP = XWOBA_BLEND_RECENT_LHP = 0.9187
XWOBA_BLEND_SPLITS_RHP = XWOBA_BLEND_CAREER_RHP = XWOBA_BLEND_TOTAL_RHP = XWOBA_BLEND_RECENT_RHP = 0.80

# Aging curve (v13, embedded; AGING_STRENGTH = 0.0 — CV search failed to clear bar, not applied)
AGING_STRENGTH = 0.0

# Historical test scope
TEST_YEARS = [2019, 2021, 2022, 2023, 2024, 2025]  # skips 2020 (COVID-shortened)
CUTOFFS    = ["june_1", "july_1", "aug_1"]
```

**Why parameters are at these values:** LHP was fully walk-forward-CV tuned in v10
(all 14 knobs jointly). RHP received partial tuning: only XWOBA_BLEND_RHP (0.80 via
v11 isolated 1-D CV) was shipped; every other RHP knob either failed CV or was never
individually validated. The component-level XWOBA_BLEND split (v12) is a pure
refactor — values haven't moved from their hand-level starting points yet.
Task #25 (full re-tune over the 18-window expanded set) is still technically open
but may be superseded by the incremental per-knob approach used since v10.

---

## 4. Validation Results (as of 2026-07-03)

### Historical forward test (2019–2025)
Source: `lineup_forward_test_detail_2026-06-26.csv` (988 rows, with n_diff column)

**Overall:** n=988, **win rate 66.1%, mean gap +0.0059**
- Year-clustered bootstrap 95% CI: win rate [63.4%, 69.4%], gap [+0.0048, +0.0070]
- The year-clustered CI is virtually identical width to the naive CI (1.0×), meaning the edge replicates year-over-year rather than being driven by one lucky season
- Sign-flip permutation test p=0.0000 (easily distinguishable from pure noise)
- 27 of 30 MLB teams favor the model lineup over the real lineup

**By pitcher hand:**
- vs LHP: n=480, win rate 63.7%, gap +0.0061 (significant)
- vs RHP: n=508, win rate 68.3%, gap +0.0057 (significant)
- LHP vs RHP gap: observed +4.6pp, but year-clustered CI straddles zero — not distinguishable from season-to-season noise

**By cutoff:**
- June 1: n=298, 61.7%, gap +0.0043 (significant)
- July 1: n=356, higher (full results in historical_uncertainty_analysis_results.txt)
- Aug 1:  higher still

### n_diff (lineup-change-magnitude) breakdown — historical
Source: `lineup_change_magnitude_analysis_results.txt`

| Bucket | n | Win rate | Mean gap |
|---|---|---|---|
| 1-swap | 207 | 65.7% | +0.0042 |
| 2-swap | 397 | 66.8% | +0.0052 |
| 3+-swap | 371 | 67.9% | +0.0078 |

All three buckets: p=0.0000 (sign-flip permutation test). The edge is **real even when the model changes only one player** — not an artifact of large-scale roster overhauls.
The 3+-swap vs 1-swap difference (observed +2.2pp) has a CI of [-6.1%, +8.5%] — straddles zero, so "more swaps = higher edge" is not statistically confirmed.

*Note: Live 2026 detail CSV did not yet have the n_diff column as of 2026-06-26 (the column was added after that run). The next run (July 3 refresh) will populate it.*

### Live 2026 forward validation
The live validation was run as of 2026-06-26. Specific numbers not captured here;
see `live_forward_validation_summary.json` and `live_uncertainty_analysis_results.txt`
(in `old_results_archive/results_and_logs/` — moved there in the July 3 cleanup).
**The July 3 data refresh and re-run is the immediate next task (see Section 7).**

### Ablation study — historical (2019–2025)
Ran `lineup_ablation_test.py` (via `lineup_ablation_test_run2.py` workaround — see
Known Bugs). Full results saved to `lineup_ablation_test_results.txt`.

Year-clustered paired bootstrap (10,000 draws, seed=20260629):

| Config | n | Win rate | Mean gap | Δ win% | Significant? |
|---|---|---|---|---|---|
| FULL (baseline) | 988 | 66.1% | +0.0059 | — | — |
| no_career | 990 | 54.9% | +0.0035 | **-11.1pp** | YES |
| no_xwoba | 978 | 61.3% | +0.0042 | **-4.7pp** | YES |
| no_splits | 980 | 62.9% | +0.0053 | -3.2pp | YES |
| no_recent | 985 | 63.8% | +0.0055 | -2.3pp | YES |
| no_platoon | 986 | 64.6% | +0.0056 | -1.5pp | YES |
| no_total | 980 | 65.8% | +0.0060 | -0.3pp | NO |
| no_decay (LHP only) | 984 | 66.3% | +0.0061 | +0.2pp | NO |
| clamp_on | 988 | 66.4% | +0.0060 | +0.3pp | YES (positive — clamping hurts) |

**Interpretation:** Career data is the dominant driver. xWOBA blending, current-season
splits, recent form, and the platoon prior all contribute real signal. Current-season
totals and career decay are inert in this win-rate metric (they were tuned on Spearman-r).
The NO_CLAMP=True production choice is validated: re-enabling clamping hurts win rate.

---

## 5. Known Bugs / Environment Quirks

### Bash stale-file-cache bug (CRITICAL for any future editing)
Once `mcp__workspace__bash` reads or executes a file, the sandbox caches its byte
content. Any subsequent `Edit()` / `Write()` changes to that same file path are
**invisible to bash for the rest of that session** (appears inode-keyed, not
path-keyed — survives rename, doesn't resolve with waiting). The Read/Edit/Write
tools always see the correct on-disk content; bash does not.

**Workaround (proven twice):** Write the corrected file content to a brand-new
filename (e.g. `foo_run2.py`), run/verify via that fresh file, then rename it to
`zz_DELETE_ME_foo_run2.py.bak`. Do NOT use the stale original filename in bash.

**Currently affected file:** `lineup_ablation_test.py` — bash has a stale, truncated
view of it (line 424 appears cut off as `print("{:<4`). The **actual file on disk is
correct and complete** (verified via Read tool). The successful full ablation run was
done via `lineup_ablation_test_run2.py` (now in `old_results_archive/deletable_backups/`).
In a **fresh session** this bug resets — bash will see the correct file on first access.

### rm / unlink blocked on OneDrive mount
`rm` returns "Operation not permitted" on the bash-side mount of the OneDrive folder.
**Workaround:** rename files to `zz_DELETE_ME_*.bak` instead. The user can then
delete them manually from Windows Explorer.

### No absolute file deletion possible from bash
All "deleted" files pile up as `zz_DELETE_ME_*.bak` artifacts. Currently 4 of
these are in `old_results_archive/deletable_backups/` — safe to delete manually.

---

## 6. Folder Structure (as of 2026-07-03 cleanup)

The project folder is intentionally flat — all live `.py`, `.bat`, and active `.csv`
files must remain in the **root** because scripts reference each other and their data
via `os.path.dirname(__file__)` and `cd /d "%~dp0"` relative paths. Moving any live
file into a subfolder will break it without code changes.

```
lineup optimization project/
  ├── backtest.py, test.py, lineup_forward_test.py, live_forward_validation.py
  ├── lineup_ablation_test.py, lineup_change_magnitude_analysis.py
  ├── historical_uncertainty_analysis.py, live_uncertainty_analysis.py
  ├── update_live_data.py, baseline_comparison.py
  ├── tuner*.py, download_*.py, build_*.py
  ├── run_*.bat  (all batch launchers — stay flat with their .py counterparts)
  ├── 2026_total.csv, 2026_vs_left.csv, 2026_vs_right.csv  (live data)
  ├── career_vs_left.csv, career_vs_right.csv, recent.csv   (live career/recent)
  ├── lineup_forward_test_detail_2026-06-26.csv   (historical detail, read by analysis scripts)
  ├── live_forward_validation_detail_2026-06-26.csv   (live detail — will be refreshed July 3)
  ├── live_forward_validation_summary.json, live_forward_validation_build_log.csv
  ├── player_ages.csv, aging_curve_pairs.csv, roster_resource.xlsx
  ├── tuner_decay_regress_rhp_checkpoint.json
  ├── historical testing/  ← DATA_DIR; contains all 18-window before_*/after_* CSVs
  ├── old_results_archive/
  │   ├── results_and_logs/  ← 46 old *_results*.txt and run_*_log.txt files
  │   └── deletable_backups/ ← 4 zz_DELETE_ME_*.bak files (safe to delete manually)
  └── ziJddmLS  ← mysterious 0-byte file, unknown origin, left alone
```

---

## 7. Next Steps (Immediate)

**The user's stated plan for the next session:** Update the 2026 data and re-run the
live forward validation now that it is July 3.

### Step 1 — Update live 2026 data
Run `run_update_live_data.bat` (or `python update_live_data.py`).
This pulls fresh 2026 stats through ~July 3 and regenerates:
`2026_total.csv`, `2026_vs_left.csv`, `2026_vs_right.csv`,
`career_vs_left.csv`, `career_vs_right.csv`, `recent.csv`.

### Step 2 — Re-run live forward validation
Run `run_live_forward_validation.bat` (or `python live_forward_validation.py`).
This will produce a new `live_forward_validation_detail_YYYY-MM-DD.csv`
**that now includes the `n_diff` column** (it was added to the script before the last
historical run; the live script should have it too — verify the CSV has that column
before proceeding to Step 4).

### Step 3 — Re-run live uncertainty analysis
Run `python live_uncertainty_analysis.py` on the new live detail CSV.
Reports win rate, bootstrap CIs, permutation test for the 2026 live validation.

### Step 4 — Re-run lineup change magnitude analysis
Run `python lineup_change_magnitude_analysis.py`. Now that the live detail CSV has
`n_diff`, this will run both the historical bucket analysis and the new live bucket
analysis in one pass.

### Step 5 — Continue validation interpretation
Compare live 2026 results to historical baselines. Key questions:
- Is the live 2026 win rate in the same ballpark as the 66.1% historical figure?
- Does the n_diff breakdown look similar live vs historical?
- Do any anomalies show up in the build log (fallback slots, data gaps, etc.)?

---

## 8. Open / Deferred Items

- **Task #25** (full joint re-tune across all 18 windows) — technically still open;
  may be superseded by the per-knob v10–v15 tuning approach already done.
- **Per-component xWOBA tuning** — the v12 split into 8 constants was a pure refactor;
  `tuner_xwoba_components.py` was built but its shipped values have not been reported
  here. Check `tuner_xwoba_components_results_*.txt` in `old_results_archive/` to see
  whether the search produced any values worth shipping.
- **Real batting-order sequencing validation** — the current "real lineup" is the
  top-9 by pre-cutoff PA (a proxy, not the actual manager's chosen order). Retrosheet
  event files and Baseball-Reference box scores carry actual per-game batting order —
  a possible future validation avenue if lineup *sequencing* (not just *personnel*)
  ever becomes the focus.
- **Live ablation** — `lineup_ablation_test.py` only covers historical data
  (TEST_YEARS/CUTOFFS). No live-data ablation variant has been built yet.
- **ziJddmLS** — 0-byte file of unknown origin in project root. Left alone.
