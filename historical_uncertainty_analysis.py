#!/usr/bin/env python3
"""
Historical Uncertainty Analysis - Lineup Optimization Project
================================================================
A cheap, model-independent statistical layer on top of
lineup_forward_test.py's already-validated per-combo results. Doesn't
touch any scoring constant, weight, or pipeline file -- reads the
detail CSV that script now writes (lineup_forward_test_detail_<date>.csv,
one row per team-window-hand combo: year, cutoff, team, hand, real_avg,
opt_avg, gap, n_real, n_opt, win) and asks: how much of the headline
66.1% win rate / +0.0059 mean wOBA gap is solid signal vs. how much
could be sampling noise?

THREE CHECKS, EACH ANSWERING A DIFFERENT QUESTION
---------------------------------------------------
1. Naive row-level bootstrap CI
   Treats all 988 combos as independent draws and resamples them with
   replacement. This is the "textbook" bootstrap, but it's optimistic
   here: combos from the same year share that year's specific rosters,
   so they're correlated, not independent. This CI is a lower bound on
   true uncertainty.

2. Year-clustered bootstrap CI
   Resamples whole YEARS (the 6 available seasons) with replacement,
   keeping every combo from a chosen year together. This is the honest
   version: it treats "how did the model do in 2022" as one data point,
   not the ~165 separate team-window-hand combos inside it. Comparing
   this CI's width to the naive one shows exactly how much the
   within-year correlation was hiding.

3. Sign-flip permutation test
   Tests H0: the model has zero true skill, i.e. each combo's signed
   gap (optimal - real) could just as easily have come out negative by
   chance. Randomly flips the sign of every combo's gap many times,
   rebuilds the null distribution of the mean gap, and reports what
   fraction of null draws are as extreme as what was actually observed.
   This is a non-parametric way to ask "is +0.0059 actually
   distinguishable from 0," using only the sign-exchangeability
   assumption -- no claim about independence across years needed.

USAGE
  python historical_uncertainty_analysis.py
  Reads the newest lineup_forward_test_detail_*.csv in this folder.
"""
import os
import glob
import numpy as np
import pandas as pd

FOLDER = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 10000
N_PERM = 10000
RNG_SEED = 20260626  # fixed seed -- reproducible CIs run to run

rng = np.random.default_rng(RNG_SEED)


def latest_detail_csv():
    # Date-only pattern (????-??-??.csv) deliberately EXCLUDES suffixed
    # experiment outputs (e.g. *_handagnostic.csv, added 2026-07-09) --
    # this analysis should always run on the production detail file.
    candidates = sorted(glob.glob(os.path.join(
        FOLDER, "lineup_forward_test_detail_????-??-??.csv")))
    if not candidates:
        raise FileNotFoundError(
            "No lineup_forward_test_detail_*.csv found. Run lineup_forward_test.py first.")
    return candidates[-1]


def naive_bootstrap_ci(wins, gaps, n_boot=N_BOOT):
    """Resample individual rows with replacement. Returns dict of point
    estimate + 95% CI for win rate and mean gap."""
    n = len(wins)
    idx = rng.integers(0, n, size=(n_boot, n))
    win_draws = wins[idx].mean(axis=1)
    gap_draws = gaps[idx].mean(axis=1)
    return {
        "win_rate": (wins.mean(), np.percentile(win_draws, 2.5), np.percentile(win_draws, 97.5)),
        "mean_gap": (gaps.mean(), np.percentile(gap_draws, 2.5), np.percentile(gap_draws, 97.5)),
    }


def year_clustered_bootstrap_ci(by_year_wins, by_year_gaps, years, n_boot=N_BOOT):
    """Resample whole years with replacement (6 years available here).
    Each draw picks 6 years with replacement from the 6 available, pools
    every combo from the chosen years, and recomputes the stat. This is
    the structurally-correct unit of resampling, since combos within a
    year share that year's specific rosters/injuries/form."""
    n_years = len(years)
    win_draws = np.empty(n_boot)
    gap_draws = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, n_years, size=n_years)
        w = np.concatenate([by_year_wins[years[i]] for i in chosen])
        g = np.concatenate([by_year_gaps[years[i]] for i in chosen])
        win_draws[b] = w.mean()
        gap_draws[b] = g.mean()
    all_wins = np.concatenate([by_year_wins[y] for y in years])
    all_gaps = np.concatenate([by_year_gaps[y] for y in years])
    return {
        "win_rate": (all_wins.mean(), np.percentile(win_draws, 2.5), np.percentile(win_draws, 97.5)),
        "mean_gap": (all_gaps.mean(), np.percentile(gap_draws, 2.5), np.percentile(gap_draws, 97.5)),
    }


def sign_flip_permutation_test(gaps, n_perm=N_PERM):
    """H0: mean gap == 0, gaps are sign-symmetric noise. Randomly flips
    the sign of every combo's gap n_perm times, builds the null
    distribution of the mean, and returns the observed mean plus a
    two-sided empirical p-value."""
    n = len(gaps)
    observed = gaps.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null_means = (signs * gaps[None, :]).mean(axis=1)
    p_value = (np.abs(null_means) >= abs(observed)).mean()
    return observed, p_value, null_means


def fmt_ci(label, triple, as_pct=False, decimals=4):
    point, lo, hi = triple
    if as_pct:
        return "  {:<10s} {:6.1%}   95% CI [{:6.1%}, {:6.1%}]".format(label, point, lo, hi)
    fmt = "  {{:<10s}} {{:+.{0}f}}   95% CI [{{:+.{0}f}}, {{:+.{0}f}}]".format(decimals)
    return fmt.format(label, point, lo, hi)


def analyze_subset(name, df, lines):
    wins = df["win"].astype(bool).to_numpy()
    gaps = df["gap"].astype(float).to_numpy()
    years = sorted(df["year"].unique())
    by_year_wins = {y: df.loc[df["year"] == y, "win"].astype(bool).to_numpy() for y in years}
    by_year_gaps = {y: df.loc[df["year"] == y, "gap"].astype(float).to_numpy() for y in years}

    lines.append("-" * 66)
    lines.append(name + "  (n=" + str(len(df)) + ", " + str(len(years)) + " seasons: " +
                 ", ".join(str(y) for y in years) + ")")
    lines.append("-" * 66)

    naive = naive_bootstrap_ci(wins, gaps)
    lines.append("  Naive row-level bootstrap (treats all combos as independent -- optimistic):")
    lines.append(fmt_ci("win rate", naive["win_rate"], as_pct=True))
    lines.append(fmt_ci("mean gap", naive["mean_gap"]))

    if len(years) >= 2:
        clustered = year_clustered_bootstrap_ci(by_year_wins, by_year_gaps, years)
        lines.append("  Year-clustered bootstrap (resamples whole seasons -- honest):")
        lines.append(fmt_ci("win rate", clustered["win_rate"], as_pct=True))
        lines.append(fmt_ci("mean gap", clustered["mean_gap"]))
        naive_width = naive["win_rate"][2] - naive["win_rate"][1]
        clus_width = clustered["win_rate"][2] - clustered["win_rate"][1]
        ratio = clus_width / naive_width if naive_width > 0 else float("nan")
        lines.append("  Year-clustered CI is " + format(ratio, ".1f") +
                     "x as wide as the naive CI on win rate.")
    else:
        lines.append("  (only 1 season in this subset -- year-clustered bootstrap skipped)")

    observed, p_value, _ = sign_flip_permutation_test(gaps)
    lines.append("  Sign-flip permutation test (H0: mean gap == 0):")
    lines.append("    observed mean gap " + format(observed, "+.4f") +
                 "   empirical two-sided p = " + format(p_value, ".4f") +
                 ("  (< 0.05 -- distinguishable from pure noise)" if p_value < 0.05
                  else "  (NOT distinguishable from pure noise at p<0.05)"))
    lines.append("")


def analyze_hand_difference(df, lines):
    """Is the vs-LHP (63.8%) vs vs-RHP (68.3%) gap itself real, or could
    that split arise from year-to-year noise alone? Cluster-bootstraps
    BOTH hand subsets from the SAME resampled year list each draw, so
    the within-year pairing (same teams/seasons on both sides) is
    preserved -- the right way to test a difference-of-subsets when
    both subsets come from identical clusters."""
    years = sorted(df["year"].unique())
    by_year = {}
    for y in years:
        sub = df[df["year"] == y]
        by_year[y] = {
            "LHP": sub.loc[sub["hand"] == "LHP", "win"].astype(bool).to_numpy(),
            "RHP": sub.loc[sub["hand"] == "RHP", "win"].astype(bool).to_numpy(),
        }
    n_years = len(years)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        chosen = rng.integers(0, n_years, size=n_years)
        lhp = np.concatenate([by_year[years[i]]["LHP"] for i in chosen])
        rhp = np.concatenate([by_year[years[i]]["RHP"] for i in chosen])
        diffs[b] = rhp.mean() - lhp.mean()

    rhp_rate = df.loc[df["hand"] == "RHP", "win"].mean()
    lhp_rate = df.loc[df["hand"] == "LHP", "win"].mean()
    observed_diff = rhp_rate - lhp_rate
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    straddles_zero = lo <= 0 <= hi

    lines.append("-" * 66)
    lines.append("vs RHP win rate minus vs LHP win rate (year-clustered)")
    lines.append("-" * 66)
    lines.append("  observed: RHP " + format(rhp_rate, ".1%") + " - LHP " + format(lhp_rate, ".1%") +
                 " = " + format(observed_diff, "+.1%"))
    lines.append("  95% CI on the difference: [" + format(lo, "+.1%") + ", " + format(hi, "+.1%") + "]")
    lines.append("  " + ("CI straddles 0 -- the RHP/LHP split is NOT clearly distinguishable "
                          "from season-to-season noise once year-clustering is accounted for."
                          if straddles_zero else
                          "CI excludes 0 -- the RHP/LHP split holds up even after year-clustering."))
    lines.append("")


def main():
    path = latest_detail_csv()
    df = pd.read_csv(path)
    df["win"] = df["win"].astype(str).str.lower().isin(["true", "1"])
    print("Loaded " + str(len(df)) + " combos from " + os.path.basename(path))
    print("(" + str(N_BOOT) + " bootstrap draws, " + str(N_PERM) + " permutation draws, seed=" +
          str(RNG_SEED) + ")")
    print()

    lines = []
    lines.append("=" * 66)
    lines.append("HISTORICAL UNCERTAINTY ANALYSIS")
    lines.append("Source: " + os.path.basename(path))
    lines.append("=" * 66)
    lines.append("")

    analyze_subset("OVERALL", df, lines)
    analyze_subset("vs LHP", df[df["hand"] == "LHP"], lines)
    analyze_subset("vs RHP", df[df["hand"] == "RHP"], lines)
    for cutoff in ["june_1", "july_1", "aug_1"]:
        analyze_subset(cutoff.replace("_", " ").upper(), df[df["cutoff"] == cutoff], lines)
    analyze_hand_difference(df, lines)

    report = "\n".join(lines)
    print(report)

    out_path = os.path.join(FOLDER, "historical_uncertainty_analysis_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print()
    print("Full report written to: " + out_path)


if __name__ == "__main__":
    main()
