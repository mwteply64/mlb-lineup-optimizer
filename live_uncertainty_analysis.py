#!/usr/bin/env python3
"""
Live (2026) Uncertainty Analysis - Lineup Optimization Project
================================================================
Same statistical layer as historical_uncertainty_analysis.py, applied to
live_forward_validation.py's per-combo detail CSV (live_forward_validation_
detail_<date>.csv: cutoff, team, hand, real_avg, opt_avg, gap, n_real,
n_opt, win) instead of the historical one. Doesn't touch any scoring
constant, weight, or pipeline file.

KEY DIFFERENCE FROM THE HISTORICAL VERSION -- READ THIS FIRST
---------------------------------------------------------------
The historical analysis clustered on YEAR (6 independent seasons). Live
data has only ONE season (2026) sliced into 4 rolling cutoffs (04-15,
04-29, 05-13, 05-27) that all share the same rosters, the same injuries,
the same hot/cold streaks carrying over from one cutoff to the next --
they are far more correlated with each other than two different seasons
ever were. CUTOFF is still the right cluster unit (it's the only
available proxy for "an independent slice of the season"), but with only
4 clusters instead of 6, and with much higher correlation between them,
the cluster-bootstrap CI here will be both WIDER and a more aggressive
correction than its historical counterpart. Read the "Nx as wide" ratio
in the output as a lower bound on how much wider reality probably is --
4 clusters is not enough to characterize a resampling distribution well,
so treat these CIs as a rough, conservative signal, not a precise one.

THREE CHECKS, EACH ANSWERING A DIFFERENT QUESTION (see
historical_uncertainty_analysis.py for the full rationale of each --
identical logic here, just swapping the cluster unit from year to cutoff)
1. Naive row-level bootstrap CI       (optimistic: ignores cutoff correlation)
2. Cutoff-clustered bootstrap CI      (honest, but crude with only 4 clusters)
3. Sign-flip permutation test         (H0: mean gap == 0)

USAGE
  python live_uncertainty_analysis.py
  Reads the newest live_forward_validation_detail_*.csv in this folder.
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
    # Date-only pattern (????-??-??.csv) deliberately EXCLUDES the suffixed
    # experiment outputs live_forward_validation.py can now also write
    # (e.g. *_histcareer.csv, *_handagnostic.csv, added 2026-07-08) --
    # this analysis should always run on the production detail file.
    candidates = sorted(glob.glob(os.path.join(
        FOLDER, "live_forward_validation_detail_????-??-??.csv")))
    if not candidates:
        raise FileNotFoundError(
            "No live_forward_validation_detail_<date>.csv found. Run live_forward_validation.py first.")
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


def cutoff_clustered_bootstrap_ci(by_cutoff_wins, by_cutoff_gaps, cutoffs, n_boot=N_BOOT):
    """Resample whole cutoffs with replacement (only 4 available here, vs
    6 years historically). Each draw picks len(cutoffs) cutoffs with
    replacement from the available set, pools every combo from the chosen
    cutoffs, and recomputes the stat. Cruder than the year-clustered
    historical version -- 4 clusters is a thin basis for a resampling
    distribution -- but still the structurally-correct unit, since combos
    within a cutoff share that cutoff's specific in-season state."""
    n_cutoffs = len(cutoffs)
    win_draws = np.empty(n_boot)
    gap_draws = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, n_cutoffs, size=n_cutoffs)
        w = np.concatenate([by_cutoff_wins[cutoffs[i]] for i in chosen])
        g = np.concatenate([by_cutoff_gaps[cutoffs[i]] for i in chosen])
        win_draws[b] = w.mean()
        gap_draws[b] = g.mean()
    all_wins = np.concatenate([by_cutoff_wins[c] for c in cutoffs])
    all_gaps = np.concatenate([by_cutoff_gaps[c] for c in cutoffs])
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
    cutoffs = sorted(df["cutoff"].unique())
    by_cutoff_wins = {c: df.loc[df["cutoff"] == c, "win"].astype(bool).to_numpy() for c in cutoffs}
    by_cutoff_gaps = {c: df.loc[df["cutoff"] == c, "gap"].astype(float).to_numpy() for c in cutoffs}

    lines.append("-" * 66)
    lines.append(name + "  (n=" + str(len(df)) + ", " + str(len(cutoffs)) + " cutoffs: " +
                 ", ".join(str(c) for c in cutoffs) + ")")
    lines.append("-" * 66)

    naive = naive_bootstrap_ci(wins, gaps)
    lines.append("  Naive row-level bootstrap (treats all combos as independent -- optimistic):")
    lines.append(fmt_ci("win rate", naive["win_rate"], as_pct=True))
    lines.append(fmt_ci("mean gap", naive["mean_gap"]))

    if len(cutoffs) >= 2:
        clustered = cutoff_clustered_bootstrap_ci(by_cutoff_wins, by_cutoff_gaps, cutoffs)
        lines.append("  Cutoff-clustered bootstrap (resamples whole cutoffs -- honest, but crude with only " +
                     str(len(cutoffs)) + " clusters):")
        lines.append(fmt_ci("win rate", clustered["win_rate"], as_pct=True))
        lines.append(fmt_ci("mean gap", clustered["mean_gap"]))
        naive_width = naive["win_rate"][2] - naive["win_rate"][1]
        clus_width = clustered["win_rate"][2] - clustered["win_rate"][1]
        ratio = clus_width / naive_width if naive_width > 0 else float("nan")
        lines.append("  Cutoff-clustered CI is " + format(ratio, ".1f") +
                     "x as wide as the naive CI on win rate. (With only " + str(len(cutoffs)) +
                     " clusters, treat this as a rough lower bound on true uncertainty, not a precise figure.)")
    else:
        lines.append("  (only 1 cutoff in this subset -- cluster bootstrap skipped, no resampling unit available)")

    observed, p_value, _ = sign_flip_permutation_test(gaps)
    lines.append("  Sign-flip permutation test (H0: mean gap == 0):")
    lines.append("    observed mean gap " + format(observed, "+.4f") +
                 "   empirical two-sided p = " + format(p_value, ".4f") +
                 ("  (< 0.05 -- distinguishable from pure noise)" if p_value < 0.05
                  else "  (NOT distinguishable from pure noise at p<0.05)"))
    lines.append("")


def analyze_hand_difference(df, lines):
    """Is the vs-LHP vs vs-RHP win-rate gap itself real, or could that
    split arise from cutoff-to-cutoff noise alone? Cluster-bootstraps
    BOTH hand subsets from the SAME resampled cutoff list each draw, so
    the within-cutoff pairing (same teams/season-state on both sides) is
    preserved. With only 4 cutoffs, this is a much blunter instrument
    than the historical 6-year version -- expect a wide CI."""
    cutoffs = sorted(df["cutoff"].unique())
    by_cutoff = {}
    for c in cutoffs:
        sub = df[df["cutoff"] == c]
        by_cutoff[c] = {
            "LHP": sub.loc[sub["hand"] == "LHP", "win"].astype(bool).to_numpy(),
            "RHP": sub.loc[sub["hand"] == "RHP", "win"].astype(bool).to_numpy(),
        }
    n_cutoffs = len(cutoffs)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        chosen = rng.integers(0, n_cutoffs, size=n_cutoffs)
        lhp = np.concatenate([by_cutoff[cutoffs[i]]["LHP"] for i in chosen])
        rhp = np.concatenate([by_cutoff[cutoffs[i]]["RHP"] for i in chosen])
        diffs[b] = rhp.mean() - lhp.mean()

    rhp_rate = df.loc[df["hand"] == "RHP", "win"].mean()
    lhp_rate = df.loc[df["hand"] == "LHP", "win"].mean()
    observed_diff = rhp_rate - lhp_rate
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    straddles_zero = lo <= 0 <= hi

    lines.append("-" * 66)
    lines.append("vs RHP win rate minus vs LHP win rate (cutoff-clustered, only " +
                 str(n_cutoffs) + " clusters -- wide by construction)")
    lines.append("-" * 66)
    lines.append("  observed: RHP " + format(rhp_rate, ".1%") + " - LHP " + format(lhp_rate, ".1%") +
                 " = " + format(observed_diff, "+.1%"))
    lines.append("  95% CI on the difference: [" + format(lo, "+.1%") + ", " + format(hi, "+.1%") + "]")
    lines.append("  " + ("CI straddles 0 -- the RHP/LHP split is NOT distinguishable from "
                          "cutoff-to-cutoff noise given only " + str(n_cutoffs) + " clusters."
                          if straddles_zero else
                          "CI excludes 0 -- the RHP/LHP split holds up even after cutoff-clustering, "
                          "despite only " + str(n_cutoffs) + " clusters to work with."))
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
    lines.append("LIVE (2026) UNCERTAINTY ANALYSIS")
    lines.append("Source: " + os.path.basename(path))
    lines.append("NOTE: cluster unit is CUTOFF (only " + str(df["cutoff"].nunique()) +
                 " available, all within one season) -- see module docstring.")
    lines.append("=" * 66)
    lines.append("")

    analyze_subset("OVERALL", df, lines)
    analyze_subset("vs LHP", df[df["hand"] == "LHP"], lines)
    analyze_subset("vs RHP", df[df["hand"] == "RHP"], lines)
    analyze_hand_difference(df, lines)

    report = "\n".join(lines)
    print(report)

    out_path = os.path.join(FOLDER, "live_uncertainty_analysis_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print()
    print("Full report written to: " + out_path)


if __name__ == "__main__":
    main()
