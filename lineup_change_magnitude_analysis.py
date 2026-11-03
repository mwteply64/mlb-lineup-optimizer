#!/usr/bin/env python3
"""
Lineup-Change-Magnitude Analysis - Lineup Optimization Project
================================================================
Answers a specific question raised in review of the historical/live
forward-test results: does the model's edge only show up when it makes
big personnel swaps, or does it persist even when it barely touches the
real lineup?

Uses n_diff -- the per-combo "personnel-change magnitude" column added
to lineup_forward_test.py and live_forward_validation.py on 2026-06-26
(len(set(optimal_lineup) - set(real_lineup)): how many of the model's 9
picks were NOT in the real lineup). n_diff is independent of win/loss or
gap size by construction -- it's purely "how much did the model change."

Reuses every statistical primitive from historical_uncertainty_analysis.py
unchanged (same bootstrap/permutation machinery, same year-clustering
design, same seed) -- this script only adds the n_diff bucketing on top.
Does not touch any scoring constant, weight, or pipeline file.

BUCKETS
  n_diff == 0   1 of the 9 optimal picks happens to equal the real
                lineup exactly -- trivial by definition (win is always
                False, gap is always 0, since optimal IS real). Reported
                separately, excluded from the main comparison.
  n_diff == 1   model swapped exactly 1 player
  n_diff == 2   model swapped exactly 2 players
  n_diff >= 3   model swapped 3 or more players

USAGE
  python lineup_change_magnitude_analysis.py
  Reads the newest lineup_forward_test_detail_*.csv in this folder.
  (A live-data companion run happens automatically too, if a
  live_forward_validation_detail_*.csv with an n_diff column is present.)
"""
import os
import glob
import numpy as np
import pandas as pd

from historical_uncertainty_analysis import (
    naive_bootstrap_ci, year_clustered_bootstrap_ci, sign_flip_permutation_test,
    fmt_ci, N_BOOT, rng,
)

FOLDER = os.path.dirname(os.path.abspath(__file__))


def analyze_subset_clustered(name, df, cluster_col, lines):
    """Cluster-generic version of historical_uncertainty_analysis.py's
    analyze_subset(), which hardcodes df['year'] and therefore crashed
    (KeyError: 'year') the first time the LIVE half of this script ran
    (2026-07-08 -- the live detail CSV clusters on 'cutoff' instead, and
    before that date it had no n_diff column, so this path never
    executed). Identical statistics; the only change is that the cluster
    column is a parameter. The imported bootstrap primitives were
    already cluster-agnostic -- year_clustered_bootstrap_ci just takes
    dicts keyed by cluster value."""
    wins = df["win"].astype(bool).to_numpy()
    gaps = df["gap"].astype(float).to_numpy()
    clusters = sorted(df[cluster_col].unique())
    by_cluster_wins = {c: df.loc[df[cluster_col] == c, "win"].astype(bool).to_numpy() for c in clusters}
    by_cluster_gaps = {c: df.loc[df[cluster_col] == c, "gap"].astype(float).to_numpy() for c in clusters}

    lines.append("-" * 66)
    lines.append(name + "  (n=" + str(len(df)) + ", " + str(len(clusters)) + " " + cluster_col +
                 " clusters: " + ", ".join(str(c) for c in clusters) + ")")
    lines.append("-" * 66)

    naive = naive_bootstrap_ci(wins, gaps)
    lines.append("  Naive row-level bootstrap (treats all combos as independent -- optimistic):")
    lines.append(fmt_ci("win rate", naive["win_rate"], as_pct=True))
    lines.append(fmt_ci("mean gap", naive["mean_gap"]))

    if len(clusters) >= 2:
        clustered = year_clustered_bootstrap_ci(by_cluster_wins, by_cluster_gaps, clusters)
        lines.append("  " + cluster_col.capitalize() + "-clustered bootstrap (resamples whole " +
                     cluster_col + "s -- honest):")
        lines.append(fmt_ci("win rate", clustered["win_rate"], as_pct=True))
        lines.append(fmt_ci("mean gap", clustered["mean_gap"]))
        naive_width = naive["win_rate"][2] - naive["win_rate"][1]
        clus_width = clustered["win_rate"][2] - clustered["win_rate"][1]
        ratio = clus_width / naive_width if naive_width > 0 else float("nan")
        lines.append("  Clustered CI is " + format(ratio, ".1f") +
                     "x as wide as the naive CI on win rate.")
    else:
        lines.append("  (only 1 cluster in this subset -- clustered bootstrap skipped)")

    observed, p_value, _ = sign_flip_permutation_test(gaps)
    lines.append("  Sign-flip permutation test (H0: mean gap == 0):")
    lines.append("    observed mean gap " + format(observed, "+.4f") +
                 "   empirical two-sided p = " + format(p_value, ".4f") +
                 ("  (< 0.05 -- distinguishable from pure noise)" if p_value < 0.05
                  else "  (NOT distinguishable from pure noise at p<0.05)"))
    lines.append("")


def latest_csv(pattern):
    candidates = sorted(glob.glob(os.path.join(FOLDER, pattern)))
    return candidates[-1] if candidates else None


def bucket_label(n):
    if n <= 2:
        return str(int(n))
    return "3+"


def analyze_bucket_difference(df, lo_label, hi_label, cluster_col, lines):
    """Is the (hi-change minus lo-change) win-rate gap itself real, or
    could it arise from cluster-to-cluster noise alone? Cluster-bootstraps
    BOTH buckets from the SAME resampled cluster list each draw, exactly
    mirroring analyze_hand_difference's pairing logic in
    historical_uncertainty_analysis.py / live_uncertainty_analysis.py."""
    clusters = sorted(df[cluster_col].unique())
    by_cluster = {}
    for c in clusters:
        sub = df[df[cluster_col] == c]
        by_cluster[c] = {
            "lo": sub.loc[sub["bucket"] == lo_label, "win"].astype(bool).to_numpy(),
            "hi": sub.loc[sub["bucket"] == hi_label, "win"].astype(bool).to_numpy(),
        }
    n_clusters = len(clusters)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        lo = np.concatenate([by_cluster[clusters[i]]["lo"] for i in chosen if len(by_cluster[clusters[i]]["lo"])])
        hi = np.concatenate([by_cluster[clusters[i]]["hi"] for i in chosen if len(by_cluster[clusters[i]]["hi"])])
        diffs[b] = (hi.mean() if len(hi) else np.nan) - (lo.mean() if len(lo) else np.nan)

    lo_rate = df.loc[df["bucket"] == lo_label, "win"].mean()
    hi_rate = df.loc[df["bucket"] == hi_label, "win"].mean()
    observed_diff = hi_rate - lo_rate
    valid = diffs[~np.isnan(diffs)]
    lo_ci, hi_ci = np.percentile(valid, [2.5, 97.5])
    straddles_zero = lo_ci <= 0 <= hi_ci

    lines.append("-" * 66)
    lines.append("n_diff=" + hi_label + " win rate minus n_diff=" + lo_label +
                 " win rate (" + cluster_col + "-clustered, " + str(n_clusters) + " clusters)")
    lines.append("-" * 66)
    lines.append("  observed: " + hi_label + "-swap " + format(hi_rate, ".1%") +
                 " - " + lo_label + "-swap " + format(lo_rate, ".1%") +
                 " = " + format(observed_diff, "+.1%"))
    lines.append("  95% CI on the difference: [" + format(lo_ci, "+.1%") + ", " + format(hi_ci, "+.1%") + "]")
    lines.append("  " + ("CI straddles 0 -- bigger swaps are NOT clearly better than small ones "
                          "given this much clustering."
                          if straddles_zero else
                          "CI excludes 0 -- bigger swaps show a reliably higher win rate than small ones."))
    lines.append("")


def run_dataset(label, path, cluster_col, lines):
    df = pd.read_csv(path)
    df["win"] = df["win"].astype(str).str.lower().isin(["true", "1"])
    if "n_diff" not in df.columns:
        lines.append(label + ": " + os.path.basename(path) +
                     " has no n_diff column yet -- rerun the forward-test script to regenerate it.")
        lines.append("")
        return

    lines.append("=" * 66)
    lines.append(label + " -- LINEUP-CHANGE-MAGNITUDE BREAKDOWN")
    lines.append("Source: " + os.path.basename(path) + "  (n=" + str(len(df)) + ")")
    lines.append("=" * 66)
    lines.append("")

    trivial = df[df["n_diff"] == 0]
    if len(trivial):
        lines.append("n_diff=0 (optimal lineup == real lineup, no change made): " +
                     str(len(trivial)) + " combos, " +
                     format(len(trivial) / len(df), ".1%") + " of the data.")
        lines.append("These are trivial by construction (win is always False, gap is always 0 -- "
                     "there's nothing for the model to be tested on). Excluded below.")
        lines.append("")

    df = df[df["n_diff"] > 0].copy()
    df["bucket"] = df["n_diff"].apply(bucket_label)

    lines.append("Quick summary (no resampling yet):")
    lines.append("  {:<6s} {:>6s} {:>10s} {:>12s} {:>12s}".format(
        "bucket", "n", "win rate", "mean gap", "median gap"))
    for b in ["1", "2", "3+"]:
        sub = df[df["bucket"] == b]
        if len(sub) == 0:
            continue
        lines.append("  {:<6s} {:>6d} {:>9.1%} {:>+12.4f} {:>+12.4f}".format(
            b, len(sub), sub["win"].mean(), sub["gap"].mean(), sub["gap"].median()))
    lines.append("")

    for b in ["1", "2", "3+"]:
        sub = df[df["bucket"] == b]
        if len(sub) == 0:
            continue
        analyze_subset_clustered(b + "-swap combos", sub, cluster_col, lines)

    if (df["bucket"] == "1").sum() and (df["bucket"] == "3+").sum():
        analyze_bucket_difference(df, "1", "3+", cluster_col, lines)


def main():
    lines = []

    # Date-only pattern excludes suffixed experiment outputs
    # (e.g. *_handagnostic.csv, added 2026-07-09) -- always analyze production.
    hist_path = latest_csv("lineup_forward_test_detail_????-??-??.csv")
    if hist_path:
        run_dataset("HISTORICAL (2019-2025)", hist_path, "year", lines)
    else:
        lines.append("No lineup_forward_test_detail_*.csv found -- run lineup_forward_test.py first.")
        lines.append("")

    # Date-only pattern excludes the suffixed experiment outputs
    # live_forward_validation.py can now also write (e.g. *_histcareer.csv,
    # *_handagnostic.csv, added 2026-07-08) -- always analyze production.
    live_path = latest_csv("live_forward_validation_detail_????-??-??.csv")
    if live_path:
        run_dataset("LIVE (2026)", live_path, "cutoff", lines)
    else:
        lines.append("No live_forward_validation_detail_*.csv found -- skipping live-data half.")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    out_path = os.path.join(FOLDER, "lineup_change_magnitude_analysis_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print()
    print("Full report written to: " + out_path)


if __name__ == "__main__":
    main()
