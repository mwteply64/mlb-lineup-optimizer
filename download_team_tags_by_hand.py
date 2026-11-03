#!/usr/bin/env python3
"""
Hand-Specific Team-Tagging Download - Lineup Optimization Project
===================================================================
Companion to download_team_tags.py (2026-07-09). That script produced
totals_before_<cutoff>_<year>_with_team.csv -- each player's TOTAL
pre-cutoff PA per team, no pitcher-hand split. This one produces the
hand-split equivalents:

    before_<cutoff>_<year>_vs_left_with_team.csv
    before_<cutoff>_<year>_vs_right_with_team.csv

(6 test years x 3 cutoffs x 2 hands = 36 files), columns:
    last_name, first_name, player_id, team, pa, woba

WHY THIS EXISTS
---------------
lineup_forward_test.py's "real lineup" benchmark was top-9 by TOTAL
pre-cutoff PA -- hand-agnostic, the same 9 players graded vs both hands.
live_forward_validation.py's benchmark is top-9 by PA vs the SPECIFIC
pitcher hand, which encodes managers' actual platoon deployment and is
a much stronger opponent (confirmed 2026-07-08: switching the live
benchmark to hand-agnostic moved live results from 41% to 46% with the
mean gap flipping positive). To measure the exact same thing in both
past and present, the historical test needs hand-specific real lineups
too -- which requires knowing each player's pre-cutoff PA vs each hand
PER TEAM. These files supply that.

Reuses download_team_tags.py's own fetch_range/aggregate_with_team/save
helpers; the only addition is the p_throws filter before aggregating
(same pattern as download_v3_expansion.py's before_* splits). All 18
pre-cutoff date ranges are identical to the totals_before pulls, so
pybaseball's cache should serve every one without a network fetch.

Idempotent: a window is skipped entirely (no fetch) when both of its
hand files already exist. Safe to re-run after a partial failure.

USAGE:
    python download_team_tags_by_hand.py
    (or double-click run_team_tags_by_hand.bat)
"""

import os
import sys
import time
import warnings

from download_v3_expansion import (
    SAVE_DIR, TEST_YEARS, CUTOFFS, HANDEDNESS, build_name_lookup,
)
from download_team_tags import fetch_range, aggregate_with_team, save


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    total_expected = len(TEST_YEARS) * len(CUTOFFS) * len(HANDEDNESS)
    files_saved = [0]

    print("Building " + str(total_expected) + " hand-specific team-tagged pre-cutoff files.")
    print("Saving to: " + SAVE_DIR)
    print("Already-existing files are skipped (idempotent).")
    print()

    name_lookup = build_name_lookup()
    print()

    for year in TEST_YEARS:
        print("  [" + str(year) + "]")
        for cutoff_label, before_fn, _after_fn in CUTOFFS:
            out_paths = {
                hand_label: os.path.join(
                    SAVE_DIR, "before_" + cutoff_label + "_" + str(year) + "_" + hand_label + "_with_team.csv")
                for hand_label, _ in HANDEDNESS
            }
            if all(os.path.exists(p) for p in out_paths.values()):
                print("    Skipping " + cutoff_label + " -- both hand files already exist.")
                files_saved[0] += 0
                continue

            b_start, b_end = before_fn(year)
            raw = fetch_range(b_start, b_end, "before_" + cutoff_label + "_" + str(year) + " (hand+team tagged)")
            if raw is None:
                print("    SKIPPING " + cutoff_label + " " + str(year) + " -- no data.")
                continue
            for hand_label, hand_code in HANDEDNESS:
                if os.path.exists(out_paths[hand_label]):
                    continue
                hand_data = raw[raw["p_throws"] == hand_code]
                result = aggregate_with_team(hand_data, name_lookup, year)
                save(result, os.path.basename(out_paths[hand_label]), files_saved)
            time.sleep(0.5)
        print()

    print("Done. Saved " + str(files_saved[0]) + " new files (target set: " + str(total_expected) + ").")
    print("Re-run to retry any that failed (cached ranges reload instantly).")


if __name__ == "__main__":
    main()
