#!/usr/bin/env python3
"""
Hand-Specific Team-Tagging Download - Lineup Optimization Project
===================================================================
Companion to download_team_tags.py (added 2026-07-08). That script
produced totals_before_<cutoff>_<year>_with_team.csv -- each player's
pre-cutoff TEAM and total PA -- which lineup_forward_test.py used to
rank the "real lineup" benchmark by TOTAL pre-cutoff PA.

But the live validation (live_forward_validation.py) ranks its real
lineup by PA vs the SPECIFIC pitcher hand -- the manager's actual
platoon deployment -- which is a strictly stronger, more meaningful
benchmark: platoon lineups are the whole point of this project
(user direction 2026-07-08). To bring the historical test up to the
same standard, this script produces the hand-split equivalent:

    before_<cutoff>_<year>_vs_left_with_team.csv
    before_<cutoff>_<year>_vs_right_with_team.csv

(6 test years x 3 cutoffs x 2 hands = 36 files), columns:
    last_name, first_name, player_id, team, pa, woba

Same date ranges as the existing before_* files, same per-pitch team
derivation as download_team_tags.py (home_team/away_team +
inning_topbot), just additionally filtered by p_throws before
aggregating. One Statcast fetch per window, both hands sliced from it.

Idempotent: windows whose two output files both already exist are
skipped without a fetch. All 18 pre-cutoff ranges should already be in
pybaseball's local cache from download_v3_expansion.py runs, so a full
run is mostly cache reads.

USAGE:
    python download_hand_team_tags.py
"""

import os
import sys
import time
import warnings

from download_v3_expansion import (
    SAVE_DIR, TEST_YEARS, CUTOFFS, HANDEDNESS, build_name_lookup,
)
from download_team_tags import fetch_range, aggregate_with_team, save

try:
    from pybaseball import cache
    cache.enable()
except ImportError:
    print("ERROR: pybaseball not installed.")
    print("Run: pip install pybaseball pandas")
    sys.exit(1)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    total_expected = len(TEST_YEARS) * len(CUTOFFS) * len(HANDEDNESS)
    files_saved = [0]

    print("Building up to " + str(total_expected) + " hand-specific team-tag files in:")
    print("  " + SAVE_DIR)
    print()

    name_lookup = build_name_lookup()
    print()

    for year in TEST_YEARS:
        print("  [" + str(year) + "]")
        for cutoff_label, before_fn, _after_fn in CUTOFFS:
            out_names = {
                hand_label: "before_" + cutoff_label + "_" + str(year) + "_" + hand_label + "_with_team.csv"
                for hand_label, _ in HANDEDNESS
            }
            if all(os.path.exists(os.path.join(SAVE_DIR, n)) for n in out_names.values()):
                print("    [skip] " + cutoff_label + " -- both hand files already exist.")
                continue

            b_start, b_end = before_fn(year)
            raw = fetch_range(b_start, b_end, "before_" + cutoff_label + "_" + str(year) + " (hand+team tagged)")
            if raw is None:
                print("    SKIPPING " + cutoff_label + "_" + str(year) + " -- no data.")
                continue
            for hand_label, hand_code in HANDEDNESS:
                out_name = out_names[hand_label]
                if os.path.exists(os.path.join(SAVE_DIR, out_name)):
                    continue
                hand_data = raw[raw["p_throws"] == hand_code]
                result = aggregate_with_team(hand_data, name_lookup, year)
                save(result, out_name, files_saved)
            time.sleep(0.5)
        print()

    print("Done. Saved " + str(files_saved[0]) + " new files (of " + str(total_expected) + " possible).")
    print("Re-run to retry any that failed (cached ranges reload instantly).")


if __name__ == "__main__":
    main()
