#!/usr/bin/env python3
"""
Batter-Side Download - Lineup Optimization Project
=====================================================
Builds "historical testing/batter_sides.csv": which side each batter
actually hits from against each pitcher hand, derived from Statcast's
per-pitch `stand` column across every cached full-season pull
(2016-2025) plus the 2026 season-to-date.

WHY THIS EXISTS (2026-07-10, LHP diagnostics follow-up)
--------------------------------------------------------
The platoon prior in backtest.py/test.py is built purely from each
player's OWN career splits -- deliberately, because some hitters have
reverse splits. But the LHP swap-level diagnostic showed the model's
marginal vs-LHP picks average only ~27 pre-cutoff PA vs LHP, i.e. the
platoon ratio is being estimated from very thin evidence exactly where
the decision matters most. The planned fix is an EVIDENCE-WEIGHTED
prior: blend the player's own career platoon ratio with a league-mean
ratio for his batter side, weighted by how much career data he has, so
established (including reverse-split) hitters keep their own ratio and
thin-sample hitters get regressed toward the typical same-side penalty.
That needs each batter's side vs each pitcher hand -- this file.

WHY `stand` AND NOT A ROSTER/REGISTER LOOKUP
---------------------------------------------
`stand` is the side the batter ACTUALLY stood on for that pitch, so a
switch hitter is handled correctly by matchup (bats R vs LHP, L vs RHP)
with no special-casing, and coverage is exact for every player who ever
appears in this project's data (the data itself is the source). The
Chadwick register carries no bats column, and roster_resource.xlsx only
covers current rosters.

OUTPUT COLUMNS (one row per batter):
    player_id, side_vs_L, side_vs_R, pa_vs_L, pa_vs_R
side_vs_<H> is the majority stand across all PA-ending events vs that
pitcher hand, pooled over every season processed ("L"/"R", blank if the
player never faced that hand). pa_vs_<H> is the pooled PA count behind
it, so downstream code can ignore thin/ambiguous rows if needed.

All source date ranges are already in pybaseball's local cache from
download_v3_expansion.py / update_live_data.py runs, so this is a
(long) local re-read, not a network pull. Safe to re-run; overwrites
the output.

USAGE:
    python download_batter_sides.py
"""

import os
import sys
import warnings
from datetime import date, timedelta

import pandas as pd

import download_v3_expansion as dv3

OUTPUT = os.path.join(dv3.SAVE_DIR, "batter_sides.csv")
SEASON_2026_START = "2026-03-25"  # mirrors update_live_data.py


def main():
    warnings.filterwarnings("ignore")

    # (batter, p_throws) -> {"L": n, "R": n} stand counts over PA events
    counts = {}

    ranges = [(y, dv3.SEASON_DATES[y]["start"], dv3.SEASON_DATES[y]["end"])
              for y in sorted(dv3.SEASON_DATES)]
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    ranges.append((2026, SEASON_2026_START, yesterday))

    for year, start, end in ranges:
        raw = dv3.fetch_range(start, end, "batter sides " + str(year))
        if raw is None:
            print("  SKIPPING " + str(year) + " -- no data.")
            continue
        needed = ["batter", "events", "stand", "p_throws"]
        missing = [c for c in needed if c not in raw.columns]
        if missing:
            print("  SKIPPING " + str(year) + " -- missing columns: " + ", ".join(missing))
            continue
        pa = raw[raw["events"].notna()][needed]
        grouped = pa.groupby(["batter", "p_throws", "stand"]).size()
        for (batter, p_throws, stand), n in grouped.items():
            key = (str(int(batter)), str(p_throws))
            counts.setdefault(key, {"L": 0, "R": 0})
            if str(stand) in ("L", "R"):
                counts[key][str(stand)] += int(n)
        del raw, pa, grouped

    players = sorted({pid for pid, _ in counts})
    rows = []
    for pid in players:
        row = {"player_id": pid}
        for hand in ("L", "R"):
            c = counts.get((pid, hand), {"L": 0, "R": 0})
            total = c["L"] + c["R"]
            row["pa_vs_" + hand] = total
            if total == 0:
                row["side_vs_" + hand] = ""
            else:
                row["side_vs_" + hand] = "L" if c["L"] >= c["R"] else "R"
        rows.append(row)

    df = pd.DataFrame(rows, columns=["player_id", "side_vs_L", "side_vs_R",
                                     "pa_vs_L", "pa_vs_R"])
    df.to_csv(OUTPUT, index=False)
    print()
    print("Wrote " + OUTPUT + "  (" + str(len(df)) + " batters)")
    both = ((df["side_vs_L"] != "") & (df["side_vs_R"] != "")).sum()
    switch = ((df["side_vs_L"] == "R") & (df["side_vs_R"] == "L")).sum()
    print("Batters with sides vs both hands: " + str(both))
    print("Of those, switch-hitter pattern (R vs LHP, L vs RHP): " + str(switch))


if __name__ == "__main__":
    main()
