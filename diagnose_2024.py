#!/usr/bin/env python3
"""
Diagnostic: why does 2024 Statcast data have more PA than MLB Stats API?
Run this locally — it reads the pybaseball cache and checks game_type distribution.
"""
import pandas as pd
import sys

try:
    from pybaseball import statcast, cache
    cache.enable()
except ImportError:
    print("pybaseball not installed"); sys.exit(1)

print("Downloading/loading 2024 Statcast (cached — should be fast)...")
raw = statcast(start_dt="2024-03-20", end_dt="2024-09-29")
print(f"Total rows: {len(raw):,}")

# 1. Check game_type distribution
print("\n=== game_type distribution ===")
print(raw["game_type"].value_counts().to_string())

# 2. Check date range — any rows outside 2024-03-20 to 2024-09-29?
raw["game_date"] = pd.to_datetime(raw["game_date"])
out_of_range = raw[(raw["game_date"] < "2024-03-20") | (raw["game_date"] > "2024-09-29")]
print(f"\nRows outside date range: {len(out_of_range)}")
if not out_of_range.empty:
    print(out_of_range["game_date"].value_counts().sort_index().head(10))

# 3. Check for duplicate pitches
before = len(raw)
deduped = raw.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
print(f"\nDuplicate rows (same game_pk+at_bat+pitch): {before - len(deduped)}")

# 4. For Aaron Judge specifically — how many PA events does Statcast have vs API?
judge_pa = raw[(raw["batter"] == 592450) & raw["events"].notna()]
print(f"\nJudge PA events in Statcast: {len(judge_pa)}")
print(f"Judge PA events by game_type:")
print(judge_pa["game_type"].value_counts().to_string())

# 5. Non-R game types for Judge
non_regular = judge_pa[judge_pa["game_type"] != "R"]
if not non_regular.empty:
    print(f"\nJudge non-regular-season events:")
    print(non_regular[["game_date","game_type","events","p_throws"]].to_string())
