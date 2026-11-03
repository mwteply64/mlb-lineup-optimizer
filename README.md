# MLB Lineup Optimizer

Projects every MLB hitter's wOBA against left- and right-handed pitching, then
asks a simple question: **if a team had played the nine hitters the model likes
best, would it have hit better than the nine it actually played?**

## About this project

Built by Max Teply (Florida State, sport management with statistics and
economics minors), spring and summer 2026. This was my first large analytics
project; [mlb-game-simulator](https://github.com/mwteply64/mlb-game-simulator)
grew out of it.

**How it was built.** The code was written with Claude Code, an AI coding
assistant. My part was the analysis: designing the test, choosing what counts
as a fair comparison, reading the results, and deciding which changes shipped.
`PROJECT_LOG.md` is the running log of those decisions, including the ones that
turned out wrong.

---

## The model

Each hitter's projection vs a given pitcher hand blends four sources, each
shrunk toward league average in proportion to its sample size:

| component | what it is |
|---|---|
| current-season splits | this season vs that hand |
| career splits | prior seasons vs that hand, recency-weighted |
| season total | this season, both hands |
| recent form | last 14 days |

Each component mixes actual wOBA with expected wOBA (xwOBA) from Statcast
contact quality. A platoon adjustment then scales the result by the hitter's
own career platoon ratio, backed by a league-wide prior for hitters with little
evidence against one hand.

Every weight was tuned with **walk-forward cross-validation**: train on earlier
seasons, test on a later one never seen during tuning.

## How it is tested

For each team, season, pitcher hand, and cutoff date (June 1, July 1, Aug 1):

- **Real lineup:** the nine players the team actually used most against that
  hand before the cutoff.
- **Model lineup:** the team's nine best by projection, using only data
  available before the cutoff.
- **Grade:** both lineups' actual wOBA *after* the cutoff.

A "win" means the model's nine out-hit the manager's nine. No future
information enters any projection.

## Results

### Historical, 2019 and 2021–2025 (1,047 team-hand-cutoff comparisons)

| split | win rate | mean wOBA gap | year-clustered 95% CI (win rate) |
|---|---|---|---|
| overall | 63.4% | +0.0049 | 60.5% – 65.9% |
| vs RHP | 68.2% | +0.0055 | 66.0% – 70.3% |
| vs LHP | 58.3% | +0.0043 | 54.5% – 62.0% |

A sign-flip permutation test puts p below 0.0001 for every split. The edge
holds even when the model changes only one player.

**What drives it** (ablation: remove one component, re-run the full test):
career splits are the biggest piece (−11.1 points of win rate without them),
then xwOBA blending (−4.7), current splits (−3.2), recent form (−2.3) and the
platoon adjustment (−1.5).

### Live 2026 season (620 comparisons, 11 cutoffs from April 15 to September 2)

The real test: a season that didn't exist when the model was tuned, graded
through the final regular-season game on September 27.

| split | win rate | mean wOBA gap | 95% CI (win rate) | permutation p |
|---|---|---|---|---|
| overall | 57.3% | +0.0027 | 53.4% – 61.1% | < 0.0001 |
| vs RHP | 64.9% | +0.0042 | 59.8% – 69.8% | < 0.0001 |
| vs LHP | 48.6% | +0.0010 | 42.8% – 54.5% | 0.29 |

**Against right-handed pitching the edge carried over almost intact**
(64.9% live against 68.2% historically). **Against left-handers it did not**:
48.6% is a coin flip. That matches what the historical diagnostics predicted.
Managers platoon mostly against lefties, so that's where their real lineup is
hardest to beat, and the hitters the model wants to promote have the least
direct evidence against left-handed pitching (about 27 plate appearances on
average).

The edge also grew as the season went on: about 55% at the spring cutoffs,
about 60% from July onward, once there was more 2026 data to work with.

*A caveat on the intervals:* every cutoff is graded through the end of the
season, so later cutoffs' grading windows overlap earlier ones. The intervals
above treat comparisons as independent, which makes them the wider (more
cautious) of the two bootstraps the analysis script reports.

## What I learned

- **The benchmark matters as much as the model.** The first live results looked
  far worse than history. The cause was an unfair comparison: the historical
  test compared the model with a lineup that ignored pitcher hand, while the
  live test compared it with managers' actual platoon decisions, which are much
  harder to beat. Measuring both the same way closed most of the gap.
- **Two good ideas can fix the same problem.** Two changes for left-handed
  pitching each passed cross-validation alone, but together they scored *worse*,
  because both were correcting the same thin-sample hitters. I shipped one.
- **Tuning on the right target.** Earlier versions were tuned on rank
  correlation, which isn't the decision being made. Re-tuning on the actual
  win-rate metric is what found the left-handed-pitching improvement.
- **A "verbatim" port wasn't.** The live script's platoon formula had never
  matched the validated one: it was about twice as strong, with no sample-size
  protection. It was found by testing the two side by side, case by case.

## Files

| file | role |
|---|---|
| `backtest.py` | scoring engine: every weight and the projection formula |
| `test.py` | live optimizer for the current season |
| `lineup_forward_test.py` | historical test, 2019–2025 |
| `live_forward_validation.py` | the same test on the current season (`--as-of DATE` pins the end date) |
| `historical_uncertainty_analysis.py`, `live_uncertainty_analysis.py` | bootstrap CIs and permutation tests |
| `lineup_ablation_test.py` | removes one component at a time |
| `tuner_*.py` | walk-forward cross-validation searches |
| `download_*.py`, `update_live_data.py` | rebuild the data from Statcast |
| `results/` | output from the runs quoted above |

**Data is not included.** The `download_*.py` scripts rebuild it from Baseball
Savant through [pybaseball](https://github.com/jldbc/pybaseball).

```bash
pip install -r requirements.txt
python download_v3_expansion.py
python lineup_forward_test.py
```
