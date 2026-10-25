# Premier League Forecaster

[![Python](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-111827.svg)](LICENSE)
[![Backtest](https://img.shields.io/badge/backtest-3%2C044%20matches-2563eb.svg)](#results)

[中文说明](README.md) · A reproducible, auditable probabilistic forecaster for Premier League results.

It does not advertise "62% accuracy". It answers one question: **with the data, metrics
and benchmarks all on the table, how far is a model from the bookmaker's closing line?**

**About 0.021 log loss — and that gap is statistically significant.** This repository
treats that as the headline finding instead of burying it.

---

## Findings

1. **The model does not beat the closing line.** Over 3,044 out-of-sample matches the
   blend scores `0.9767` log loss against `0.9558` for the de-vigged closing price —
   a gap of `+0.0209` (95% bootstrap CI `+0.0147` to `+0.0271`, excludes zero).
2. **The market prices home advantage better than the model.** 53.0% vs 55.0% accuracy —
   though accuracy is never the point, calibration is.
3. **The model is systematically under-confident about draws.** Predicted `P(draw)≈0.30`
   outcomes turn up about 23% of the time. That is the most stable bias in the project.
4. **Stacking is not estimable at this sample size.** One season of 380 matches cannot
   identify a 3-simplex: equal weights give `0.9770`, tuned weights `0.9767–0.9803`.
   The default is therefore a fixed "shrink 50% toward equal" rule, not the best cell
   of a grid search.

![Models vs the closing line](reports/leaderboard.png)

---

## Results

<!-- RESULTS:START -->

**Out-of-sample: 3,044 matches, 9 seasons (2018-19 → 2026-27).** Lower log loss is better.

| Model | Log loss | Brier | RPS | Accuracy |
| --- | --- | --- | --- | --- |
| Closing line (de-vigged) | 0.9558 | 0.5660 | 0.1952 | 0.5503 |
| Blend (Elo + DC + GBDT) | 0.9767 | 0.5803 | 0.2020 | 0.5302 |
| Elo + softmax | 0.9788 | 0.5815 | 0.2024 | 0.5342 |
| Gradient boosting | 0.9911 | 0.5893 | 0.2054 | 0.5204 |
| Dixon-Coles | 0.9941 | 0.5921 | 0.2077 | 0.5138 |
| Base rates | 1.0686 | 0.6468 | 0.2342 | 0.4373 |

Gap to the closing line: **+0.0209** log loss (95% CI +0.0147 to +0.0271). Not statistically distinguishable from the market — by design.

| Club | Projected pts | Title | Top 4 | Relegated |
| --- | --- | --- | --- | --- |
| Man City | 83.47 | 55.5% | 99.9% | 0.0% |
| Arsenal | 81.68 | 41.8% | 99.7% | 0.0% |
| Brighton | 68.91 | 2.4% | 82.6% | 0.0% |
| Chelsea | 60.02 | 0.1% | 35.5% | 0.1% |
| Liverpool | 58.77 | 0.1% | 29.1% | 0.1% |
| Man United | 56.67 | 0.0% | 19.4% | 0.4% |

<!-- RESULTS:END -->

![Cumulative log loss](reports/cumulative_logloss.png)

---

## Next round

Refreshed weekly by GitHub Actions. Model probabilities use only pre-match
information; the price column is the de-vigged Pinnacle opening line.

<!-- FIXTURES:START -->

**Round of 2026-10-23: 10 fixtures, 10 of them already kicked off and waiting for a result.** Prices are de-vigged Pinnacle openers (`PPH+PPD+PPA`).

| Kick-off | Fixture | Model H / D / A | Price H / D / A | Pick |
| --- | --- | --- | --- | --- |
| 10-23 20:00 | Ipswich v Nott'm Forest | 0.33 / 0.26 / 0.41 | 0.31 / 0.28 / 0.41 | A 41% |
| 10-24 12:30 | Aston Villa v Man City | 0.21 / 0.26 / 0.53 | 0.20 / 0.21 / 0.59 | A 53% |
| 10-24 15:00 | Arsenal v Everton | 0.65 / 0.22 / 0.13 | 0.66 / 0.21 / 0.13 | H 65% |
| 10-24 15:00 | Coventry v Fulham | 0.36 / 0.28 / 0.36 | 0.30 / 0.28 / 0.42 | H 36% |
| 10-24 17:30 | Chelsea v Tottenham | 0.54 / 0.25 / 0.20 | 0.56 / 0.23 / 0.21 | H 54% |
| 10-25 14:00 | Crystal Palace v Newcastle | 0.38 / 0.27 / 0.35 | 0.31 / 0.27 / 0.42 | H 38% |
| 10-25 14:00 | Hull v Brentford | 0.38 / 0.27 / 0.34 | 0.20 / 0.25 / 0.56 | H 38% |
| 10-25 14:00 | Liverpool v Brighton | 0.45 / 0.24 / 0.30 | 0.45 / 0.25 / 0.30 | H 45% |
| 10-25 14:00 | Man United v Bournemouth | 0.49 / 0.26 / 0.25 | 0.44 / 0.26 / 0.30 | H 49% |
| 10-25 16:30 | Sunderland v Leeds | 0.38 / 0.26 / 0.36 | 0.31 / 0.30 / 0.39 | H 38% |


![Next round](reports/next_round.png)

<!-- FIXTURES:END -->

---

## Calibration

![Calibration](reports/calibration.png)

Home-win forecasts track the diagonal closely; draws are under-predicted in the upper
tail. That is the usual cost of treating "draw" as an independent class.

---

## Why this is not another accuracy-chasing repo

| Common practice | Here |
| --- | --- |
| Random K-fold CV | Strict walk-forward: a season is forecast only from data before its opening day |
| Accuracy as the headline | Log loss first, Brier and RPS second, accuracy for reference only |
| No benchmark | Base rates, Elo, each single model, and the **de-vigged closing line** |
| Only wins reported | All 9 test seasons, including 2020-21 and 2025-26 where the gap widens |
| Irreproducible claims | `plpred fetch && plpred report` rebuilds every figure |
| No leakage guard | `tests/test_leakage.py` pins features by truncation *and* by mutating future results |

The four findings are credible precisely because the test suite exists to stop the
author from fooling himself.

---

## Method

**Data** — [football-data.co.uk](https://www.football-data.co.uk/englandm.php), 2014-15 to
2026-27: 4,630 matches with scores, shots, shots on target and multiple bookmakers'
opening and closing prices. The closing-price source is resolved per season — Pinnacle
`PS*` up to 2024-25, renamed to `PP*` from 2026-27, and Bet365 for 2025-26 where
Pinnacle coverage falls to 55%. The source used is recorded on every row rather than
pretending the whole sample is homogeneous.

**Features** (all pre-match): exponentially weighted goals/points/shots-on-target over
the last 5 matches, rest days, match number within the season, a newly-promoted flag,
and Elo difference (60-point home advantage, one-third regression to 1500 each summer,
goal-margin weighting). Every rolling window is `shift(1)`-ed.

**Models** — `base_rates` (the floor), `elo_logit` (one feature, heavy regularisation,
and the strongest single model), `dixon_coles` (bivariate Poisson with 540-day time decay
and the low-score correction), `gbdt` (HistGradientBoosting, sample-weighted by the same
decay), and `blend` (linear pool of the last three).

**Evaluation** — nine test seasons, refitted each season; paired bootstrap for the
log-loss difference; GBDT hyper-parameters selected on 2017-18, a season that sits
*before* the test window, not inside it.

---

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

plpred fetch          # 13 seasons of results + the upcoming fixture list
plpred build          # feature table -> data/processed/features.parquet
plpred backtest       # walk-forward, ~10 s, prints the leaderboard
plpred report         # figures + docs/index.html + injects the Results block above

plpred fixtures --save              # next round: model vs opening price
plpred predict Arsenal Liverpool
plpred simulate --sims 20000
plpred value --min-edge 0.05
```

---

## This season

![Simulated table](reports/simulated_table.png)

Remaining fixtures are recovered as "full double round-robin minus what has been played",
then simulated 20,000 times from the Dixon-Coles scoreline distribution fitted on the
last two seasons (so the parameter table contains exactly the 20 clubs in the league).
Points already banked are real, not simulated.

---

## Repository layout

```text
src/plpred/
├─ ingest/       # results + fixture feed, column-name-adaptive parsing
├─ features/     # Elo and lagged form
├─ models/       # base_rates, elo_logit, dixon_coles, gbdt, blend
├─ eval/         # walk-forward, metrics, calibration
├─ forecast.py   # next-round forecasts and a running hit record
├─ simulate.py   # Monte Carlo of the remaining schedule
├─ plots.py      # every figure
├─ report.py     # tables, README injection, single-file HTML dashboard
└─ cli.py
tests/           # 39 tests, mostly about leakage
reports/         # generated figures and CSVs (committed for convenience)
docs/index.html  # self-contained dashboard, ready for GitHub Pages
```

---

## Appendix B — value betting probe (research only)

When the model disagrees with the opening price by a wide margin, is it right? Flat
1-unit stakes against the opening line:

| Minimum edge | Bets | Share | Flat ROI |
| --- | --- | --- | --- |
| 2% | 2,614 | 92.2% | −5.2% |
| 5% | 2,280 | 80.4% | −5.6% |
| 8% | 1,994 | 70.3% | −3.2% |
| 10% | 1,795 | 63.3% | −4.0% |
| 15% | 1,372 | 48.4% | −4.8% |
| 20% | 1,069 | 37.7% | −6.6% |

Every threshold loses money, and the median claimed edge is 14% — which is itself the
answer: that is not value, it is the model being more confident than the market in the
tails. Mean CLV is ~0, so none of these bets beat the closing line.

**The model has no exploitable edge.** The section exists to show the falsification, not
because it makes money.

---

## Limitations

- No xG features: football-data only ships `HxG/AxG` from 2026-27, so shots on target are
  a proxy. Adding Understat is the obvious next step.
- No injuries, lineups or European fixture congestion — part of the gap to the market.
- Promoted clubs start from a league-average prior; the first five matchdays are the
  weakest point of the season.
- No post-hoc calibration (temperature/isotonic) yet; that is the most likely small win.

## Disclaimer

Research and engineering demonstration only. **Not betting advice**, no profit claims.
Data remains the property of its sources.

## License

[MIT](LICENSE)
