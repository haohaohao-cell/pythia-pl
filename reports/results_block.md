**Out-of-sample: 3,054 matches, 9 seasons (2018-19 → 2026-27).** Lower log loss is better.

| Model | Log loss | Brier | RPS | Accuracy |
| --- | --- | --- | --- | --- |
| Closing line (de-vigged) | 0.9563 | 0.5663 | 0.1953 | 0.5494 |
| Blend (Elo + DC + GBDT) | 0.9774 | 0.5808 | 0.2022 | 0.5301 |
| Elo + softmax | 0.9795 | 0.5820 | 0.2026 | 0.5331 |
| Gradient boosting | 0.9919 | 0.5898 | 0.2055 | 0.5200 |
| Dixon-Coles | 0.9945 | 0.5924 | 0.2078 | 0.5131 |
| Base rates | 1.0688 | 0.6470 | 0.2342 | 0.4368 |

Gap to the closing line: **+0.0211** log loss (95% CI +0.0150 to +0.0273). Not statistically distinguishable from the market — by design.

| Club | Projected pts | Title | Top 4 | Relegated |
| --- | --- | --- | --- | --- |
| Arsenal | 82.99 | 57.8% | 99.9% | 0.0% |
| Man City | 80.90 | 40.6% | 99.6% | 0.0% |
| Brighton | 66.47 | 1.3% | 75.1% | 0.0% |
| Liverpool | 61.81 | 0.2% | 48.3% | 0.1% |
| Chelsea | 56.82 | 0.1% | 21.2% | 0.4% |
| Brentford | 54.71 | 0.0% | 13.0% | 0.6% |