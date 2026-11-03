**Out-of-sample: 3,064 matches, 9 seasons (2018-19 → 2026-27).** Lower log loss is better.

| Model | Log loss | Brier | RPS | Accuracy |
| --- | --- | --- | --- | --- |
| Closing line (de-vigged) | 0.9566 | 0.5665 | 0.1954 | 0.5493 |
| Blend (Elo + DC + GBDT) | 0.9777 | 0.5811 | 0.2023 | 0.5300 |
| Elo + softmax | 0.9798 | 0.5822 | 0.2028 | 0.5333 |
| Gradient boosting | 0.9925 | 0.5902 | 0.2058 | 0.5189 |
| Dixon-Coles | 0.9949 | 0.5927 | 0.2080 | 0.5134 |
| Base rates | 1.0687 | 0.6469 | 0.2342 | 0.4370 |

Gap to the closing line: **+0.0211** log loss (95% CI +0.0149 to +0.0273). Not statistically distinguishable from the market — by design.

| Club | Projected pts | Title | Top 4 | Relegated |
| --- | --- | --- | --- | --- |
| Arsenal | 84.70 | 74.0% | 99.9% | 0.0% |
| Man City | 77.97 | 22.5% | 98.8% | 0.0% |
| Brighton | 69.80 | 3.2% | 86.4% | 0.0% |
| Liverpool | 60.42 | 0.1% | 37.3% | 0.1% |
| Chelsea | 59.21 | 0.1% | 29.9% | 0.1% |
| Brentford | 56.20 | 0.0% | 16.1% | 0.4% |