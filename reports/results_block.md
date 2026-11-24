**Out-of-sample: 3,084 matches, 9 seasons (2018-19 → 2026-27).** Lower log loss is better.

| Model | Log loss | Brier | RPS | Accuracy |
| --- | --- | --- | --- | --- |
| Closing line (de-vigged) | 0.9572 | 0.5669 | 0.1956 | 0.5490 |
| Blend (Elo + DC + GBDT) | 0.9777 | 0.5810 | 0.2023 | 0.5295 |
| Elo + softmax | 0.9798 | 0.5822 | 0.2027 | 0.5334 |
| Gradient boosting | 0.9918 | 0.5896 | 0.2055 | 0.5198 |
| Dixon-Coles | 0.9951 | 0.5928 | 0.2080 | 0.5133 |
| Base rates | 1.0683 | 0.6466 | 0.2341 | 0.4381 |

Gap to the closing line: **+0.0206** log loss (95% CI +0.0145 to +0.0272). Not statistically distinguishable from the market — by design.

| Club | Projected pts | Title | Top 4 | Relegated |
| --- | --- | --- | --- | --- |
| Arsenal | 84.13 | 85.9% | 99.9% | 0.0% |
| Man City | 72.70 | 9.3% | 92.5% | 0.0% |
| Brighton | 69.30 | 3.6% | 83.0% | 0.0% |
| Liverpool | 65.01 | 1.0% | 61.6% | 0.0% |
| Brentford | 58.20 | 0.1% | 19.5% | 0.1% |
| Bournemouth | 57.31 | 0.1% | 16.0% | 0.2% |