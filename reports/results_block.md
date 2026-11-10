**Out-of-sample: 3,074 matches, 9 seasons (2018-19 → 2026-27).** Lower log loss is better.

| Model | Log loss | Brier | RPS | Accuracy |
| --- | --- | --- | --- | --- |
| Closing line (de-vigged) | 0.9565 | 0.5664 | 0.1954 | 0.5491 |
| Blend (Elo + DC + GBDT) | 0.9775 | 0.5809 | 0.2023 | 0.5299 |
| Elo + softmax | 0.9796 | 0.5821 | 0.2027 | 0.5335 |
| Gradient boosting | 0.9920 | 0.5898 | 0.2056 | 0.5208 |
| Dixon-Coles | 0.9949 | 0.5927 | 0.2080 | 0.5133 |
| Base rates | 1.0685 | 0.6467 | 0.2342 | 0.4375 |

Gap to the closing line: **+0.0210** log loss (95% CI +0.0145 to +0.0268). Not statistically distinguishable from the market — by design.

| Club | Projected pts | Title | Top 4 | Relegated |
| --- | --- | --- | --- | --- |
| Arsenal | 85.82 | 84.2% | 100.0% | 0.0% |
| Man City | 74.79 | 10.6% | 96.4% | 0.0% |
| Brighton | 71.67 | 4.9% | 90.9% | 0.0% |
| Liverpool | 62.52 | 0.2% | 48.4% | 0.0% |
| Chelsea | 57.17 | 0.1% | 18.6% | 0.2% |
| Bournemouth | 55.94 | 0.0% | 13.4% | 0.3% |