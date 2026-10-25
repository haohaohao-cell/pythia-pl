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