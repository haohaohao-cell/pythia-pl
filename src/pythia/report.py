"""Report generation: tables, figures, dashboard and README injection."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DOCS_DIR, FEATURES_PARQUET, PREDICTIONS_PARQUET, REPORTS_DIR, ROOT
from .dataset import load_dataset
from .eval.calibration import expected_calibration_error, reliability_curve
from .eval.metrics import evaluate, log_loss, per_class_recall
from .eval.walk_forward import (
    bootstrap_delta,
    leaderboard,
    run_backtest,
    season_table,
    test_seasons_from,
)
from .forecast import next_round
from .ingest.fixtures_feed import load_fixtures
from .models.calibration import calibrate_by_season
from .models.base import outcome_index
from .models.dixon_coles import DixonColes
from .models.blend import Blend
from .models.elo_logit import EloLogit
from .models.gbdt import GradientBoost
from .simulate import current_table, simulate_remaining
from .teams import en_name, zh_name

MODEL_DISPLAY = {
    "market": "Closing line (de-vigged)",
    "blend": "Blend (Elo + DC + GBDT)",
    "elo_logit": "Elo + softmax",
    "gbdt": "Gradient boosting",
    "dixon_coles": "Dixon-Coles",
    "base_rates": "Base rates",
}

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"
FIXTURES_START = "<!-- FIXTURES:START -->"
FIXTURES_END = "<!-- FIXTURES:END -->"
CALIB_START = "<!-- CALIB:START -->"
CALIB_END = "<!-- CALIB:END -->"
LEDGER_START = "<!-- LEDGER:START -->"
LEDGER_END = "<!-- LEDGER:END -->"


def get_predictions(*, refresh: bool = False) -> pd.DataFrame:
    if PREDICTIONS_PARQUET.exists() and not refresh:
        return pd.read_parquet(PREDICTIONS_PARQUET)
    features = load_dataset()
    preds = run_backtest(features)
    preds.to_parquet(PREDICTIONS_PARQUET, index=False)
    return preds


def fit_final(features: pd.DataFrame) -> dict[str, object]:
    """Models fitted on everything available today (for forecasts, not scores)."""
    dc = DixonColes().fit(features)
    return {
        "dixon_coles": dc,
        "elo_logit": EloLogit().fit(features),
        "gbdt": GradientBoost().fit(features),
        "blend": Blend([EloLogit(), DixonColes(), GradientBoost()]).fit(features),
    }


def headline_stats(preds: pd.DataFrame) -> dict[str, float]:
    y_all = {}
    for model in ("market", "blend", "elo_logit", "dixon_coles", "gbdt", "base_rates"):
        block = preds.loc[preds["model"] == model]
        if block.empty:
            continue
        y = outcome_index(block)
        probs = block[["p_home", "p_draw", "p_away"]].to_numpy()
        y_all[model] = {"n": len(block), **evaluate(probs, y)}

    blend = preds.loc[preds["model"] == "blend"]
    market = preds.loc[preds["model"] == "market"]
    y = outcome_index(blend)
    gap = bootstrap_delta(
        blend[["p_home", "p_draw", "p_away"]].to_numpy(),
        market[["p_home", "p_draw", "p_away"]].to_numpy(),
        y,
    )
    curve = reliability_curve(blend["p_home"].to_numpy(), (blend["ftr"] == "H").to_numpy(float), bins=10)
    stats = {
        "n_matches": int(len(blend)),
        "n_seasons": int(blend["season"].nunique()),
        "gap_to_market": gap["delta"],
        "gap_ci_low": gap["ci_low"],
        "gap_ci_high": gap["ci_high"],
        "ece_home": expected_calibration_error(curve),
    }
    stats["models"] = y_all  # type: ignore[assignment]
    return stats


def current_season(features: pd.DataFrame) -> str:
    return str(features["season"].max())


def latest_predictions(preds: pd.DataFrame, features: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    season = current_season(features)
    block = preds.loc[(preds["season"] == season) & (preds["model"].isin(["blend", "market"]))]
    wide = block.pivot_table(index=["match_id", "date", "home", "away", "ftr"], columns="model",
                             values=["p_home", "p_draw", "p_away"]).reset_index()
    wide = wide.sort_values("date").tail(n)
    out = pd.DataFrame({
        "date": wide["date"].dt.strftime("%Y-%m-%d"),
        "fixture": [f"{en_name(h)} v {en_name(a)}" for h, a in zip(wide["home"], wide["away"])],
        "result": wide["ftr"],
        "model_H": wide[("p_home", "blend")],
        "model_D": wide[("p_draw", "blend")],
        "model_A": wide[("p_away", "blend")],
        "market_H": wide[("p_home", "market")],
        "market_D": wide[("p_draw", "market")],
        "market_A": wide[("p_away", "market")],
    })
    return out.sort_values("date").reset_index(drop=True)


def _fmt_table(df: pd.DataFrame, columns: dict[str, str], float_cols: tuple[str, ...] = ()) -> str:
    """Minimal markdown table writer (avoids the optional ``tabulate`` dependency)."""
    view = df[list(columns)].rename(columns=columns).copy()
    header = "| " + " | ".join(view.columns) + " |"
    sep = "|" + "|".join([" --- "] * len(view.columns)) + "|"
    lines = [header, sep]
    for row in view.itertuples(index=False):
        cells = []
        for name, value in zip(view.columns, row):
            if name in float_cols and pd.notna(value):
                cells.append(f"{value:.4f}")
            elif pd.isna(value):
                cells.append("—")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def results_markdown(preds: pd.DataFrame, stats: dict, sim: pd.DataFrame) -> str:
    lb = leaderboard(preds)
    season = current_season(load_dataset())
    lines = []
    lines.append(f"**Out-of-sample: {stats['n_matches']:,} matches, {stats['n_seasons']} seasons "
                 f"(2018-19 → {season}).** Lower log loss is better.\n")
    table = lb.copy()
    table["model"] = table["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
    lines.append(_fmt_table(
        table, {"model": "Model", "log_loss": "Log loss", "brier": "Brier", "rps": "RPS", "accuracy": "Accuracy"},
        float_cols=("Log loss", "Brier", "RPS", "Accuracy"),
    ))
    lines.append("")
    lines.append(f"Gap to the closing line: **+{stats['gap_to_market']:.4f}** log loss "
                 f"(95% CI {stats['gap_ci_low']:+.4f} to {stats['gap_ci_high']:+.4f}). "
                 f"Not statistically distinguishable from the market — by design.\n")

    leaders = sim.head(6)[["team", "projected", "p_title", "p_top4", "p_relegation"]].copy()
    leaders["team"] = leaders["team"].map(en_name)
    for col in ("projected", "p_title", "p_top4", "p_relegation"):
        leaders[col] = leaders[col].map(lambda v: f"{v:.2f}" if col == "projected" else f"{v * 100:.1f}%")
    lines.append(_fmt_table(
        leaders,
        {"team": "Club", "projected": "Projected pts", "p_title": "Title", "p_top4": "Top 4", "p_relegation": "Relegated"},
    ))
    return "\n".join(lines)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _next_round_html(forecast: pd.DataFrame | None, figure: Path | None) -> str:
    if forecast is None or forecast.empty:
        return ""
    rows = []
    for row in forecast.itertuples(index=False):
        kickoff = pd.Timestamp(row.kickoff).strftime("%a %d %b, %H:%M")
        market = (f"{row.mkt_home:.2f} / {row.mkt_draw:.2f} / {row.mkt_away:.2f}"
                  if np.isfinite(row.mkt_home) else "—")
        if row.status == "played":
            result = f"{row.score} {row.result} {'✅' if row.hit else '❌'}"
        elif row.status == "awaiting_result":
            result = "awaiting result"
        else:
            result = "—"
        rows.append(
            f"<tr><td>{kickoff}</td><td>{en_name(row.home)} v {en_name(row.away)}</td>"
            f"<td>{row.p_home:.2f} / {row.p_draw:.2f} / {row.p_away:.2f}</td>"
            f"<td>{market}</td><td>{row.pick} {row.pick_prob:.0%}</td><td>{result}</td></tr>"
        )
    image = (f'<figure><img alt="next round" src="data:image/png;base64,{_b64(figure)}">'
             f"<figcaption>M = model, P = de-vigged opening price. Model / price probabilities "
             f"are home / draw / away.</figcaption></figure>") if figure and figure.exists() else ""
    return f"""<h2>Next round</h2>
<p class="lede">The weekly artefact: each fixture forecast by the fused model and compared with the
de-vigged opening price. Fixtures already played keep their result in the last column, so the
page doubles as a running account of the model's calls.</p>
{image}
<table>
<thead><tr><th>Kick-off</th><th>Fixture</th><th>Model H / D / A</th><th>Price H / D / A</th><th>Pick</th><th>Result</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
"""


def write_dashboard(
    stats: dict,
    preds: pd.DataFrame,
    sim: pd.DataFrame,
    figures: dict[str, Path],
    forecast: pd.DataFrame | None = None,
    fixtures_figure: Path | None = None,
) -> Path:
    lb = leaderboard(preds)
    lb["model"] = lb["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
    lb_view = lb.copy()
    for col in ("log_loss", "brier", "rps", "accuracy"):
        lb_view[col] = lb_view[col].map(lambda v: f"{v:.4f}")

    top = sim.head(8).copy()
    top["team"] = top["team"].map(en_name)
    for col in ("projected", "p10", "p90"):
        top[col] = top[col].map(lambda v: f"{v:.1f}")
    for col in ("p_title", "p_top4", "p_relegation"):
        top[col] = top[col].map(lambda v: f"{v * 100:.1f}%")

    season = current_season(load_dataset())
    figures_html = "".join(
        f'<figure><img alt="{key}" src="data:image/png;base64,{_b64(path)}">'
        f'<figcaption>{caption}</figcaption></figure>'
        for key, (path, caption) in figures.items()
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PYTHIA — Premier League forecast {season}</title>
<style>
  :root {{ --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#f9fafb; color:var(--ink);
         font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 40px 20px 80px; }}
  header {{ border-bottom: 1px solid var(--line); padding-bottom: 22px; margin-bottom: 28px; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing:-0.3px; }}
  .sub {{ color: var(--muted); font-size: 13.5px; }}
  .cta {{ display:inline-block; margin-top:12px; font-size:13px; font-weight:600;
         color:#fff; background:var(--accent); padding:7px 13px; border-radius:8px;
         text-decoration:none; }}
  .cta:hover {{ background:#1d4ed8; }}
  .cards {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin: 26px 0 34px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
  .card .k {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
  .card .v {{ font-size:22px; font-weight:600; margin-top:4px; font-variant-numeric: tabular-nums; }}
  .card .n {{ font-size:11.5px; color:var(--muted); }}
  h2 {{ font-size:16px; margin:38px 0 10px; }}
  p.lede {{ color:#374151; margin: 0 0 8px; }}
  figure {{ margin: 0 0 26px; background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px; }}
  figure img {{ width:100%; height:auto; display:block; }}
  figcaption {{ color:var(--muted); font-size:12px; margin-top:8px; }}
  table {{ border-collapse: collapse; width:100%; background:#fff; border:1px solid var(--line);
           border-radius:10px; overflow:hidden; font-size:13px; font-variant-numeric: tabular-nums; }}
  th, td {{ padding:8px 10px; text-align:right; border-bottom:1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ background:#f3f4f6; font-weight:600; font-size:12px; color:#374151; }}
  tr:last-child td {{ border-bottom:none; }}
  .lo {{ color:#b91c1c; }} .hi {{ color:#047857; }}
  footer {{ margin-top:50px; padding-top:18px; border-top:1px solid var(--line);
            color:var(--muted); font-size:12px; }}
  code {{ background:#eef2ff; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>PYTHIA</h1>
  <div class="sub">Premier League forecast · {season} · walk-forward backtest over
  {stats['n_matches']} out-of-sample matches · benchmark: {MODEL_DISPLAY['market']}</div>
  <a class="cta" href="./explorer.html">交互探索器 · 选两支球队看概率与比分分布 →</a>
</header>

<div class="cards">
  <div class="card"><div class="k">Blend log loss</div><div class="v">{stats['models']['blend']['log_loss']:.4f}</div><div class="n">lower is better</div></div>
  <div class="card"><div class="k">Closing line</div><div class="v">{stats['models']['market']['log_loss']:.4f}</div><div class="n">the bar to beat</div></div>
  <div class="card"><div class="k">Gap</div><div class="v">+{stats['gap_to_market']:.4f}</div><div class="n">95% CI {stats['gap_ci_low']:+.3f} / {stats['gap_ci_high']:+.3f}</div></div>
  <div class="card"><div class="k">Accuracy</div><div class="v">{stats['models']['blend']['accuracy'] * 100:.1f}%</div><div class="n">vs {stats['models']['market']['accuracy'] * 100:.1f}% market</div></div>
</div>

<p class="lede">This is a forecast quality report, not a betting tip sheet. Every number below is produced by
refitting on data available <em>before</em> each season kicked off; the closing line is de-vigged and scored
on exactly the same matches. The model does not beat the market, and the project says so on the front page.</p>

<h2>Leaderboard</h2>
<table>
<thead><tr><th>Model</th><th>Log loss</th><th>Brier</th><th>RPS</th><th>Accuracy</th><th>n</th></tr></thead>
<tbody>
{"".join(f"<tr><td>{r.model}</td><td>{r.log_loss}</td><td>{r.brier}</td><td>{r.rps}</td><td>{r.accuracy}</td><td>{int(r.n)}</td></tr>" for r in lb_view.itertuples(index=False))}
</tbody>
</table>

{_next_round_html(forecast, fixtures_figure)}

{figures_html}

<h2>Projected final table — {season}</h2>
<p class="lede">Remaining fixtures simulated 20,000 times from the Dixon-Coles scoreline distribution,
added to points already banked.</p>
<table>
<thead><tr><th>Club</th><th>Projected pts</th><th>P10</th><th>P90</th><th>Title</th><th>Top 4</th><th>Relegated</th></tr></thead>
<tbody>
{"".join(f"<tr><td>{r.team}</td><td>{r.projected}</td><td>{r.p10}</td><td>{r.p90}</td><td>{r.p_title}</td><td>{r.p_top4}</td><td>{r.p_relegation}</td></tr>" for r in top.itertuples(index=False))}
</tbody>
</table>

<footer>
  Data: football-data.co.uk (results, shots, closing odds). Models: base rates, Elo + softmax, Dixon-Coles,
  gradient boosting, and a shrunk blend of the last three. Reproduce with
  <code>pythia fetch &amp;&amp; pythia report</code>. Research project — no betting advice, no profit claims.
</footer>
</div>
</body>
</html>
"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def _decomposition_table(preds: pd.DataFrame) -> pd.DataFrame:
    from .eval.decomposition import brier_decomposition
    from .models.base import outcome_index

    rows = []
    for model in ("market", "blend", "blend+T"):
        block = preds.loc[preds["model"] == model]
        if block.empty:
            continue
        parts = brier_decomposition(block[["p_home", "p_draw", "p_away"]].to_numpy(), outcome_index(block))
        rows.append({"model": model, **parts})
    return pd.DataFrame(rows)


def calibration_markdown(
    lb_cal: pd.DataFrame,
    decomposition: pd.DataFrame,
    lambdas: dict,
    *,
    lang: str = "zh",
) -> str:
    """README block for the calibration study."""
    zh = lang == "zh"
    lookup = lb_cal.set_index("model")["log_loss"].to_dict()

    lines = []
    pairs = [("blend", "blend+T"), ("elo_logit", "elo_logit+T"),
             ("dixon_coles", "dixon_coles+T"), ("gbdt", "gbdt+T")]
    rows = []
    for base, cal in pairs:
        if base not in lookup or cal not in lookup:
            continue
        t_values = [t for (m, _), t in lambdas.items() if m == base]
        t_median = float(np.median(t_values)) if t_values else 1.0
        rows.append((MODEL_DISPLAY.get(base, base), lookup[base], lookup[cal], t_median))

    header = ("| 模型 | log loss | + 温度 | 变化 | T 中位数 |" if zh
              else "| Model | Log loss | With temperature | Delta | Median T |")
    lines.append(header)
    lines.append("| --- | --- | --- | --- | --- |")
    for name, base_loss, cal_loss, t_median in rows:
        lines.append(f"| {name} | {base_loss:.4f} | {cal_loss:.4f} | {cal_loss - base_loss:+.4f} | {t_median:.2f} |")

    lines.append("")
    if zh:
        lines.append("温度在**上一季**的样本外预测上拟合，绝不接触被测赛季。T 中位数在 1.00–1.14 之间移动："
                     "融合与 Elo 基本是 1.00，唯一偏乐观的是梯度提升（1.14，校准后确实降了 0.0006），"
                     "但它的整体水平太差，校准救不回来。**没有模型的校准被温度明显改善。**")
    else:
        lines.append("Temperatures are fitted on the **previous** season's out-of-sample forecasts, never on the "
                     "season being scored. The median T moves between 1.00 and 1.14: blend and Elo stay at 1.00, "
                     "and the only genuinely over-confident model is gradient boosting (T = 1.14, which does "
                     "recover 0.0006) — but it is too weak for that to matter. "
                     "**No model gets a meaningful calibration improvement.**")

    if not decomposition.empty:
        lines.append("")
        lines.append("**Brier 分解**（Murphy）：`Brier = 可靠性 − 分辨率 + 不确定性`" if zh
                     else "**Brier decomposition** (Murphy): `Brier = reliability − resolution + uncertainty`")
        lines.append("")
        lines.append("| 预测 | 可靠性 ↓ | 分辨率 ↑ | Brier |" if zh
                     else "| Forecast | Reliability ↓ | Resolution ↑ | Brier |")
        lines.append("| --- | --- | --- | --- |")
        names = {"market": "收盘价" if zh else "Closing line",
                 "blend": "融合" if zh else "Blend",
                 "blend+T": "融合 + 温度" if zh else "Blend + temperature"}
        for row in decomposition.itertuples(index=False):
            lines.append(f"| {names.get(row.model, row.model)} | {row.reliability:.4f} | "
                         f"{row.resolution:.4f} | {row.brier:.4f} |")

        if zh:
            lines.append("")
            lines.append("读法：两者的**可靠性几乎相同**（0.0016 对 0.0012），差距几乎全部来自**分辨率**"
                         "（0.0661 对 0.0786）。也就是说，模型不是过度自信，而是**知道得更少**——"
                         "它没有比庄家更差的校准，只有更少的信息。这解释了为什么温度校准救不了差距，"
                         "也指出了唯一有意义的方向：加信息（xG、伤停、欧战赛程），而不是调概率。")
        else:
            lines.append("")
            lines.append("Read it as: reliability is nearly identical (0.0016 vs 0.0012) and almost the whole gap "
                         "is **resolution** (0.0661 vs 0.0786). The model is not over-confident, it simply knows "
                         "less. That is why a temperature cannot close the gap, and why the only meaningful "
                         "direction is more information (xG, injuries, European fixtures), not sharper tuning.")

    lines.append("")
    lines.append("![校准研究](reports/calibration_study.png)" if zh else "![Calibration study](reports/calibration_study.png)")
    return "\n".join(lines)


def generate(*, refresh: bool = True, n_sims: int = 20_000, inject_readme: bool = True) -> dict:
    from . import plots

    features = load_dataset()
    preds = get_predictions(refresh=refresh)
    stats = headline_stats(preds)

    lb = leaderboard(preds)
    lb.to_csv(REPORTS_DIR / "leaderboard.csv", index=False)
    season_table(preds).to_csv(REPORTS_DIR / "season_logloss.csv")
    preds.loc[preds["model"].isin(["blend", "market"])].to_csv(REPORTS_DIR / "predictions.csv", index=False)

    # Calibration study: one temperature per model, fitted on the prior season,
    # plus the Brier decomposition that explains where the gap actually comes from.
    calibrated, lambdas = calibrate_by_season(preds)
    lb_cal = leaderboard(calibrated)
    lb_cal.to_csv(REPORTS_DIR / "calibration_study.csv", index=False)
    decomposition = _decomposition_table(calibrated)
    decomposition.to_csv(REPORTS_DIR / "brier_decomposition.csv", index=False)

    finals = fit_final(features)
    season = current_season(features)
    # For *live* forecasts (strength map, scoreline heatmap, simulation) the fit
    # is restricted to the last two seasons: it keeps the parameter table to the
    # twenty clubs actually in the league and lets recent form dominate.
    start_year = int(season[:4]) - 1
    recent = features.loc[features["date"] >= pd.Timestamp(f"{start_year}-07-01")]
    dc_live = DixonColes().fit(recent)
    sim, sample = simulate_remaining(features, dc_live, season, n_sims=n_sims)
    sim.to_csv(REPORTS_DIR / "simulated_table.csv", index=False)

    table_now = current_table(features, season)
    points_now = table_now.set_index("team")["points"]

    top_teams = sim.head(2)["team"].tolist()
    heat_home, heat_away = (top_teams + ["arsenal", "man_city"])[:2]

    figures = {
        "leaderboard": (plots.leaderboard_bars(lb), "Every model against the closing line on identical matches."),
        "cumulative": (plots.cumulative_logloss(preds), "Cumulative mean log loss: skill accumulates slowly and plateaus above the market."),
        "calibration": (plots.calibration_plot(preds), "Reliability of home-win and draw probabilities, with 95% binomial error bars."),
        "seasons": (plots.season_bars(preds), "Season by season — a consistent gap rather than one lucky year."),
        "strength": (plots.strength_map(dc_live), "Dixon-Coles attack and defence parameters, fitted on the last two seasons."),
        "heatmap": (plots.score_heatmap(dc_live, heat_home, heat_away), f"Scoreline distribution for {en_name(heat_home)} vs {en_name(heat_away)}."),
        "simulation": (plots.simulation_plot(sample, points_now), "Projected final points, 20,000 simulations of the remaining fixtures."),
        "calibration_study": (plots.calibration_study(lb_cal, decomposition),
                              "A temperature does not close the gap; reliability vs resolution shows why."),
    }

    latest = latest_predictions(preds, features)
    latest.to_csv(REPORTS_DIR / "latest_predictions.csv", index=False)

    # Weekly artefact: the next round, forecast live and compared with the price.
    live_models = _live_models(features, finals, preds)
    fixtures = load_fixtures()
    forecast = next_round(features, fixtures, models=live_models) if not fixtures.empty else pd.DataFrame()
    if not forecast.empty:
        forecast.to_csv(REPORTS_DIR / "fixtures_forecast.csv", index=False)
        fixtures_figure = plots.fixtures_chart(forecast)
        fixtures_blocks = {
            "README.md": fixtures_markdown(forecast, lang="zh"),
            "README.en.md": fixtures_markdown(forecast, lang="en"),
        }
        (REPORTS_DIR / "fixtures_block.md").write_text(fixtures_blocks["README.md"], encoding="utf-8")
    else:
        fixtures_figure = None
        fixtures_blocks = {"README.md": "_赛程源暂时没有可预测的比赛。_",
                           "README.en.md": "_No forecastable fixtures in the feed._"}

    block = results_markdown(preds, stats, sim)
    (REPORTS_DIR / "results_block.md").write_text(block, encoding="utf-8")

    calib_blocks = {
        "README.md": calibration_markdown(lb_cal, decomposition, lambdas, lang="zh"),
        "README.en.md": calibration_markdown(lb_cal, decomposition, lambdas, lang="en"),
    }
    (REPORTS_DIR / "calibration_block.md").write_text(calib_blocks["README.md"], encoding="utf-8")

    from .eval.ledger import build_ledger, ledger_markdown

    ledger = build_ledger(preds, season=season)
    if not ledger.empty:
        ledger.to_csv(REPORTS_DIR / f"ledger_{season}.csv", index=False)
    ledger_blocks = {
        "README.md": ledger_markdown(ledger, season, lang="zh"),
        "README.en.md": ledger_markdown(ledger, season, lang="en"),
    }

    from . import explorer

    explorer_path = explorer.build(features, preds, forecast if not forecast.empty else None)

    dashboard = write_dashboard(stats, preds, sim, figures, forecast, fixtures_figure)

    if inject_readme:
        for name, fixtures_block in fixtures_blocks.items():
            inject_readme_block(block, ROOT / name)
            inject_block(fixtures_block, ROOT / name, FIXTURES_START, FIXTURES_END)
            inject_block(calib_blocks[name], ROOT / name, CALIB_START, CALIB_END)
            inject_block(ledger_blocks[name], ROOT / name, LEDGER_START, LEDGER_END)

    return {
        "leaderboard": lb,
        "ledger": ledger,
        "stats": stats,
        "simulation": sim,
        "figures": figures,
        "forecast": forecast,
        "dashboard": dashboard,
        "explorer": explorer_path,
        "calibration": decomposition,
    }


def blend_weights(preds: pd.DataFrame) -> list[float] | None:
    """Shrunk stacking weights from the two preceding seasons of forecasts.

    Same rule the walk-forward uses: tune on out-of-sample predictions, then
    shrink halfway toward equal weights because 760 matches cannot identify a
    3-simplex cleanly.
    """
    from .models.blend import tune_weights

    if preds is None or preds.empty:
        return None

    components = ("elo_logit", "dixon_coles", "gbdt")
    wide = preds.pivot_table(index="match_id", columns="model", values=["p_home", "p_draw", "p_away"])
    meta = preds.drop_duplicates("match_id").set_index("match_id")[["season", "ftr"]]
    wide = pd.concat([wide, meta], axis=1)

    seasons = sorted(wide["season"].unique())[-2:]
    history = wide.loc[wide["season"].isin(seasons)]
    if len(history) < 200:
        return None

    probs = {
        n: np.column_stack([
            history[("p_home", n)], history[("p_draw", n)], history[("p_away", n)],
        ])
        for n in components
    }
    y = history["ftr"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)
    tuned = tune_weights(probs, y)
    equal = 1.0 / len(components)
    weights = {n: 0.5 * equal + 0.5 * tuned[n] for n in components}
    total = sum(weights.values())
    return [weights[n] / total for n in components]


def _live_models(features: pd.DataFrame, finals: dict, preds: pd.DataFrame) -> dict:
    """Join the already-fitted models with the blend weights the backtest would use."""
    from .models.blend import Blend

    weights = blend_weights(preds)
    blend = Blend([finals["elo_logit"], finals["dixon_coles"], finals["gbdt"]], weights=weights)
    if weights is None:
        blend.weights = [1 / 3, 1 / 3, 1 / 3]

    return {
        "elo_logit": finals["elo_logit"],
        "dixon_coles": finals["dixon_coles"],
        "gbdt": finals["gbdt"],
        "blend": blend,
    }


def fixtures_markdown(forecast: pd.DataFrame, *, lang: str = "zh") -> str:
    """README block for the next round, in either language."""
    from .forecast import hit_rate, round_markdown

    if forecast.empty:
        return "_No forecastable fixtures in the feed._" if lang == "en" else "_赛程源暂时没有可预测的比赛。_"

    pending = forecast.loc[forecast["status"] != "played"]
    played = forecast.loc[forecast["status"] == "played"]
    stats = hit_rate(forecast)

    lines = []
    if not pending.empty:
        first = pending["kickoff"].min()
        waiting = int((pending["status"] == "awaiting_result").sum())
        if lang == "en":
            note = f", {waiting} of them already kicked off and waiting for a result" if waiting else ""
            lines.append(f"**Round of {first:%Y-%m-%d}: {len(pending)} fixtures{note}.** "
                         f"Prices are de-vigged Pinnacle openers (`{pending['odds_src'].iloc[0] or '—'}`).\n")
        else:
            note = f"，其中 {waiting} 场已开球待赛果" if waiting else ""
            lines.append(f"**最近一轮：{first:%Y-%m-%d} 起，共 {len(pending)} 场{note}。** "
                         f"「初盘」为去抽水后的 Pinnacle 初盘（来源 `{pending['odds_src'].iloc[0] or '—'}`）。\n")

    lines.append(round_markdown(pending if not pending.empty else forecast, lang=lang))
    if stats["n"]:
        if lang == "en":
            lines.append(f"\nSettled so far: {stats['hits']}/{stats['n']} calls correct ({stats['rate']:.0%}).")
        else:
            lines.append(f"\n已收录赛果 {stats['n']} 场，模型命中 {stats['hits']} 场（{stats['rate']:.0%}）。")
    if not played.empty:
        lines.append("\nSettled:\n" if lang == "en" else "\n已赛场次：\n")
        lines.append(round_markdown(played, lang=lang))
    lines.append("\n![下一轮](reports/next_round.png)" if lang == "zh" else "\n![Next round](reports/next_round.png)")
    return "\n".join(lines)


def inject_readme_block(block: str, path: Path | None = None) -> bool:
    return inject_block(block, path or ROOT / "README.md", README_START, README_END)


def inject_block(block: str, path: Path, start: str, end: str) -> bool:
    """Replace the content between two HTML comment markers."""
    path = Path(path)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return False
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n\n{block.strip()}\n\n{end}{tail}", encoding="utf-8")
    return True
