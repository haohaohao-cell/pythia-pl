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


def write_dashboard(stats: dict, preds: pd.DataFrame, sim: pd.DataFrame, figures: dict[str, Path]) -> Path:
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
<title>PL Predictor — {season}</title>
<style>
  :root {{ --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#f9fafb; color:var(--ink);
         font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 40px 20px 80px; }}
  header {{ border-bottom: 1px solid var(--line); padding-bottom: 22px; margin-bottom: 28px; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing:-0.3px; }}
  .sub {{ color: var(--muted); font-size: 13.5px; }}
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
  <h1>Premier League predictor</h1>
  <div class="sub">{season} · walk-forward backtest over {stats['n_matches']} out-of-sample matches ·
  benchmark: {MODEL_DISPLAY['market']}</div>
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
  <code>plpred fetch &amp;&amp; plpred report</code>. Research project — no betting advice, no profit claims.
</footer>
</div>
</body>
</html>
"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def generate(*, refresh: bool = True, n_sims: int = 20_000, inject_readme: bool = True) -> dict:
    from . import plots

    features = load_dataset()
    preds = get_predictions(refresh=refresh)
    stats = headline_stats(preds)

    lb = leaderboard(preds)
    lb.to_csv(REPORTS_DIR / "leaderboard.csv", index=False)
    season_table(preds).to_csv(REPORTS_DIR / "season_logloss.csv")
    preds.loc[preds["model"].isin(["blend", "market"])].to_csv(REPORTS_DIR / "predictions.csv", index=False)

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
    }

    latest = latest_predictions(preds, features)
    latest.to_csv(REPORTS_DIR / "latest_predictions.csv", index=False)

    block = results_markdown(preds, stats, sim)
    (REPORTS_DIR / "results_block.md").write_text(block, encoding="utf-8")

    dashboard = write_dashboard(stats, preds, sim, figures)

    if inject_readme:
        for name in ("README.md", "README.en.md"):
            inject_readme_block(block, ROOT / name)

    return {
        "leaderboard": lb,
        "stats": stats,
        "simulation": sim,
        "figures": figures,
        "dashboard": dashboard,
    }


def inject_readme_block(block: str, path: Path | None = None) -> bool:
    path = Path(path or ROOT / "README.md")
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    path.write_text(f"{head}{README_START}\n\n{block}\n\n{README_END}{tail}", encoding="utf-8")
    return True
