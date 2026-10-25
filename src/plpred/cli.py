"""Command line entry point: ``plpred <command>``."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, season_start_year, season_label


def _cmd_fetch(args: argparse.Namespace) -> None:
    from .ingest.football_data import download_season
    from .config import SEASON_CODES

    for season, code in SEASON_CODES.items():
        path = download_season(code, force=args.force)
        print(f"  {season}  {path.name}  {path.stat().st_size / 1024:.0f} KB")
    print(f"fetched {len(SEASON_CODES)} seasons")


def _cmd_build(args: argparse.Namespace) -> None:
    from .dataset import build_dataset

    df = build_dataset(force=args.force)
    print(f"built feature table: {len(df)} matches, {df.shape[1]} columns")
    print(df.groupby("season").size().to_string())


def _cmd_backtest(args: argparse.Namespace) -> None:
    from .config import PREDICTIONS_PARQUET
    from .dataset import load_dataset
    from .eval.walk_forward import leaderboard, run_backtest, season_table

    features = load_dataset()
    preds = run_backtest(features)
    preds.to_parquet(PREDICTIONS_PARQUET, index=False)
    print()
    print(leaderboard(preds).round(4).to_string(index=False))
    print()
    print(season_table(preds).round(4).to_string())


def _cmd_report(args: argparse.Namespace) -> None:
    from .report import generate

    result = generate(refresh=not args.no_refresh, n_sims=args.sims, inject_readme=not args.no_readme)
    stats = result["stats"]
    print(f"dashboard : {result['dashboard']}")
    print(f"matches   : {stats['n_matches']} over {stats['n_seasons']} seasons")
    print(f"blend     : {stats['models']['blend']['log_loss']:.4f} log loss")
    print(f"market    : {stats['models']['market']['log_loss']:.4f} log loss")
    print(f"gap       : {stats['gap_to_market']:+.4f} "
          f"(95% CI {stats['gap_ci_low']:+.4f} to {stats['gap_ci_high']:+.4f})")
    for name, figure in result["figures"].items():
        print(f"figure    : {figure[0].name}")


def _cmd_predict(args: argparse.Namespace) -> None:
    from .dataset import load_dataset
    from .fixtures import fixture_row
    from .models.blend import Blend
    from .models.dixon_coles import DixonColes
    from .models.elo_logit import EloLogit
    from .models.gbdt import GradientBoost
    from .teams import en_name, team_key, zh_name

    features = load_dataset()
    home, away = team_key(args.home), team_key(args.away)
    try:
        row = fixture_row(features, home, away)
    except KeyError as exc:
        print(f"unknown club: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    dc = DixonColes().fit(features)
    models = {
        "Dixon-Coles": dc,
        "Elo + softmax": EloLogit().fit(features),
        "Gradient boosting": GradientBoost().fit(features),
    }
    models["Blend"] = Blend([EloLogit(), DixonColes(), GradientBoost()]).fit(features)

    season = features["season"].max()
    print(f"\n{zh_name(home)} ({en_name(home)})  vs  {zh_name(away)} ({en_name(away)})   [{season}]\n")
    print(f"{'model':<22}{'H':>8}{'D':>8}{'A':>8}   {'pick':>6}")
    for name, model in models.items():
        p = model.predict_proba(row)[0]
        pick = ["H", "D", "A"][int(np.argmax(p))]
        print(f"{name:<22}{p[0]:>8.3f}{p[1]:>8.3f}{p[2]:>8.3f}   {pick:>6}")

    lam, mu = dc._rates(home, away)
    print(f"\nexpected goals: {en_name(home)} {lam:.2f} — {mu:.2f} {en_name(away)}")
    matrix = dc.score_matrix(home, away)
    order = np.argsort(matrix.ravel())[::-1][:5]
    print("most likely scorelines:")
    for flat in order:
        i, j = divmod(int(flat), matrix.shape[1])
        print(f"  {i}-{j}   {matrix[i, j] * 100:5.1f}%")


def _cmd_simulate(args: argparse.Namespace) -> None:
    from .dataset import load_dataset
    from .models.dixon_coles import DixonColes
    from .simulate import simulate_remaining
    from .teams import en_name

    features = load_dataset()
    season = args.season or features["season"].max()
    # Fitted on everything available today; the time decay keeps recent matches
    # dominant, which is what a live forecast should reflect.
    dc = DixonColes().fit(features)
    summary, _ = simulate_remaining(features, dc, season, n_sims=args.sims)
    summary.to_csv(REPORTS_DIR / f"simulation_{season}.csv", index=False)

    view = summary.head(args.top).copy()
    view["club"] = view["team"].map(en_name)
    view["p_title"] = (view["p_title"] * 100).round(1)
    view["p_relegation"] = (view["p_relegation"] * 100).round(1)
    print(view[["club", "played", "points_now", "projected", "p10", "p90", "p_title", "p_relegation"]]
          .round(1).to_string(index=False))


def _cmd_value(args: argparse.Namespace) -> None:
    from .report import get_predictions
    from .dataset import load_dataset
    from .value import season_summary, summarise, value_bets

    features = load_dataset()
    preds = get_predictions()
    bets = value_bets(preds, features, model=args.model, min_edge=args.min_edge)
    n = len(preds.loc[preds["model"] == args.model])
    overall = summarise(bets, n)
    print(f"model={args.model}  min_edge={args.min_edge:.2f}  bets={overall['bets']} "
          f"({overall['share'] * 100:.1f}% of matches)")
    print(f"flat-stake ROI: {overall['roi'] * 100:+.1f}%   "
          f"profit: {overall['profit']:+.1f}u   hit rate: {overall['hit_rate'] * 100:.1f}%   "
          f"mean CLV: {overall['clv'] * 100:+.2f}pp")
    table = season_summary(bets, preds, model=args.model)
    if not table.empty:
        table["roi"] = (table["roi"] * 100).round(1)
        table["hit_rate"] = (table["hit_rate"] * 100).round(1)
        print(table[["season", "bets", "roi", "profit", "hit_rate", "avg_edge", "clv"]].round(3).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plpred", description="Premier League forecast benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download raw CSVs from football-data.co.uk")
    p.add_argument("--force", action="store_true", help="ignore cached files")
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("build", help="build the feature table")
    p.add_argument("--force", action="store_true", help="re-download and rebuild")
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("backtest", help="walk-forward backtest and leaderboard")
    p.set_defaults(func=_cmd_backtest)

    p = sub.add_parser("report", help="regenerate tables, figures and the dashboard")
    p.add_argument("--sims", type=int, default=20_000)
    p.add_argument("--no-refresh", action="store_true", help="reuse cached predictions")
    p.add_argument("--no-readme", action="store_true", help="do not inject into README")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("predict", help="forecast a single fixture")
    p.add_argument("home")
    p.add_argument("away")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("simulate", help="Monte Carlo the rest of a season")
    p.add_argument("--season", default=None)
    p.add_argument("--sims", type=int, default=20_000)
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("value", help="appendix: flat-stake value-bet probe")
    p.add_argument("--model", default="blend")
    p.add_argument("--min-edge", type=float, default=0.05)
    p.set_defaults(func=_cmd_value)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
