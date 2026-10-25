"""Command line entry point: ``pythia <command>``."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, season_start_year, season_label


def _cmd_fetch(args: argparse.Namespace) -> None:
    from .config import SEASON_CODES
    from .ingest.fixtures_feed import download_fixtures
    from .ingest.football_data import download_season

    for season, code in SEASON_CODES.items():
        path = download_season(code, force=args.force)
        print(f"  {season}  {path.name}  {path.stat().st_size / 1024:.0f} KB")
    fixtures = download_fixtures(force=args.force)
    print(f"  next round  {fixtures.name}  {fixtures.stat().st_size / 1024:.0f} KB")
    print(f"fetched {len(SEASON_CODES)} seasons + 1 fixture feed")

    if getattr(args, "xg", False):
        from .ingest import understat

        data = understat.fetch(force=args.force, verbose=True)
        print(f"  understat xG  {len(data)} matches  {understat.CACHE_PATH.stat().st_size / 1024:.0f} KB")
        print("rebuild the feature table with `pythia build` to pick it up")


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


def _cmd_fixtures(args: argparse.Namespace) -> None:
    from .dataset import load_dataset
    from .forecast import hit_rate, next_round
    from .ingest.fixtures_feed import load_fixtures
    from .teams import en_name

    features = load_dataset()
    fixtures = load_fixtures(force=args.refresh)
    forecast = next_round(features, fixtures)
    if forecast.empty:
        print("no upcoming fixtures in the feed")
        return

    view = forecast.assign(
        fixture=[f"{en_name(h)} v {en_name(a)}" for h, a in zip(forecast["home"], forecast["away"])],
        kickoff=forecast["kickoff"].dt.strftime("%a %d %b %H:%M"),
    )
    print()
    print(view[["kickoff", "fixture", "p_home", "p_draw", "p_away",
                "mkt_home", "mkt_draw", "mkt_away", "pick", "pick_prob", "status"]]
          .round(3).to_string(index=False))
    stats = hit_rate(forecast)
    if stats["n"]:
        print(f"\nplayed so far: {stats['hits']}/{stats['n']} correct ({stats['rate']:.0%})")
    if args.save:
        from .config import REPORTS_DIR
        path = REPORTS_DIR / "fixtures_forecast.csv"
        forecast.to_csv(path, index=False)
        print(f"saved {path}")


def _cmd_ablate(args: argparse.Namespace) -> None:
    from .config import REPORTS_DIR
    from .dataset import load_dataset
    from .eval.ablation import has_xg, run_ablation
    from .report import get_predictions

    features = load_dataset()
    if not has_xg(features):
        print('xG features are missing. Run `pythia fetch --xg` then `pythia build` first.')
        return

    preds = get_predictions()
    table, summary = run_ablation(features, preds)
    print(table.round(4).to_string(index=False))
    print()
    print(f"{'variant':<12}{'log loss':>10}{'delta':>10}")
    base_blend = summary.get("blend")
    for name in ("gbdt", "gbdt_xg", "blend", "blend_xg"):
        if name in summary:
            delta = summary[name] - summary["gbdt"]
            print(f"{name:<12}{summary[name]:>10.4f}{delta:>+10.4f}")
    if "delta_blend" in summary:
        d = summary["delta_blend"]
        print(f"\nblend with xG minus blend: {d['delta']:+.4f} "
              f"(95% CI {d['ci_low']:+.4f} to {d['ci_high']:+.4f}, P(better)={d['p_better']:.2f})")

    path = REPORTS_DIR / "xg_ablation.csv"
    table.to_csv(path, index=False)
    print(f"\nsaved {path}")


def _cmd_fpl(args: argparse.Namespace) -> None:
    from .ingest import fpl
    from .teams import en_name

    bootstrap = fpl.fetch_bootstrap()
    players = fpl.players_frame(bootstrap)
    table = fpl.team_availability(bootstrap)

    gw = fpl.current_gameweek(bootstrap)
    flagged = int((players["status"] != "a").sum())
    print(f"gameweek {gw}: {len(players)} players listed, {flagged} not fully available")

    if not table.empty:
        view = table.head(args.top).copy()
        view["club"] = view["team"].map(en_name)
        view["avail"] = (view["avail_score"] * 100).round(1)
        print(view[["club", "n_injured", "n_doubtful", "n_suspended",
                    "minutes_available", "avail"]].to_string(index=False))

    if args.snapshot:
        path = fpl.save_snapshot(bootstrap)
        print(f"\nsnapshot saved: {path}")


def _cmd_markets(args: argparse.Namespace) -> None:
    from .config import REPORTS_DIR
    from .dataset import load_dataset
    from .eval.derived import evaluate_over_under
    from . import plots

    features = load_dataset()
    if features["ou_over"].notna().sum() < 100:
        print("no over/under odds in the feature table; rebuild with `pythia build --force`")
        return

    table, summary = evaluate_over_under(features)
    print(table.round(4).to_string(index=False))
    print()
    print(f"{'':<8}{'log loss':>10}")
    for name in ("model", "market", "base"):
        print(f"{name:<8}{summary[name]:>10.4f}")
    print(f"\nmodel minus market: {summary['delta']:+.4f} "
          f"(95% CI {summary['ci_low']:+.4f} to {summary['ci_high']:+.4f}, "
          f"P(better)={summary['p_better']:.2f})")

    path = REPORTS_DIR / "ou_25.csv"
    table.to_csv(path, index=False)
    plots.derived_market_chart(table)
    print(f"saved {path} and reports/ou_25.png")


def _cmd_prior(args: argparse.Namespace) -> None:
    from .config import REPORTS_DIR
    from .dataset import load_dataset
    from .eval.prior import run_prior_study
    from .report import get_predictions
    from . import plots

    features = load_dataset()
    preds = get_predictions()
    table, summary = run_prior_study(features, preds)
    if table.empty:
        print("no overlapping matches with opening prices")
        return

    print(table.round(4).to_string(index=False))
    print()
    for name in ("model", "opening", "hybrid", "closing"):
        print(f"{name:<8}{summary[name]:>9.4f}")
    print(f"\nmedian mix weight: {summary['median_weight']:.2f}  "
          f"(1.00 means the optimiser discarded the model entirely)")
    print(f"gap closed by anchoring on the opening line: {summary['gap_closed'] * 100:.0f}%")
    print(f"hybrid minus closing: {summary['hybrid_vs_closing']['delta']:+.4f} "
          f"(95% CI {summary['hybrid_vs_closing']['ci_low']:+.4f} to "
          f"{summary['hybrid_vs_closing']['ci_high']:+.4f})")

    table.to_csv(REPORTS_DIR / "prior_study.csv", index=False)
    plots.prior_ladder(summary)
    print(f"saved {REPORTS_DIR / 'prior_study.csv'} and reports/prior_ladder.png")


def _cmd_ledger(args: argparse.Namespace) -> None:
    from .config import REPORTS_DIR
    from .dataset import load_dataset
    from .eval.ledger import build_ledger, ledger_markdown
    from .report import get_predictions

    features = load_dataset()
    season = args.season or str(features["season"].max())
    preds = get_predictions()
    ledger = build_ledger(preds, season=season)
    if ledger.empty:
        print(f"no settled matches for {season}")
        return

    print(ledger.round(4).to_string(index=False))
    print()
    print(ledger_markdown(ledger, season))

    path = REPORTS_DIR / f"ledger_{season}.csv"
    ledger.to_csv(path, index=False)
    print(f"\nsaved {path}")


def _cmd_doctor(args: argparse.Namespace) -> None:
    from .doctor import format_checks, has_failures, run_checks

    checks = run_checks()
    print(format_checks(checks))
    raise SystemExit(1 if has_failures(checks) else 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pythia", description="Premier League forecast benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download raw CSVs from football-data.co.uk")
    p.add_argument("--force", action="store_true", help="ignore cached files")
    p.add_argument("--xg", action="store_true", help="also fetch Understat xG (needs the [xg] extra)")
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

    p = sub.add_parser("fixtures", help="forecast the next round from the fixtures feed")
    p.add_argument("--refresh", action="store_true", help="re-download fixtures.csv")
    p.add_argument("--save", action="store_true", help="write reports/fixtures_forecast.csv")
    p.set_defaults(func=_cmd_fixtures)

    p = sub.add_parser("simulate", help="Monte Carlo the rest of a season")
    p.add_argument("--season", default=None)
    p.add_argument("--sims", type=int, default=20_000)
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("ablate", help="xG ablation: same protocol, one variable changed")
    p.set_defaults(func=_cmd_ablate)

    p = sub.add_parser("markets", help="derived market: over/under 2.5 goals")
    p.set_defaults(func=_cmd_markets)

    p = sub.add_parser("prior", help="how much of the market can the model replicate?")
    p.set_defaults(func=_cmd_prior)

    p = sub.add_parser("ledger", help="running season account: model vs closing line")
    p.add_argument("--season", default=None)
    p.set_defaults(func=_cmd_ledger)

    p = sub.add_parser("doctor", help="data source and cache health check")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("fpl", help="squad availability snapshot from the FPL API")
    p.add_argument("--snapshot", action="store_true", help="commit today's team-level snapshot")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=_cmd_fpl)

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
