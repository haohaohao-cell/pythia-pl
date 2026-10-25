"""Forecast the next round of fixtures and score last week's calls.

This is the weekly artefact: for each upcoming match, the model's three-way
probabilities, the de-vigged opening price, and the gap between the two.  Once a
fixture has been played the row is filled with the result, so the page keeps an
honest running account of the model's calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fixtures import fixture_row
from .ingest.fixtures_feed import load_fixtures
from .models.base import outcome_index
from .models.blend import Blend
from .models.dixon_coles import DixonColes
from .models.elo_logit import EloLogit
from .models.gbdt import GradientBoost
from .odds import implied_probs
from .teams import en_name

OUTCOME_LABEL = {0: "H", 1: "D", 2: "A"}


def default_models(features: pd.DataFrame) -> dict[str, object]:
    """Everything fitted on all matches available today."""
    return {
        "dixon_coles": DixonColes().fit(features),
        "elo_logit": EloLogit().fit(features),
        "gbdt": GradientBoost().fit(features),
        "blend": Blend([EloLogit(), DixonColes(), GradientBoost()]).fit(features),
    }


def _played_lookup(features: pd.DataFrame) -> pd.DataFrame:
    return features[["home", "away", "date", "ftr", "fthg", "ftag"]].copy()


def next_round(
    features: pd.DataFrame,
    fixtures: pd.DataFrame | None = None,
    *,
    models: dict[str, object] | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per fixture, with model and market probabilities."""
    fixtures = load_fixtures() if fixtures is None else fixtures
    if fixtures.empty:
        return pd.DataFrame()

    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    models = models or default_models(features)
    played = _played_lookup(features)
    rows = []

    for fixture in fixtures.itertuples(index=False):
        try:
            frame = fixture_row(features, fixture.home, fixture.away)
        except KeyError:
            continue
        frame.loc[0, "date"] = fixture.kickoff

        probs = {name: np.asarray(model.predict_proba(frame))[0] for name, model in models.items()}
        blend = probs["blend"] / probs["blend"].sum()

        odds = np.array([fixture.odds_h_h, fixture.odds_h_d, fixture.odds_h_a], dtype=float)
        market = implied_probs(odds)
        has_market = bool(np.isfinite(market).all())

        pick = int(np.argmax(blend))

        # Has this fixture already been played?  Match on clubs and a +/- 6 day window.
        candidate = played.loc[(played["home"] == fixture.home) & (played["away"] == fixture.away)]
        if not candidate.empty:
            gap = (candidate["date"] - fixture.kickoff).dt.total_seconds().abs() / 86400.0
            hit_row = candidate.loc[gap.sort_values().index[0]] if gap.min() <= 6 else None
        else:
            hit_row = None

        row = {
            "kickoff": fixture.kickoff,
            "home": fixture.home,
            "away": fixture.away,
            "p_home": blend[0],
            "p_draw": blend[1],
            "p_away": blend[2],
            "dc_home": probs["dixon_coles"][0],
            "dc_away": probs["dixon_coles"][2],
            "mkt_home": market[0] if has_market else np.nan,
            "mkt_draw": market[1] if has_market else np.nan,
            "mkt_away": market[2] if has_market else np.nan,
            "pick": OUTCOME_LABEL[pick],
            "pick_prob": float(blend[pick]),
            "odds_src": fixture.odds_src,
            # "awaiting_result" is the honest state when kick-off has passed but the
            # results feed has not caught up yet.
            "status": "played" if hit_row is not None else
                      ("awaiting_result" if fixture.kickoff < now else "upcoming"),
            "result": hit_row["ftr"] if hit_row is not None else "",
            "score": f"{int(hit_row['fthg'])}-{int(hit_row['ftag'])}" if hit_row is not None else "",
        }
        row["hit"] = (row["result"] == row["pick"]) if hit_row is not None else None
        if has_market:
            row["gap_home"] = row["p_home"] - row["mkt_home"]
            row["gap_draw"] = row["p_draw"] - row["mkt_draw"]
            row["gap_away"] = row["p_away"] - row["mkt_away"]
            row["max_gap"] = float(np.max(np.abs([row["gap_home"], row["gap_draw"], row["gap_away"]])))
        else:
            row["gap_home"] = row["gap_draw"] = row["gap_away"] = row["max_gap"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("kickoff").reset_index(drop=True)


HEADERS = {
    "zh": {
        "kickoff": "开球", "fixture": "对阵", "model": "模型 主/平/客",
        "price": "初盘 主/平/客", "pick": "倾向", "result": "结果", "await": "待赛果",
        "empty": "_赛程源暂时没有可预测的比赛。_",
    },
    "en": {
        "kickoff": "Kick-off", "fixture": "Fixture", "model": "Model H / D / A",
        "price": "Price H / D / A", "pick": "Pick", "result": "Result", "await": "awaiting",
        "empty": "_No forecastable fixtures in the feed._",
    },
}


def round_markdown(forecast: pd.DataFrame, *, lang: str = "zh", limit: int | None = None) -> str:
    """Compact markdown table used in the README and the dashboard."""
    labels = HEADERS[lang]
    if forecast.empty:
        return labels["empty"] + "\n"
    view = forecast if limit is None else forecast.head(limit)
    show_result = bool((view["status"] == "played").any())

    header = (f"| {labels['kickoff']} | {labels['fixture']} | {labels['model']} "
              f"| {labels['price']} | {labels['pick']} |")
    rule = "| --- | --- | --- | --- | --- |"
    if show_result:
        header += f" {labels['result']} |"
        rule += " --- |"
    lines = [header, rule]

    for row in view.itertuples(index=False):
        kickoff = pd.Timestamp(row.kickoff).strftime("%m-%d %H:%M")
        fixture = f"{en_name(row.home)} v {en_name(row.away)}"
        model = f"{row.p_home:.2f} / {row.p_draw:.2f} / {row.p_away:.2f}"
        if np.isfinite(row.mkt_home):
            market = f"{row.mkt_home:.2f} / {row.mkt_draw:.2f} / {row.mkt_away:.2f}"
        else:
            market = "—"
        line = f"| {kickoff} | {fixture} | {model} | {market} | {row.pick} {row.pick_prob:.0%} |"
        if show_result:
            if row.status == "played":
                result = f"{row.score} {row.result} {'✅' if row.hit else '❌'}"
            elif row.status == "awaiting_result":
                result = labels["await"]
            else:
                result = "—"
            line += f" {result} |"
        lines.append(line)
    return "\n".join(lines) + "\n"


def hit_rate(forecast: pd.DataFrame) -> dict[str, float]:
    done = forecast.loc[forecast["status"] == "played"]
    if done.empty:
        return {"n": 0, "hits": 0, "rate": float("nan")}
    hits = int(done["hit"].sum())
    return {"n": int(len(done)), "hits": hits, "rate": hits / len(done)}
