"""Interactive explorer: a static page that runs the Dixon-Coles maths in the browser.

Everything that needs a trained model is precomputed into a small JSON payload
(20 clubs, 380 ordered pairings, a handful of parameters).  The browser then
rebuilds the scoreline distribution from the Poisson parameters, so the page is
a genuine model rather than a table of stored numbers — and it needs no server,
no API key and no cold start.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DOCS_DIR, ROOT
from .fixtures import team_state
from .forecast import next_round
from .models.dixon_coles import DixonColes
from .teams import en_name, zh_name

MAX_GOALS = 8          # parameterisation depth for the browser
DISPLAY_GOALS = 6      # heatmap size
TEMPLATE = Path(__file__).with_name("explorer_template.html")


def _pair_key(home: str, away: str) -> str:
    return f"{home}|{away}"


def build_payload(
    features: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    forecast: pd.DataFrame | None = None,
) -> dict:
    """Assemble every number the page needs, rounded to keep the file small."""
    from .fixtures import fixture_row
    from .models.blend import Blend
    from .models.elo_logit import EloLogit
    from .models.gbdt import GradientBoost
    from .report import blend_weights

    season = str(features["season"].max())
    start_year = int(season[:4]) - 1
    recent = features.loc[features["date"] >= pd.Timestamp(f"{start_year}-07-01")]

    # The structural model is fitted on the current-squad window so that the
    # parameter table, the heatmap and the blend all describe the same 20 clubs.
    dc = DixonColes(max_goals=MAX_GOALS).fit(recent)
    weights = blend_weights(preds)
    blend = Blend([EloLogit().fit(features), dc, GradientBoost().fit(features)], weights=weights)
    if weights is None:
        blend.weights = [1 / 3, 1 / 3, 1 / 3]

    teams = sorted(set(features.loc[features["season"] == season, "home"]) |
                   set(features.loc[features["season"] == season, "away"]))

    table = dc.strength_table().set_index("team")
    team_rows = []
    for key in teams:
        state = team_state(features, key)
        team_rows.append({
            "key": key,
            "en": en_name(key),
            "zh": zh_name(key),
            "elo": round(float(state["elo"]), 1),
            "attack": round(float(table.loc[key, "attack"]), 4) if key in table.index else 0.0,
            "defence": round(float(table.loc[key, "defence"]), 4) if key in table.index else 0.0,
        })

    pairs: dict[str, list[float]] = {}
    for home in teams:
        for away in teams:
            if home == away:
                continue
            try:
                frame = fixture_row(features, home, away)
            except KeyError:
                continue
            probs = np.asarray(blend.predict_proba(frame), dtype=float)[0]
            probs = np.clip(probs, 1e-9, 1)
            probs = probs / probs.sum()
            pairs[_pair_key(home, away)] = [round(float(p), 4) for p in probs]

    market: dict[str, list[float]] = {}
    fixtures_meta: dict[str, dict] = {}
    if forecast is not None and not forecast.empty:
        for row in forecast.itertuples(index=False):
            if not np.isfinite(row.mkt_home):
                continue
            key = _pair_key(row.home, row.away)
            market[key] = [round(float(row.mkt_home), 4),
                           round(float(row.mkt_draw), 4),
                           round(float(row.mkt_away), 4)]
            fixtures_meta[key] = {"kickoff": pd.Timestamp(row.kickoff).strftime("%Y-%m-%d %H:%M"),
                                  "status": row.status}

    return {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "season": season,
        "teams": team_rows,
        "pairs": pairs,
        "market": market,
        "fixtures": fixtures_meta,
        "dc": {
            "home_adv": round(float(dc.home_adv_), 4),
            "rho": round(float(dc.rho_), 4),
            "max_goals": MAX_GOALS,
            "display_goals": DISPLAY_GOALS,
        },
        "weights": getattr(blend, "weights", None),
    }


def render_html(payload: dict) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    return template.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def write_explorer(payload: dict, path: Path | None = None) -> Path:
    path = Path(path or DOCS_DIR / "explorer.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(payload), encoding="utf-8")
    return path


def build(features: pd.DataFrame, preds: pd.DataFrame, forecast: pd.DataFrame | None = None) -> Path:
    payload = build_payload(features, preds, forecast=forecast)
    return write_explorer(payload)
