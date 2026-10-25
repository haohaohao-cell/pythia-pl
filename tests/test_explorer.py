"""The interactive explorer: payload completeness and template rendering."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pythia.explorer import build_payload, render_html

PRED_COLUMNS = ["match_id", "season", "model", "p_home", "p_draw", "p_away", "ftr"]


@pytest.fixture
def payload(features):
    empty = pd.DataFrame(columns=PRED_COLUMNS)
    return build_payload(features, empty)


def test_every_ordered_pair_is_present(payload) -> None:
    n = len(payload["teams"])
    assert n >= 4
    assert len(payload["pairs"]) == n * (n - 1)
    assert "alpha|bravo" in payload["pairs"]


def test_probabilities_are_distributions(payload) -> None:
    for key, probs in payload["pairs"].items():
        assert len(probs) == 3, key
        assert abs(sum(probs) - 1.0) < 0.005, key
        assert all(0.0 <= p <= 1.0 for p in probs), key


def test_teams_carry_model_parameters(payload) -> None:
    for team in payload["teams"]:
        assert set(team) >= {"key", "en", "zh", "elo", "attack", "defence"}
        assert team["en"] and team["zh"]


def test_dc_block_has_what_the_browser_needs(payload) -> None:
    dc = payload["dc"]
    assert dc["max_goals"] > dc["display_goals"] >= 3
    assert -0.3 < dc["rho"] < 0.3
    assert 0 < dc["home_adv"] < 1.5


def test_render_replaces_the_placeholder(payload) -> None:
    html = render_html(payload)
    assert "__PAYLOAD__" not in html
    assert html.count("<script>") == 1
    assert "PYTHIA" in html
    # The JSON must survive round-tripping inside the page.
    start = html.index("const P = ") + len("const P = ")
    end = html.index(";", start)
    assert json.loads(html[start:end])["season"] == payload["season"]


def test_payload_is_compact(payload) -> None:
    """The page is a single file served from Pages; keep it small."""
    assert len(render_html(payload)) < 400_000


def test_no_market_is_fine(features) -> None:
    payload = build_payload(features, pd.DataFrame(columns=PRED_COLUMNS), forecast=None)
    assert payload["market"] == {}
