"""Season ledger aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pythia.eval.ledger import build_ledger, ledger_markdown


def _preds() -> pd.DataFrame:
    rows = []
    for i in range(20):
        outcome = ["H", "D", "A"][i % 3]
        date = pd.Timestamp("2026-08-10") + pd.Timedelta(days=7 * (i // 5) + i % 5)
        for model, probs in (
            ("blend", [0.5, 0.3, 0.2]),
            ("market", [0.45, 0.25, 0.30]),
        ):
            rows.append({
                "match_id": f"m{i}",
                "season": "2026-27",
                "date": date,
                "home": "alpha",
                "away": "bravo",
                "ftr": outcome,
                "model": model,
                "p_home": probs[0], "p_draw": probs[1], "p_away": probs[2],
            })
    return pd.DataFrame(rows)


def test_ledger_has_one_row_per_week() -> None:
    ledger = build_ledger(_preds(), season="2026-27")
    assert len(ledger) == 4
    assert ledger["n"].sum() == 20


def test_hits_are_counted_per_model() -> None:
    ledger = build_ledger(_preds(), season="2026-27")
    # blend always predicts H; market always predicts H as well (0.45 is the max)
    assert ledger["model_hits"].sum() == ledger["market_hits"].sum()


def test_running_totals_are_cumulative() -> None:
    ledger = build_ledger(_preds(), season="2026-27")
    assert ledger["n_cum"].is_monotonic_increasing
    assert ledger["n_cum"].iloc[-1] == 20
    assert ledger["model_ll_cum"].iloc[-1] == pytest.approx(ledger["model_ll"].mul(ledger["n"]).sum() / 20)


def test_running_log_loss_matches_manual_computation() -> None:
    preds = _preds()
    ledger = build_ledger(preds, season="2026-27")
    blend = preds.loc[preds["model"] == "blend"]
    y = blend["ftr"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    probs = blend[["p_home", "p_draw", "p_away"]].to_numpy()
    expected = float(-np.log(probs[np.arange(len(y)), y]).mean())
    assert ledger["model_ll_cum"].iloc[-1] == pytest.approx(expected, abs=1e-9)


def test_unknown_season_is_empty() -> None:
    assert build_ledger(_preds(), season="1999-00").empty


def test_markdown_reports_both_models() -> None:
    text = ledger_markdown(build_ledger(_preds(), season="2026-27"), "2026-27")
    assert "2026-27" in text
    assert "log loss" in text
    assert text.count("|") > 20


def test_markdown_is_bilingual() -> None:
    ledger = build_ledger(_preds(), season="2026-27")
    assert "赛季记账" in ledger_markdown(ledger, "2026-27", lang="zh")
    assert "ledger" in ledger_markdown(ledger, "2026-27", lang="en")


def test_empty_ledger_markdown_is_safe() -> None:
    assert ledger_markdown(pd.DataFrame(), "2026-27")
