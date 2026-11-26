"""Health check output."""

from __future__ import annotations

import pandas as pd
import pytest

from pythia import doctor
from pythia.doctor import Check, format_checks, has_failures, run_checks

VALID = {"ok", "warn", "fail"}


def test_checks_return_valid_statuses() -> None:
    checks = run_checks()
    assert checks, "doctor should always produce at least one check"
    assert all(c.status in VALID for c in checks)
    assert all(c.name and c.detail for c in checks)


def test_format_aligns_and_summarises() -> None:
    text = format_checks([Check("a", "ok", "fine"), Check("bb", "warn", "careful")])
    assert "ok" in text and "WARN" in text
    assert "warnings" in text


def test_failures_are_loud() -> None:
    text = format_checks([Check("a", "fail", "broken")])
    assert "FAIL" in text
    assert "need attention" in text


def test_has_failures() -> None:
    assert has_failures([Check("a", "warn", "")]) is False
    assert has_failures([Check("a", "fail", "")]) is True


def test_empty_list_is_safe() -> None:
    assert format_checks([])
    assert has_failures([]) is False


def _round_at(tmp_path, when: str):
    (tmp_path / "fixtures_forecast.csv").write_text(
        "kickoff,home,away\n" + when + ",alpha,bravo\n", encoding="utf-8"
    )
    return tmp_path


def _published_check(checks):
    return next(c for c in checks if c.name == "published round")


def test_future_round_is_ok(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "REPORTS_DIR", _round_at(tmp_path, "2030-01-01T15:00:00Z"))
    check = _published_check(run_checks(today=pd.Timestamp("2029-12-30")))
    assert check.status == "ok"
    assert "in the future" in check.detail


def test_past_round_warns(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "REPORTS_DIR", _round_at(tmp_path, "2026-10-25T15:00:00Z"))
    check = _published_check(run_checks(today=pd.Timestamp("2026-11-26")))
    assert check.status == "warn"
    assert "past" in check.detail


def test_missing_round_warns(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "REPORTS_DIR", tmp_path)
    check = _published_check(run_checks())
    assert check.status == "warn"
