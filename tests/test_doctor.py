"""Health check output."""

from __future__ import annotations

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
