"""Data health check: are the sources alive and are the caches current?

Weekly automation is only as good as its inputs.  This command answers "is
anything silently broken?" — a source that stops updating, a cache that went
missing, a feature block that quietly turned all-NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import FEATURES_PARQUET, MATCHES_PARQUET, PREDICTIONS_PARQUET, ROOT, season_boundaries

STALE_DAYS = 14


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str


def _age_days(latest: pd.Timestamp, today: pd.Timestamp) -> int:
    return int((today - pd.Timestamp(latest)).days)


def run_checks(today: pd.Timestamp | None = None) -> list[Check]:
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    checks: list[Check] = []

    # 1. canonical matches
    if not MATCHES_PARQUET.exists():
        checks.append(Check("matches cache", "fail", "data/processed/matches.parquet missing — run `pythia build`"))
        matches = None
    else:
        matches = pd.read_parquet(MATCHES_PARQUET)
        latest = matches["date"].max()
        age = _age_days(latest, today)
        status = "ok" if age <= STALE_DAYS else "warn"
        checks.append(Check(
            "matches cache", status,
            f"{len(matches):,} matches, {matches['season'].nunique()} seasons, latest {latest:%Y-%m-%d} ({age}d ago)",
        ))

    # 2. feature table
    if not FEATURES_PARQUET.exists():
        checks.append(Check("feature table", "fail", "features.parquet missing — run `pythia build`"))
        features = None
    else:
        features = pd.read_parquet(FEATURES_PARQUET)
        null_ratio = float(features[["elo_h", "elo_a"]].isna().mean().max())
        status = "ok" if null_ratio < 0.01 else "warn"
        checks.append(Check("feature table", status,
                            f"{features.shape[1]} columns, {null_ratio:.1%} missing Elo"))

    # 3. optional blocks
    if features is not None:
        xg_ready = all(c in features.columns and features[c].notna().mean() > 0.9
                       for c in ("home_xg_form", "away_xg_form"))
        checks.append(Check("xG block", "ok" if xg_ready else "warn",
                            "present" if xg_ready else 'absent — `pip install -e ".[xg]"` then `pythia fetch --xg`'))

        ou_ready = "ou_over" in features.columns and features["ou_over"].notna().mean() > 0.5
        checks.append(Check("over/under odds", "ok" if ou_ready else "warn",
                            "present (2019-20 onwards)" if ou_ready else "missing — rebuild with --force"))

    # 4. predictions cache
    if not PREDICTIONS_PARQUET.exists():
        checks.append(Check("backtest predictions", "warn", "missing — run `pythia backtest`"))
    else:
        preds = pd.read_parquet(PREDICTIONS_PARQUET)
        checks.append(Check("backtest predictions", "ok",
                            f"{len(preds):,} rows, {preds['season'].nunique()} seasons, "
                            f"{preds['model'].nunique()} models"))

    # 5. season freshness against the calendar
    if matches is not None:
        current = str(matches["season"].max())
        start, end = season_boundaries(current)
        upcoming = (start <= today < end)
        checks.append(Check("season clock", "ok" if upcoming else "warn",
                            f"{current} runs {start:%Y-%m-%d} to {end:%Y-%m-%d}"))

    # 6. FPL snapshots (committed history)
    from .ingest import fpl

    history = fpl.load_history()
    if history.empty:
        checks.append(Check("FPL snapshots", "warn", "none yet — `pythia fpl --snapshot`"))
    else:
        latest = pd.to_datetime(history["date"]).max()
        age = _age_days(latest, today)
        checks.append(Check("FPL snapshots", "ok" if age <= STALE_DAYS else "warn",
                            f"{len(history)} rows, {history['date'].nunique()} dates, latest {latest:%Y-%m-%d}"))

    # 7. raw fixtures feed
    from .ingest.fixtures_feed import FIXTURE_PATH

    if FIXTURE_PATH.exists():
        checks.append(Check("fixture feed", "ok",
                            f"{FIXTURE_PATH.stat().st_size / 1024:.0f} KB, "
                            f"updated {pd.Timestamp(FIXTURE_PATH.stat().st_mtime, unit='s'):%Y-%m-%d}"))
    else:
        checks.append(Check("fixture feed", "warn", "missing — `pythia fetch`"))

    return checks


def format_checks(checks: list[Check]) -> str:
    icons = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}
    width = max(len(c.name) for c in checks) if checks else 10
    lines = [f"{'':4}  {'check':<{width}}  detail", f"{'':4}  {'-' * width}  {'-' * 40}"]
    for check in checks:
        lines.append(f"{icons.get(check.status, '?   ')}  {check.name:<{width}}  {check.detail}")
    worst = "fail" if any(c.status == "fail" for c in checks) else (
        "warn" if any(c.status == "warn" for c in checks) else "ok")
    lines.append("")
    lines.append({"ok": "all checks passed", "warn": "passed with warnings",
                  "fail": "failures need attention"}[worst])
    return "\n".join(lines)


def has_failures(checks: list[Check]) -> bool:
    return any(c.status == "fail" for c in checks)


def repo_root() -> str:
    return str(ROOT)
