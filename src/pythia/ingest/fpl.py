"""FPL availability snapshots: squad news, collected forward in time.

The research question this module exists to answer is *whether* squad
availability can be used as a pre-match feature.  It cannot be answered from
today's API: ``bootstrap-static`` describes the present squad, and the per-player
history returned by ``element-summary`` carries minutes but no availability flag.
There is no free historical series, so the feature cannot be backtested.

What is possible is to start collecting: one small snapshot per refresh, stored
in the repository, so that in a season or two the question becomes testable with
a real out-of-sample period.  Until then the data is displayed, never fed to a
model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from ..config import DATA_DIR
from ..teams import team_key
from .fixtures_feed import UK_TZ

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
SNAPSHOT_DIR = DATA_DIR / "fpl"
RAW_JSON = DATA_DIR / "raw" / "fpl_bootstrap.json"

#: FPL status codes -> human label.
STATUS_LABELS = {"a": "available", "d": "doubtful", "i": "injured", "s": "suspended", "u": "unavailable"}


def fetch_bootstrap(*, timeout: int = 30) -> dict:
    resp = requests.get(BOOTSTRAP_URL, timeout=timeout, headers={"User-Agent": "pythia/0.1"})
    resp.raise_for_status()
    payload = resp.json()
    RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
    RAW_JSON.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def current_gameweek(bootstrap: dict) -> int | None:
    for event in bootstrap.get("events", []):
        if event.get("is_current"):
            return int(event["id"])
    for event in bootstrap.get("events", []):
        if event.get("is_next"):
            return int(event["id"])
    return None


def fetch_fixtures(bootstrap: dict | None = None, *, timeout: int = 30) -> pd.DataFrame:
    """The full season schedule, with kick-offs converted to UK local time.

    Football-data's ``fixtures.csv`` is the natural source for prices, but it is
    not a reliable schedule: it lists whichever divisions are playing in the next
    few days, and the Premier League can be absent for weeks at a time. The FPL
    endpoint always carries the whole season, so it is used for the calendar and
    the prices are joined on where they exist.
    """
    bootstrap = fetch_bootstrap() if bootstrap is None else bootstrap

    resp = requests.get(FIXTURES_URL, timeout=timeout, headers={"User-Agent": "pythia/0.1"})
    resp.raise_for_status()
    raw = pd.DataFrame(resp.json())

    teams = {t["id"]: team_key(t["name"]) for t in bootstrap.get("teams", [])}
    frame = pd.DataFrame({
        "event": pd.to_numeric(raw.get("event"), errors="coerce"),
        "kickoff": pd.to_datetime(raw["kickoff_time"], errors="coerce", utc=True).dt.tz_convert(UK_TZ),
        "home": raw["team_h"].map(teams),
        "away": raw["team_a"].map(teams),
        "finished": raw["finished"].astype(bool),
        "started": raw.get("started", False).astype(bool),
    })
    return frame.dropna(subset=["kickoff", "home", "away"]).sort_values("kickoff").reset_index(drop=True)


def next_round_fixtures(fixtures: pd.DataFrame, *, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """The earliest gameweek that still has an unfinished match."""
    if fixtures.empty:
        return fixtures
    pending = fixtures.loc[~fixtures["finished"]]
    if pending.empty:
        return pending
    event = pending["event"].min()
    block = pending.loc[pending["event"] == event].copy()
    return block.sort_values("kickoff").reset_index(drop=True)


def players_frame(bootstrap: dict) -> pd.DataFrame:
    """One row per listed player with availability and season minutes."""
    teams = {t["id"]: team_key(t["name"]) for t in bootstrap.get("teams", [])}
    rows = []
    for element in bootstrap.get("elements", []):
        rows.append({
            "player": element.get("web_name"),
            "team": teams.get(element.get("team")),
            "position": element.get("element_type"),
            "status": element.get("status"),
            "chance": element.get("chance_of_playing_next_round"),
            "minutes": element.get("minutes", 0) or 0,
            "cost": (element.get("now_cost") or 0) / 10.0,
            "news": (element.get("news") or "").strip(),
        })
    frame = pd.DataFrame(rows)
    frame["status_label"] = frame["status"].map(STATUS_LABELS).fillna("unknown")
    return frame


def team_availability(bootstrap: dict) -> pd.DataFrame:
    """Minutes-weighted share of each squad that is currently selectable.

    The weight is season minutes, i.e. "how much of this team's actual football
    would go missing".  It is a heuristic, not a model — and it is deliberately
    kept out of the forecast until it can be validated.
    """
    players = players_frame(bootstrap)
    if players.empty:
        return pd.DataFrame()

    players["selectable"] = (players["status"] == "a") | (
        (players["status"] == "d") & (players["chance"].fillna(0) >= 75)
    )

    rows = []
    for team, block in players.groupby("team"):
        minutes = float(block["minutes"].sum())
        available_minutes = float(block.loc[block["selectable"], "minutes"].sum())
        rows.append({
            "team": team,
            "n_players": len(block),
            "n_injured": int((block["status"] == "i").sum()),
            "n_doubtful": int((block["status"] == "d").sum()),
            "n_suspended": int((block["status"] == "s").sum()),
            "n_unavailable": int((block["status"] == "u").sum()),
            "minutes_total": minutes,
            "minutes_available": available_minutes,
            "avail_score": available_minutes / minutes if minutes > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("avail_score").reset_index(drop=True)


def save_snapshot(bootstrap: dict, *, when: pd.Timestamp | None = None) -> Path:
    """Append today's team-level availability to the committed snapshot history."""
    when = pd.Timestamp.now().normalize() if when is None else pd.Timestamp(when).normalize()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    table = team_availability(bootstrap)
    table.insert(0, "date", when)
    table.insert(1, "gameweek", current_gameweek(bootstrap))
    path = SNAPSHOT_DIR / f"availability_{when:%Y-%m-%d}.parquet"
    table.to_parquet(path, index=False)
    return path


def load_history() -> pd.DataFrame:
    """Every snapshot ever committed, oldest first."""
    if not SNAPSHOT_DIR.exists():
        return pd.DataFrame()
    paths = sorted(SNAPSHOT_DIR.glob("availability_*.parquet"))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True).sort_values(["date", "team"]).reset_index(drop=True)
