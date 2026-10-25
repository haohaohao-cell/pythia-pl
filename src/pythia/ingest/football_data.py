"""Downloading and normalising Premier League results from football-data.co.uk.

The site publishes one CSV per season (``mmz4281/2425/E0.csv``).  Column names
are *not* stable across seasons — most famously Pinnacle is ``PS*`` up to
2024-25 and ``PP*`` from 2025-26 — so odds columns are resolved per season
instead of being hard-coded.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from ..config import (
    BASE_URL,
    DIVISION,
    MIN_ODDS_COVERAGE,
    RAW_DIR,
    SEASON_CODES,
    season_code,
    season_label,
)
from ..teams import team_key

#: Preferred bookmaker families, best first.  Pinnacle closing is the sharpest
#: publicly available line and therefore the benchmark of record.
CLOSING_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("PSCH", "PSCD", "PSCA"),
    ("PPCH", "PPCD", "PPCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("AvgCH", "AvgCD", "AvgCA"),
)
OPENING_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("PSH", "PSD", "PSA"),
    ("PPH", "PPD", "PPA"),
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
)

#: Over/under 2.5 goals. football-data only ships these from 2019-20 onwards.
OU_FAMILIES: tuple[tuple[str, str], ...] = (
    ("B365C>2.5", "B365C<2.5"),
    ("B365>2.5", "B365<2.5"),
    ("AvgC>2.5", "AvgC<2.5"),
    ("Avg>2.5", "Avg<2.5"),
)

#: Canonical columns kept by :func:`normalise`.
_FLOAT_COLS = ("fthg", "ftag", "hthg", "htag", "hs", "as_", "hst", "ast", "hxg", "axg")


def season_path(code: str) -> Path:
    return RAW_DIR / f"E0_{code}.csv"


def download_season(code: str, *, force: bool = False, timeout: int = 60) -> Path:
    """Fetch one season CSV, skipping the request when a cached copy exists."""
    path = season_path(code)
    if path.exists() and not force and path.stat().st_size > 10_000:
        return path

    url = BASE_URL.format(code=code)
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "pythia-pl/0.1"})
            resp.raise_for_status()
            # The payload is UTF-8 with a BOM; decode explicitly and write UTF-8.
            text = resp.content.decode("utf-8-sig")
            if "HomeTeam" not in text.splitlines()[0]:
                raise ValueError(f"unexpected header for {code}")
            path.write_text(text, encoding="utf-8")
            return path
        except Exception as exc:  # noqa: BLE001 - retried below
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not download season {code}: {last_err}")


def _parse_dates(values: pd.Series) -> pd.Series:
    """football-data uses dd/mm/yyyy in recent seasons and dd/mm/yy before."""
    raw = values.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        todo = out.isna()
        if not todo.any():
            break
        parsed = pd.to_datetime(raw[todo], format=fmt, errors="coerce")
        out.loc[todo] = parsed
    todo = out.isna()
    if todo.any():
        out.loc[todo] = pd.to_datetime(raw[todo], dayfirst=True, errors="coerce")
    return out


def _pick_families(df: pd.DataFrame, families, coverage: float):
    """Return the first odds family whose columns are sufficiently complete."""
    for family in families:
        cols = list(family)
        if all(col in df.columns for col in cols):
            complete = df[cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
            if complete.mean() >= coverage:
                return tuple(cols)
    return None


def normalise(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Map a raw season frame onto the canonical match schema."""
    df = df.loc[df["Div"].astype(str).str.strip() == DIVISION].copy()
    df = df.loc[df["FTHG"].notna() & df["FTAG"].notna()].copy()

    out = pd.DataFrame(index=df.index)
    out["date"] = _parse_dates(df["Date"])
    out.loc[out["date"].isna(), "date"] = pd.NaT
    out["home_raw"] = df["HomeTeam"].astype(str).str.strip()
    out["away_raw"] = df["AwayTeam"].astype(str).str.strip()
    out["home"] = out["home_raw"].map(team_key)
    out["away"] = out["away_raw"].map(team_key)
    out["fthg"] = pd.to_numeric(df["FTHG"], errors="coerce").astype("Int64")
    out["ftag"] = pd.to_numeric(df["FTAG"], errors="coerce").astype("Int64")

    for src, dst in (
        ("HTHG", "hthg"), ("HTAG", "htag"),
        ("HS", "hs"), ("AS", "as_"), ("HST", "hst"), ("AST", "ast"),
        ("HF", "hf"), ("AF", "af"), ("HC", "hc"), ("AC", "ac"),
        ("HxG", "hxg"), ("AxG", "axg"),
    ):
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else float("nan")

    for name, families in (("close", CLOSING_FAMILIES), ("open", OPENING_FAMILIES)):
        picked = _pick_families(df, families, MIN_ODDS_COVERAGE)
        for slot, side in enumerate(("h", "d", "a")):
            col = f"odds_{name}_{side}"
            if picked is None:
                out[col] = float("nan")
            else:
                out[col] = pd.to_numeric(df[picked[slot]], errors="coerce")
        out[f"odds_{name}_src"] = "+".join(picked) if picked else ""

    picked_ou = _pick_families(df, OU_FAMILIES, MIN_ODDS_COVERAGE)
    for slot, side in enumerate(("over", "under")):
        out[f"ou_{side}"] = (pd.to_numeric(df[picked_ou[slot]], errors="coerce")
                             if picked_ou else float("nan"))
    out["ou_src"] = "+".join(picked_ou) if picked_ou else ""

    out["season"] = season
    out = out.dropna(subset=["date", "home", "away", "fthg", "ftag"])
    out = out.sort_values(["date", "home"]).reset_index(drop=True)
    out["match_id"] = [f"{season}:{i:03d}" for i in range(len(out))]
    return out


def read_season(code: str) -> pd.DataFrame:
    path = season_path(code)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `pythia fetch` first")
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig", on_bad_lines="skip")
    raw.columns = [c.strip() for c in raw.columns]
    return normalise(raw, season_label(int(code[:2]) + 2000))


def build_matches(*, force: bool = False) -> pd.DataFrame:
    """Download (or reuse) every season and concatenate canonical match rows."""
    frames = []
    for season, code in SEASON_CODES.items():
        download_season(code, force=force)
        frames.append(read_season(code))
    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values(["date", "home"]).reset_index(drop=True)
    matches["match_id"] = (
        matches["season"].astype(str) + "-" + matches.groupby("season").cumcount().astype(str).str.zfill(3)
    )
    matches["ftr"] = matches.apply(
        lambda r: "H" if r.fthg > r.ftag else ("A" if r.fthg < r.ftag else "D"), axis=1
    )
    return matches
