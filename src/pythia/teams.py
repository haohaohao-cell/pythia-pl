"""Canonical team identifiers.

football-data.co.uk ships names such as ``Man United``, ``Nott'm Forest`` and
``Sheffield United``.  Any join across sources (FBref, Understat, ClubElo) dies
on those variations, so every name used by the project goes through this table.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    key: str        # canonical id used in every table
    name_en: str    # display name (English)
    name_zh: str    # display name (Chinese), used by the dashboard
    aliases: tuple[str, ...] = ()


def _t(key: str, en: str, zh: str, *aliases: str) -> Team:
    return Team(key=key, name_en=en, name_zh=zh, aliases=aliases)


#: Premier League clubs that appear in football-data.co.uk E0 files, 2014-2026.
TEAMS: tuple[Team, ...] = (
    _t("arsenal", "Arsenal", "阿森纳"),
    _t("aston_villa", "Aston Villa", "阿斯顿维拉"),
    _t("bournemouth", "Bournemouth", "伯恩茅斯"),
    _t("brentford", "Brentford", "布伦特福德"),
    _t("brighton", "Brighton", "布莱顿"),
    _t("burnley", "Burnley", "伯恩利"),
    _t("cardiff", "Cardiff", "卡迪夫城"),
    _t("chelsea", "Chelsea", "切尔西"),
    _t("coventry", "Coventry", "考文垂", "Coventry City"),
    _t("crystal_palace", "Crystal Palace", "水晶宫"),
    _t("everton", "Everton", "埃弗顿"),
    _t("fulham", "Fulham", "富勒姆"),
    _t("huddersfield", "Huddersfield", "哈德斯菲尔德"),
    _t("hull", "Hull", "赫尔城"),
    _t("ipswich", "Ipswich", "伊普斯维奇"),
    _t("leeds", "Leeds", "利兹联"),
    _t("leicester", "Leicester", "莱斯特城"),
    _t("liverpool", "Liverpool", "利物浦"),
    _t("luton", "Luton", "卢顿"),
    _t("man_city", "Man City", "曼城", "Manchester City"),
    _t("man_united", "Man United", "曼联", "Manchester United", "Man Utd"),
    _t("middlesbrough", "Middlesbrough", "米德尔斯堡"),
    _t("newcastle", "Newcastle", "纽卡斯尔联", "Newcastle United"),
    _t("norwich", "Norwich", "诺维奇"),
    _t("nottm_forest", "Nott'm Forest", "诺丁汉森林", "Nottingham Forest"),
    _t("qpr", "QPR", "女王公园巡游者", "Queens Park Rangers"),
    _t("sheffield_united", "Sheffield United", "谢菲尔德联", "Sheff United"),
    _t("southampton", "Southampton", "南安普顿"),
    _t("stoke", "Stoke", "斯托克城"),
    _t("sunderland", "Sunderland", "桑德兰"),
    _t("swansea", "Swansea", "斯旺西"),
    _t("tottenham", "Tottenham", "托特纳姆热刺", "Spurs"),
    _t("watford", "Watford", "沃特福德"),
    _t("west_brom", "West Brom", "西布罗姆维奇", "West Bromwich Albion"),
    _t("west_ham", "West Ham", "西汉姆联"),
    _t("wigan", "Wigan", "维冈竞技"),
    _t("wolves", "Wolves", "狼队", "Wolverhampton Wanderers"),
)


def _norm(name: str) -> str:
    """Lowercase, strip accents/punctuation so lookups are forgiving."""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return "".join(ch for ch in name.lower() if ch.isalnum())


#: Lookup table: normalised alias -> Team.
_LOOKUP: dict[str, Team] = {}
for _team in TEAMS:
    for _alias in (_team.key, _team.name_en, _team.name_zh, *_team.aliases):
        _LOOKUP[_norm(_alias)] = _team

_ZH: dict[str, str] = {t.key: t.name_zh for t in TEAMS}
_EN: dict[str, str] = {t.key: t.name_en for t in TEAMS}


def team_key(name: str) -> str:
    """Map any known spelling to the canonical key; unknown names pass through.

    Unknown names are normalised (never dropped) so a newly promoted club cannot
    silently vanish from the dataset.
    """
    team = _LOOKUP.get(_norm(name))
    return team.key if team else _norm(name)


def zh_name(key: str) -> str:
    return _ZH.get(key, key)


def en_name(key: str) -> str:
    return _EN.get(key, key)


def known_teams() -> list[Team]:
    return list(TEAMS)
