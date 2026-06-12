"""Parse NBA play-by-play CSVs into ground-truth game records, with a JSONL cache."""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import pandas as pd

FILENAME_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2})\]-(\d+)-([A-Z]{3})@([A-Z]{3})\.csv")

LINEUP_COLS = {"away": ["a1", "a2", "a3", "a4", "a5"], "home": ["h1", "h2", "h3", "h4", "h5"]}


@dataclass
class PlayerLine:
    side: str  # "home" or "away"
    pts: int = 0
    reb: int = 0
    ast: int = 0
    minutes: float = 0.0


@dataclass
class Game:
    game_id: str
    date: str  # YYYY-MM-DD
    away: str  # team abbreviation
    home: str
    away_score: int
    home_score: int
    periods: List[List[int]]  # per period [away_pts, home_pts]
    players: Dict[str, PlayerLine]
    starters: Dict[str, List[str]]  # side -> 5 names

    @property
    def winner(self) -> str:
        return self.home if self.home_score > self.away_score else self.away

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Game":
        d = dict(d)
        d["players"] = {name: PlayerLine(**p) for name, p in d["players"].items()}
        return Game(**d)


def parse_filename(filename: str):
    m = FILENAME_RE.match(os.path.basename(filename))
    if not m:
        raise ValueError(f"Cannot parse filename: {filename}")
    date, game_id, away, home = m.groups()
    return date, game_id, away, home


def _length_to_seconds(value: str) -> int:
    h, m, s = str(value).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_game(path: str) -> Game:
    date, game_id, away, home = parse_filename(path)
    df = pd.read_csv(path)

    away_score = int(df["away_score"].iloc[-1])
    home_score = int(df["home_score"].iloc[-1])

    # Per-period scoring (handles OT periods beyond 4).
    periods = []
    prev_away, prev_home = 0, 0
    for period in sorted(df["period"].unique()):
        last = df[df["period"] == period].iloc[-1]
        a, h = int(last["away_score"]), int(last["home_score"])
        periods.append([a - prev_away, h - prev_home])
        prev_away, prev_home = a, h

    # Which side each player belongs to, from lineup columns.
    side_of: Dict[str, str] = {}
    for side, cols in LINEUP_COLS.items():
        for col in cols:
            for name in df[col].dropna().unique():
                name = str(name).strip()
                if name:
                    side_of[name] = side

    players = {name: PlayerLine(side=side) for name, side in side_of.items()}

    # Minutes: sum of play_length over rows where the player is on the floor.
    seconds = df["play_length"].map(_length_to_seconds)
    for cols in LINEUP_COLS.values():
        for col in cols:
            for name, total in seconds.groupby(df[col]).sum().items():
                name = str(name).strip()
                if name in players:
                    players[name].minutes += total / 60.0

    # Points: the points column is attributed to the scorer for shots and free throws.
    scored = df[df["points"].fillna(0) > 0]
    for name, pts in scored.groupby("player")["points"].sum().items():
        name = str(name).strip()
        if name in players:
            players[name].pts = int(pts)

    rebounds = df[df["event_type"] == "rebound"]
    for name, count in rebounds.groupby("player").size().items():
        name = str(name).strip()
        if name in players:
            players[name].reb = int(count)

    for name, count in df.groupby("assist").size().items():
        name = str(name).strip()
        if name in players:
            players[name].ast = int(count)

    starters = {
        side: [str(df[col].iloc[0]).strip() for col in cols]
        for side, cols in LINEUP_COLS.items()
    }

    for line in players.values():
        line.minutes = round(line.minutes, 1)

    return Game(
        game_id=game_id,
        date=date,
        away=away,
        home=home,
        away_score=away_score,
        home_score=home_score,
        periods=periods,
        players=players,
        starters=starters,
    )


def build_cache(data_dir: str, cache_path: str) -> int:
    """Parse every game CSV in data_dir into a JSONL cache. Returns game count."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    count = 0
    with open(cache_path, "w") as out:
        for filename in files:
            game = parse_game(os.path.join(data_dir, filename))
            out.write(json.dumps(game.to_dict()) + "\n")
            count += 1
    return count


def load_games(cache_path: str) -> List[Game]:
    """Load all games from the cache, sorted by date."""
    games = []
    with open(cache_path) as f:
        for line in f:
            games.append(Game.from_dict(json.loads(line)))
    games.sort(key=lambda g: (g.date, g.game_id))
    return games
