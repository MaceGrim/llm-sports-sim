"""Statcast data loading and the MLB-ID -> name map.

The season parquet (statcast_2024.parquet) is pulled once via pybaseball
(see DESIGN.md). Rows are pitches; load_season() returns them chronologically
sorted within each game, iter_games() yields one frame per game.
"""

import json
import os
from collections import Counter

import pandas as pd

COLS = ["game_pk", "game_date", "game_type", "away_team", "home_team",
        "inning", "inning_topbot", "at_bat_number", "pitch_number",
        "batter", "pitcher", "pitch_type", "description", "events",
        "home_score", "away_score", "post_home_score", "post_away_score",
        "release_speed", "release_spin_rate", "bb_type", "launch_speed",
        "launch_angle", "hc_x", "hc_y", "plate_x", "plate_z",
        "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b"]


def load_season(parquet_path: str) -> pd.DataFrame:
    """Regular-season pitches, chronologically sorted within each game.

    (game_pk, at_bat_number, pitch_number) is unique and sorting by it
    preserves half-inning order — both verified on the full 2024 season."""
    df = pd.read_parquet(parquet_path, columns=COLS)
    df = df[df.game_type == "R"]
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"])


def iter_games(df: pd.DataFrame):
    yield from df.groupby("game_pk", sort=True)


def player_names(df: pd.DataFrame, cache_path: str) -> dict:
    """MLB ID -> readable name for every batter/pitcher in df, cached as JSON.

    Built via pybaseball's Chadwick-register lookup (network on first run).
    Colliding names (five pairs in 2024 — two Will Smiths, two Luis Ortizes,
    ...) get an ID suffix so one token never means two players."""
    ids = sorted(set(df.batter) | set(df.pitcher))
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            names = {int(k): v for k, v in json.load(f).items()}
        missing = set(ids) - set(names)
        if missing:
            raise ValueError(f"{cache_path} is missing {len(missing)} player "
                             f"IDs (new data?) — delete it to rebuild")
        return names

    from pybaseball import playerid_reverse_lookup
    lookup = playerid_reverse_lookup(ids, key_type="mlbam")
    unresolved = set(ids) - set(lookup.key_mlbam)
    if unresolved:
        raise ValueError(f"IDs with no Chadwick entry: {sorted(unresolved)}")
    names = {int(r.key_mlbam): f"{r.name_first} {r.name_last}".title()
             for r in lookup.itertuples()}
    dupes = {n for n, k in Counter(names.values()).items() if k > 1}
    names = {i: f"{n} ({i})" if n in dupes else n for i, n in names.items()}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(names, f, indent=0)
    return names
