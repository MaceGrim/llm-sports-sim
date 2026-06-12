"""Verify the parser against a known game: PHI @ BOS, 2022-10-18 (final 117-126)."""

import os

import pytest

from sim.games import parse_game, parse_filename

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "nba_data")
SAMPLE = os.path.join(DATA_DIR, "[2022-10-18]-0022200001-PHI@BOS.csv")


@pytest.fixture(scope="module")
def game():
    return parse_game(SAMPLE)


def test_filename_parsing():
    date, game_id, away, home = parse_filename(SAMPLE)
    assert (date, away, home) == ("2022-10-18", "PHI", "BOS")


def test_final_score(game):
    assert game.away_score == 117
    assert game.home_score == 126
    assert game.winner == "BOS"


def test_periods_sum_to_final(game):
    assert sum(p[0] for p in game.periods) == game.away_score
    assert sum(p[1] for p in game.periods) == game.home_score
    assert len(game.periods) == 4  # no OT in this game


def test_player_points_sum_to_final(game):
    away_pts = sum(p.pts for p in game.players.values() if p.side == "away")
    home_pts = sum(p.pts for p in game.players.values() if p.side == "home")
    assert away_pts == game.away_score
    assert home_pts == game.home_score


def test_minutes_total_240_per_team(game):
    # 5 players on the floor for 48 minutes = 240 player-minutes per team.
    for side in ("away", "home"):
        total = sum(p.minutes for p in game.players.values() if p.side == side)
        assert total == pytest.approx(240, abs=2)


def test_starters(game):
    assert "Jayson Tatum" in game.starters["home"]
    assert "Joel Embiid" in game.starters["away"]
    assert len(game.starters["home"]) == 5
    assert len(game.starters["away"]) == 5


def test_known_player_line(game):
    # Embiid's actual line that night: 26 pts, 15 reb, 5 ast.
    embiid = game.players["Joel Embiid"]
    assert embiid.side == "away"
    assert embiid.pts == 26
    assert embiid.reb == 15
    assert embiid.ast == 5
    assert 30 < embiid.minutes < 42
