"""Grammar and replay tests: a handcrafted sequence exercising every
mechanic (no data needed), plus a pinned real game when the parquet and
name cache are present."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sim.tokenizer import Replay, build_vocab, encode_game  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "..", "statcast_2024.parquet")
PLAYERS = os.path.join(HERE, "..", "cache", "players.json")

# Top 1: strikeout, then a 2-run homer, then a balk-type run ([MID]).
# Bot 1: a PA with a mid-PA pitching change AND a pinch hitter, ending in an
# RBI single — charged to the new pitcher and the new batter.
MINI = [
    "[GAME]", "TEAM:AAA", "TEAM:HHH",
    "[LINEUP_A]", "P:A1", "P:A2", "[LINEUP_H]", "P:H1",
    "[BENCH_A]", "[BENCH_H]", "P:H2",
    "[PEN_A]", "P:Ap", "P:Ap2", "[PEN_H]", "P:Hp",
    "[HALF]",
    "[PA]", "P:Hp", "P:A1",
    "T:FF", "Z:5", "R:swinging_strike", "E:strikeout",
    "[PA]", "P:Hp", "P:A2",
    "T:FF", "Z:unk", "R:hit_into_play", "E:home_run", "+2",
    "[MID]", "+1",
    "[HALF]",
    "[PA]", "P:Ap", "P:H1",
    "T:SL", "Z:3", "R:foul",
    "[NEWP]", "P:Ap2",
    "[NEWB]", "P:H2",
    "T:CH", "Z:9", "R:hit_into_play", "E:single", "+1",
    "[EOG]",
]


def test_mini_replay_score_and_halves():
    r = Replay(MINI).run()
    assert (r.away_score, r.home_score) == (3, 1)
    assert r.half_runs == [3, 1]
    assert (r.away_team, r.home_team) == ("AAA", "HHH")


def test_mini_batter_lines():
    r = Replay(MINI).run()
    assert r.bat["A1"] == {"pa": 1, "h": 0, "hr": 0, "bb": 0, "k": 1}
    assert r.bat["A2"] == {"pa": 1, "h": 1, "hr": 1, "bb": 0, "k": 0}
    # the PA belongs to the pinch hitter, not the starter he replaced
    assert r.bat["H2"] == {"pa": 1, "h": 1, "hr": 0, "bb": 0, "k": 0}
    assert "H1" not in r.bat


def test_mini_pitcher_lines():
    r = Replay(MINI).run()
    assert r.arm["Hp"] == {"bf": 2, "h": 1, "bb": 0, "k": 1}
    # the PA belongs to the relief pitcher, not the one he replaced
    assert r.arm["Ap2"] == {"bf": 1, "h": 1, "bb": 0, "k": 0}
    assert "Ap" not in r.arm


def test_unexpected_token_raises():
    bad = MINI[:17] + ["dt:5"] + MINI[17:]
    with pytest.raises(ValueError, match="unexpected token"):
        Replay(bad).run()


def test_vocab_padded_and_specials_first():
    vocab = build_vocab([MINI])
    assert vocab[0] == "[PAD]"
    assert vocab.index("[GAME]") < vocab.index("+1")
    for tok in ("Z:14", "+4", "T:KN", "T:unk", "Z:unk"):
        assert tok in vocab  # programmatic padding, even if unseen
    assert len(vocab) == len(set(vocab))


@pytest.mark.skipif(not (os.path.exists(PARQUET) and os.path.exists(PLAYERS)),
                    reason="needs statcast_2024.parquet and cache/players.json")
def test_round_trip_pinned_game():
    """Game 747004 (WSH@BAL): the 2024 game where a run scores between the
    last pitch of Top 1 and the first pitch of Bot 1 — the [MID] straddle."""
    from run import game_truth
    from sim.data import load_season, player_names

    df = load_season(PARQUET)
    g = df[df.game_pk == 747004]
    names = player_names(df, PLAYERS)

    tokens = encode_game(g, names)
    assert tokens[0] == "[GAME]" and tokens[-1] == "[EOG]"
    assert "[MID]" in tokens

    replay = Replay(tokens).run()
    final, half_runs, bat, arm = game_truth(g, names)
    assert (replay.away_score, replay.home_score) == final == (9, 3)
    assert replay.half_runs == half_runs and len(half_runs) == 18
    assert dict(replay.bat) == bat
    assert dict(replay.arm) == arm
