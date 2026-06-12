"""Grammar and replay tests: a handcrafted sequence exercising every
mechanic (no data needed), plus a pinned real game when the parquet and
name cache are present."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sim.tokenizer import (Replay, bucket, build_vocab,  # noqa: E402
                           encode_game, spray_degrees)

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "..", "statcast_2024.parquet")
PLAYERS = os.path.join(HERE, "..", "cache", "players.json")

# Top 1: strikeout, then a 2-run homer, then a balk-type run ([MID]).
# Bot 1: a PA with a mid-PA pitching change AND a pinch hitter, ending in an
# RBI single — charged to the new pitcher and the new batter.
MINI = [
    "[GAME]", "TEAM:AAA", "TEAM:HHH", "PARK:HHH", "MONTH:6",
    "[LINEUP_A]", "P:A1", "P:A2", "[LINEUP_H]", "P:H1",
    "[BENCH_A]", "[BENCH_H]", "P:H2",
    "[PEN_A]", "P:Ap", "P:Ap2", "[PEN_H]", "P:Hp",
    "[HALF]",
    "[PA]", "P:Hp", "P:A1",
    "T:FF", "V:95", "S:2300", "PX:0", "PZ:24", "R:swinging_strike",
    "E:strikeout", "O:1", "B:000",
    "[PA]", "P:Hp", "P:A2",
    "T:FF", "V:91", "S:unk", "PX:-3", "PZ:30", "R:ball",
    "B:100",                          # ball four would walk him; call it a steal
    "T:FF", "V:91", "S:unk", "PX:unk", "PZ:unk", "R:hit_into_play",
    "BB:fly_ball", "EV:104", "LA:25", "SP:-20", "E:home_run", "+2", "O:2",
    "[MID]", "+1",
    "[HALF]",
    "[PA]", "P:Ap", "P:H1",
    "T:SL", "V:84", "S:2600", "PX:12", "PZ:18", "R:foul",
    "[NEWP]", "P:Ap2",
    "[NEWB]", "P:H2",
    "T:CH", "V:87", "S:1700", "PX:-6", "PZ:12", "R:hit_into_play",
    "BB:ground_ball", "EV:92", "LA:-5", "SP:10", "E:single", "+1",
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


def test_mini_state_trace():
    r = Replay(MINI).run()
    # (balls, strikes, outs, bases) before each pitch: count from R:,
    # outs from O:, bases from B:, all reset at [HALF]/[PA]
    assert r.pitch_state == [
        (0, 0, 0, "000"),  # A1's only pitch
        (0, 0, 1, "000"),  # A2 first pitch, one out
        (1, 0, 1, "100"),  # ball + the B:100 steal landed
        (0, 0, 0, "000"),  # Bot 1 resets outs/bases; H1 fresh count
        (0, 1, 0, "000"),  # foul made it 0-1; count survives NEWP/NEWB
    ]


def test_unexpected_token_raises():
    k = MINI.index("[HALF]") + 1
    bad = MINI[:k] + ["dt:5"] + MINI[k:]
    with pytest.raises(ValueError, match="unexpected token"):
        Replay(bad).run()


def test_ball_in_play_requires_contact_tokens():
    bad = [tok for tok in MINI
           if tok not in ("BB:fly_ball", "EV:104", "LA:25", "SP:-20")]
    with pytest.raises(ValueError, match="without contact tokens"):
        Replay(bad).run()


def test_bucket_edges():
    assert bucket("V", 94.4) == "V:94"
    assert bucket("V", 45) == "V:60-"      # position-player lob
    assert bucket("V", 106.1) == "V:106+"
    assert bucket("LA", -7) == "LA:-10"    # floors toward negative
    assert bucket("EV", None) == "EV:unk"
    assert bucket("SP", 130) == "SP:90+"   # caught behind the plate
    assert bucket("S", 2288) == "S:2200"


def test_spray_orientation():
    # straight up the middle from home plate is 0; left field negative
    assert abs(spray_degrees(125.42, 100.0)) < 0.01
    assert spray_degrees(50.0, 100.0) < -20
    assert spray_degrees(200.0, 100.0) > 20


def test_vocab_padded_and_specials_first():
    vocab = build_vocab([MINI])
    assert vocab[0] == "[PAD]"
    assert vocab.index("[GAME]") < vocab.index("+1")
    for tok in ("+4", "T:KN", "T:unk", "V:60-", "V:106+", "PX:-33", "PZ:63",
                "S:3600+", "EV:118", "LA:-90", "SP:90+", "BB:popup",
                "B:111", "O:3", "MONTH:9"):
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
    final, half_runs, bat, arm, state = game_truth(g, names)
    assert (replay.away_score, replay.home_score) == final == (9, 3)
    assert replay.half_runs == half_runs and len(half_runs) == 18
    assert dict(replay.bat) == bat
    assert dict(replay.arm) == arm
    assert replay.pitch_state == state  # count/outs/bases on every pitch


@pytest.mark.skipif(not (os.path.exists(PARQUET) and os.path.exists(PLAYERS)),
                    reason="needs statcast_2024.parquet and cache/players.json")
@pytest.mark.parametrize("game_pk", [
    744802,  # 10 innings: extra halves open with the Manfred runner (replay
             # rule, not a token — it's deterministic)
    746526,  # the Manfred runner takes third BEFORE the first pitch
    746820,  # the Manfred runner is picked off before the first pitch:
             # the half opens 1 out, bases empty (O:/B: after [HALF])
])
def test_round_trip_extra_innings(game_pk):
    from run import game_truth
    from sim.data import load_season, player_names

    df = load_season(PARQUET)
    g = df[df.game_pk == game_pk]
    names = player_names(df, PLAYERS)
    replay = Replay(encode_game(g, names)).run()
    _, half_runs, _, _, state = game_truth(g, names)
    assert len(half_runs) > 18
    assert replay.pitch_state == state
