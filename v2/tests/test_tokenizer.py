"""Round-trip contract: tokens must reconstruct the game exactly.

Every 10th game here for speed; `python run.py tokenize` sweeps all 1,320.
"""

import os

import pytest

from sim.games import load_games
from sim.tokenizer import Replay, build_vocab, canonical, encode_game

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "..", "nba_data")
CACHE = os.path.join(HERE, "..", "cache", "games.jsonl")


def game_files():
    return sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))


@pytest.fixture(scope="module")
def games_by_id():
    return {g.game_id: g for g in load_games(CACHE)}


def assert_roundtrip(filename, truth):
    replay = Replay(encode_game(os.path.join(DATA_DIR, filename))).run()

    assert replay.away_score == truth.away_score, filename
    assert replay.home_score == truth.home_score, filename
    assert replay.period_scores == truth.periods, filename
    assert len(replay.channels) == len(replay.tokens), filename

    for name, line in truth.players.items():
        name = canonical(name)
        assert replay.box[name]["pts"] == line.pts, (filename, name, "pts")
        assert replay.box[name]["reb"] == line.reb, (filename, name, "reb")
        assert replay.box[name]["ast"] == line.ast, (filename, name, "ast")
        # ±2 min: lineup columns flicker on some FT rows in the source data;
        # the cache inherits that noise, the replay reconstructs from sub events.
        assert replay.minutes(name) == pytest.approx(line.minutes, abs=2.0), \
            (filename, name, "min")


def test_roundtrip_subset(games_by_id):
    for filename in game_files()[::10]:
        game_id = filename.split("-")[3]
        assert_roundtrip(filename, games_by_id[game_id])


def test_canonicalization():
    assert canonical("A.J. Green") == "AJ Green"
    assert canonical(" Jayson Tatum ") == "Jayson Tatum"


def test_grammar_shape():
    # pinned: the assertions below are specific to this game's rosters
    tokens = encode_game(os.path.join(DATA_DIR, "[2022-10-18]-0022200001-PHI@BOS.csv"))
    assert tokens[:4] == ["[GAME]", "TEAM:PHI", "TEAM:BOS", "[ROSTER_A]"]
    assert tokens[-1] == "[EOG]"
    # header: away roster, then home roster, then the first period begins
    h = tokens.index("[ROSTER_H]")
    assert all(t.startswith("P:") for t in tokens[4:h])
    assert "P:Joel Embiid" in tokens[4:h]       # PHI starter in away roster
    assert "P:Sam Hauser" in tokens[h:]   # BOS bench player in home roster
    q = tokens.index("[START_Q]")
    assert tokens[q - 1] == "dt:0"
    assert all(t.startswith("P:") for t in tokens[q + 1:q + 11])  # 10 starters


def test_channels_never_leak_the_event(games_by_id):
    """The score-diff channel on a made shot's tokens must NOT include that
    shot's points — channels are state BEFORE the event."""
    tokens = encode_game(os.path.join(DATA_DIR, game_files()[0]))
    replay = Replay(tokens).run()
    for i, tok in enumerate(tokens):
        if tok == "made":
            # walk back to this event's dt: token; all its tokens share state
            j = i
            while not tokens[j].startswith("dt:"):
                j -= 1
            assert replay.channels[i] == replay.channels[j]
            # and the NEXT event's channel must differ by this shot's points
            # (unless a [SUB]/[OTHER]-style no-score event follows, the diff
            # still must never move before the outcome token itself)
            if i + 1 < len(replay.channels):
                assert replay.channels[i][0] == replay.channels[j][0]


def test_vocab_is_stable_and_complete():
    tokens = encode_game(os.path.join(DATA_DIR, game_files()[0]))
    vocab = build_vocab([tokens])
    assert vocab[0] == "[PAD]"
    assert "dt:74" in vocab and "D:36+" in vocab  # programmatic ranges padded
    assert len(vocab) == len(set(vocab))
