"""Free-throw grammar in the sampler: FTs exist only as consequences of fouls,
and a started sequence (1/2 -> 2/2) must finish, surviving subs and timeouts.

These drive GameState directly — no model needed.
"""

import json
import os

import pytest

from sim.sample import GameState, TokenSets, ft_remaining

HERE = os.path.dirname(__file__)
VOCAB = os.path.join(HERE, "..", "cache", "vocab.json")


@pytest.fixture(scope="module")
def ts():
    with open(VOCAB) as f:
        return TokenSets(json.load(f))


@pytest.fixture
def st(ts):
    names = sorted(ts.player_id)[:16]
    away, home = names[:8], names[8:]
    state = GameState(ts, away, home)
    state.push("dt:0")
    state.push("[START_Q]")
    for p in away[:5] + home[:5]:
        state.push("P:" + p)
    return state


def legal_tokens(state):
    return {state.ts.vocab[i] for i in state.legal()}


def begin_event(state, actor, dt="dt:2"):
    state.push(dt)
    state.push("P:" + actor)


def test_ft_remaining():
    assert ft_remaining("free throw 1/2") == ["free throw 2/2"]
    assert ft_remaining("free throw 1/3") == ["free throw 2/3", "free throw 3/3"]
    assert ft_remaining("free throw 1/1") == []
    assert ft_remaining("free throw technical") == []
    assert ft_remaining("free throw flagrant 1/2") == ["free throw flagrant 2/2"]
    assert ft_remaining("free throw clear path 1/2") == ["free throw clear path 2/2"]


def test_fts_illegal_without_a_foul(st):
    begin_event(st, st.roster["away"][0])
    assert not any(t.startswith("A:free throw") for t in legal_tokens(st))


def shooting_foul(st, fouler, fouled):
    begin_event(st, fouler)
    st.push("F:s.foul")
    st.push("[VS]")
    st.push("P:" + fouled)


def test_shooting_foul_arms_fouled_player(st):
    a0, h0 = st.roster["away"][0], st.roster["home"][0]
    shooting_foul(st, a0, h0)
    begin_event(st, h0)
    legal = legal_tokens(st)
    assert {"A:free throw 1/1", "A:free throw 1/2", "A:free throw 1/3"} <= legal
    assert "A:free throw 2/2" not in legal  # can't start mid-sequence
    assert "A:free throw technical" not in legal


def test_arming_is_for_the_fouled_player_only_and_expires(st):
    a0, h0, h1 = st.roster["away"][0], st.roster["home"][0], st.roster["home"][1]
    shooting_foul(st, a0, h0)
    begin_event(st, h1)  # someone else acts: no FTs for them...
    assert not any(t.startswith("A:free throw") for t in legal_tokens(st))
    st.push("A:driving layup")
    st.push("D:1")
    st.push("made")
    st.push("dt:0")
    begin_event(st, h0)  # ...and normal play expired the arming
    assert not any(t.startswith("A:free throw") for t in legal_tokens(st))


def test_charge_does_not_arm(st):
    a0, h0 = st.roster["away"][0], st.roster["home"][0]
    begin_event(st, a0)
    st.push("F:offensive charge foul")
    st.push("[VS]")
    st.push("P:" + h0)
    begin_event(st, h0)
    assert not any(t.startswith("A:free throw") for t in legal_tokens(st))


def test_sequence_is_forced_to_completion(st):
    a0, h0 = st.roster["away"][0], st.roster["home"][0]
    shooting_foul(st, a0, h0)
    begin_event(st, h0)
    st.push("A:free throw 1/2")
    st.push("made")
    assert st.score == {"away": 0, "home": 1}
    st.push("dt:0")
    assert legal_tokens(st) == {"P:" + h0, "[TIMEOUT]", "[SUB]"}
    st.push("P:" + h0)
    assert legal_tokens(st) == {"A:free throw 2/2"}
    st.push("A:free throw 2/2")
    st.push("miss")
    assert st.score == {"away": 0, "home": 1}
    begin_event(st, h0)  # sequence over: FTs illegal again
    assert not any(t.startswith("A:free throw") for t in legal_tokens(st))


def test_shooter_subbed_out_still_shoots_and_scores(st):
    a0, h0, h5 = st.roster["away"][0], st.roster["home"][0], st.roster["home"][5]
    shooting_foul(st, a0, h0)
    begin_event(st, h0)
    st.push("A:free throw 1/2")
    st.push("made")
    st.push("dt:0")
    st.push("[SUB]")
    st.push("P:" + h0)
    st.push("P:" + h5)
    assert h0 not in st.on_floor
    st.push("dt:0")
    assert "P:" + h0 in legal_tokens(st)  # off the floor, but owed a shot
    st.push("P:" + h0)
    st.push("A:free throw 2/2")
    st.push("made")
    assert st.score == {"away": 0, "home": 2}


def test_technical_foul_arms_either_team_then_expires(st):
    a0 = st.roster["away"][0]
    st.push("dt:2")
    st.push("[NONE]")
    st.push("TF:coach technical foul")
    st.push("dt:0")  # no [VS]
    st.push("P:" + a0)
    legal = legal_tokens(st)
    assert "A:free throw technical" in legal
    assert "A:free throw 1/2" not in legal  # techs don't open sequences
    st.push("A:free throw technical")
    st.push("made")
    assert st.score == {"away": 1, "home": 0}
    begin_event(st, a0)
    assert "A:free throw technical" in legal_tokens(st)  # a second is allowed
    st.push("A:pullup jump shot")
    st.push("D:15")
    st.push("miss")
    st.push("[BLK]")
    st.push("P:" + st.roster["home"][0])
    begin_event(st, a0)  # normal play expired it
    assert "A:free throw technical" not in legal_tokens(st)
