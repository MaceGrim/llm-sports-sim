"""Token grammar for Statcast pitch-by-pitch, plus the replay state machine.

encode_game() turns one game's pitch rows into a token sequence. Replay()
consumes a token sequence and reconstructs the game — final score, runs per
half-inning, batter and pitcher box lines — from tokens alone. The round-trip
against the parquet is the correctness contract (v2's lesson: a tokenizer
that cannot reconstruct the score would train a model that cannot keep score).

Grammar — the header declares the matchup and everyone who appears. Starting
batting orders are pregame information, so they are listed in order; bench
and bullpen usage are in-game coaching decisions the model should learn, so
those sections are alphabetical (v2's roster-ordering principle):

    [GAME] TEAM:away TEAM:home
    [LINEUP_A] P: x9                away batting order   (then [LINEUP_H])
    [BENCH_A] P: ...                other batters who appeared, alphabetical
    [PEN_A] P:starter P: ...        starter first, relievers alphabetical
    [HALF]                          every half-inning; the replay counts
                                    Top 1, Bot 1, Top 2, ... (a skipped
                                    bottom half only ever ends the game)
    [PA] P:pitcher P:batter
    T:FF Z:5 R:called_strike        one triple per pitch (T:unk Z:unk when
                                    untracked: pitch-clock violations, the
                                    odd lost reading)
    E:strikeout                     PA outcome, on the final pitch only
    +1 .. +4                        runs scored on the play, after R:/E:
    [NEWP] P:name | [NEWB] P:name   mid-PA pitching change / pinch hitter
    [MID] +n                        runs scored between pitches (balk-type
                                    plays, 54 in 2024)
    [EOG]

Everything derivable is absent: no inning numbers, no scores, no outs, no
counts — the replay reconstructs score and innings, which is what makes the
round-trip test meaningful. Unseen source values raise (closed sets below).

Data quirks, all measured on the full 2024 season (trust the state columns
over event labels — the principle that resolved every v2 quirk):
- Runs can score BETWEEN pitch rows (balks, steals of home, no-pitch plays):
  the absolute score columns jump across consecutive rows. The side whose
  score jumped identifies the half-inning the run belongs to — the fielding
  team never scores (0 rows in 2024) — including the one case where the jump
  straddles a half boundary.
- 104 PAs end with no event (inning-ending baserunner outs, two walk-offs):
  no E: token, and no PA is charged — official scoring does the same.
- Mid-PA pitcher (58) and batter (14) changes: the PA's outcome is charged
  to the pitcher and batter of the final pitch, matching the parquet's rows.
"""

import json
from collections import defaultdict
from typing import Dict, List

import pandas as pd

# Closed sets, frozen from the 2024 season. A new value (new season, rule
# change, source rename) must be added here deliberately, not absorbed.
PITCH_TYPES = {"FF", "SI", "SL", "CH", "FC", "ST", "CU", "FS", "KC", "SV",
               "KN", "FA", "EP", "SC", "FO", "CS", "PO"}
DESCRIPTIONS = {"ball", "foul", "hit_into_play", "called_strike",
                "swinging_strike", "blocked_ball", "foul_tip",
                "swinging_strike_blocked", "automatic_ball", "hit_by_pitch",
                "foul_bunt", "missed_bunt", "automatic_strike", "pitchout",
                "bunt_foul_tip"}
EVENTS = {"field_out", "strikeout", "single", "walk", "double", "home_run",
          "force_out", "grounded_into_double_play", "hit_by_pitch",
          "sac_fly", "field_error", "triple", "intent_walk", "sac_bunt",
          "fielders_choice", "double_play", "truncated_pa",
          "fielders_choice_out", "strikeout_double_play", "catcher_interf",
          "sac_fly_double_play", "triple_play"}

HITS = {"single", "double", "triple", "home_run"}
WALKS = {"walk", "intent_walk"}
STRIKEOUTS = {"strikeout", "strikeout_double_play"}


def encode_game(g: pd.DataFrame, names: Dict[int, str]) -> List[str]:
    """One game's sorted pitch rows -> token sequence."""
    rows = list(g.itertuples(index=False))
    away, home = rows[0].away_team, rows[0].home_team

    # Batting orders (first 9 distinct batters per side, in PA order) and
    # pitchers (appearance order). Away bats in Top halves; the pitcher on a
    # Top row therefore belongs to the home staff.
    order = {"Top": [], "Bot": []}
    staff = {"Top": [], "Bot": []}  # keyed by the half they PITCH in
    for r in rows:
        if r.batter not in order[r.inning_topbot]:
            order[r.inning_topbot].append(r.batter)
        if r.pitcher not in staff[r.inning_topbot]:
            staff[r.inning_topbot].append(r.pitcher)
    if len(order["Top"]) < 9 or len(order["Bot"]) < 9:
        raise ValueError(f"game {rows[0].game_pk}: fewer than 9 batters")

    def section(marker, ids, ordered):
        if not ordered:
            ids = sorted(ids, key=names.get)
        return [marker] + [f"P:{names[i]}" for i in ids]

    tokens = ["[GAME]", f"TEAM:{away}", f"TEAM:{home}"]
    tokens += section("[LINEUP_A]", order["Top"][:9], ordered=True)
    tokens += section("[LINEUP_H]", order["Bot"][:9], ordered=True)
    tokens += section("[BENCH_A]", order["Top"][9:], ordered=False)
    tokens += section("[BENCH_H]", order["Bot"][9:], ordered=False)
    for marker, staff_ids in (("[PEN_A]", staff["Bot"]), ("[PEN_H]", staff["Top"])):
        tokens += [marker, f"P:{names[staff_ids[0]]}"]
        tokens += [f"P:{names[i]}" for i in sorted(staff_ids[1:], key=names.get)]

    half = None  # (inning, topbot) of the half currently open
    cur_ab = cur_pitcher = cur_batter = None
    prev = None
    for r in rows:
        # Runs that scored between this row and the previous one (no-pitch
        # plays). The scoring side must be a batting side; it decides whether
        # the run belongs to the half just ended or the one about to open.
        if prev is not None:
            jump_h = r.home_score - prev.post_home_score
            jump_a = r.away_score - prev.post_away_score
            if jump_h < 0 or jump_a < 0 or (jump_h and jump_a):
                raise ValueError(f"game {r.game_pk}: unexplained score change "
                                 f"({jump_a}, {jump_h}) before ab {r.at_bat_number}")
            jump, scorer_bats_in = ((jump_a, "Top") if jump_a
                                    else (jump_h, "Bot"))
            if jump and half[1] == scorer_bats_in:
                tokens += ["[MID]", f"+{jump}"]
                jump = 0
        else:
            jump = 0
        if (r.inning, r.inning_topbot) != half:
            tokens.append("[HALF]")
            half = (r.inning, r.inning_topbot)
        if jump:
            if half[1] != scorer_bats_in:
                raise ValueError(f"game {r.game_pk}: between-row run fits "
                                 f"neither adjacent half")
            tokens += ["[MID]", f"+{jump}"]

        if r.at_bat_number != cur_ab:
            tokens += ["[PA]", f"P:{names[r.pitcher]}", f"P:{names[r.batter]}"]
            cur_ab, cur_pitcher, cur_batter = r.at_bat_number, r.pitcher, r.batter
        else:
            if r.pitcher != cur_pitcher:  # mid-PA pitching change
                tokens += ["[NEWP]", f"P:{names[r.pitcher]}"]
                cur_pitcher = r.pitcher
            if r.batter != cur_batter:  # mid-PA pinch hitter
                tokens += ["[NEWB]", f"P:{names[r.batter]}"]
                cur_batter = r.batter

        ptype = str(r.pitch_type) if pd.notna(r.pitch_type) else "unk"
        if ptype != "unk" and ptype not in PITCH_TYPES:
            raise ValueError(f"unseen pitch_type {ptype!r}")
        zone = f"Z:{int(r.zone)}" if pd.notna(r.zone) else "Z:unk"
        if r.description not in DESCRIPTIONS:
            raise ValueError(f"unseen description {r.description!r}")
        tokens += [f"T:{ptype}", zone, f"R:{r.description}"]
        if pd.notna(r.events):
            if r.events not in EVENTS:
                raise ValueError(f"unseen events {r.events!r}")
            tokens.append(f"E:{r.events}")
        runs = ((r.post_home_score - r.home_score)
                + (r.post_away_score - r.away_score))
        if runs:
            tokens.append(f"+{runs}")
        prev = r
    tokens.append("[EOG]")
    return tokens


def _line():
    return {"pa": 0, "h": 0, "hr": 0, "bb": 0, "k": 0}


class Replay:
    """Replays a token sequence, reconstructing the game token by token.

    After run(): away_score/home_score, half_runs (one total per half-inning,
    in order — Top 1, Bot 1, ...), bat (per batter: pa/h/hr/bb/k), and arm
    (per pitcher: bf/h/bb/k). State channels for training (score diff, inning,
    outs, count, bases) come later — outs and bases need the advancement
    tokens that are deliberately not in grammar v1.
    """

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.away_score = self.home_score = 0
        self.half_runs: List[int] = []
        self.bat = defaultdict(_line)
        self.arm = defaultdict(lambda: {"bf": 0, "h": 0, "bb": 0, "k": 0})

    def run(self) -> "Replay":
        t = self.tokens
        assert t[0] == "[GAME]", "sequence must start with [GAME]"
        self.away_team, self.home_team = t[1][5:], t[2][5:]
        i = t.index("[HALF]")  # header is declarative; play starts here
        pitcher = batter = None
        while t[i] != "[EOG]":
            tok = t[i]
            if tok == "[HALF]":
                self.half_runs.append(0)
                i += 1
            elif tok == "[PA]":
                pitcher, batter = t[i + 1][2:], t[i + 2][2:]
                i += 3
            elif tok == "[NEWP]":
                pitcher = t[i + 1][2:]
                i += 2
            elif tok == "[NEWB]":
                batter = t[i + 1][2:]
                i += 2
            elif tok == "[MID]":
                self._score(int(t[i + 1]))
                i += 2
            elif tok.startswith("T:"):
                i += 3  # T: Z: R:
                if t[i].startswith("E:"):
                    self._outcome(t[i][2:], batter, pitcher)
                    i += 1
                if t[i].startswith("+"):
                    self._score(int(t[i]))
                    i += 1
            else:
                raise ValueError(f"unexpected token {tok!r}")
        return self

    def _outcome(self, event: str, batter: str, pitcher: str):
        b, p = self.bat[batter], self.arm[pitcher]
        b["pa"] += 1
        p["bf"] += 1
        if event in HITS:
            b["h"] += 1
            p["h"] += 1
        if event == "home_run":
            b["hr"] += 1
        if event in WALKS:
            b["bb"] += 1
            p["bb"] += 1
        if event in STRIKEOUTS:
            b["k"] += 1
            p["k"] += 1

    def _score(self, runs: int):
        # Top halves are even (0-indexed); the away team bats in them. Runs
        # only ever score for the batting team (verified: 0 exceptions).
        if (len(self.half_runs) - 1) % 2 == 0:
            self.away_score += runs
        else:
            self.home_score += runs
        self.half_runs[-1] += runs


# -- vocabulary ---------------------------------------------------------------

def build_vocab(token_lists) -> List[str]:
    """All distinct tokens, padded with the full programmatic ranges, in a
    stable readable order: specials, then by namespace."""
    seen = set()
    for tokens in token_lists:
        seen.update(tokens)
    seen.update(f"Z:{z}" for z in range(1, 15))
    seen.update(f"+{n}" for n in range(1, 5))
    seen.update(f"T:{p}" for p in PITCH_TYPES)
    seen.update(["T:unk", "Z:unk"])

    specials = ["[PAD]", "[GAME]", "[LINEUP_A]", "[LINEUP_H]", "[BENCH_A]",
                "[BENCH_H]", "[PEN_A]", "[PEN_H]", "[HALF]", "[PA]",
                "[NEWP]", "[NEWB]", "[MID]", "[EOG]"]
    rest = sorted(tok for tok in seen if tok not in specials)
    return specials + rest


def save_vocab(vocab: List[str], path: str):
    with open(path, "w") as f:
        json.dump(vocab, f, indent=0)


def load_vocab(path: str) -> List[str]:
    with open(path) as f:
        return json.load(f)
