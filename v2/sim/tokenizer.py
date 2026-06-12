"""Token grammar for NBA play-by-play, plus the replay state machine.

encode_game() turns one game CSV into a token sequence. Replay() consumes a
token sequence and reconstructs the game — score, box lines, clock, lineups —
and emits per-token state channels (score diff, period, clock) for training.

The round-trip against cache/games.jsonl is the correctness contract: a
tokenizer that cannot reconstruct the score would train a model that cannot
keep score.

Grammar — header declares the matchup and full available rosters (every
player who took the floor; the data cannot see DNPs), then one line per
event, actor first, outcome after action:

    [GAME] TEAM:away TEAM:home
    [ROSTER_A] P:... (alphabetical -- rotation order would leak coaching
    [ROSTER_H] P:...  decisions the model is supposed to learn)
    dt:0 [START_Q] P:a1 .. P:a5 P:h1 .. P:h5    every period (lineups change
                                                 at breaks without sub events)
    dt:N P:name A:<shot type> D:<feet> made|miss  (+ [AST] P:x | + [BLK] P:x)
    dt:N P:name A:<free throw type> made|miss
    dt:N P:name|[TEAM] reb_off|reb_def|reb_team
    dt:N P:name|[TEAM] TO:<reason>                (+ [STL] P:x)
    dt:N P:name|[TEAM] F:<reason>                 (+ [VS] P:fouled)
    dt:N P:name|[NONE] TF:<type> | V:<type> | EJ:<type>  (+ [VS] P:fouled on TF)
    dt:N [SUB] P:out P:in
    dt:N [TIMEOUT] TEAM:abbr
    dt:N [JUMP] P:away P:home P:gains|[NONE]
    dt:N [OTHER]                                  rare empty rows that carry clock
    dt:N [END_Q]
    [EOG]

Everything derivable is absent: no scores, no points, no absolute clock —
the replay reconstructs all of it, which is what makes the round-trip test
meaningful.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from .games import parse_filename

# One player, two spellings in the source data — each pair verified to never
# share a game (experiments/name_audit.py over all six seasons, 2026-06-11).
# Alias -> the spelling with more games. Without this, the player's token
# (and embedding) splits in half.
NAME_FIXES = {
    # mechanical variants (punctuation / Jr-Sr-III suffixes)
    "A.J. Green": "AJ Green",
    "Brandon Boston": "Brandon Boston Jr.",
    "Jeff Dowtin": "Jeff Dowtin Jr.",
    "Jimmy Butler III": "Jimmy Butler",
    "O.G. Anunoby": "OG Anunoby",
    "P.J. Dozier": "PJ Dozier",
    "Xavier Tillman Sr.": "Xavier Tillman",
    # nickname / legal-name switches (manually reviewed)
    "Nicolas Claxton": "Nic Claxton",
    "Kenyon Martin Jr.": "KJ Martin",
    "Carlton Carrington": "Bub Carrington",
    "Alex Sarr": "Alexandre Sarr",
    "Enes Freedom": "Enes Kanter",
    "Marcos Louzada Silva": "Didi Louzada",
}

LINEUP_COLS = ["a1", "a2", "a3", "a4", "a5", "h1", "h2", "h3", "h4", "h5"]

USECOLS = LINEUP_COLS + [
    "period", "play_length", "event_type", "type", "player", "assist", "block",
    "steal", "entered", "left", "away", "home", "possession", "shot_distance",
    "result", "reason", "team", "points", "opponent", "description",
]


def canonical(name) -> str:
    name = str(name).strip()
    return NAME_FIXES.get(name, name)


def seconds(play_length: str) -> int:
    h, m, s = str(play_length).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def dist_token(value) -> str:
    if pd.isna(value):
        return "D:unk"
    feet = int(float(value))
    return f"D:{feet}" if feet <= 35 else "D:36+"


def _actor(row, fallback: str) -> str:
    """The acting player, or a non-person fallback ([TEAM]/[NONE])."""
    return f"P:{canonical(row.player)}" if pd.notna(row.player) else fallback


def _reason(row) -> str:
    return str(row.reason).strip() if pd.notna(row.reason) else "unknown"


# Two games have a shooter who steps in just for flagrant FTs — invisible to
# the lineup columns, blank in the player column, surname-only in the
# description (Freeman: game 0022400986; Chandler: game 0041900175).
FT_SHOOTER_FIXES = {"Freeman": "Enrique Freeman", "Chandler": "Tyson Chandler"}


def _ft_shooter_from_description(row, roster) -> str:
    """A few FT rows have a blank player column; the description
    ('Freeman Free Throw Flagrant 1 of 2') still names the shooter."""
    desc = str(row.description).removeprefix("MISS ")
    surname = desc.split(" Free Throw")[0].strip()
    matches = [p for p in roster if p.endswith(surname)]
    if len(matches) == 1:
        return matches[0]
    if surname in FT_SHOOTER_FIXES:
        return FT_SHOOTER_FIXES[surname]
    raise ValueError(f"cannot resolve FT shooter from {desc!r}")


def has_corrupt_points(csv_path: str) -> bool:
    """True if any made shot's points column conflicts with its type+result
    (4 games corpus-wide: a made dunk recorded as 1 point, made 3PTs as 2 or 1
    — descriptions and the historical box scores confirm the type+result
    derivation, which the encoder uses, and refute the points column, which
    the cache sums). The round-trip sweep waives score/points (only) for
    these games; rebounds, assists, and minutes still must match."""
    df = pd.read_csv(csv_path, usecols=["event_type", "type", "result", "points"])
    shots = df[(df.event_type.astype(str).str.strip() == "shot")
               & (df.result == "made")]
    derived = [3 if ("3pt" in str(t) or p == 3) else 2
               for t, p in zip(shots["type"], shots["points"])]
    return any(p != d for p, d in zip(shots["points"], derived))


def has_corrupt_lineups(csv_path: str) -> bool:
    """True if any row shows fewer than 5 distinct players in one side's
    lineup columns — a player listed twice, or a blank cell (source-data
    corruption, 7 games corpus-wide). Minutes cannot be reconstructed for the
    affected stretch — the cache miscounts and the true fifth man is
    unrecorded — so the round-trip sweep waives the minutes check (only)."""
    df = pd.read_csv(csv_path, usecols=LINEUP_COLS, dtype=str, keep_default_na=False)
    for side in (LINEUP_COLS[:5], LINEUP_COLS[5:]):
        distinct = df[side].apply(lambda r: len({x.strip() for x in r} - {""}), axis=1)
        if (distinct < 5).any():
            return True
    return False


def encode_game(csv_path: str) -> List[str]:
    """One game CSV -> token sequence."""
    _, _, away, home = parse_filename(csv_path)
    df = pd.read_csv(csv_path, usecols=USECOLS)

    # Rosters: everyone who appears in the lineup columns over the whole game,
    # plus any blank-player FT shooter recovered from the description — he took
    # the floor, so the legality mask must know him (see FT_SHOOTER_FIXES).
    away_set = {canonical(p) for c in LINEUP_COLS[:5] for p in df[c].dropna()}
    home_set = {canonical(p) for c in LINEUP_COLS[5:] for p in df[c].dropna()}
    blank_fts = df[(df["event_type"] == "free throw") & df["player"].isna()]
    for row in blank_fts.itertuples(index=False):
        side_set = away_set if str(row.team).strip() == away else home_set
        side_set.add(_ft_shooter_from_description(row, sorted(side_set)))
    away_roster, home_roster = sorted(away_set), sorted(home_set)

    tokens = ["[GAME]", f"TEAM:{away}", f"TEAM:{home}"]
    tokens += ["[ROSTER_A]"] + [f"P:{p}" for p in away_roster]
    tokens += ["[ROSTER_H]"] + [f"P:{p}" for p in home_roster]
    # On-floor players per our own sub tracking (flicker-immune), per side —
    # sides are diffed separately so one side's corruption can't mispair a
    # legitimate sub on the other side.
    expected = {"a": set(), "h": set()}
    q_open = False
    for row in df.itertuples(index=False):
        ev = str(row.event_type).strip() if pd.notna(row.event_type) else ""
        tokens.append(f"dt:{seconds(row.play_length)}")

        if ev == "start of period":
            starters = []
            for c in LINEUP_COLS:
                v = getattr(row, c)
                if pd.notna(v):
                    starters.append(canonical(v))
                    continue
                # One game (0021901281, 2OT) leaves a lineup cell blank across
                # a period boundary. Lineups carry over the break, so the
                # missing starter comes from our tracked on-floor set; if the
                # corruption leaves several candidates, pick deterministically
                # — that game's minutes are waived as corrupt anyway.
                listed = {canonical(getattr(row, k)) for k in LINEUP_COLS
                          if k.startswith(c[0]) and pd.notna(getattr(row, k))}
                candidate = expected[c[0]] - listed
                if not candidate:
                    raise ValueError(f"cannot infer missing starter in {c}")
                starters.append(min(candidate))
            tokens.append("[START_Q]")
            tokens += [f"P:{p}" for p in starters]
            expected["a"], expected["h"] = set(starters[:5]), set(starters[5:])
            q_open = True
        elif ev == "end of period":
            # one game has a duplicated end-of-period row; a second [END_Q]
            # would open a phantom 0-0 period in the replay
            tokens.append("[END_Q]" if q_open else "[OTHER]")
            q_open = False
        elif ev == "shot":
            kind = str(row.type).strip()
            # 8 made threes corpus-wide have a mislabeled type ("unknown",
            # 25-32 ft); the points column is authoritative, so correct the
            # type token and keep "3pt in type <=> worth 3" derivable.
            if row.points == 3 and "3pt" not in kind:
                kind = f"3pt {kind}"
            tokens += [_actor(row, "[TEAM]"), f"A:{kind}",
                       dist_token(row.shot_distance),
                       "made" if row.result == "made" else "miss"]
            if pd.notna(row.assist):
                tokens += ["[AST]", f"P:{canonical(row.assist)}"]
            if pd.notna(row.block):
                tokens += ["[BLK]", f"P:{canonical(row.block)}"]
        elif ev == "free throw":
            actor = _actor(row, "[TEAM]")
            if actor == "[TEAM]":  # blank player column; recover the shooter
                roster = away_roster if str(row.team).strip() == away else home_roster
                actor = f"P:{_ft_shooter_from_description(row, roster)}"
            tokens += [actor, f"A:{str(row.type).strip()}",
                       "made" if row.result == "made" else "miss"]
        elif ev == "rebound":
            kind = str(row.type)
            reb = ("reb_off" if "offensive" in kind
                   else "reb_def" if "defensive" in kind else "reb_team")
            tokens += [_actor(row, "[TEAM]"), reb]
        elif ev == "turnover":
            tokens += [_actor(row, "[TEAM]"), f"TO:{_reason(row)}"]
            if pd.notna(row.steal):
                tokens += ["[STL]", f"P:{canonical(row.steal)}"]
        elif ev == "foul":
            tokens += [_actor(row, "[TEAM]"), f"F:{_reason(row)}"]
            if pd.notna(row.opponent):  # the player who drew the foul
                tokens += ["[VS]", f"P:{canonical(row.opponent)}"]
        elif ev == "technical foul":
            tokens += [_actor(row, "[NONE]"), f"TF:{str(row.type).strip()}"]
            if pd.notna(row.opponent):
                tokens += ["[VS]", f"P:{canonical(row.opponent)}"]
        elif ev == "violation":
            tokens += [_actor(row, "[NONE]"), f"V:{str(row.type).strip()}"]
        elif ev == "ejection":
            tokens += [_actor(row, "[NONE]"), f"EJ:{str(row.type).strip()}"]
        elif ev == "substitution":
            # The entered/left fields are occasionally wrong or duplicated
            # (e.g. game 0042200214 OT: "Brogdon out" while the lineup columns
            # show Brogdon staying and Brown gone). The lineup columns are
            # authoritative: diff them against our tracked lineup instead.
            pairs = []
            for s in ("a", "h"):
                actual = {canonical(getattr(row, c)) for c in LINEUP_COLS
                          if c.startswith(s) and pd.notna(getattr(row, c))}
                if len(actual) < 5:
                    # NaN cell or a player listed twice (6 games have such
                    # stretches): this side's view is unreliable — keep our
                    # tracking and wait for the columns to heal
                    continue
                outs, ins = sorted(expected[s] - actual), sorted(actual - expected[s])
                pairs += list(zip(outs, ins))
                expected[s] = actual
            if not pairs:
                tokens.append("[OTHER]")  # duplicate/corrupt sub row, no change
            for k, (out_p, in_p) in enumerate(pairs):
                if k:  # rare multi-player correction: one [SUB] event per pair
                    tokens.append("dt:0")
                tokens += ["[SUB]", f"P:{out_p}", f"P:{in_p}"]
        elif ev == "timeout":
            tokens += ["[TIMEOUT]", f"TEAM:{str(row.team).strip()}"]
        elif ev == "jump ball":
            jumpers = [f"P:{canonical(getattr(row, c))}" if pd.notna(getattr(row, c))
                       else "[NONE]" for c in ("away", "home", "possession")]
            tokens += ["[JUMP]"] + jumpers
        elif ev == "":
            tokens.append("[OTHER]")  # rare empty rows; kept because they carry clock
        else:
            raise ValueError(f"Unknown event_type {ev!r} in {csv_path}")
    tokens.append("[EOG]")
    return tokens


class Replay:
    """Replays a token sequence, reconstructing game state token by token.

    After run(): away_score/home_score, period_scores, box (pts/reb/ast/min
    per player), and channels — one (score_diff, period, clock_remaining)
    tuple per token, each reflecting state BEFORE that token's event, so a
    model conditioned on channels never sees its own label.
    """

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.away_score = self.home_score = 0
        self.period = 0
        self.clock = 0  # seconds remaining in current period
        self.period_scores: List[List[int]] = []  # per period [away, home]
        self.on_floor: Dict[str, str] = {}  # player -> "away"|"home"
        self.side: Dict[str, str] = {}  # persistent: a player's side never changes
        self.poss = 0  # possession channel: 0 unknown, 1 away, 2 home
        self.box = defaultdict(lambda: {"pts": 0, "reb": 0, "ast": 0, "sec": 0})
        self.channels: List[Tuple[int, int, int]] = []
        # Per-token on-floor lineup (away five then home five, "" while the
        # floor is unknown), same state-BEFORE convention as channels.
        self.lineups: List[Tuple[str, ...]] = []
        self._bucket_open = False  # True between [END_Q] and the next [START_Q]

    def run(self) -> "Replay":
        t = self.tokens
        assert t[0] == "[GAME]", "sequence must start with [GAME]"
        i = t.index("[ROSTER_H]")
        for p in t[4:i]:  # [GAME] TEAM TEAM [ROSTER_A] ...
            self.side[p[2:]] = "away"
        j = i + 1
        while t[j].startswith("P:"):
            self.side[t[j][2:]] = "home"
            j += 1
        self._mark(j)  # entire header
        i = j
        while t[i] != "[EOG]":
            i = self._event(i)
        self._mark(1)  # [EOG]
        if self._bucket_open:  # bucket opened by the final [END_Q], never used
            self.period_scores.pop()
        return self

    # -- event parsing ------------------------------------------------------

    def _event(self, i: int) -> int:
        """Parse one event starting at its dt: token; return the next index."""
        t = self.tokens
        start = i
        # Channels must reflect state BEFORE this event: marking post-event
        # state would hand the model the answer (e.g. the score-diff channel
        # on a shot's tokens would already contain the made basket).
        state_before = (self.home_score - self.away_score, self.period,
                        self.clock, self.poss)
        floor_before = self._floor()
        dt = int(t[i].split(":")[1])
        i += 1
        head = t[i]

        if head == "[START_Q]":
            lineup = [p[2:] for p in t[i + 1:i + 11]]
            self.on_floor = {p: "away" for p in lineup[:5]}
            self.on_floor.update({p: "home" for p in lineup[5:]})
            self.side.update(self.on_floor)
            self.period += 1
            self.clock = 720 if self.period <= 4 else 300  # 12:00 / 5:00 OT
            self.poss = 0  # unknown at every period start
            if not self._bucket_open:  # else END_Q already opened this period's bucket
                self.period_scores.append([0, 0])
            self._bucket_open = False
            i += 11
        elif head == "[END_Q]":
            # Open the next period's bucket now: fouls/technical FTs recorded
            # between periods belong to the following period's score.
            self.period_scores.append([0, 0])
            self._bucket_open = True
            i += 1
        elif head == "[OTHER]":
            i += 1
        elif head == "[SUB]":
            out_p, in_p = t[i + 1][2:], t[i + 2][2:]
            self.on_floor[in_p] = self.side[in_p] = self.on_floor.pop(out_p)
            i += 3
        elif head == "[TIMEOUT]":
            i += 2
        elif head == "[JUMP]":
            gains = t[i + 3]
            if gains.startswith("P:") and gains[2:] in self.side:
                self.poss = 1 if self.side[gains[2:]] == "away" else 2
            i += 4
        else:  # an actor (P:name, [TEAM], or [NONE]) followed by an action
            i = self._actor_event(i)

        # Time is credited to the post-event lineup, matching the source
        # data's convention (lineup columns reflect the row's lineup).
        for player in self.on_floor:
            self.box[player]["sec"] += dt
        self.clock -= dt
        if self.tokens[start + 1] == "[END_Q]" and self.clock != 0:
            raise ValueError(f"period {self.period} ended with {self.clock}s left")
        self._mark(i - start, state_before, floor_before)
        return i

    def _actor_event(self, i: int) -> int:
        t = self.tokens
        actor = t[i][2:] if t[i].startswith("P:") else None
        action = t[i + 1]
        i += 2

        if action.startswith("A:"):  # shot or free throw
            kind = action[2:]
            is_ft = kind.startswith("free throw")
            if not is_ft:
                i += 1  # D: distance token
            made = t[i] == "made"
            i += 1
            if made and actor:
                points = 1 if is_ft else (3 if "3pt" in kind else 2)
                self._score(actor, points)
                self.poss = possession_after_score(self.side[actor], kind)
            if t[i] == "[AST]":
                self.box[t[i + 1][2:]]["ast"] += 1
                i += 2
            if t[i] == "[BLK]":
                i += 2
        elif action.startswith("reb"):
            if actor:
                self.box[actor]["reb"] += 1
                if action == "reb_def":  # defense claims the ball
                    self.poss = 1 if self.side[actor] == "away" else 2
        elif action.startswith("TO:"):
            if actor:
                self.poss = 2 if self.side[actor] == "away" else 1
            elif self.poss:  # team turnover: ball flips from whoever had it
                self.poss = 3 - self.poss
            if t[i] == "[STL]":
                i += 2
        elif action.startswith(("F:", "TF:")):
            if t[i] == "[VS]":  # the player who drew the foul
                i += 2
        elif not (action.startswith(("V:", "EJ:"))):
            raise ValueError(f"unexpected action token {action!r}")
        return i

    def _score(self, player: str, points: int):
        # Not on_floor: a player subbed out mid-FT-sequence still shoots his
        # remaining free throws (the sub row precedes the final FT in the data).
        side = self.side[player]
        self.box[player]["pts"] += points
        if side == "away":
            self.away_score += points
            self.period_scores[-1][0] += points
        else:
            self.home_score += points
            self.period_scores[-1][1] += points

    def _floor(self) -> Tuple[str, ...]:
        away = sorted(p for p, s in self.on_floor.items() if s == "away")
        home = sorted(p for p, s in self.on_floor.items() if s == "home")
        return tuple((away + [""] * 5)[:5] + (home + [""] * 5)[:5])

    def _mark(self, n: int, state=None, floor=None):
        """Record state channels for the n tokens just consumed."""
        if state is None:
            state = (self.home_score - self.away_score, self.period,
                     self.clock, self.poss)
        self.channels += [state] * n
        self.lineups += [floor if floor is not None else self._floor()] * n

    # -- results ------------------------------------------------------------

    def minutes(self, player: str) -> float:
        return round(self.box[player]["sec"] / 60.0, 1)


def possession_after_score(scorer_side: str, kind: str) -> int:
    """Who has the ball after a made basket or free throw.

    Made field goals and the final regular free throw flip possession to the
    opponent. Technical and flagrant FTs don't (the shooting team keeps the
    ball), and non-final FTs leave it undecided until the sequence resolves.
    """
    opponent = 2 if scorer_side == "away" else 1
    if not kind.startswith("free throw"):
        return opponent
    if "technical" in kind or "flagrant" in kind or "clear path" in kind:
        return 1 if scorer_side == "away" else 2
    n_of = kind.rsplit(" ", 1)[-1]  # "1/2", "2/2", "1/1", ...
    if "/" in n_of and n_of.split("/")[0] == n_of.split("/")[1]:
        return opponent  # final FT made
    return 1 if scorer_side == "away" else 2  # sequence continues


# -- vocabulary --------------------------------------------------------------

def build_vocab(token_lists) -> List[str]:
    """All distinct tokens, padded with the full programmatic ranges, in a
    stable readable order: specials, then by namespace."""
    seen = set()
    for tokens in token_lists:
        seen.update(tokens)
    seen.update(f"dt:{s}" for s in range(75))
    seen.update(f"D:{d}" for d in range(36))
    seen.update(["D:36+", "D:unk"])

    specials = ["[PAD]", "[GAME]", "[ROSTER_A]", "[ROSTER_H]", "[START_Q]",
                "[END_Q]", "[EOG]", "[SUB]", "[TIMEOUT]", "[JUMP]", "[AST]",
                "[BLK]", "[STL]", "[VS]", "[TEAM]", "[NONE]", "[OTHER]",
                "made", "miss", "reb_off", "reb_def", "reb_team"]
    rest = sorted(tok for tok in seen if tok not in specials)
    return specials + rest


def save_vocab(vocab: List[str], path: str):
    with open(path, "w") as f:
        json.dump(vocab, f, indent=0)


def load_vocab(path: str) -> List[str]:
    with open(path) as f:
        return json.load(f)
