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
    T:FF V:94 S:2200 PX:-3 PZ:24    one pitch: the pitcher's choice — type,
    R:called_strike                 velocity (1 mph), spin (100 rpm), plate
                                    location (3-inch bins, catcher's view,
                                    PX negative = inside to a RHB) — then
                                    the batter's result (:unk when
                                    untracked: pitch-clock violations, the
                                    odd lost reading)
    BB:fly_ball EV:98 LA:25 SP:-10  contact physics after R:hit_into_play —
                                    trajectory, exit velo (2 mph), launch
                                    angle (5 deg), spray direction (10 deg,
                                    negative = left field; distance is
                                    omitted as derivable physics)
    E:strikeout                     PA outcome, on the final pitch only
    +1 .. +4                        runs scored on the play, after R:/E:
    O:n B:xyz                       state transition, after the play: outs
                                    recorded, then base occupancy 1st-2nd-
                                    3rd (B:101 = corners) — how the runners
                                    actually advanced. B: follows every PA
                                    that leaves the half-inning open; O:/B:
                                    after a non-final pitch are steals,
                                    pickoffs, caught stealings. Outs reach
                                    3 exactly at [HALF]; count/outs/bases
                                    are replay-tracked and verified against
                                    the source state columns on every pitch
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
import math
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

BB_TYPES = {"ground_ball", "fly_ball", "line_drive", "popup"}

HITS = {"single", "double", "triple", "home_run"}
WALKS = {"walk", "intent_walk"}
STRIKEOUTS = {"strikeout", "strikeout_double_play"}

# Numeric token ranges: (prefix, lo, hi, step). Values bucket to
# prefix:floor(v/step)*step within [lo, hi); the tails are prefix:lo-
# (below) and prefix:hi+ (at or above), v2's D:36+ pattern. Bounds chosen
# from the 2024 distributions (plate bounds at p0.1/p99.9; inches).
RANGES = {"V": (60, 106, 1), "S": (1000, 3600, 100),
          "EV": (30, 120, 2), "LA": (-90, 90, 5), "SP": (-90, 90, 10),
          "PX": (-33, 33, 3), "PZ": (-12, 66, 3)}

# Count bookkeeping by pitch result. Fouls only add a strike below two;
# hit_into_play and hit_by_pitch end the PA and never touch the count.
# Verified against the balls/strikes columns on every 2024 pitch.
BALL_DESCS = {"ball", "blocked_ball", "automatic_ball", "pitchout"}
STRIKE_DESCS = {"called_strike", "swinging_strike", "swinging_strike_blocked",
                "missed_bunt", "automatic_strike", "foul_tip"}
FOUL_DESCS = {"foul", "foul_bunt", "bunt_foul_tip"}


def bucket(prefix: str, value) -> str:
    lo, hi, step = RANGES[prefix]
    if pd.isna(value):
        return f"{prefix}:unk"
    if value < lo:
        return f"{prefix}:{lo}-"
    if value >= hi:
        return f"{prefix}:{hi}+"
    return f"{prefix}:{int(value // step) * step}"


def spray_degrees(hc_x, hc_y) -> float:
    """Spray direction from Statcast hit coordinates: 0 = straightaway
    center, negative = left field. Home plate is at (125.42, 198.27) in
    the coordinate frame Baseball Savant uses."""
    return math.degrees(math.atan2(hc_x - 125.42, 198.27 - hc_y))


def bases_str(row) -> str:
    """Base occupancy 1st-2nd-3rd from a row's pre-pitch state columns
    (on_1b/2b/3b hold the runner's MLB ID, null when the base is empty)."""
    return "".join("1" if pd.notna(v) else "0"
                   for v in (row.on_1b, row.on_2b, row.on_3b))


def has_voided_pitch(g: pd.DataFrame) -> bool:
    """True if some pitch's count, per the rules above, fails to land on
    the next same-PA row's count (one 2024 game: a 3-2 ball after which
    the batter stayed at 3-2 — a voided/dead pitch the source kept as a
    row). The sweep waives per-pitch state (only) for these games."""
    prev = None
    for r in g.itertuples(index=False):
        if prev is not None and r.at_bat_number == prev.at_bat_number:
            b, s = int(prev.balls), int(prev.strikes)
            if prev.description in BALL_DESCS:
                b += 1
            elif prev.description in STRIKE_DESCS:
                s += 1
            elif prev.description in FOUL_DESCS:
                s = min(2, s + 1)
            if (int(r.balls), int(r.strikes)) != (b, s):
                return True
        prev = r
    return False


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

    # Park and calendar conditioning (Mason, 2026-06-12): the venue shapes
    # offense (Coors) and the month proxies weather (cold April suppresses
    # carry). PARK is the home team — right for all but the handful of
    # neutral-site games a year (Seoul, London); a venue lookup can refine
    # it later. month is from game_date (YYYY-MM-DD).
    month = int(str(rows[0].game_date)[5:7])
    tokens = ["[GAME]", f"TEAM:{away}", f"TEAM:{home}",
              f"PARK:{home}", f"MONTH:{month}"]
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
    for k, r in enumerate(rows):
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
            # No-pitch plays before the half's first pitch (Manfred runner
            # picked off or advanced): emit the deviation from the rule
            # start state the replay assumes (empty, or runner on second
            # in extras).
            if int(r.outs_when_up):
                tokens.append(f"O:{int(r.outs_when_up)}")
            start_bases = "010" if r.inning >= 10 else "000"
            if bases_str(r) != start_bases:
                tokens.append(f"B:{bases_str(r)}")
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
        if r.description not in DESCRIPTIONS:
            raise ValueError(f"unseen description {r.description!r}")
        tokens += [f"T:{ptype}", bucket("V", r.release_speed),
                   bucket("S", r.release_spin_rate),
                   bucket("PX", r.plate_x * 12), bucket("PZ", r.plate_z * 12),
                   f"R:{r.description}"]
        if r.description == "hit_into_play":
            bb = str(r.bb_type) if pd.notna(r.bb_type) else "unk"
            if bb != "unk" and bb not in BB_TYPES:
                raise ValueError(f"unseen bb_type {bb!r}")
            spray = (spray_degrees(r.hc_x, r.hc_y)
                     if pd.notna(r.hc_x) and pd.notna(r.hc_y) else None)
            tokens += [f"BB:{bb}", bucket("EV", r.launch_speed),
                       bucket("LA", r.launch_angle), bucket("SP", spray)]
        if pd.notna(r.events):
            if r.events not in EVENTS:
                raise ValueError(f"unseen events {r.events!r}")
            tokens.append(f"E:{r.events}")
        runs = ((r.post_home_score - r.home_score)
                + (r.post_away_score - r.away_score))
        if runs:
            tokens.append(f"+{runs}")

        # State transition, from the NEXT row's pre-pitch columns: outs
        # recorded (a half always closes at exactly 3) and the resulting
        # base occupancy. B: is omitted when the half ends (bases wiped)
        # and after the game's final pitch; after a non-final pitch these
        # only appear for steals/pickoffs/caught-stealings.
        nxt = rows[k + 1] if k + 1 < len(rows) else None
        if nxt is not None:
            if (nxt.inning, nxt.inning_topbot) == half:
                d_outs = int(nxt.outs_when_up) - int(r.outs_when_up)
                if d_outs < 0:
                    raise ValueError(f"game {r.game_pk}: outs went backwards "
                                     f"at ab {nxt.at_bat_number}")
                if d_outs:
                    tokens.append(f"O:{d_outs}")
                nxt_bases = bases_str(nxt)
                if pd.notna(r.events) or nxt_bases != bases_str(r):
                    tokens.append(f"B:{nxt_bases}")
            else:
                tokens.append(f"O:{3 - int(r.outs_when_up)}")
        prev = r
    tokens.append("[EOG]")
    return tokens


def _line():
    return {"pa": 0, "h": 0, "hr": 0, "bb": 0, "k": 0}


class Replay:
    """Replays a token sequence, reconstructing the game token by token.

    After run(): away_score/home_score, half_runs (one total per half-inning,
    in order — Top 1, Bot 1, ...), bat (per batter: pa/h/hr/bb/k), arm
    (per pitcher: bf/h/bb/k), and pitch_state — the (balls, strikes, outs,
    bases) before every pitch, tracked from tokens alone: count from R:
    results, outs from O:, bases from B:. The round-trip sweep verifies
    pitch_state against the source state columns on every pitch, which is
    what makes generated state transitions trustworthy.

    channels holds one (score_diff, inning, half, outs, balls, strikes,
    bases) tuple per token, each reflecting state BEFORE that token's group,
    so a model conditioned on channels never sees its own label.
    """

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.away_score = self.home_score = 0
        self.half_runs: List[int] = []
        self.bat = defaultdict(_line)
        self.arm = defaultdict(lambda: {"bf": 0, "h": 0, "bb": 0, "k": 0})
        self.balls = self.strikes = self.outs = 0
        self.bases = "000"
        self.pitch_state: List[tuple] = []
        self.channels: List[tuple] = []

    def run(self) -> "Replay":
        t = self.tokens
        assert t[0] == "[GAME]", "sequence must start with [GAME]"
        self.away_team, self.home_team = t[1][5:], t[2][5:]
        i = t.index("[HALF]")  # header is declarative; play starts here
        self.channels += [self._state()] * i
        pitcher = batter = None
        while t[i] != "[EOG]":
            tok = t[i]
            group_start, state = i, self._state()
            if tok == "[HALF]":
                self.half_runs.append(0)
                inning = (len(self.half_runs) + 1) // 2
                # Manfred runner: regular-season extra innings start with a
                # runner placed on second (2020+ rule; postseason differs —
                # revisit when game_type P is added).
                self.outs, self.bases = 0, "010" if inning >= 10 else "000"
                i += 1
            elif tok == "[PA]":
                pitcher, batter = t[i + 1][2:], t[i + 2][2:]
                self.balls = self.strikes = 0
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
            elif tok.startswith("O:"):
                self.outs += int(tok[2:])
                i += 1
            elif tok.startswith("B:"):
                self.bases = tok[2:]
                i += 1
            elif tok.startswith("T:"):
                self.pitch_state.append(
                    (self.balls, self.strikes, self.outs, self.bases))
                desc = t[i + 5][2:]
                i += 6  # T: V: S: PX: PZ: R:
                if desc == "hit_into_play":
                    if not t[i].startswith("BB:"):
                        raise ValueError(f"ball in play without contact "
                                         f"tokens at {t[i]!r}")
                    i += 4  # BB: EV: LA: SP:
                if t[i].startswith("E:"):
                    self._outcome(t[i][2:], batter, pitcher)
                    i += 1
                if t[i].startswith("+"):
                    self._score(int(t[i]))
                    i += 1
                # post-play O:/B: are standalone branches of the main loop
                if desc in BALL_DESCS:
                    self.balls += 1
                elif desc in STRIKE_DESCS:
                    self.strikes += 1
                elif desc in FOUL_DESCS:
                    self.strikes = min(2, self.strikes + 1)
            else:
                raise ValueError(f"unexpected token {tok!r}")
            self.channels += [state] * (i - group_start)
        self.channels.append(self._state())  # [EOG]
        return self

    def _state(self) -> tuple:
        n = len(self.half_runs)  # current half is n - 1 (0 during header)
        return (self.home_score - self.away_score, (n + 1) // 2,
                (n - 1) % 2 if n else 0, self.outs,
                self.balls, self.strikes, int(self.bases, 2))

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
    seen.update(f"+{n}" for n in range(1, 5))
    seen.update(f"T:{p}" for p in PITCH_TYPES)
    seen.update(f"BB:{b}" for b in BB_TYPES)
    seen.update(f"B:{n >> 2 & 1}{n >> 1 & 1}{n & 1}" for n in range(8))
    seen.update(f"O:{n}" for n in range(1, 4))
    seen.update(f"MONTH:{m}" for m in range(3, 11))
    seen.update(["T:unk", "BB:unk"])
    for prefix, (lo, hi, step) in RANGES.items():
        seen.update(f"{prefix}:{v}" for v in range(lo, hi, step))
        seen.update([f"{prefix}:{lo}-", f"{prefix}:{hi}+", f"{prefix}:unk"])

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
