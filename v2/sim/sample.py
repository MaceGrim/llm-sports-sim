"""Generate complete games from the trained model, one token at a time.

The sampler enforces at generation time exactly what training assumed:
the legality mask over player slots, plus the hard rules the data guaranteed
implicitly — clock arithmetic (dt <= time remaining, periods end at 0:00),
actors must be on the floor, subs come from the bench, periods are forced
([END_Q] -> [START_Q], overtime if tied, [EOG] otherwise), and free throws
exist only as the consequence of a foul. Deterministic structure is the state
machine's job; the model only ever samples among genuinely legal choices.

Free-throw rules (measured on the real corpus, see TODO #1):
- A defensive foul with a [VS] player arms free throws for that player —
  whether they're actually taken stays the model's choice (s.foul -> FT 97%,
  p.foul ~24% from penalty situations, charge 0%).
- Once a multi-shot sequence starts (e.g. "free throw 1/2"), the remaining
  shots are forced in order; only subs and timeouts may intervene (the only
  thing that does in real data). The shooter keeps shooting even if subbed out.
- Technical FTs are armed by technical fouls / delay of game / ejections,
  shooter chosen by the model from either team.
"""

from collections import Counter
from typing import Dict, List, Optional

import torch

from .model import KVCache, bucketize_channels
from .tokenizer import possession_after_score

# Fouls that never award free throws to a [VS] player.
NO_FT_FOULS = {"F:offensive charge foul"}
# Events that arm a technical free throw (shooter from either team).
TECH_FT_EVENTS = ("TF:", "V:delay of game", "EJ:")


def slot_temperatures(ts: "TokenSets", temperature):
    """Normalize the temperature argument into a per-slot lookup. A float
    applies everywhere (the old behavior); a dict maps slot classes
    ('action' / 'actor' / 'outcome' / 'dt') to temperatures, with 'default'
    (1.0) covering unlisted classes. A slot's class is the plurality class
    of its legal tokens (legal sets are near-homogeneous by construction)."""
    if isinstance(temperature, dict):
        default = temperature.get("default", 1.0)
        return lambda legal: temperature.get(
            Counter(ts.slot_class[j] for j in legal).most_common(1)[0][0],
            default)
    return lambda legal: temperature


def ft_remaining(kind: str) -> List[str]:
    """'free throw flagrant 1/3' -> ['free throw flagrant 2/3', 'free throw flagrant 3/3']."""
    head, _, frac = kind.rpartition(" ")
    if "/" not in frac:
        return []  # 'free throw technical' is a single shot
    n, m = (int(x) for x in frac.split("/"))
    return [f"{head} {k}/{m}" for k in range(n + 1, m + 1)]


class TokenSets:
    """Vocabulary indices grouped by grammar role."""

    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.id = {t: i for i, t in enumerate(vocab)}
        by = lambda pred: [i for i, t in enumerate(vocab) if pred(t)]
        self.dt = {s: self.id[f"dt:{s}"] for s in range(75)}
        self.player_id = {t[2:]: i for t, i in self.id.items() if t.startswith("P:")}
        self.player_name = {i: n for n, i in self.player_id.items()}
        self.shot = by(lambda t: t.startswith("A:") and not t[2:].startswith("free throw"))
        self.ft = by(lambda t: t.startswith("A:") and t[2:].startswith("free throw"))
        # FTs that can begin a sequence: 'free throw [flagrant|clear path] 1/m'
        self.ft_start = by(lambda t: t.startswith("A:free throw")
                           and t.rsplit(" ", 1)[-1].startswith("1/"))
        self.ft_tech = self.id["A:free throw technical"]
        self.dist = by(lambda t: t.startswith("D:"))
        self.to = by(lambda t: t.startswith("TO:"))
        self.foul = by(lambda t: t.startswith("F:"))
        self.tf = by(lambda t: t.startswith("TF:"))
        self.v_ej = by(lambda t: t.startswith(("V:", "EJ:")))
        # Slot class per vocab id, for per-slot sampling temperature (#17):
        # 'outcome' = made/miss, 'dt' = clock, 'actor' = players, else 'action'.
        self.slot_class = ["outcome" if t in ("made", "miss")
                           else "dt" if t.startswith("dt:")
                           else "actor" if t.startswith("P:")
                           else "action" for t in vocab]


class GameState:
    """Incremental grammar/state machine for one game being generated."""

    def __init__(self, ts: TokenSets, away_roster: List[str], home_roster: List[str]):
        self.ts = ts
        self.roster = {"away": list(away_roster), "home": list(home_roster)}
        self.on_floor: Dict[str, str] = {}  # player -> side
        self.side: Dict[str, str] = {}  # like on_floor but never forgets (FT scoring)
        self.score = {"away": 0, "home": 0}
        self.period = 0
        self.clock = 0
        self.zero_clock_events = 0
        self.done = False
        self.expect = "dt"
        self._lineup_picks: List[str] = []
        self._actor: Optional[str] = None
        self._shot_kind = ""
        self._foul_kind = ""
        self.ft_queue: List[str] = []  # forced remaining shots of a started sequence
        self.ft_shooter: Optional[str] = None
        self.ft_armed: Optional[str] = None  # fouled player who may start a sequence
        self.tech_armed = 0  # technical FTs available (any on-floor shooter)
        self.poss = 0  # 0 unknown, 1 away, 2 home — same convention as Replay
        self._channel = (0, 0, 0, 0)  # state at current event start
        self._floor_snap = [0] * 10  # on-floor ids at current event start

    # -- helpers -------------------------------------------------------------

    def _pids(self, names) -> List[int]:
        return [self.ts.player_id[n] for n in names]

    def _floor(self, side=None, exclude=None) -> List[str]:
        return [p for p, s in self.on_floor.items()
                if (side is None or s == side) and p != exclude]

    def _bench(self, side) -> List[str]:
        return [p for p in self.roster[side] if p not in self.on_floor]

    def channel(self):
        return self._channel

    def floor_ids(self) -> List[int]:
        """On-floor vocab ids (away five then home five, PAD-padded), same
        event-start snapshot convention as channel()."""
        return self._floor_snap

    # -- legality ------------------------------------------------------------

    def legal(self) -> List[int]:
        ts, id_ = self.ts, self.ts.id
        e = self.expect
        if e == "dt":
            hi = min(74, self.clock)
            return [ts.dt[s] for s in range(hi + 1)]
        if e == "postq_dt":
            return [ts.dt[0]]
        if e == "postq":
            if self.period >= 4 and self.score["away"] != self.score["home"]:
                return [id_["[EOG]"]]
            return [id_["[START_Q]"]]
        if e == "head":
            if self.ft_queue:  # a started FT sequence must finish
                heads = [ts.player_id[self.ft_shooter]]
                if self.clock > 0 or self.zero_clock_events < 3:
                    heads.append(id_["[TIMEOUT]"])
                    if any(self._bench(s) for s in ("away", "home")):
                        heads.append(id_["[SUB]"])
                return heads
            heads = []
            if self.clock == 0:
                heads.append(id_["[END_Q]"])
                if self.zero_clock_events >= 3:  # don't loiter at the buzzer
                    return heads
            actors = self._pids(self._floor()) + [id_["[TEAM]"], id_["[NONE]"]]
            heads += actors + [id_["[TIMEOUT]"], id_["[JUMP]"], id_["[OTHER]"]]
            if any(self._bench(s) for s in ("away", "home")):
                heads.append(id_["[SUB]"])
            return heads
        if e.startswith("lineup"):
            k = len(self._lineup_picks)
            side = "away" if k < 5 else "home"
            pool = [p for p in self.roster[side] if p not in self._lineup_picks]
            return self._pids(pool)
        if e == "action":
            if self._actor == "[TEAM]":
                return [id_["reb_off"], id_["reb_def"], id_["reb_team"]] + ts.to + ts.foul
            if self._actor == "[NONE]":
                return ts.tf + ts.v_ej
            if self.ft_queue and self._actor == self.ft_shooter:
                return [id_["A:" + self.ft_queue[0]]]
            acts = (ts.shot + ts.to + ts.foul + ts.tf + ts.v_ej
                    + [id_["reb_off"], id_["reb_def"]])
            if self._actor == self.ft_armed:
                acts = acts + ts.ft_start
            if self.tech_armed:
                acts = acts + [ts.ft_tech]
            return acts
        if e == "dist":
            return ts.dist
        if e == "outcome":
            return [id_["made"], id_["miss"]]
        if e == "after_shot_made":
            return self.legal_dt_or(id_["[AST]"])
        if e == "after_shot_miss":
            return self.legal_dt_or(id_["[BLK]"])
        if e == "after_to":
            return self.legal_dt_or(id_["[STL]"])
        if e == "after_foul":
            return self.legal_dt_or(id_["[VS]"])
        if e == "ast_p":  # teammate of the scorer, on the floor
            side = self.on_floor.get(self._actor)
            return self._pids(self._floor(side, exclude=self._actor))
        if e in ("blk_p", "stl_p", "vs_p"):  # opponent on the floor
            side = self.on_floor.get(self._actor)
            if side is None:  # [TEAM]-attributed event: side unknown
                return self._pids(self._floor())
            return self._pids(self._floor("home" if side == "away" else "away"))
        if e == "sub_out":
            return self._pids([p for p in self._floor()
                               if self._bench(self.on_floor[p])])
        if e == "sub_in":
            return self._pids(self._bench(self.on_floor[self._actor]))
        if e == "timeout_t":
            return [i for i, t in enumerate(self.ts.vocab) if t.startswith("TEAM:")]
        if e == "jump_a":
            return self._pids(self._floor("away"))
        if e == "jump_h":
            return self._pids(self._floor("home"))
        if e == "jump_g":
            return self._pids(self._floor()) + [id_["[NONE]"]]
        raise ValueError(f"unknown slot {e}")

    def legal_dt_or(self, extra: int) -> List[int]:
        hi = min(74, self.clock)
        return [self.ts.dt[s] for s in range(hi + 1)] + [extra]

    # -- transitions ----------------------------------------------------------

    def push(self, tok: str):
        e = self.expect
        if tok.startswith("dt:"):
            self._channel = (self.score["home"] - self.score["away"],
                             self.period, self.clock, self.poss)
            away = sorted(p for p, s in self.on_floor.items() if s == "away")
            home = sorted(p for p, s in self.on_floor.items() if s == "home")
            self._floor_snap = ([self.ts.player_id[p] for p in away[:5]]
                                + [0] * (5 - min(len(away), 5))
                                + [self.ts.player_id[p] for p in home[:5]]
                                + [0] * (5 - min(len(home), 5)))
            dt = int(tok[3:])
            self.clock -= dt
            self.zero_clock_events += 1 if self.clock == 0 and dt == 0 else 0
            if self.clock == 0 and dt > 0:
                self.zero_clock_events = 0
            self.expect = "postq" if e == "postq_dt" else "head"
        elif tok == "[START_Q]":
            self.period += 1
            self.clock = 720 if self.period <= 4 else 300
            self.zero_clock_events = 0
            self.poss = 0
            self._lineup_picks = []
            self.expect = "lineup"
        elif e == "lineup":
            self._lineup_picks.append(tok[2:])
            if len(self._lineup_picks) == 10:
                self.on_floor = {p: "away" for p in self._lineup_picks[:5]}
                self.on_floor.update({p: "home" for p in self._lineup_picks[5:]})
                self.side.update(self.on_floor)
                self.expect = "dt"
        elif tok == "[END_Q]":
            self.ft_armed = None
            self.tech_armed = 0
            game_over = self.period >= 4 and self.score["away"] != self.score["home"]
            # real grammar: [EOG] follows [END_Q] directly; a new period
            # starts with its own dt:0 [START_Q]
            self.expect = "postq" if game_over else "postq_dt"
        elif tok == "[EOG]":
            self.done = True
        elif e == "head":
            if tok == "[SUB]":
                self.expect = "sub_out"
            elif tok == "[TIMEOUT]":
                self.expect = "timeout_t"
            elif tok == "[JUMP]":
                self.ft_armed, self.tech_armed = None, 0
                self.expect = "jump_a"
            elif tok == "[OTHER]":
                self.ft_armed, self.tech_armed = None, 0
                self.expect = "dt"
            else:  # actor: P:, [TEAM], [NONE]
                self._actor = tok[2:] if tok.startswith("P:") else tok
                self.expect = "action"
        elif e == "action":
            if tok.startswith("A:"):
                self._shot_kind = tok[2:]
                if self._shot_kind.startswith("free throw"):
                    if self.ft_queue and self._actor == self.ft_shooter:
                        self.ft_queue.pop(0)
                        if not self.ft_queue:
                            self.ft_shooter = None
                    elif self._shot_kind == "free throw technical":
                        self.tech_armed -= 1
                    else:  # start the sequence the foul armed
                        self.ft_queue = ft_remaining(self._shot_kind)
                        self.ft_shooter = self._actor if self.ft_queue else None
                    self.ft_armed = None
                    self.expect = "outcome"
                else:
                    self.ft_armed, self.tech_armed = None, 0
                    self.expect = "dist"
            elif tok.startswith("TO:"):
                self.ft_armed, self.tech_armed = None, 0
                if self._actor in self.on_floor:
                    self.poss = 2 if self.on_floor[self._actor] == "away" else 1
                elif self.poss:
                    self.poss = 3 - self.poss
                self.expect = "after_to"
            elif tok.startswith(("F:", "TF:")):
                self._foul_kind = tok
                self.ft_armed = None
                if tok.startswith("TF:"):
                    self.tech_armed = 2
                self.expect = "after_foul"
            else:  # rebounds, violations, ejections
                if tok.startswith(TECH_FT_EVENTS):
                    self.tech_armed = 2
                else:
                    self.ft_armed, self.tech_armed = None, 0
                if tok == "reb_def" and self._actor in self.on_floor:
                    self.poss = 1 if self.on_floor[self._actor] == "away" else 2
                self.expect = "dt"
        elif e == "dist":
            self.expect = "outcome"
        elif e == "outcome":
            made = tok == "made"
            if made and self._actor in self.side:
                side = self.side[self._actor]
                pts = (1 if self._shot_kind.startswith("free throw")
                       else 3 if "3pt" in self._shot_kind else 2)
                self.score[side] += pts
                self.poss = possession_after_score(side, self._shot_kind)
            if self._shot_kind.startswith("free throw"):
                self.expect = "dt"
            else:
                self.expect = "after_shot_made" if made else "after_shot_miss"
        elif e in ("after_shot_made", "after_shot_miss", "after_to", "after_foul"):
            self.expect = {"[AST]": "ast_p", "[BLK]": "blk_p",
                           "[STL]": "stl_p", "[VS]": "vs_p"}[tok]
        elif e in ("ast_p", "blk_p", "stl_p", "vs_p"):
            if (e == "vs_p" and self._foul_kind.startswith("F:")
                    and self._foul_kind not in NO_FT_FOULS):
                self.ft_armed = tok[2:]
            self.expect = "dt"
        elif e == "sub_out":
            self._actor = tok[2:]
            self.expect = "sub_in"
        elif e == "sub_in":
            side = self.on_floor.pop(self._actor)
            self.on_floor[tok[2:]] = side
            self.side[tok[2:]] = side
            self.expect = "dt"
        elif e == "timeout_t":
            self.expect = "dt"
        elif e == "jump_a":
            self.expect = "jump_h"
        elif e == "jump_h":
            self.expect = "jump_g"
        elif e == "jump_g":
            if tok.startswith("P:") and tok[2:] in self.on_floor:
                self.poss = 1 if self.on_floor[tok[2:]] == "away" else 2
            self.expect = "dt"
        else:
            raise ValueError(f"cannot push {tok!r} in slot {e}")


@torch.no_grad()
def generate_games(model, vocab: List[str], headers: List[List[str]], device: str,
                   seed: int = 0, temperature: float = 1.0,
                   max_len: int = 3300) -> List[List[str]]:
    """Generate a batch of games in lockstep with KV caching.

    Every game appends exactly one token per step (forced or sampled), so a
    batch of grammar machines advances together: one model.step() per token
    position serves the whole batch. Headers may differ in length — shorter
    prefixes are right-padded and the padding keys masked out of attention.
    """
    ts = TokenSets(vocab)
    B = len(headers)
    use_lineup = getattr(model.cfg, "lineup_channel", False)
    states, tokens, channels, floors = [], [], [], []
    for header in headers:
        h = header.index("[ROSTER_H]")
        away = [t[2:] for t in header[4:h] if t.startswith("P:")]
        home = [t[2:] for t in header[h + 1:] if t.startswith("P:")]
        st = GameState(ts, away, home)
        chans, flrs = [], []
        for tok in header:
            chans.append(st.channel())
            flrs.append(st.floor_ids())
            if tok.startswith(("dt:", "[START_Q]")) or st.expect == "lineup":
                st.push(tok)
        states.append(st)
        tokens.append(list(header))
        channels.append(chans)
        floors.append(flrs)

    def tensors(rows, pad_to=None):
        L = pad_to or max(len(r) for r in rows)
        out = torch.zeros((B, L), dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            out[i, :len(r)] = torch.tensor(r)
        return out

    # Prime the caches on the (right-padded) headers.
    Lp = max(len(t) for t in tokens)
    ids = tensors([[ts.id[t] for t in seq] for seq in tokens])
    chan = [bucketize_channels(c + [(0, 0, 0, 0)] * (Lp - len(c))) for c in channels]
    diff, period, clock, poss = (torch.stack([c[k] for c in chan]).to(device)
                                 for k in range(4))
    lineup = None
    if use_lineup:
        lineup = torch.zeros((B, Lp, 10), dtype=torch.long, device=device)
        for i, f in enumerate(floors):
            lineup[i, :len(f)] = torch.tensor(f)

    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")
    model.eval()
    cache = KVCache(model.cfg, B, device,
                    torch.bfloat16 if device == "cuda" else torch.float32)
    with autocast:
        logits_all = model.prime(ids, diff, period, clock, poss, cache,
                                 lineup=lineup)
    lengths = [len(t) for t in tokens]
    last_logits = torch.stack([logits_all[i, lengths[i] - 1] for i in range(B)])
    key_valid = torch.zeros((B, Lp), dtype=torch.bool, device=device)
    for i, n in enumerate(lengths):
        key_valid[i, :n] = True

    gen = torch.Generator(device="cpu").manual_seed(seed)
    temp_of = slot_temperatures(ts, temperature)

    while any(not s.done for s in states) and max(len(t) for t in tokens) < max_len:
        # Whole-batch sampling: one GPU->CPU transfer and one multinomial per
        # step (per-row calls dominated wall time at large batch sizes).
        legals = {i: st.legal() for i, st in enumerate(states) if not st.done}
        choices = {i: l[0] for i, l in legals.items() if len(l) == 1}
        sample_rows = [i for i, l in legals.items() if len(l) > 1]
        if sample_rows:
            t_vec = torch.tensor([temp_of(legals[i]) for i in sample_rows])
            rows = last_logits[sample_rows].float().cpu() / t_vec[:, None]
            mask = torch.full_like(rows, float("-inf"))
            for r, i in enumerate(sample_rows):
                mask[r, legals[i]] = 0.0
            probs = torch.softmax(rows + mask, dim=-1)
            draws = torch.multinomial(probs, 1, generator=gen)[:, 0]
            choices.update(zip(sample_rows, draws.tolist()))

        next_ids, next_chan, next_floor = [], [], []
        for i, st in enumerate(states):
            if st.done:
                next_ids.append(0)  # PAD; masked out of attention below
                next_chan.append((0, 0, 0, 0))
                next_floor.append([0] * 10)
                continue
            choice = choices[i]
            tok = vocab[choice]
            # A dt: token STARTS its event: push first so channel/floor are
            # this event's start state, exactly as Replay marks it in
            # training. (push snapshots before applying the dt itself.)
            if tok.startswith("dt:"):
                st.push(tok)
                next_chan.append(st.channel())
                next_floor.append(st.floor_ids())
            else:
                next_chan.append(st.channel())
                next_floor.append(st.floor_ids())
                st.push(tok)
            tokens[i].append(tok)
            next_ids.append(choice)

        ids = torch.tensor(next_ids, device=device)[:, None]
        d, p, c, po = bucketize_channels(next_chan)
        step_lineup = (torch.tensor(next_floor, device=device)[:, None]
                       if use_lineup else None)
        pos = torch.tensor([len(t) - 1 if not states[i].done else 0
                            for i, t in enumerate(tokens)], device=device)[:, None]
        alive = torch.tensor([not s.done for s in states], device=device)
        key_valid = torch.cat([key_valid, alive[:, None]], dim=1)
        with autocast:
            last_logits = model.step(
                ids, d[:, None].to(device), p[:, None].to(device),
                c[:, None].to(device), po[:, None].to(device),
                pos, cache, key_valid, lineup=step_lineup)
    return tokens


@torch.no_grad()
def generate_game(model, vocab: List[str], header: List[str], device: str,
                  seed: int = 0, temperature: float = 1.0,
                  max_len: int = 3300) -> List[str]:
    """Condition on a real header (teams, rosters, Q1 starters); generate the rest."""
    ts = TokenSets(vocab)
    h = header.index("[ROSTER_H]")
    away_roster = [t[2:] for t in header[4:h] if t.startswith("P:")]
    home_roster = [t[2:] for t in header[h + 1:] if t.startswith("P:")]
    state = GameState(ts, away_roster, home_roster)
    use_lineup = getattr(model.cfg, "lineup_channel", False)

    tokens, channels, floors = [], [], []
    for tok in header:  # forced conditioning prefix
        channels.append(state.channel())
        floors.append(state.floor_ids())
        if tok.startswith(("dt:", "[START_Q]")) or state.expect == "lineup":
            state.push(tok)
        tokens.append(tok)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    temp_of = slot_temperatures(ts, temperature)
    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")
    model.eval()

    while not state.done and len(tokens) < max_len:
        legal = state.legal()
        if len(legal) == 1:
            choice = legal[0]
        else:
            ids = torch.tensor([[ts.id[t] for t in tokens]], device=device)
            diff, period, clock, poss = bucketize_channels(channels)
            lineup = (torch.tensor(floors, device=device)[None]
                      if use_lineup else None)
            with autocast:
                logits, _ = model(ids, diff[None].to(device),
                                  period[None].to(device), clock[None].to(device),
                                  poss[None].to(device), lineup=lineup)
            row = logits[0, -1].float().cpu() / temp_of(legal)
            mask = torch.full_like(row, float("-inf"))
            mask[legal] = 0.0
            probs = torch.softmax(row + mask, dim=-1)
            choice = int(torch.multinomial(probs, 1, generator=gen))
        tok = vocab[choice]
        if tok.startswith("dt:"):  # event start: state must be post-push
            state.push(tok)
            channels.append(state.channel())
            floors.append(state.floor_ids())
        else:
            channels.append(state.channel())
            floors.append(state.floor_ids())
            state.push(tok)
        tokens.append(tok)
    return tokens
