# Tokenizer Specification

Derived from a full-corpus scan of all 1,320 games (`experiments/token_analysis.py`,
results in `results/token_analysis.json`). Every number below is measured, not
assumed.

## Principles: verbatim vs programmatic vs derived

**Take verbatim from cells** (closed sets, measured cardinality):

| source column | tokens | notes |
|---|---|---|
| player names (all 9 name columns) | 543 | only 41 names appear <50 times; keep all, no UNK needed this season |
| shot `type` | 31 | "3pt step back jump shot", "driving layup"… — the playstyle signal; only 5 values occur <100 times |
| free-throw `type` | 15 | "free throw 1/2", "…technical", "…flagrant 2/3"… |
| turnover `reason` | 5 | bad pass, unknown, offensive foul, traveling, out of bounds |
| foul `reason` | 6 | s.foul, p.foul, unknown, l.b.foul, charge, take foul |
| violation/technical/ejection `type` | 8 | kicked ball, goaltending, delay, lane, 3 technical kinds, ejection |
| team abbreviations (from filename) | 30 | |

**Define programmatically** (numeric, open-ended in raw form):

| quantity | scheme | tokens |
|---|---|---|
| clock | `Δt` = play_length in seconds, one token per second 0–74 | 75 |
| shot distance | one token per foot 0–35, plus `36+` (heaves) | 37 |

- Clock: 99.0% of events are Δ≤24s (the shot clock); max observed is 74s.
  Per-second tokens keep the absolute clock **exactly** reconstructible —
  verified: Σ play_length = 12:00 (or 5:00 OT) in 100.0% of 5,374 periods.
  Absolute times like "11:24" are therefore never tokenized.
- Distance: 2pt mass is at the rim (p50 = 4 ft), 3pt at 26–29 ft. Per-foot
  preserves the mid-range/rim/arc structure the embeddings should learn.

**Drop entirely — derivable or redundant** (the model never sees these):

- `remaining_time`, `elapsed` — reconstructed from Δt exactly (see above)
- `points` — implied by shot type (3pt prefix) + result; FT = 1
- `num`/`outof` — already inside the FT type string
- `away_score`/`home_score` — running sum of derived points
- `team` — implied by the acting player's roster
- coordinates (`original_x/y`, `converted_x/y`) — present on 99.4% of shots but
  redundant with distance for v1; revisit for shot-chart realism
- `description` — free text, redundant
- lineup columns `a1–a5`/`h1–h5` — implied by starters + substitution events

**Structural tokens** (~15): `[GAME] [LINEUP_A] [LINEUP_H] [START_Q] [END_Q]
[JUMP] [AST] [BLK] [STL] [SUB] [TIMEOUT] [TEAM] [NONE] [EOG]` + made/miss.

**Total vocabulary: ≈ 745** (543 players + 95 actions/reasons + 112 numeric + structural).

## Event grammar

Header (15 tokens):
```
[GAME] [AWAY_TEAM] [HOME_TEAM] [LINEUP_A] p p p p p [LINEUP_H] p p p p p
```

Events — `Δt` first (time advances), then actor, action, qualifiers, outcome:

```
shot          Δt player shot_type dist made|miss   (+ [AST] player | + [BLK] player)
free throw    Δt player ft_type made|miss
rebound       Δt player|[TEAM] reb_off|reb_def
turnover      Δt player|[TEAM] tov_reason          (+ [STL] player)
foul          Δt player foul_reason
substitution  Δt [SUB] player_out player_in
timeout       Δt [TIMEOUT] team
jump ball     Δt [JUMP] player player player|[NONE]   (possession absent in ~4%)
violation &c. Δt event_type_token player|[NONE]
period        [START_Q] / [END_Q]
end of game   [EOG]
```

Outcome (`made|miss`) is a separate position from action choice so the model
factors "what was attempted" from "did it work" — and so simulation can
intervene on either independently.

## Sequencing: within-event order is a modeling decision

The chain rule makes each slot its own prediction problem, so the order
determines what each conditional means. Actor-first maps every slot to a
basketball skill:

| slot | conditional | what it learns |
|---|---|---|
| `Δt` | P(Δt \| history) | pace, tempo |
| `actor` | P(player \| state, on-floor) | **usage** — who takes the shot |
| `action` | P(action \| actor, state) | **playstyle** — what this player does |
| `dist` | P(dist \| actor, action) | shot diet |
| `outcome` | P(made \| actor, action, dist) | **efficiency** |

`P(action | actor)` is exactly the own-action conditioning that won the
embedding lab — the loss pays the player embedding to encode playstyle.
Action-before-actor would instead learn posterior attribution ("a dunk
happened; who probably dunked?"), which composes poorly with the core
simulation intervention: swap a player in and ask what *they* would do.

Deliberate exclusions from the event frame:
- **No TEAM slot** — possession is ~deterministic from the previous outcome
  (make → other team, defensive rebound → rebounder's team, turnover → other
  team). An explicit token costs +1/event (~+24% sequence) for near-zero
  information; possession enters via a state channel instead (below).
- **No points slot** — deterministic given action + outcome.
- **Rebound is its own event**, not a shot suffix: different actor, own Δt.
  Assist/block ARE same-instant suffixes, and the data fixes their position:
  assists exist only on makes, blocks only on misses, so both must follow the
  outcome token.

Fixed slot order also gives structured decoding for free: every position has a
known role, so rollouts can mask illegal tokens per slot (e.g. actor ∈ the 10
on floor).

## Game state: channels, not tokens

No `state → action → state` alternation: in an autoregressive transformer the
prefix *is* the state, and interleaving full state would inflate sequences
3–5×. But "derivable in principle" ≠ "computed reliably by a small model" —
a running score is a sum over hundreds of tokens, small transformers are weak
at implicit long-range arithmetic, and score-dependent behavior is real
(intentional fouls, end-quarter heaves, garbage-time rotations).

Resolution: **additive state channels.** At each position the dataloader adds
embeddings for (score-diff bucket, period, clock-remaining bucket, possession
side, optionally an on-floor lineup summary) to the token embedding — the same
mechanism as positional embeddings. Zero sequence cost, exact state, no
arithmetic burden. Constraints:

- Channels are computed from the **prefix only** (state before the current
  token) — never from the token being predicted, or they leak the label.
- The round-trip validation contract already requires exact score/clock/lineup
  tracking at every token, so the channel computation is free infrastructure.
- The lineup/possession channel is also where team context enters the model —
  the separate channel the embedding lab prescribed so player embeddings stay
  team-free.
- At rollout time the simulator maintains the same state machine (required
  anyway for legality masks) and feeds the channels back in.

## Division of labor: state machine vs model

Token positions fall into three classes, and each is handled differently —
in **both** training and decoding, not just one:

1. **Fully deterministic given the prefix** — FT 2/2 follows FT 1/2, [END_Q]
   fires when the clock hits 0:00, a regulation tie forces OT, score updates
   after a made 2. The score-type facts never enter the token stream at all
   (they're state channels); the sequence-structural ones are **forced at
   decode** (probability 1) and **masked out of the training loss** — no
   gradient is spent learning what the state machine hardcodes, and held-out
   perplexity stays honest (not deflated by free tokens).
2. **Constrained but stochastic** — the actor must be one of the 10 on floor,
   Δt can't exceed the remaining clock, a fouled-out player can't act. Apply
   legality masks to the softmax **during training too**, not just decoding:
   the model never spends capacity separating legal from illegal, the
   probability mass is normalized over the true support, and train/inference
   distributions match (decode-time-only masking renormalizes a distribution
   the model never trained under).
3. **Genuinely stochastic** — who acts, what they attempt, whether it goes in.
   The model's entire job.

Principle: **deterministic structure is the state machine's job, stochastic
choice is the model's job, and the mask is the interface.** The dataloader
already tracks exact state at every token (round-trip contract), so training
masks are free.

## Sequence length (measured on the implemented tokenizer, all 7,590 games)

| | tokens/game |
|---|---|
| mean | 2,141 |
| p95 | 2,381 |
| max (4OT) | 3,124 |

- **A whole game fits in the model's 3,328-token context** (lineups are
  re-stated after every [START_Q] — 91% of period breaks change lineups
  without sub events; full rosters are declared in the header; fouls carry
  the fouled player).
- **Corpus: 16.25M tokens** over six seasons, vocab **1,386**.
- Roster caveat: play-by-play only sees players who took the floor, so the
  header roster is "who played", not "who dressed" — DNPs are invisible.
- Implementation notes from the full sweep (`python run.py tokenize`,
  7,580/7,590 exact round-trip, 10 documented waivers): 8 made threes
  corpus-wide have mislabeled types (points column is authoritative — type
  corrected at encode); 4 games have the REVERSE corruption (made shots with
  points=1, made 3PTs with points=2/1) where descriptions + historical box
  scores confirm type+result and refute the points column — the cache sums
  raw points, so score is waived for those 4; substitution entered/left
  fields are occasionally wrong or duplicated (lineup columns are
  authoritative — subs derived by diffing, per side); 6 games duplicate a
  player inside one side's lineup columns for a stretch (minutes waived);
  lineup columns flicker on some FT rows (replay reconstructs minutes from
  sub events, ±2 min tolerance vs the row-based cache); a player subbed out
  mid-FT-sequence still shoots his remaining free throws; two games have
  blank-player flagrant FTs whose shooter never appears in lineup columns
  (recovered from descriptions via FT_SHOOTER_FIXES and added to the header
  roster); one game duplicates an end-of-period row.

## Multi-year notes

- Player vocab grows ~400–500 per added season (this season alone uses 543);
  ten seasons ≈ 2,500–3,500 player tokens — still a small vocabulary.
- Player-season factorization (DESIGN.md) happens at the **embedding layer**
  (`e_player + δ_season + e_season`), not the token layer: one token per
  player, season supplied by the header. This keeps sequences identical in
  shape across eras.
- Action/clock/distance vocab is era-stable; new shot-type strings from other
  seasons' data sources would extend the verbatim set, so the tokenizer must
  fail loudly on unseen cell values rather than silently bucketing them.

## Validation contract

The tokenizer must round-trip: `tokens → (final score, quarter scores, box
lines, clock)` reproduced **exactly** against `cache/games.jsonl` for all
1,320 games. The games.py parser is the independent referee. This is the first
test to write — a tokenizer that can't reconstruct the score will train a
model that can't keep score.
