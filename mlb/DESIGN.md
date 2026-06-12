# MLB / Statcast Simulation — Design Notes

The MLB twin of `v2/` (NBA): `simulate(lineup, lineup) -> full game`, player
embeddings as a first-class deliverable. Same playbook — measure the data,
fix a token grammar with a round-trip contract, train a small event
transformer from scratch with legality masks and state channels — but
baseball's structure moves more of the game into the state machine.

## Data source

**Statcast via pybaseball** (Baseball Savant). Public and free — none of the
BigDataBall licensing constraints. Rows are *pitches* (~294/game, 119
columns), grouped by `game_pk -> at_bat_number -> pitch_number`. Coverage is
2015+ (the Statcast era), ~2,430 regular-season games/season.

Measured on 2024-06-05..11 (90 games, 26,422 pitches; `explore_statcast.py`):

| quantity | value |
|---|---|
| pitches/game | 294 mean, 399 max |
| PAs/game | 75, 3.90 pitches/PA |
| pitch_type | 16 values (FF/SI/SL/CH/FC/CU/ST/FS...) |
| description (per-pitch result) | 15 values (ball/foul/hit_into_play/...) |
| events (PA outcome) | 20 values (field_out/strikeout/single/...) |
| bb_type | 4 values |
| zone | 14 values, 99.7% coverage |
| batters / pitchers in one week | 402 / 417 |
| state nulls (inning/outs/count/bases/score) | 0 |

## What transfers from v2, and what changes

- **Usage becomes deterministic.** v2's hardest learned slot — who acts —
  is the batting order here. The state machine deals the next batter; the
  model only learns matchup outcomes. (Pinch hitters/pitching changes are
  explicit events, like v2's [SUB].)
- **State channels are free.** v2 needed an exact replay to compute channels;
  Statcast carries (inning, top/bot, outs, balls, strikes, bases, score) on
  every row. Channels: score-diff, inning, outs, count, base-occupancy bits.
- **Two-sided conditioning is the core.** Event factorization, actor-first:
  P(pitch_type, zone | pitcher, count, state) is the pitcher's arsenal;
  P(result | pitch, batter, ...) is the batter's eye/contact. Both
  embeddings get paid directly, mirroring v2's P(action|actor) lesson.
- **Team-leakage risk is lower but real** (same-team batters share lineups).
  The v2 legality-mask discovery applies verbatim: restrict player softmax
  to the game's rosters during training, and probe teammate@k vs style@k
  before believing any embedding.
- **Token budget ~1,258/game** (PA header + ~3/pitch) — roomy. Pitch-level
  granularity preserved, full game in ~1.3k context.

## Candidate grammar (to validate against a real encoder + round-trip)

```
[GAME] TEAM:away TEAM:home [LINEUP_A] B:... x9 [LINEUP_H] B:... x9
per PA:    [PA] P:<pitcher> B:<batter>          (state machine deals batter;
                                                 tokens needed only on change?)
per pitch: T:<pitch_type> Z:<zone> R:<description>
final:     E:<events> (+ bb:<type> + EV/LA buckets when hit into play)
inning/half/outs: deterministic -> state machine + channels, never tokens
```

Open questions, in the order to resolve them:

1. ~~Round-trip mismatch~~ DIAGNOSED 2026-06-11: both failing games have a
   run scoring *between* pitch rows (balk-type events with a runner on third
   — the between-row score jump appears in no pitch's bat_score delta). Not a
   data bug; pitch-level deltas are simply incomplete.
2. Baserunning/steals/wild pitches: per #1, non-pitch events (SB, CS, balk,
   pickoff, scoring plays between pitches) must be **derived from
   between-row state diffs** — score, bases, outs — and emitted as explicit
   tokens, exactly v2's sub-derivation lesson (trust state columns over
   event labels). The encoder's round-trip contract then covers them.
3. Batter identity is an MLB ID int — map to names via pybaseball's
   playerid_reverse_lookup for readable tokens (v2's readability principle).
4. Pitcher fatigue: pitch count is derivable; `n_thruorder_pitcher` is on-row.
   Channel or token? (Channel, probably — same as score.)
5. Season scope: start with 2024 (one season, ~700k pitches ≈ 3M tokens —
   between v2's single-season 2.8M and six-season 16M), add 2015-2023 after
   the pipeline round-trips.

## Status

- 2026-06-11: directory created; pybaseball verified; one-week schema audit
  done (`explore_statcast.py` output above); round-trip mismatch diagnosed
  (runs between pitches -> derive non-pitch events from state diffs); **full
  2024 season pulled: 712,274 pitches in `statcast_2024.parquet`** (105MB,
  gitignored, also in the pybaseball cache). Next: write `encode_game` +
  `Replay` with the same exact-reconstruction contract as v2 — final score,
  per-inning runs, and batter/pitcher box lines from tokens alone.
