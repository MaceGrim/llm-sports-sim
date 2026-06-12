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

## Grammar (implemented in sim/tokenizer.py; round-trips all of 2024)

```
[GAME] TEAM:away TEAM:home
[LINEUP_A] P: x9 batting order        [LINEUP_H] P: x9
[BENCH_A] / [BENCH_H] P:...           later batters, alphabetical
[PEN_A] / [PEN_H] P:starter P:...     starter first, relievers alphabetical
[HALF]                                each half-inning, by count (Top 1, ...)
per PA:    [PA] P:<pitcher> P:<batter>
per pitch: T:<type> V:<mph> S:<rpm/100> Z:<zone> R:<description>
           — the pitcher's choice, then the batter's result
in play:   BB:<trajectory> EV:<mph/2> LA:<deg/5> SP:<deg/10>
           — contact physics after R:hit_into_play; spray direction from
           the hit coordinates (negative = left field), distance omitted
           as derivable from EV/LA/spray
final:     E:<events> on the last pitch, "+n" when runs score on the play
mid-PA:    [NEWP] P:x (pitching change, 58/season) | [NEWB] P:x (PH, 14)
no-pitch:  [MID] +n  (runs between pitch rows, 54/season)
inning/half/outs: deterministic -> state machine + channels, never tokens
```

One deliberate deviation from the first sketch: a single `P:` namespace
instead of `B:`/`P:` — one player, one token, one embedding (a two-way
player would otherwise split in half; pitcher/batter role is positional in
the `[PA]` header).

Open questions, in the order to resolve them:

1. ~~Round-trip mismatch~~ DIAGNOSED 2026-06-11: both failing games have a
   run scoring *between* pitch rows (balk-type events with a runner on third
   — the between-row score jump appears in no pitch's bat_score delta). Not a
   data bug; pitch-level deltas are simply incomplete. RESOLVED 2026-06-12:
   `[MID] +n` tokens derived from between-row jumps in the absolute score
   columns; the scoring side identifies the half the run belongs to (the
   fielding team never scores — 0 rows in 2024).
2. ~~Names~~ DONE 2026-06-12: pybaseball's playerid_reverse_lookup, cached
   in cache/players.json; five colliding names (two Will Smiths, ...) get an
   ID suffix so one token never means two players.
3. **Outs/bases for state channels**: grammar v1 carries runs but not
   baserunner advancement, so the replay can reconstruct score and innings
   but not outs or base occupancy. Training channels (count, outs, bases)
   need either advancement tokens (out-delta + post-play base state per PA,
   from between-row diffs of on_1b/2b/3b and outs_when_up) or channels
   computed from rows at encode time. Decide when training starts; count
   (balls/strikes) is already derivable from R: tokens alone.
4. Pitcher fatigue: pitch count is derivable; `n_thruorder_pitcher` is on-row.
   Channel or token? (Channel, probably — same as score.)
5. Season scope: start with 2024 (3.0M tokens — between v2's single-season
   2.8M and six-season 16M), add 2015-2023 after the pipeline round-trips.

## Status

- 2026-06-11: directory created; pybaseball verified; one-week schema audit
  done (`explore_statcast.py` output above); round-trip mismatch diagnosed
  (runs between pitches -> derive non-pitch events from state diffs); **full
  2024 season pulled: 712,274 pitches in `statcast_2024.parquet`** (105MB,
  gitignored, also in the pybaseball cache).
- 2026-06-12: `encode_game` + `Replay` landed (sim/tokenizer.py, sim/data.py,
  run.py mirroring v2's layout). **Round-trip: 2,427/2,427 regular-season
  games exact, zero waivers** — final score, per-half-inning runs, batter
  lines (PA/H/HR/BB/K) and pitcher lines (BF/H/BB/K) from tokens alone,
  verified against parquet-derived truth (`python run.py tokenize`).
- 2026-06-12 (later): grammar enriched per Mason — per-pitch velocity and
  spin (V:/S:), per-contact trajectory/exit-velo/launch-angle/spray
  (BB:/EV:/LA:/SP:, spray from the hit coordinates). Round-trip still
  2,427/2,427 exact. Tokens: mean 2,034/game, p95 2,402, max 3,053 (train
  at max_len 3,200); corpus 4,936,560; vocab 1,761. Physics sanity: median
  HR is EV:104/LA:25, HR spray peaks at the pull gaps, league four-seam
  median V:94 (real 2024 average: 94.2).
  Tests in tests/test_tokenizer.py (handcrafted grammar sequence + the
  [MID]-straddle game 747004 pinned). Next: training — reuse v2's EventGPT/
  KVCache by import, settle open question #3 (outs/bases channels), then
  the sampler state machine (deal batters from the lineup, force [HALF] at
  three outs).
