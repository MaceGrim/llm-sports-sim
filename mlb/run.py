#!/usr/bin/env python3
"""CLI for the MLB simulator.

  python run.py tokenize          # encode all games, verify round-trip,
                                  # write cache/tokens.jsonl + vocab.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.data import iter_games, load_season, player_names
from sim.tokenizer import (HITS, STRIKEOUTS, WALKS, Replay, bases_str,
                           build_vocab, encode_game, has_voided_pitch,
                           save_vocab)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
DEFAULT_PARQUET = os.path.join(HERE, "statcast_2024.parquet")


def game_truth(g, names):
    """Score, per-half runs, and box lines straight from the rows — the
    independent side of the round-trip check. Final score comes from the
    last row's post columns (not summed deltas); box lines from the events
    column; per-half runs from per-row score deltas, with between-row jumps
    credited to the half whose batting side scored; per-pitch pre-state
    straight from the balls/strikes/outs/bases columns."""
    rows = list(g.itertuples(index=False))
    final = (int(rows[-1].post_away_score), int(rows[-1].post_home_score))
    state = [(int(r.balls), int(r.strikes), int(r.outs_when_up), bases_str(r))
             for r in rows]

    half_runs, half = [], None
    prev = None
    for r in rows:
        jump_top = jump_bot = 0
        if prev is not None:
            jump_top = r.away_score - prev.post_away_score  # away bats in Top
            jump_bot = r.home_score - prev.post_home_score
        if jump_top and half[1] == "Top":
            half_runs[-1] += jump_top
            jump_top = 0
        if jump_bot and half[1] == "Bot":
            half_runs[-1] += jump_bot
            jump_bot = 0
        if (r.inning, r.inning_topbot) != half:
            half = (r.inning, r.inning_topbot)
            half_runs.append(0)
        half_runs[-1] += jump_top + jump_bot  # jump belonging to the new half
        half_runs[-1] += int((r.post_away_score - r.away_score)
                             + (r.post_home_score - r.home_score))
        prev = r

    bat, arm = {}, {}
    for r in rows:
        if not isinstance(r.events, str):
            continue
        b = bat.setdefault(names[r.batter],
                           {"pa": 0, "h": 0, "hr": 0, "bb": 0, "k": 0})
        p = arm.setdefault(names[r.pitcher], {"bf": 0, "h": 0, "bb": 0, "k": 0})
        b["pa"] += 1
        p["bf"] += 1
        b["h"] += r.events in HITS
        p["h"] += r.events in HITS
        b["hr"] += r.events == "home_run"
        b["bb"] += r.events in WALKS
        p["bb"] += r.events in WALKS
        b["k"] += r.events in STRIKEOUTS
        p["k"] += r.events in STRIKEOUTS
    return final, half_runs, bat, arm, state


def cmd_tokenize(args):
    """Encode every game, verify the round-trip contract, write tokens + vocab."""
    df = load_season(args.parquet)
    names = player_names(df, os.path.join(CACHE, "players.json"))

    all_tokens, lengths, failures = [], [], []
    waived = {"per-pitch state (voided pitch in source)": []}
    out_path = os.path.join(CACHE, "tokens.jsonl")
    n = 0
    with open(out_path, "w") as out:
        for game_pk, g in iter_games(df):
            tokens = encode_game(g, names)
            replay = Replay(tokens).run()
            final, half_runs, bat, arm, state = game_truth(g, names)

            scores_ok = ((replay.away_score, replay.home_score) == final
                         and replay.half_runs == half_runs
                         and dict(replay.bat) == bat
                         and dict(replay.arm) == arm)
            if scores_ok and replay.pitch_state == state:
                pass
            elif scores_ok and has_voided_pitch(g):
                # see has_voided_pitch docstring
                waived["per-pitch state (voided pitch in source)"].append(
                    int(game_pk))
            else:
                failures.append(int(game_pk))
            row0 = g.iloc[0]
            out.write(json.dumps({
                "game_pk": int(game_pk), "date": str(row0.game_date)[:10],
                "away": row0.away_team, "home": row0.home_team,
                "tokens": tokens}) + "\n")
            all_tokens.append(tokens)
            lengths.append(len(tokens))
            n += 1
            if n % 250 == 0:
                print(f"  {n} games...")

    vocab = build_vocab(all_tokens)
    save_vocab(vocab, os.path.join(CACHE, "vocab.json"))

    import numpy as np
    lengths = np.array(lengths)
    n_waived = sum(len(v) for v in waived.values())
    print(f"Round-trip: {n - len(failures) - n_waived}/{n} games exact"
          + (f"  FAILURES: {failures[:5]}" if failures else ""))
    for reason, pks in waived.items():
        if pks:
            print(f"  waived {reason}: {pks}")
    print(f"Tokens/game: mean {lengths.mean():.0f}, "
          f"p95 {np.percentile(lengths, 95):.0f}, max {lengths.max()}")
    print(f"Corpus: {lengths.sum():,} tokens   Vocab: {len(vocab)}")
    print(f"Wrote {out_path} and cache/vocab.json")
    if failures:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MLB game simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tokenize",
                       help="encode all games, verify round-trip, write vocab")
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.set_defaults(func=cmd_tokenize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
