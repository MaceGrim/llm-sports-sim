#!/usr/bin/env python3
"""CLI for the v2 NBA simulator.

  python run.py build-cache                 # parse nba_data/ into cache/games.jsonl
  python run.py simulate PHI BOS            # simulate PHI @ BOS (away first)
  python run.py evaluate --start-date 2023-01-15
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.evaluate import backtest
from sim.form import team_form
from sim.games import build_cache, load_games, parse_filename
from sim.simulate import StatisticalSimulator, simulate_matchup
from sim.tokenizer import (Replay, build_vocab, canonical, encode_game,
                           has_corrupt_lineups, has_corrupt_points, save_vocab)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "games.jsonl")
DEFAULT_DATA = os.path.join(HERE, "..", "nba_data")


def cmd_build_cache(args):
    n = build_cache(args.data_dir, CACHE)
    print(f"Cached {n} games to {CACHE}")


def _load_games():
    if not os.path.exists(CACHE):
        sys.exit(f"No cache at {CACHE} - run: python run.py build-cache")
    return load_games(CACHE)


def cmd_simulate(args):
    games = _load_games()
    as_of = args.date or max(g.date for g in games) + "~"  # after last game: full-season form
    sim = StatisticalSimulator(games)

    summary = simulate_matchup(sim, args.away, args.home, as_of,
                               n_sims=args.sims, seed=args.seed)
    sample = summary["sample_game"]

    print(f"\n{args.away} @ {args.home}  (form as of {args.date or 'end of dataset'}, "
          f"{args.sims} simulations)\n")
    print(f"  {args.home} win probability: {summary['home_win_prob']:.0%}")
    print(f"  Mean score: {args.away} {summary['mean_score']['away']} - "
          f"{summary['mean_score']['home']} {args.home}")
    p10, p50, p90 = summary["margin_p10_p50_p90"]
    print(f"  Home margin p10/p50/p90: {p10:+d} / {p50:+d} / {p90:+d}\n")

    print(f"  One simulated game: {args.away} {sample.away_score} - "
          f"{sample.home_score} {args.home}")
    quarters = "  ".join(f"Q{i+1} {a}-{h}" for i, (a, h) in enumerate(sample.periods))
    print(f"  {quarters}\n")
    for team in (args.away, args.home):
        form = sim._form(team, as_of)
        names = {p.name for p in form.players}
        print(f"  {team}:")
        lines = sorted(((n, l) for n, l in sample.players.items() if n in names),
                       key=lambda x: -x[1].pts)
        for name, line in lines:
            print(f"    {name:<24} {line.pts:>3} pts {line.reb:>3} reb {line.ast:>3} ast")
        print()


def cmd_evaluate(args):
    games = _load_games()
    sim = StatisticalSimulator(games)
    result = backtest(sim, games, start_date=args.start_date,
                      min_prior_games=args.min_prior_games,
                      n_sims=args.sims, seed=args.seed, limit=args.limit)

    s = result["summary"]
    print(f"\nBacktest: {s['n_games']} games from {args.start_date}, "
          f"{args.sims} sims/game\n")
    print(f"  Pick accuracy:    {s['pick_accuracy']:.1%}   "
          f"(home-always {s['baselines']['home_always']['pick_accuracy']:.1%}, "
          f"better-record {s['baselines']['better_record_wins']['pick_accuracy']:.1%})")
    print(f"  Brier score:      {s['brier']:.4f}  "
          f"(home-always@0.58 {s['baselines']['home_always']['brier']:.4f})")
    print(f"  Log loss:         {s['log_loss']:.4f}")
    print(f"  Margin MAE:       {s['margin_mae']:.1f} pts")
    print(f"  Total MAE:        {s['total_mae']:.1f} pts")
    print(f"  Margin coverage (p10-p90): {s['margin_coverage_p10_p90']:.1%}  (target ~80%)")
    print("\n  Calibration (predicted home-win prob vs actual):")
    for b in s["calibration"]:
        print(f"    {b['bucket']}: predicted {b['predicted']:.2f}, "
              f"actual {b['actual']:.2f}  (n={b['n']})")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"backtest_{args.start_date}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull results: {out}")


def cmd_tokenize(args):
    """Encode every game, verify the round-trip contract, write tokens + vocab."""
    truth = {g.game_id: g for g in _load_games()}
    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".csv"))

    all_tokens, lengths, failures = [], [], []
    waived = {"minutes (corrupt source lineups)": [],
              "score (corrupt source points)": []}
    out_path = os.path.join(HERE, "cache", "tokens.jsonl")
    with open(out_path, "w") as out:
        for filename in files:
            date, game_id, away, home = parse_filename(filename)
            path = os.path.join(args.data_dir, filename)
            tokens = encode_game(path)
            replay = Replay(tokens).run()

            g = truth[game_id]
            points_ok = (replay.away_score == g.away_score
                         and replay.home_score == g.home_score
                         and replay.period_scores == g.periods
                         and all(replay.box[canonical(n)]["pts"] == p.pts
                                 for n, p in g.players.items()))
            counts_ok = all(replay.box[canonical(n)]["reb"] == p.reb
                            and replay.box[canonical(n)]["ast"] == p.ast
                            for n, p in g.players.items())
            # ±2 min: source lineup columns flicker on some FT rows
            # (data noise the cache inherits; replay uses sub events)
            minutes_ok = all(abs(replay.minutes(canonical(n)) - p.minutes) <= 2.0
                             for n, p in g.players.items())
            if points_ok and counts_ok and minutes_ok:
                pass
            elif points_ok and counts_ok and has_corrupt_lineups(path):
                # see has_corrupt_lineups docstring
                waived["minutes (corrupt source lineups)"].append(filename)
            elif counts_ok and minutes_ok and has_corrupt_points(path):
                # see has_corrupt_points docstring
                waived["score (corrupt source points)"].append(filename)
            else:
                failures.append(filename)
            out.write(json.dumps({"game_id": game_id, "date": date,
                                  "tokens": tokens}) + "\n")
            all_tokens.append(tokens)
            lengths.append(len(tokens))

    vocab = build_vocab(all_tokens)
    save_vocab(vocab, os.path.join(HERE, "cache", "vocab.json"))

    import numpy as np
    lengths = np.array(lengths)
    n_waived = sum(len(v) for v in waived.values())
    print(f"Round-trip: {len(files) - len(failures) - n_waived}/{len(files)} games exact"
          + (f"  FAILURES: {failures[:5]}" if failures else ""))
    for reason, names in waived.items():
        if names:
            print(f"  waived {reason}: {names}")
    print(f"Tokens/game: mean {lengths.mean():.0f}, p95 {np.percentile(lengths, 95):.0f}, "
          f"max {lengths.max()}")
    print(f"Corpus: {lengths.sum():,} tokens   Vocab: {len(vocab)}")
    print(f"Wrote {out_path} and cache/vocab.json")
    if failures:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="NBA game simulator (v2)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-cache", help="parse all game CSVs into the cache")
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.set_defaults(func=cmd_build_cache)

    p = sub.add_parser("tokenize", help="encode all games, verify round-trip, write vocab")
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.set_defaults(func=cmd_tokenize)

    p = sub.add_parser("simulate", help="simulate AWAY @ HOME")
    p.add_argument("away", help="away team abbreviation, e.g. PHI")
    p.add_argument("home", help="home team abbreviation, e.g. BOS")
    p.add_argument("--date", help="use only games before this date (YYYY-MM-DD)")
    p.add_argument("--sims", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("evaluate", help="backtest the simulator on held-out games")
    p.add_argument("--start-date", default="2023-01-15")
    p.add_argument("--min-prior-games", type=int, default=15)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="cap number of games (0 = all)")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
