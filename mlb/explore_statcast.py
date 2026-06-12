#!/usr/bin/env python3
"""First look at Statcast as a token corpus: what's the event structure, what
are the closed sets, what would a game cost in tokens, and can we round-trip
the score? (The MLB analog of v2/experiments/token_analysis.py.)

Statcast rows are PITCHES, grouped by game_pk -> at_bat_number -> pitch_number.
The candidate grammar, following v2's actor-first lesson (here the "actors"
are the pitcher-batter matchup, and WHO bats next is deterministic from the
lineup — baseball moves v2's hardest slot, usage, into the state machine):

    per PA:    [PA] P:pitcher B:batter  then per pitch:
               T:<pitch_type> Z:<zone> R:<result desc>
               ... final pitch carries E:<events> (+ batted-ball suffixes)

Usage: python explore_statcast.py [--start 2024-06-05] [--end 2024-06-11]
"""

import argparse
import warnings
from collections import Counter

import pandas as pd

warnings.filterwarnings("ignore")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-06-05")
    p.add_argument("--end", default="2024-06-11")
    args = p.parse_args()

    from pybaseball import cache, statcast
    cache.enable()
    df = statcast(start_dt=args.start, end_dt=args.end, verbose=False)
    df = df[df.game_type == "R"]  # regular season
    games = df.groupby("game_pk")

    print(f"{len(df):,} pitches, {games.ngroups} regular-season games "
          f"({args.start}..{args.end})")
    print(f"pitches/game: mean {len(df)/games.ngroups:.0f}, "
          f"max {games.size().max()}")
    pa = df.groupby(["game_pk", "at_bat_number"])
    print(f"PAs/game: {pa.ngroups/games.ngroups:.0f}, "
          f"pitches/PA: {len(df)/pa.ngroups:.2f}")

    print("\n-- closed sets (verbatim token candidates) --")
    for col in ("pitch_type", "description", "events", "bb_type"):
        vals = Counter(df[col].dropna().astype(str))
        print(f"{col}: {len(vals)} values")
        for v, c in vals.most_common(30):
            print(f"   {v}: {c}")

    print("\n-- numeric (programmatic token candidates) --")
    for col, desc in [("zone", "pitch location zone (1-14)"),
                      ("launch_speed", "exit velo mph"),
                      ("launch_angle", "degrees"),
                      ("hit_distance_sc", "feet"),
                      ("release_speed", "pitch mph")]:
        s = df[col].dropna()
        print(f"{col} ({desc}): coverage {len(s)/len(df):.1%}, "
              f"p5 {s.quantile(.05):.0f}, p50 {s.quantile(.5):.0f}, "
              f"p95 {s.quantile(.95):.0f}")

    print("\n-- players --")
    print(f"distinct batters: {df.batter.nunique()}, "
          f"pitchers: {df.pitcher.nunique()}")

    print("\n-- state already on every row (channel candidates) --")
    print("inning/topbot/outs/balls/strikes/on_1b/on_2b/on_3b/score: "
          f"nulls = {df[['inning','outs_when_up','balls','strikes']].isna().sum().sum()}")

    # Round-trip feasibility: per-row score deltas must sum to the final.
    print("\n-- round-trip check: sum of per-PA run deltas == final score --")
    bad = 0
    for pk, g in games:
        g = g.sort_values(["at_bat_number", "pitch_number"])
        runs = (g.post_bat_score - g.bat_score).clip(lower=0).sum()
        final = g.iloc[-1][["post_home_score", "post_away_score"]].sum()
        if runs != final:
            bad += 1
    print(f"games where per-row deltas mismatch the final: {bad}/{games.ngroups}")

    # Token budget estimate: PA header (3) + 3/pitch + ~2 extra on the last.
    est = pa.ngroups / games.ngroups * 3 + len(df) / games.ngroups * 3 + 76 * 2
    print(f"\nestimated tokens/game at 3/pitch + PA headers: ~{est:.0f}")


if __name__ == "__main__":
    main()
