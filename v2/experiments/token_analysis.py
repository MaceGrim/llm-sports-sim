#!/usr/bin/env python3
"""Full-corpus analysis to pin down the tokenization strategy.

Measures, across all game CSVs:
  1. events per game (sequence length raw material)
  2. cardinality of every candidate-token column (what we can take verbatim)
  3. play_length distribution + clock derivability (can we drop absolute times?)
  4. shot_distance distribution (programmatic distance tokens)
  5. secondary-actor rates (assist/block/steal sub-token frequency)
  6. exact tokens-per-game under the proposed scheme, and total vocab

Run from v2/:  python experiments/token_analysis.py
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "nba_data")
RESULTS = os.path.join(HERE, "..", "results", "token_analysis.json")

USECOLS = ["period", "play_length", "remaining_time", "event_type", "type",
           "player", "assist", "block", "steal", "entered", "left",
           "away", "home", "possession", "num", "outof", "shot_distance",
           "result", "reason", "team", "points", "original_x", "converted_x"]


def seconds(value) -> int:
    h, m, s = str(value).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def clock_token_count(play_len_s: int) -> int:
    """Programmatic clock vocab: one token per second 0..24, one for 25+."""
    return 1  # always exactly one token; vocab size handled separately


def tokens_for_game(df) -> dict:
    """Exact token count for one game under the proposed scheme.

    Header: [GAME] [away_team] [home_team] [LINEUP_A] p*5 [LINEUP_H] p*5 = 15
    Events (clock token = play_length seconds, always 1 token):
      start/end period       [START_Q] / [END_Q]                          1
      jump ball              [clk] [JUMP] [p_away] [p_home] [p_gains]     5
      shot                   [clk] [player] [shot_type] [dist] [made|miss] 5
        +assisted            [AST] [player]                              +2
        +blocked             [BLK] [player]                              +2
      free throw             [clk] [player] [ft_num_of] [made|miss]       4
      rebound                [clk] [player|TEAM] [reb_off|reb_def]        3
      turnover               [clk] [player|TEAM] [tov_reason]             3
        +stolen              [STL] [player]                              +2
      foul                   [clk] [player] [foul_reason]                 3
      substitution           [clk] [SUB] [out] [in]                       4
      timeout                [clk] [TIMEOUT] [team]                       3
      other/rare             [clk] [EVENT_TYPE] [player?]                 3
    """
    n = 15
    counts = Counter()
    for row in df.itertuples(index=False):
        ev = row.event_type
        if pd.isna(ev):
            continue
        ev = str(ev).strip()
        if ev in ("start of period", "end of period"):
            n += 1
        elif ev == "jump ball":
            n += 5
        elif ev == "shot":
            n += 5
            if pd.notna(row.assist):
                n += 2
            if pd.notna(row.block):
                n += 2
        elif ev == "free throw":
            n += 4
        elif ev in ("rebound", "foul"):
            n += 3
        elif ev == "turnover":
            n += 3
            if pd.notna(row.steal):
                n += 2
        elif ev == "substitution":
            n += 4
        elif ev == "timeout":
            n += 3
        else:
            n += 3
            counts[ev] += 1
    return {"tokens": n, "other_events": counts}


def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    print(f"Scanning {len(files)} games...\n")

    rows_per_game, tokens_per_game = [], []
    event_types = Counter()
    shot_types, ft_types, tov_reasons, foul_reasons, other_events = (
        Counter(), Counter(), Counter(), Counter(), Counter())
    results_vals, num_outof = Counter(), Counter()
    all_names = Counter()
    play_lengths = Counter()
    shot_dists_2pt, shot_dists_3pt = [], []
    coord_null = coord_total = 0
    clock_mismatch_periods = total_periods = 0
    ot_games = 0
    assisted = blocked = stolen = made_shots = shots = turnovers = 0

    for filename in files:
        df = pd.read_csv(os.path.join(DATA_DIR, filename), usecols=USECOLS)
        rows_per_game.append(len(df))
        event_types.update(df.event_type.dropna().str.strip())

        t = tokens_for_game(df)
        tokens_per_game.append(t["tokens"])
        other_events.update(t["other_events"])

        evt = df.event_type.fillna("")
        shots_df = df[evt == "shot"]
        shots += len(shots_df)
        made = shots_df.result == "made"
        made_shots += int(made.sum())
        assisted += int(shots_df.assist.notna().sum())
        blocked += int(shots_df.block.notna().sum())
        shot_types.update(shots_df.type.dropna().str.strip())
        is3 = shots_df.type.fillna("").str.contains("3pt")
        shot_dists_2pt.extend(shots_df.shot_distance[~is3].dropna().tolist())
        shot_dists_3pt.extend(shots_df.shot_distance[is3].dropna().tolist())
        coord_total += len(shots_df)
        coord_null += int(shots_df.original_x.isna().sum())

        ft_types.update(df[evt == "free throw"].type.dropna().str.strip())
        num_outof.update(
            f"{int(n)}/{int(o)}" for n, o in
            df[evt == "free throw"][["num", "outof"]].dropna().itertuples(index=False))
        tov = df[evt == "turnover"]
        turnovers += len(tov)
        stolen += int(tov.steal.notna().sum())
        tov_reasons.update(tov.reason.fillna("unknown").astype(str).str.strip())
        foul_reasons.update(df[evt == "foul"].reason.fillna("unknown").astype(str).str.strip())
        results_vals.update(df.result.dropna().astype(str).str.strip())

        for col in ("player", "assist", "block", "steal", "entered", "left",
                    "away", "home", "possession"):
            all_names.update(df[col].dropna().astype(str).str.strip())

        play_lengths.update(df.play_length.map(seconds))

        periods = sorted(df.period.unique())
        if max(periods) > 4:
            ot_games += 1
        for p in periods:
            total_periods += 1
            expect = 720 if p <= 4 else 300
            if df[df.period == p].play_length.map(seconds).sum() != expect:
                clock_mismatch_periods += 1

    rows = np.array(rows_per_game)
    toks = np.array(tokens_per_game)
    pl_total = sum(play_lengths.values())
    pl_le_24 = sum(c for s, c in play_lengths.items() if s <= 24)
    dist2 = np.array(shot_dists_2pt)
    dist3 = np.array(shot_dists_3pt)

    def stats(a):
        return {"mean": round(float(a.mean()), 1), "p50": int(np.percentile(a, 50)),
                "p95": int(np.percentile(a, 95)), "max": int(a.max())}

    report = {
        "games": len(files),
        "rows_per_game": stats(rows),
        "tokens_per_game": stats(toks),
        "total_corpus_tokens": int(toks.sum()),
        "event_types": dict(event_types.most_common()),
        "other_unmodeled_events": dict(other_events.most_common()),
        "distinct_names_all_columns": len(all_names),
        "names_under_50_appearances": sum(1 for c in all_names.values() if c < 50),
        "shot_type_cardinality": len(shot_types),
        "shot_types_top12": dict(shot_types.most_common(12)),
        "shot_types_under_100": sum(1 for c in shot_types.values() if c < 100),
        "ft_types": dict(ft_types.most_common()),
        "num_outof_values": dict(num_outof.most_common()),
        "tov_reason_cardinality": len(tov_reasons),
        "tov_reasons_top10": dict(tov_reasons.most_common(10)),
        "foul_reason_cardinality": len(foul_reasons),
        "foul_reasons_top10": dict(foul_reasons.most_common(10)),
        "result_values": dict(results_vals),
        "play_length_seconds": {
            "le_24s_pct": round(100 * pl_le_24 / pl_total, 2),
            "p99": int(np.percentile(np.repeat(list(play_lengths.keys()),
                                               list(play_lengths.values())), 99)),
            "max": max(play_lengths),
        },
        "clock_mismatch_periods_pct": round(100 * clock_mismatch_periods / total_periods, 2),
        "ot_games": ot_games,
        "shot_distance_2pt": {"p50": float(np.percentile(dist2, 50)),
                              "p95": float(np.percentile(dist2, 95)),
                              "max": float(dist2.max())},
        "shot_distance_3pt": {"p50": float(np.percentile(dist3, 50)),
                              "p95": float(np.percentile(dist3, 95)),
                              "max": float(dist3.max())},
        "shot_coord_null_pct": round(100 * coord_null / coord_total, 2),
        "assisted_made_shot_pct": round(100 * assisted / made_shots, 1),
        "blocked_shot_pct": round(100 * blocked / shots, 1),
        "stolen_turnover_pct": round(100 * stolen / turnovers, 1),
    }

    print(json.dumps(report, indent=2))
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
