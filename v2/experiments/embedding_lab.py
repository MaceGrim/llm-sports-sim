#!/usr/bin/env python3
"""Embedding lab: why do player embeddings cluster by team, and what fixes it?

Builds count-based player embeddings (PPMI + SVD) from play-by-play under four
definitions of "context", then measures team leakage vs playstyle signal:

  CTX-full   context = nearby events incl. WHO acts (what an LM over pbp sees)
  CTX-anon   context = nearby events, player identities stripped
  ACT        context = the player's own actions only
  CTX-team   CTX-full with the player's team centroid subtracted post-hoc

Metrics (players with >= MIN_MINUTES season minutes):
  teammate@10  fraction of 10 nearest cosine neighbors on the same team (leakage)
  style@10     neighbor overlap vs an independent action-profile reference
               built from the OTHER half of games (even/odd date split, so the
               reference is not circular with the embeddings)

Run from v2/:  python experiments/embedding_lab.py
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim.games import load_games

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "nba_data")
CACHE = os.path.join(HERE, "..", "cache", "games.jsonl")
RESULTS = os.path.join(HERE, "..", "results", "embedding_lab.json")

MIN_MINUTES = 500
WINDOW = 3  # events on each side that count as context
DIMS = [16, 64, 128]
K = 10

EXAMPLE_PLAYERS = ["Stephen Curry", "Joel Embiid", "Nikola Jokic", "Rudy Gobert",
                   "James Harden", "Klay Thompson"]


def action_token(row) -> str:
    """One playstyle token per event row, credited to row['player']."""
    ev = row.event_type
    if ev == "shot":
        kind = str(row.type)
        if "3pt" in kind:
            bucket = "3pt"
        else:
            d = row.shot_distance
            d = float(d) if pd.notna(d) else 10.0
            bucket = "rim" if d <= 3 else "short" if d <= 9 else "mid" if d <= 16 else "long2"
        made = "made" if row.result == "made" else "miss"
        return f"shot_{bucket}_{made}"
    if ev == "free throw":
        return "ft_made" if row.result == "made" else "ft_miss"
    if ev == "rebound":
        return "reb_off" if "offensive" in str(row.type) else "reb_def"
    if ev == "turnover":
        return "tov"
    if ev == "foul":
        return "foul"
    return ""


def extract_sequences():
    """Per game: ordered list of (player, action) incl. secondary actors.

    Returns (sequences, split) where split[i] is 0/1 by date-order parity.
    """
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    sequences, splits = [], []
    for idx, filename in enumerate(files):
        df = pd.read_csv(os.path.join(DATA_DIR, filename),
                         usecols=["player", "event_type", "type", "result",
                                  "shot_distance", "assist", "block", "steal"])
        seq = []
        for row in df.itertuples(index=False):
            act = action_token(row)
            if act and pd.notna(row.player):
                seq.append((str(row.player).strip(), act))
                made_shot = act.startswith("shot") and act.endswith("made")
            else:
                made_shot = False
            for col, tok in (("assist", "ast"), ("block", "blk"), ("steal", "stl")):
                name = getattr(row, col)
                if pd.notna(name) and str(name).strip():
                    seq.append((str(name).strip(), tok))
                    if col == "assist" and made_shot:
                        seq[-2] = (seq[-2][0], seq[-2][1] + "_assisted")
        sequences.append(seq)
        splits.append(idx % 2)
    return sequences, splits


def player_universe():
    """Players with >= MIN_MINUTES, and their primary (most-minutes) team."""
    games = load_games(CACHE)
    minutes = Counter()
    team_minutes = defaultdict(Counter)
    for g in games:
        for name, line in g.players.items():
            minutes[name] += line.minutes
            team_minutes[name][g.home if line.side == "home" else g.away] += line.minutes
    keep = sorted(n for n, m in minutes.items() if m >= MIN_MINUTES)
    primary = {n: team_minutes[n].most_common(1)[0][0] for n in keep}
    return keep, primary


def count_contexts(sequences, splits, players):
    """Count player->context-token occurrences for each variant, plus reference profiles."""
    pset = set(players)
    ctx_full = defaultdict(Counter)   # embedding half: actors + actions in window
    ctx_anon = defaultdict(Counter)   # embedding half: actions only in window
    act_own = defaultdict(Counter)    # embedding half: own actions
    ref_profile = defaultdict(Counter)  # reference half: own actions

    for seq, split in zip(sequences, splits):
        if split == 0:  # reference half
            for player, act in seq:
                if player in pset:
                    ref_profile[player][act] += 1
            continue
        for i, (player, act) in enumerate(seq):
            if player not in pset:
                continue
            act_own[player][f"A:{act}"] += 1
            lo, hi = max(0, i - WINDOW), min(len(seq), i + WINDOW + 1)
            for j in range(lo, hi):
                other_player, other_act = seq[j]
                ctx_full[player][f"A:{other_act}"] += 1
                ctx_anon[player][f"A:{other_act}"] += 1
                if j != i:
                    ctx_full[player][f"P:{other_player}"] += 1
    return ctx_full, ctx_anon, act_own, ref_profile


def ppmi_svd(counts, players, dim):
    """Counts dict -> PPMI matrix -> rank-dim SVD embeddings (rows = players)."""
    vocab = sorted({tok for p in players for tok in counts[p]})
    col = {tok: i for i, tok in enumerate(vocab)}
    M = np.zeros((len(players), len(vocab)))
    for r, p in enumerate(players):
        for tok, c in counts[p].items():
            M[r, col[tok]] = c

    total = M.sum()
    row_p = M.sum(axis=1, keepdims=True) / total
    col_p = M.sum(axis=0, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M / total) / (row_p @ col_p))
    ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)

    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    dim = min(dim, len(S))
    return U[:, :dim] * np.sqrt(S[:dim])


def cosine_neighbors(emb, k):
    """Top-k neighbor indices for each row under cosine similarity."""
    X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)
    return np.argsort(-sims, axis=1)[:, :k]


def evaluate(emb, players, primary, ref_neighbors, k=K):
    nn = cosine_neighbors(emb, k)
    teammate = np.mean([
        np.mean([primary[players[j]] == primary[players[i]] for j in nn[i]])
        for i in range(len(players))
    ])
    style = np.mean([
        len(set(nn[i]) & set(ref_neighbors[i])) / k for i in range(len(players))
    ])
    return round(float(teammate), 3), round(float(style), 3)


def main():
    print("Extracting event sequences from all games...")
    sequences, splits = extract_sequences()
    players, primary = player_universe()
    print(f"{len(sequences)} games, {len(players)} players with >= {MIN_MINUTES} min")

    ctx_full, ctx_anon, act_own, ref_profile = count_contexts(sequences, splits, players)

    # Independent playstyle reference: action distributions from the OTHER half.
    ref_emb = np.zeros((len(players), 0))
    ref_vocab = sorted({t for p in players for t in ref_profile[p]})
    rcol = {t: i for i, t in enumerate(ref_vocab)}
    R = np.zeros((len(players), len(ref_vocab)))
    for r, p in enumerate(players):
        for t, c in ref_profile[p].items():
            R[r, rcol[t]] = c
    R = R / (R.sum(axis=1, keepdims=True) + 1e-12)  # action distribution
    ref_neighbors = cosine_neighbors(R, K)

    # Expected leakage floor: same-team rate among reference (pure-style) neighbors.
    style_floor = np.mean([
        np.mean([primary[players[j]] == primary[players[i]] for j in ref_neighbors[i]])
        for i in range(len(players))
    ])
    print(f"\nReference check: same-team rate among pure-playstyle neighbors = "
          f"{style_floor:.3f} (chance ~ {9/len(players):.3f})\n")

    results = {"style_floor_teammate_rate": round(float(style_floor), 3), "runs": []}
    header = f"{'variant':<12} {'dim':>4}   {'teammate@10':>11}   {'style@10':>8}"
    print(header + "\n" + "-" * len(header))

    team_centroids = {}
    for dim in DIMS:
        embs = {
            "CTX-full": ppmi_svd(ctx_full, players, dim),
            "CTX-anon": ppmi_svd(ctx_anon, players, dim),
            "ACT": ppmi_svd(act_own, players, dim),
        }
        # CTX-team: subtract each player's primary-team centroid from CTX-full.
        base = embs["CTX-full"]
        centroids = defaultdict(lambda: np.zeros(base.shape[1]))
        counts = Counter()
        for i, p in enumerate(players):
            centroids[primary[p]] += base[i]
            counts[primary[p]] += 1
        embs["CTX-team"] = np.array([
            base[i] - centroids[primary[p]] / counts[primary[p]]
            for i, p in enumerate(players)
        ])

        for name, emb in embs.items():
            teammate, style = evaluate(emb, players, primary, ref_neighbors)
            print(f"{name:<12} {dim:>4}   {teammate:>11.3f}   {style:>8.3f}")
            results["runs"].append({"variant": name, "dim": dim,
                                    "teammate_at_10": teammate, "style_at_10": style})
        print()

    # Qualitative: nearest neighbors for well-known players, best style variant.
    emb = ppmi_svd(act_own, players, 64)
    nn = cosine_neighbors(emb, 5)
    idx = {p: i for i, p in enumerate(players)}
    print("ACT-64 nearest neighbors (qualitative):")
    examples = {}
    for name in EXAMPLE_PLAYERS:
        if name in idx:
            neighbors = [players[j] for j in nn[idx[name]]]
            examples[name] = neighbors
            print(f"  {name:<18} -> {', '.join(neighbors)}")
    results["example_neighbors_ACT_64"] = examples

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
