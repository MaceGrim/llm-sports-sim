#!/usr/bin/env python3
"""Ad hoc probe: do the trained model's player embeddings cluster by playstyle
or by team? Reuses the embedding-lab metrics so numbers are comparable:
  count-based reference points -- CTX-full: teammate 0.89 / style 0.03
                                  ACT:      teammate 0.03 / style 0.20

Run from v2/:  python test_scripts/probe_embeddings.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from experiments.embedding_lab import (EXAMPLE_PLAYERS, K, cosine_neighbors,
                                       count_contexts, extract_sequences,
                                       player_universe)
from sim.tokenizer import load_vocab

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "cache", "model.pt")


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    emb_table = ckpt["state_dict"]["tok_emb.weight"].float().numpy()
    index = {tok: i for i, tok in enumerate(vocab)}

    players, primary = player_universe()
    players = [p for p in players if f"P:{p}" in index]
    emb = np.stack([emb_table[index[f"P:{p}"]] for p in players])
    print(f"{len(players)} players (>=500 min), embedding dim {emb.shape[1]}")

    # Independent playstyle reference: action profiles from half the games
    # (same construction as the embedding lab, so style@10 is comparable).
    sequences, splits = extract_sequences()
    _, _, _, ref_profile = count_contexts(sequences, splits, players)
    ref_vocab = sorted({t for p in players for t in ref_profile[p]})
    col = {t: i for i, t in enumerate(ref_vocab)}
    R = np.zeros((len(players), len(ref_vocab)))
    for r, p in enumerate(players):
        for t, c in ref_profile[p].items():
            R[r, col[t]] = c
    R = R / (R.sum(axis=1, keepdims=True) + 1e-12)
    ref_nn = cosine_neighbors(R, K)

    nn = cosine_neighbors(emb, K)
    teammate = np.mean([
        np.mean([primary[players[j]] == primary[players[i]] for j in nn[i]])
        for i in range(len(players))])
    style = np.mean([len(set(nn[i]) & set(ref_nn[i])) / K
                     for i in range(len(players))])
    print(f"teammate@10 = {teammate:.3f}   style@10 = {style:.3f}")

    idx = {p: i for i, p in enumerate(players)}
    nn5 = cosine_neighbors(emb, 5)
    for name in EXAMPLE_PLAYERS:
        if name in idx:
            print(f"  {name:<18} -> {', '.join(players[j] for j in nn5[idx[name]])}")


if __name__ == "__main__":
    main()
