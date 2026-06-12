#!/usr/bin/env python3
"""Interactive embedding explorer: PCA / t-SNE of the trained player
embeddings, colorable by per-game stats computed from the token stream.

Writes results/embeddings.html (self-contained, open in any browser).
Run from v2/:  python experiments/embedding_viz.py
"""

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim.games import load_games

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(HERE, "..", "results", "embeddings.html")

MIN_MINUTES = 500


def collect_metrics():
    """Per-game stats per player, walked straight off the token stream."""
    stats = defaultdict(Counter)  # player -> counter of raw totals
    games_played = Counter()

    with open(os.path.join(CACHE, "tokens.jsonl")) as f:
        for line in f:
            toks = json.loads(line)["tokens"]
            first_dt = next(i for i, t in enumerate(toks) if t.startswith("dt:"))
            for t in toks[:first_dt]:
                if t.startswith("P:"):
                    games_played[t[2:]] += 1

            for i, t in enumerate(toks[first_dt:], first_dt):
                prev = toks[i - 1]
                if t.startswith("A:") and prev.startswith("P:"):
                    p, kind = prev[2:], t[2:]
                    if kind.startswith("free throw"):
                        stats[p]["fta"] += 1
                        if toks[i + 1] == "made":
                            stats[p]["pts"] += 1
                    else:
                        made = toks[i + 2] == "made"
                        three = "3pt" in kind
                        stats[p]["fga"] += 1
                        stats[p]["3pa"] += three
                        stats[p]["3pm"] += three and made
                        stats[p]["dunks"] += "dunk" in kind and made
                        stats[p]["rim"] += ("layup" in kind or "dunk" in kind)
                        if made:
                            stats[p]["fgm"] += 1
                            stats[p]["pts"] += 3 if three else 2
                elif t in ("reb_off", "reb_def") and prev.startswith("P:"):
                    stats[prev[2:]]["reb"] += 1
                elif t.startswith("TO:") and prev.startswith("P:"):
                    stats[prev[2:]]["tov"] += 1
                elif t.startswith("F:") and prev.startswith("P:"):
                    stats[prev[2:]]["fouls"] += 1
                elif t in ("[AST]", "[BLK]", "[STL]", "[VS]"):
                    key = {"[AST]": "ast", "[BLK]": "blk",
                           "[STL]": "stl", "[VS]": "drawn"}[t]
                    stats[toks[i + 1][2:]][key] += 1
    return stats, games_played


def main():
    ckpt = torch.load(os.path.join(CACHE, "model.pt"),
                      map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    table = ckpt["state_dict"]["tok_emb.weight"].float().numpy()
    index = {t: i for i, t in enumerate(vocab)}

    games = load_games(os.path.join(CACHE, "games.jsonl"))
    minutes, team_minutes = Counter(), defaultdict(Counter)
    for g in games:
        for n, line in g.players.items():
            minutes[n] += line.minutes
            team_minutes[n][g.home if line.side == "home" else g.away] += line.minutes
    players = sorted(n for n, m in minutes.items()
                     if m >= MIN_MINUTES and f"P:{n}" in index)
    team = {n: team_minutes[n].most_common(1)[0][0] for n in players}

    stats, gp = collect_metrics()

    emb = np.stack([table[index[f"P:{p}"]] for p in players])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # cosine geometry
    pca = PCA(n_components=2, random_state=0).fit_transform(emb)
    tsne = TSNE(n_components=2, perplexity=30, random_state=0,
                init="pca").fit_transform(emb)
    print(f"{len(players)} players projected")

    def per_game(p, key):
        return round(stats[p][key] / max(gp[p], 1), 2)

    metrics = {
        "points/g": [per_game(p, "pts") for p in players],
        "3PM/g": [per_game(p, "3pm") for p in players],
        "3PA/g": [per_game(p, "3pa") for p in players],
        "dunks/g": [per_game(p, "dunks") for p in players],
        "rim attempts/g": [per_game(p, "rim") for p in players],
        "blocks/g": [per_game(p, "blk") for p in players],
        "steals/g": [per_game(p, "stl") for p in players],
        "rebounds/g": [per_game(p, "reb") for p in players],
        "assists/g": [per_game(p, "ast") for p in players],
        "FTA/g": [per_game(p, "fta") for p in players],
        "fouls drawn/g": [per_game(p, "drawn") for p in players],
        "turnovers/g": [per_game(p, "tov") for p in players],
        "minutes/g": [round(minutes[p] / max(gp[p], 1), 1) for p in players],
        "team (leakage check)": [sorted(set(team.values())).index(team[p])
                                 for p in players],
    }

    hover = [f"<b>{p}</b> ({team[p]}, {gp[p]} gp)<br>"
             f"{metrics['points/g'][i]} pts | {metrics['3PM/g'][i]} 3PM | "
             f"{metrics['dunks/g'][i]} dunks<br>"
             f"{metrics['rebounds/g'][i]} reb | {metrics['assists/g'][i]} ast | "
             f"{metrics['blocks/g'][i]} blk"
             for i, p in enumerate(players)]

    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        x=tsne[:, 0], y=tsne[:, 1], mode="markers",
        marker=dict(size=9, color=metrics["points/g"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="points/g"),
                    line=dict(width=0.5, color="white")),
        text=hover, hoverinfo="text"))

    proj_menu = dict(
        buttons=[
            dict(label="t-SNE", method="restyle",
                 args=[{"x": [tsne[:, 0]], "y": [tsne[:, 1]]}]),
            dict(label="PCA", method="restyle",
                 args=[{"x": [pca[:, 0]], "y": [pca[:, 1]]}]),
        ], direction="down", x=0.0, y=1.12, showactive=True)

    color_menu = dict(
        buttons=[
            dict(label=name, method="restyle",
                 args=[{"marker.color": [vals],
                        "marker.colorbar.title.text": name}])
            for name, vals in metrics.items()
        ], direction="down", x=0.18, y=1.12, showactive=True)

    fig.update_layout(
        title="EventGPT player embeddings (368 players, ≥500 min, 2022-23)",
        updatemenus=[proj_menu, color_menu],
        width=1100, height=800, template="plotly_white",
        xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.write_html(OUT, include_plotlyjs=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
