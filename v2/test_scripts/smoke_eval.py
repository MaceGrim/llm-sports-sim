#!/usr/bin/env python3
"""Smoke test: generate full games from the trained model and compare the
distributions of scores, outcomes, and player stats against held-out reality.

Run from v2/:  python test_scripts/smoke_eval.py [--games 16] [--seed 0]
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sim.games import load_games
from sim.model import Config, EventGPT, pick_device
from sim.sample import generate_games
from sim.tokenizer import Replay

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-cutoff", default="2025-03-01",
                   help="must match the cutoff the checkpoint was trained with")
    args = p.parse_args()

    device = pick_device()
    ckpt = torch.load(os.path.join(CACHE, "model.pt"),
                      map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    model = EventGPT(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"checkpoint from step {ckpt.get('step', '?')}, device={device}")

    # Validation games (regular season, after the training cutoff).
    with open(os.path.join(CACHE, "tokens.jsonl")) as f:
        val = [json.loads(l) for l in f]
    val = [g for g in val if g["game_id"].startswith("002") and g["date"] >= args.val_cutoff]
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(val), size=args.games, replace=False)

    # Season-long per-player averages (for correlation targets).
    games = load_games(os.path.join(CACHE, "games.jsonl"))
    tot = defaultdict(lambda: Counter())
    for g in games:
        for n, line in g.players.items():
            tot[n].update({"pts": line.pts, "min": line.minutes, "gp": 1})
    season_ppg = {n: c["pts"] / c["gp"] for n, c in tot.items() if c["gp"] >= 10}
    season_mpg = {n: c["min"] / c["gp"] for n, c in tot.items() if c["gp"] >= 10}

    import time
    chosen = [val[gi] for gi in picks]
    headers = [g["tokens"][:g["tokens"].index("[START_Q]") + 11] for g in chosen]
    t0 = time.time()
    sims = []
    for start in range(0, len(headers), 16):
        sims += generate_games(model, vocab, headers[start:start + 16],
                               device, seed=args.seed + start)
    print(f"generated {len(sims)} games in {time.time() - t0:.1f}s "
          f"({(time.time() - t0) / len(sims):.2f}s/game)")

    sim_totals, sim_margins, winners_right = [], [], 0
    sim_fgm = sim_fga = sim_3pm = sim_3pa = sim_fta = 0
    pts_pairs, min_pairs, top_hits = [], [], 0

    truncated = sum(1 for s in sims if s[-1] != "[EOG]")
    if truncated:
        print(f"WARNING: {truncated} game(s) hit the length cap unfinished — skipped")

    n_used = 0
    for g, sim in zip(chosen, sims):
        if sim[-1] != "[EOG]":
            continue
        n_used += 1
        toks = g["tokens"]
        r = Replay(sim).run()
        real = Replay(toks).run()
        sim_totals += [r.away_score, r.home_score]
        sim_margins.append(r.home_score - r.away_score)
        sim_winner = "home" if r.home_score > r.away_score else "away"
        real_winner = "home" if real.home_score > real.away_score else "away"
        winners_right += sim_winner == real_winner

        # shooting from tokens
        for i, t in enumerate(sim):
            if t.startswith("A:"):
                if t[2:].startswith("free throw"):
                    sim_fta += 1
                    continue
                made = sim[i + 2] == "made"
                three = "3pt" in t
                sim_fga += 1
                sim_fgm += made
                sim_3pa += three
                sim_3pm += three and made

        # player-level: sim pts/min vs season averages
        team_real_top = sorted(real.box, key=lambda n: -real.box[n]["pts"])[:3]
        sim_top = max(r.box, key=lambda n: r.box[n]["pts"])
        top_hits += sim_top in team_real_top
        for n, line in r.box.items():
            if n in season_ppg:
                pts_pairs.append((line["pts"], season_ppg[n]))
                min_pairs.append((line["sec"] / 60, season_mpg[n]))
        print(f"  {g['date']} {toks[1][5:]}@{toks[2][5:]}: "
              f"sim {r.away_score}-{r.home_score} (real {real.away_score}-{real.home_score})")

    # Reference distributions from ALL validation games.
    real_totals, real_margins = [], []
    real_fgm = real_fga = real_3pm = real_3pa = real_fta = 0
    for g in val:
        rr = Replay(g["tokens"]).run()
        real_totals += [rr.away_score, rr.home_score]
        real_margins.append(rr.home_score - rr.away_score)
        for i, t in enumerate(g["tokens"]):
            if t.startswith("A:"):
                if t[2:].startswith("free throw"):
                    real_fta += 1
                    continue
                made = g["tokens"][i + 2] == "made"
                three = "3pt" in t
                real_fga += 1; real_fgm += made
                real_3pa += three; real_3pm += three and made

    st, rt = np.array(sim_totals), np.array(real_totals)
    sm, rm = np.array(sim_margins), np.array(real_margins)
    pp = np.array(pts_pairs); mp = np.array(min_pairs)

    print(f"\n=== SMOKE REPORT ({n_used} simulated games vs {len(val)} real val games) ===")
    print(f"Team totals:    sim {st.mean():.1f} ± {st.std():.1f}   real {rt.mean():.1f} ± {rt.std():.1f}")
    print(f"Home margin:    sim {sm.mean():+.1f} ± {sm.std():.1f}   real {rm.mean():+.1f} ± {rm.std():.1f}")
    print(f"FG%:            sim {sim_fgm/max(sim_fga,1):.3f}   real {real_fgm/real_fga:.3f}")
    print(f"3P%:            sim {sim_3pm/max(sim_3pa,1):.3f}   real {real_3pm/real_3pa:.3f}")
    print(f"FGA/game:       sim {sim_fga/max(n_used,1)/2:.1f}   real {real_fga/len(val)/2:.1f}")
    print(f"FTA/game:       sim {sim_fta/max(n_used,1)/2:.1f}   real {real_fta/len(val)/2:.1f}")
    print(f"Winner (1 sim): {winners_right}/{n_used} correct")
    print(f"Sim top scorer in real team top-3: {top_hits}/{n_used}")
    print(f"Player pts corr (sim game vs season avg): r={np.corrcoef(pp[:,0], pp[:,1])[0,1]:.3f}  (n={len(pp)})")
    print(f"Player min corr (sim game vs season avg): r={np.corrcoef(mp[:,0], mp[:,1])[0,1]:.3f}")


if __name__ == "__main__":
    main()
