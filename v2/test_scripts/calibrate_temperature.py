#!/usr/bin/env python3
"""Per-slot temperature calibration (#17, Gate A of the Definition of GOOD).

Sweeps (outcome, action) sampling temperatures on a DEV window — Jan-Feb
2025 regular-season games, before the 2025-03-01 val cutoff — and measures
margin coverage (p10-p90; target [78, 82]), per-game margin sd, and dev
Brier under the protocol-v2 smoothed win probability. The 343-game val set
is never touched here: tune on dev, evaluate there once.

Caveat (recorded in TODO.md): dev games are inside the model's training
window. Temperature is a 2-parameter sampler property, so the leak risk is
small; the next retrain moves --val-cutoff to 2025-01-01 to eliminate it.

Run from v2/:  python test_scripts/calibrate_temperature.py
               [--games 48] [--sims 50] [--temps-out 1.0,0.9,0.8]
               [--temps-act 1.0,0.9,0.8]
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sim.model import Config, EventGPT, pick_device
from sim.sample import generate_games
from sim.tokenizer import Replay

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DEV_START, DEV_END = "2025-01-01", "2025-03-01"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=48)
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temps-out", default="1.0,0.9,0.8")
    p.add_argument("--temps-act", default="1.0,0.9,0.8")
    p.add_argument("--ckpt", default=os.path.join(CACHE, "model.pt"))
    args = p.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    model = EventGPT(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(os.path.join(CACHE, "tokens.jsonl")) as f:
        dev = [g for g in (json.loads(l) for l in f)
               if g["game_id"].startswith("002")
               and DEV_START <= g["date"] < DEV_END]
    rng = np.random.default_rng(args.seed)
    dev = [dev[i] for i in rng.choice(len(dev), size=min(args.games, len(dev)),
                                      replace=False)]
    print(f"device={device}  dev games={len(dev)}  sims={args.sims}", flush=True)

    grid = [(o, a) for o in map(float, args.temps_out.split(","))
            for a in map(float, args.temps_act.split(","))]

    # Flat job list: games interleaved into full-size batches so the GPU
    # always runs at args.batch, instead of one under-sized batch per game.
    jobs, actuals = [], []
    for gi, g in enumerate(dev):
        toks = g["tokens"]
        header = toks[:toks.index("[START_Q]") + 11]
        jobs.extend((gi, header) for _ in range(args.sims))
        real = Replay(toks).run()
        actuals.append(real.home_score - real.away_score)

    results = []
    for o, a in grid:
        temps = {"outcome": o, "action": a}
        by_game = defaultdict(list)
        t0 = time.time()
        for s in range(0, len(jobs), args.batch):
            chunk = jobs[s:s + args.batch]
            sims = generate_games(model, vocab, [h for _, h in chunk], device,
                                  seed=args.seed + s, temperature=temps)
            for (gi, _), sim in zip(chunk, sims):
                if sim[-1] == "[EOG]":
                    r = Replay(sim).run()
                    by_game[gi].append(r.home_score - r.away_score)
        cov, sds, briers, maes = [], [], [], []
        for gi, actual in enumerate(actuals):
            margins = np.array(by_game[gi])
            cov.append(np.percentile(margins, 10) <= actual
                       <= np.percentile(margins, 90))
            sds.append(margins.std(ddof=1))
            mu, sd = margins.mean(), margins.std(ddof=1)
            prob = 0.5 * (1 + math.erf(mu / (sd * math.sqrt(2)))) if sd else 0.5
            briers.append((prob - (actual > 0)) ** 2)
            maes.append(abs(mu - actual))
        row = {"outcome": o, "action": a,
               "coverage": round(float(np.mean(cov)), 3),
               "margin_sd": round(float(np.mean(sds)), 1),
               "brier": round(float(np.mean(briers)), 4),
               "margin_mae": round(float(np.mean(maes)), 1),
               "secs": round(time.time() - t0)}
        results.append(row)
        print(f"out={o} act={a}  coverage {row['coverage']:.0%}  "
              f"sd {row['margin_sd']}  brier {row['brier']}  "
              f"mae {row['margin_mae']}  ({row['secs']}s)", flush=True)

    out = os.path.join(HERE, "..", "results", "calibration_grid.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dev_window": [DEV_START, DEV_END], "games": len(dev),
                   "sims": args.sims, "grid": results}, f, indent=2)
    # Gate band is [0.76, 0.84] (+/-2 SE at val n=343); at dev n=48 the SE is
    # ~5.8pp, so closest-to-0.80 does the real work and the band just trims.
    in_band = [r for r in results if 0.76 <= r["coverage"] <= 0.84]
    pick = min(in_band or results, key=lambda r: (abs(r["coverage"] - 0.80),
                                                  r["brier"]))
    print(f"\nbest: outcome={pick['outcome']} action={pick['action']} "
          f"(coverage {pick['coverage']:.0%}, brier {pick['brier']})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
