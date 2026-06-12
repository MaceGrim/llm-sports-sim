#!/usr/bin/env python3
"""Backtest the transformer as a game predictor, mirroring the statistical
baseline's protocol (sim/evaluate.py): N sims per game -> home win prob,
scored on picks / Brier / log loss / margin MAE / coverage.

Conditioning: the real game's header (teams, rosters, Q1 starters) — i.e. the
night's actual actives and starting lineups, information a pre-game predictor
plausibly has. The baseline instead uses season-to-date team form.

Protocol v2 (Mason, 2026-06-12): 50 sims/game, and the home-win probability
comes from a normal fit to the simulated margins — at 50 sims an empirical
frequency adds ~0.005 of pure estimation noise to Brier, which would swamp
the differences we're measuring. Compare only against baselines run under
the same protocol. --temps applies per-slot sampling temperatures (#17),
e.g. '{"outcome": 0.9, "action": 0.9}'.

Run from v2/:  python test_scripts/backtest_transformer.py [--sims 50]
               [--limit 0] [--batch 64] [--val-cutoff 2025-03-01]
               [--temps JSON]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sim.model import Config, EventGPT, pick_device
from sim.sample import generate_games
from sim.tokenizer import Replay

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")


def smooth_win_prob(margins) -> float:
    """P(home win) from a normal fit to the simulated margins."""
    mu, sd = margins.mean(), margins.std(ddof=1)
    if sd == 0:
        return 0.5 if mu == 0 else float(mu > 0)
    return 0.5 * (1 + math.erf(mu / (sd * math.sqrt(2))))


def simulate_all(model, vocab, headers, n_sims, batch, device, seed, temps=1.0):
    """n_sims rollouts of every header, flat-batched so the GPU always runs
    full batches -> (margins per game, totals per game, n_truncated)."""
    jobs = [(gi, h) for gi, h in enumerate(headers) for _ in range(n_sims)]
    margins = [[] for _ in headers]
    totals = [[] for _ in headers]
    truncated = 0
    t0 = time.time()
    for s in range(0, len(jobs), batch):
        chunk = jobs[s:s + batch]
        sims = generate_games(model, vocab, [h for _, h in chunk], device,
                              seed=seed + s, temperature=temps)
        for (gi, _), sim in zip(chunk, sims):
            if sim[-1] != "[EOG]":
                truncated += 1
                continue
            r = Replay(sim).run()
            margins[gi].append(r.home_score - r.away_score)
            totals[gi].append(r.home_score + r.away_score)
        done = min(s + batch, len(jobs))
        if (s // batch) % 10 == 0:
            rate = (time.time() - t0) / done
            print(f"  {done}/{len(jobs)} rollouts "
                  f"(~{rate * (len(jobs) - done) / 60:.0f} min left)", flush=True)
    return ([np.array(m) for m in margins], [np.array(t) for t in totals],
            truncated)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--limit", type=int, default=0, help="cap games (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-cutoff", default="2025-03-01")
    p.add_argument("--temps", default="", help="per-slot temperature JSON")
    p.add_argument("--ckpt", default=os.path.join(CACHE, "model.pt"))
    p.add_argument("--tag", default="", help="suffix for the results filename")
    args = p.parse_args()
    temps = json.loads(args.temps) if args.temps else 1.0

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    model = EventGPT(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"checkpoint from step {ckpt.get('step', '?')}, device={device}")

    with open(os.path.join(CACHE, "tokens.jsonl")) as f:
        val = [json.loads(l) for l in f]
    val = [g for g in val
           if g["game_id"].startswith("002") and g["date"] >= args.val_cutoff]
    if args.limit:
        val = val[:args.limit]
    print(f"{len(val)} games, {args.sims} sims each")

    headers = [g["tokens"][:g["tokens"].index("[START_Q]") + 11] for g in val]
    all_margins, all_totals, all_truncated = simulate_all(
        model, vocab, headers, args.sims, args.batch, device,
        seed=args.seed, temps=temps)

    rows = []
    for g, margins, totals in zip(val, all_margins, all_totals):
        toks = g["tokens"]
        real = Replay(toks).run()
        actual_margin = real.home_score - real.away_score
        rows.append({
            "date": g["date"], "game_id": g["game_id"],
            "matchup": f"{toks[1][5:]}@{toks[2][5:]}",
            "home_win_prob": smooth_win_prob(margins),
            "home_won": actual_margin > 0,
            "margin_err": float(margins.mean() - actual_margin),
            "total_err": float(totals.mean() - (real.home_score + real.away_score)),
            "margin_p10": float(np.percentile(margins, 10)),
            "margin_p90": float(np.percentile(margins, 90)),
            "actual_margin": actual_margin,
        })

    probs = np.array([r["home_win_prob"] for r in rows])
    wins = np.array([r["home_won"] for r in rows])
    picks = ((probs >= 0.5) == wins).mean()
    brier = ((probs - wins) ** 2).mean()
    clamped = np.clip(probs, 1e-6, 1 - 1e-6)
    log_loss = -(wins * np.log(clamped) + (1 - wins) * np.log(1 - clamped)).mean()
    margin_mae = np.mean([abs(r["margin_err"]) for r in rows])
    total_mae = np.mean([abs(r["total_err"]) for r in rows])
    coverage = np.mean([r["margin_p10"] <= r["actual_margin"] <= r["margin_p90"]
                        for r in rows])
    sim_sd = np.mean([m.std(ddof=1) for m in all_margins])
    real_sd = np.std([r["actual_margin"] for r in rows], ddof=1)

    print(f"\n=== TRANSFORMER BACKTEST ({len(rows)} games, {args.sims} sims) ===")
    print(f"  Pick accuracy:  {picks:.1%}")
    print(f"  Brier score:    {brier:.4f}")
    print(f"  Log loss:       {log_loss:.4f}")
    print(f"  Margin MAE:     {margin_mae:.1f} pts")
    print(f"  Total MAE:      {total_mae:.1f} pts")
    print(f"  Margin coverage (p10-p90): {coverage:.1%}  (target ~80%)")
    print(f"  Sim margin sd (per-game mean): {sim_sd:.1f}  "
          f"(real window sd {real_sd:.1f})")
    if all_truncated:
        print(f"  truncated rollouts skipped: {all_truncated}")

    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(
        HERE, "..", "results",
        f"backtest_transformer_{args.val_cutoff}_s{args.sims}{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    summary = {"n_games": len(rows), "sims": args.sims, "temps": temps,
               "protocol": "v2-smoothed", "picks": float(picks),
               "brier": float(brier), "log_loss": float(log_loss),
               "margin_mae": float(margin_mae), "total_mae": float(total_mae),
               "coverage_p10_p90": float(coverage), "sim_margin_sd": float(sim_sd),
               "real_margin_sd": float(real_sd), "truncated": all_truncated}
    with open(out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
