#!/usr/bin/env python3
"""Pre-registered counterfactual suite (#16, Gate C of the Definition of GOOD).

PRE-REGISTRATION (fixed 2026-06-12, before any results were generated).
Five counterfactuals, each a directional hypothesis tested on N dev-window
games (Jan-Feb 2025), paired per game against sims from the unmodified
header. Gate C passes this component if >= 4 of 5 show the predicted sign
with a 95% bootstrap CI (over per-game mean differences) excluding zero.

1. rest_star    — remove the home team's top season-PPG active (off roster
                  and starters; lowest-mpg non-starting active starts
                  instead).            METRIC home margin.    PREDICT down.
2. add_star     — Nikola Jokic joins the home roster and replaces the
                  lowest-PPG home starter (who stays on the bench).
                                       METRIC home margin.    PREDICT up.
3. rim_protector— Victor Wembanyama likewise replaces the lowest-PPG home
                  starter.             METRIC away rim FG% (shots D:0-3).
                                                              PREDICT down.
4. five_centers — the home starting five becomes five elite centers (Gobert,
                  Allen, Kessler, Zubac, B. Lopez), all added to the roster;
                  original starters stay rostered.
                                       METRIC home 3PA per sim. PREDICT down.
5. venue_flip   — identical matchup with home/away swapped. METRIC the
                  original home team's margin (home game) plus the flipped
                  game's margin (their road game) = twice the model's home
                  edge.                                       PREDICT > 0.

A game is skipped for a counterfactual if the injected player already plays
in it. Skips are reported, never silent.

Run from v2/:  python test_scripts/counterfactual_suite.py
               [--games 24] [--sims 25] [--batch 128]
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sim.form import season_start
from sim.games import load_games
from sim.model import Config, EventGPT, pick_device
from sim.sample import generate_games
from sim.tokenizer import Replay

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DEV_START, DEV_END = "2025-01-01", "2025-03-01"

STAR = "P:Nikola Jokic"
RIM = "P:Victor Wembanyama"
CENTERS = ["P:Rudy Gobert", "P:Jarrett Allen", "P:Walker Kessler",
           "P:Ivica Zubac", "P:Brook Lopez"]


def header_parts(toks):
    """Split a header into its blocks. Starters are away 5 then home 5."""
    ra, rh = toks.index("[ROSTER_A]"), toks.index("[ROSTER_H]")
    dt = toks.index("dt:0")
    sq = toks.index("[START_Q]")
    return {"teams": toks[1:3], "away": toks[ra + 1:rh], "home": toks[rh + 1:dt],
            "starters_away": toks[sq + 1:sq + 6], "starters_home": toks[sq + 6:sq + 11]}


def build_header(teams, away, home, st_away, st_home):
    return (["[GAME]"] + teams + ["[ROSTER_A]"] + sorted(away)
            + ["[ROSTER_H]"] + sorted(home) + ["dt:0", "[START_Q]"]
            + st_away + st_home)


def season_ppg_mpg(games, date):
    start = season_start(date)
    tot = defaultdict(lambda: {"g": 0, "pts": 0, "min": 0.0})
    for g in games:
        if not (start <= g.date < date):
            continue
        for name, line in g.players.items():
            t = tot["P:" + name]
            t["g"] += 1
            t["pts"] += line.pts
            t["min"] += line.minutes
    return {k: (v["pts"] / v["g"], v["min"] / v["g"]) for k, v in tot.items()
            if v["g"] > 0}


def shot_stats(sim, away_set, home_set):
    """(away rim attempts, away rim makes, home 3PA) from one rollout."""
    rim_att = rim_made = h3pa = 0
    for i, t in enumerate(sim):
        if not t.startswith("A:") or i == 0 or not sim[i - 1].startswith("P:"):
            continue
        shooter = sim[i - 1]
        if i + 2 < len(sim) and sim[i + 1].startswith("D:"):
            d = sim[i + 1][2:].rstrip("+")  # D:36+ caps; D:unk has no distance
            if (d.isdigit() and int(d) <= 3 and not t.startswith("A:3pt")
                    and shooter in away_set):
                rim_att += 1
                rim_made += sim[i + 2] == "made"
        if t.startswith("A:3pt") and shooter in home_set:
            h3pa += 1
    return rim_att, rim_made, h3pa


def make_counterfactuals(parts, form):
    """name -> (modified header parts, metric key) or None to skip."""
    home, st_home = parts["home"], parts["starters_home"]
    ppg = lambda p: form.get(p, (0.0, 0.0))[0]
    mpg = lambda p: form.get(p, (0.0, 0.0))[1]
    both = set(parts["away"]) | set(home)
    out = {}

    star = max(home, key=ppg)
    bench = [p for p in home if p not in st_home and p != star]
    if bench:
        new_home = [p for p in home if p != star]
        new_st = ([min(bench, key=mpg) if p == star else p for p in st_home]
                  if star in st_home else st_home)
        out["rest_star"] = dict(home=new_home, st_home=new_st, metric="margin")
    else:
        out["rest_star"] = None

    weakest = min(st_home, key=ppg)
    for name, player in (("add_star", STAR), ("rim_protector", RIM)):
        if player in both:
            out[name] = None
        else:
            out[name] = dict(home=home + [player],
                             st_home=[player if p == weakest else p for p in st_home],
                             metric="margin" if name == "add_star" else "rim")

    if any(c in both for c in CENTERS):
        out["five_centers"] = None
    else:
        out["five_centers"] = dict(home=home + CENTERS, st_home=list(CENTERS),
                                   metric="h3pa")

    out["venue_flip"] = dict(flip=True, metric="margin")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--sims", type=int, default=25)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=os.path.join(CACHE, "model.pt"))
    p.add_argument("--tag", default="", help="suffix for the results filename")
    args = p.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    for tok in [STAR, RIM] + CENTERS:
        if tok not in vocab:
            raise SystemExit(f"pre-registered player not in vocab: {tok}")
    model = EventGPT(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    games = load_games(os.path.join(CACHE, "games.jsonl"))
    with open(os.path.join(CACHE, "tokens.jsonl")) as f:
        dev = [g for g in (json.loads(l) for l in f)
               if g["game_id"].startswith("002")
               and DEV_START <= g["date"] < DEV_END]
    rng = np.random.default_rng(args.seed)
    dev = [dev[i] for i in rng.choice(len(dev), size=min(args.games, len(dev)),
                                      replace=False)]
    print(f"device={device}  games={len(dev)}  sims={args.sims}/condition",
          flush=True)

    # One flat job list: baseline + every counterfactual condition.
    jobs = []  # (game_idx, condition_name, header, away_set, home_set, flip)
    skips = defaultdict(int)
    for gi, g in enumerate(dev):
        toks = g["tokens"]
        parts = header_parts(toks[:toks.index("[START_Q]") + 11])
        base = build_header(parts["teams"], parts["away"], parts["home"],
                            parts["starters_away"], parts["starters_home"])
        jobs.append((gi, "base", base, set(parts["away"]), set(parts["home"])))
        form = season_ppg_mpg(games, g["date"])
        for name, cf in make_counterfactuals(parts, form).items():
            if cf is None:
                skips[name] += 1
                continue
            if cf.get("flip"):
                hdr = build_header(parts["teams"][::-1], parts["home"],
                                   parts["away"], parts["starters_home"],
                                   parts["starters_away"])
                jobs.append((gi, name, hdr, set(parts["home"]),
                             set(parts["away"])))
            else:
                hdr = build_header(parts["teams"], parts["away"], cf["home"],
                                   parts["starters_away"], cf["st_home"])
                jobs.append((gi, name, hdr, set(parts["away"]),
                             set(cf["home"])))
    flat = [(j, k) for j in jobs for k in range(args.sims)]
    print(f"{len(flat)} rollouts ({len(jobs)} conditions); skips: {dict(skips)}",
          flush=True)

    stats = defaultdict(lambda: defaultdict(list))  # cond -> game -> per-sim
    t0 = time.time()
    for s in range(0, len(flat), args.batch):
        chunk = flat[s:s + args.batch]
        sims = generate_games(model, vocab, [j[2] for j, _ in chunk], device,
                              seed=args.seed + s, temperature=1.0)
        for (gi, cond, _, away_set, home_set), sim in zip([j for j, _ in chunk],
                                                          sims):
            if sim[-1] != "[EOG]":
                continue
            r = Replay(sim).run()
            ra, rm, h3 = shot_stats(sim, away_set, home_set)
            stats[cond][gi].append({
                "margin": r.home_score - r.away_score,
                "rim_att": ra, "rim_made": rm, "h3pa": h3})
        done = min(s + args.batch, len(flat))
        if (s // args.batch) % 5 == 0:
            rate = (time.time() - t0) / done
            print(f"  {done}/{len(flat)} rollouts "
                  f"(~{rate * (len(flat) - done) / 60:.0f} min left)", flush=True)

    def metric(cond, gi, key):
        rows = stats[cond][gi]
        if not rows:
            return None
        if key == "rim":
            att = sum(r["rim_att"] for r in rows)
            return sum(r["rim_made"] for r in rows) / att if att else None
        return float(np.mean([r[key] for r in rows]))

    PREDICT = {"rest_star": ("margin", -1), "add_star": ("margin", +1),
               "rim_protector": ("rim", -1), "five_centers": ("h3pa", -1),
               "venue_flip": ("margin", +1)}
    boot = np.random.default_rng(args.seed + 1)
    results, passed = {}, 0
    print()
    for name, (key, sign) in PREDICT.items():
        diffs = []
        for gi in stats[name]:
            b, c = metric("base", gi, key), metric(name, gi, key)
            if b is None or c is None:
                continue
            # venue_flip metric is base margin PLUS flipped margin (= 2x HCA)
            diffs.append(b + c if name == "venue_flip" else c - b)
        diffs = np.array(diffs)
        bs = np.array([boot.choice(diffs, size=len(diffs)).mean()
                       for _ in range(10_000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        mean = float(diffs.mean())
        ok = (mean * sign > 0) and not (lo <= 0 <= hi)
        passed += ok
        results[name] = {"n_games": len(diffs), "metric": key,
                         "predicted_sign": sign, "mean_diff": round(mean, 3),
                         "ci95": [round(float(lo), 3), round(float(hi), 3)],
                         "pass": bool(ok)}
        print(f"{name:14s} n={len(diffs):2d}  mean {mean:+.3f}  "
              f"CI [{lo:+.3f}, {hi:+.3f}]  predict "
              f"{'+' if sign > 0 else '-'}  {'PASS' if ok else 'FAIL'}",
              flush=True)

    print(f"\n{passed}/5 passed (gate needs >= 4)")
    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(HERE, "..", "results", f"counterfactual_suite{tag}.json")
    with open(out, "w") as f:
        json.dump({"games": len(dev), "sims": args.sims, "skips": dict(skips),
                   "passed": passed, "results": results}, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
